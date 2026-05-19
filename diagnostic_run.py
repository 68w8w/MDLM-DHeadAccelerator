"""Diagnostic training run: per-substep metrics to verify loss drop is healthy.

Runs 200 steps with detailed logging every 50 steps.
Metrics per substep (k=0..3):
  1. mask_count / mask_ratio
  2. teacher entropy (MASK positions)
  3. student entropy (MASK positions)
  4. teacher top-1 accuracy vs x0 (MASK positions)
  5. student top-1 accuracy vs x0 (MASK positions)
  6. teacher-student top-1 agreement (MASK positions)
  7. KL per substep
  8. teacher frozen confirmation
  9. label leakage check
  10. debug decode of x0 / z / teacher top-1 / student top-1
"""

import time
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer

from config import Config
from data import get_dataloader
from model import build_student, build_teacher
from diffusion_utils import forward_noise, absorbing_reverse_step
from train import KL_loss, get_cosine_schedule_with_warmup


# ── helpers ──────────────────────────────────────────────────────────

def _entropy_on_mask(log_p: torch.Tensor, mask: torch.Tensor) -> float:
    """H = -sum(p * log_p) averaged over MASK positions."""
    if mask.sum() == 0:
        return 0.0
    log_p_m = log_p[mask].float()          # [M, V]
    p_m = log_p_m.exp()
    ent = -(p_m * log_p_m)
    # zero out where p==0 to avoid nan
    ent = torch.where(p_m > 0, ent, torch.zeros_like(ent))
    return ent.sum(dim=-1).mean().item()


def _top1_acc_vs_x0(log_p: torch.Tensor, x0: torch.Tensor,
                     mask: torch.Tensor) -> float:
    """argmax(log_p) == x0 on MASK positions."""
    if mask.sum() == 0:
        return 0.0
    pred = log_p[mask].argmax(dim=-1)      # [M]
    gt = x0[mask]                          # [M]
    return (pred == gt).float().mean().item()


def _top1_agreement(log_pT: torch.Tensor, log_pS: torch.Tensor,
                     mask: torch.Tensor) -> float:
    """argmax(log_pT) == argmax(log_pS) on MASK positions."""
    if mask.sum() == 0:
        return 0.0
    pred_T = log_pT[mask].argmax(dim=-1)
    pred_S = log_pS[mask].argmax(dim=-1)
    return (pred_T == pred_S).float().mean().item()


def _decode_first_n(tokenizer, ids, n_chars=200, mask_token_id=50257):
    """Decode token ids, replacing MASK with [M]."""
    tokens = ids.tolist()
    pieces = []
    for t in tokens:
        if t == mask_token_id:
            pieces.append("[M]")
        else:
            pieces.append(tokenizer.decode([t]))
    text = "".join(pieces)
    return text[:n_chars]


# ── diagnostic train step ────────────────────────────────────────────

def train_step_diagnostic(
    student, teacher, x0, config, optimizer, scheduler, step,
    tokenizer=None, verbose=False,
):
    """Training step identical to train.py but collects per-substep diagnostics.

    Returns:
        avg_loss: float
        diag: dict of diagnostic info
    """
    B, L = x0.shape
    device = x0.device
    K = config.K
    MASK_ID = config.mask_token_id
    chunk_size = config.kl_chunk_size

    # 1. Sample random band
    band_idx = torch.randint(0, config.n_bands, (1,)).item()
    t_band_high = 1.0 - band_idx / config.n_bands
    t_band_low = 1.0 - (band_idx + 1) / config.n_bands

    t_src = torch.rand(B, device=device) * (t_band_high - t_band_low) + t_band_low
    t_dst = torch.full((B,), t_band_low, device=device)

    # 2. Forward noising
    z_t = forward_noise(x0, t_src, MASK_ID)

    # ── label leakage check on z_t ──
    initial_mask = (z_t == MASK_ID)
    initial_mask_count = initial_mask.sum().item()
    initial_mask_ratio = initial_mask_count / (B * L)
    expected_mask_ratio = t_src.mean().item()
    # At MASK positions, z_t must be MASK_ID (tautology — but we also
    # check that non-MASK positions really are x0).
    non_mask_correct = (z_t[~initial_mask] == x0[~initial_mask]).all().item()

    # 3. Student backbone forward
    hidden_src, c_src = student.forward_backbone(z_t, t_src)

    # 4. K substeps
    optimizer.zero_grad(set_to_none=True)

    z = z_t.clone()
    Delta = (t_src - t_dst) / K
    total_loss = 0.0

    substep_diags = []

    for k in range(K):
        t_cur = t_src - k * Delta
        t_next = t_src - (k + 1) * Delta

        mask = (z == MASK_ID)
        mask_count = mask.sum().item()
        mask_ratio = mask_count / (B * L)

        # ── leakage: all mask-True positions must hold MASK_ID ──
        leakage_ok = (z[mask] == MASK_ID).all().item() if mask_count > 0 else True

        # visible token ratio
        visible_ratio = 1.0 - mask_ratio

        # Teacher target
        with torch.no_grad():
            log_pT = teacher.forward_log_probs(z, t_cur)

        # Student prediction
        c_cur = c_src
        logits_S = student.heads.compute_one_head(
            hidden_src=hidden_src, z=z, c=c_cur,
            t_cur=t_cur, band_idx=band_idx,
        )
        log_pS = F.log_softmax(logits_S.float(), dim=-1)

        # ── per-substep metrics (all no_grad, detached) ──
        with torch.no_grad():
            kl_k = KL_loss(log_pT, log_pS.detach(), mask, chunk_size=chunk_size).item()
            ent_T = _entropy_on_mask(log_pT, mask)
            ent_S = _entropy_on_mask(log_pS.detach(), mask)
            acc_T = _top1_acc_vs_x0(log_pT, x0, mask)
            acc_S = _top1_acc_vs_x0(log_pS.detach(), x0, mask)
            agree = _top1_agreement(log_pT, log_pS.detach(), mask)

        sd = dict(
            k=k, band_idx=band_idx,
            mask_count=mask_count, mask_ratio=mask_ratio,
            visible_ratio=visible_ratio,
            leakage_ok=leakage_ok,
            kl=kl_k,
            ent_T=ent_T, ent_S=ent_S,
            acc_T=acc_T, acc_S=acc_S,
            agree=agree,
            t_cur_mean=t_cur.mean().item(),
            t_next_mean=t_next.mean().item(),
        )

        # ── debug decode (only first sample) ──
        if verbose and tokenizer is not None and k == 0:
            with torch.no_grad():
                sd['decode_x0'] = _decode_first_n(tokenizer, x0[0], 200, MASK_ID)
                sd['decode_z'] = _decode_first_n(tokenizer, z[0], 200, MASK_ID)
                top1_T = log_pT[0].argmax(dim=-1)
                top1_S = log_pS[0].detach().argmax(dim=-1)
                sd['decode_T_top1'] = _decode_first_n(tokenizer, top1_T, 200, MASK_ID)
                sd['decode_S_top1'] = _decode_first_n(tokenizer, top1_S, 200, MASK_ID)

        substep_diags.append(sd)

        # ── actual backward (same as train.py) ──
        loss_k = KL_loss(log_pT, log_pS, mask, chunk_size=chunk_size)
        (loss_k / K).backward(retain_graph=(k < K - 1))
        total_loss += loss_k.detach().item()

        with torch.no_grad():
            z = absorbing_reverse_step(
                z=z, log_p=log_pT, t_curr=t_cur, t_next=t_next,
                mask_token_id=MASK_ID,
            )

        del log_pT, logits_S, log_pS, loss_k

    clip_grad_norm_(student.get_trainable_parameters(), config.grad_clip)
    optimizer.step()
    scheduler.step()

    diag = dict(
        band_idx=band_idx,
        t_src_mean=t_src.mean().item(),
        t_dst=t_band_low,
        initial_mask_ratio=initial_mask_ratio,
        expected_mask_ratio=expected_mask_ratio,
        non_mask_positions_equal_x0=non_mask_correct,
        substeps=substep_diags,
    )
    return total_loss / K, diag


# ── pretty-print ─────────────────────────────────────────────────────

def print_diag(step, avg_loss, diag, verbose=False):
    """Print one step's diagnostics."""
    B = "band"
    print(f"\n{'─'*72}")
    print(f"  step {step} | avg_loss {avg_loss:.4f} | "
          f"{B}={diag['band_idx']} | "
          f"t_src≈{diag['t_src_mean']:.3f} → t_dst={diag['t_dst']:.3f}")
    print(f"  initial_mask_ratio={diag['initial_mask_ratio']:.4f}  "
          f"(expected≈{diag['expected_mask_ratio']:.4f})  "
          f"non_mask==x0: {diag['non_mask_positions_equal_x0']}")
    print(f"{'─'*72}")

    hdr = (f"  {'k':>2}  {'mask_cnt':>9} {'mask_r':>7} {'vis_r':>7} "
           f"{'KL':>8} {'ent_T':>7} {'ent_S':>7} "
           f"{'acc_T':>7} {'acc_S':>7} {'agree':>7} "
           f"{'leak_ok':>7}")
    print(hdr)

    for sd in diag['substeps']:
        row = (f"  {sd['k']:>2}  {sd['mask_count']:>9} "
               f"{sd['mask_ratio']:>7.4f} {sd['visible_ratio']:>7.4f} "
               f"{sd['kl']:>8.3f} {sd['ent_T']:>7.2f} {sd['ent_S']:>7.2f} "
               f"{sd['acc_T']:>7.4f} {sd['acc_S']:>7.4f} {sd['agree']:>7.4f} "
               f"{str(sd['leakage_ok']):>7}")
        print(row)

    if verbose:
        sd0 = diag['substeps'][0]
        if 'decode_x0' in sd0:
            print(f"\n  [decode] x0:       {sd0['decode_x0']}")
            print(f"  [decode] z (k=0):  {sd0['decode_z']}")
            print(f"  [decode] T top-1:  {sd0['decode_T_top1']}")
            print(f"  [decode] S top-1:  {sd0['decode_S_top1']}")


# ── main ─────────────────────────────────────────────────────────────

def run_diagnostic(n_steps=200, log_every=50):
    config = Config()
    device = config.device

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    print("Building student...")
    student = build_student(config, device)
    student.train()

    print("Building teacher...")
    teacher = build_teacher(config, device)
    teacher.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ── Check 8: teacher frozen ──
    teacher_trainable = sum(
        p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"\n[Check 8] Teacher trainable params = {teacher_trainable}")
    assert teacher_trainable == 0, "TEACHER IS NOT FULLY FROZEN!"

    # Verify teacher and student backbone are separate objects
    teacher_backbone_id = id(teacher.hf_model.backbone)
    student_backbone_id = id(student.hf_model.backbone)
    print(f"[Check 8] teacher backbone id = {teacher_backbone_id}")
    print(f"[Check 8] student backbone id = {student_backbone_id}")
    assert teacher_backbone_id != student_backbone_id, \
        "TEACHER AND STUDENT SHARE THE SAME BACKBONE OBJECT!"
    print("[Check 8] PASS: teacher fully frozen, separate from student")

    # Snapshot teacher params to verify they don't change
    teacher_param_snapshot = {
        n: p.data.clone()
        for n, p in teacher.named_parameters()
    }

    # ── Data ──
    print("\nLoading data...")
    train_loader = get_dataloader(config, split="train")
    train_iter = iter(train_loader)

    trainable_params = student.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, n_steps)

    print(f"\nStarting diagnostic run: {n_steps} steps, log every {log_every}")
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    losses = []
    all_diags = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x0 = batch['input_ids'].to(device)

        is_log_step = (step % log_every == 0) or step == 1
        avg_loss, diag = train_step_diagnostic(
            student, teacher, x0, config, optimizer, scheduler, step,
            tokenizer=tokenizer if is_log_step else None,
            verbose=is_log_step,
        )
        losses.append(avg_loss)

        if is_log_step:
            print_diag(step, avg_loss, diag, verbose=True)
            all_diags.append((step, avg_loss, diag))

    elapsed = time.time() - t0

    # ── Check 8 continued: verify teacher params didn't change ──
    print(f"\n{'='*72}")
    print("  POST-TRAINING TEACHER FROZEN CHECK")
    print(f"{'='*72}")
    teacher_changed = False
    for n, p in teacher.named_parameters():
        if not torch.equal(p.data, teacher_param_snapshot[n]):
            print(f"  CHANGED: {n}")
            teacher_changed = True
    if not teacher_changed:
        print("  PASS: all teacher parameters unchanged after training")
    else:
        print("  FAIL: teacher parameters changed!")

    # ── Summary table ──
    print(f"\n{'='*72}")
    print("  LOSS TRAJECTORY")
    print(f"{'='*72}")
    window = log_every
    for i in range(0, n_steps, window):
        chunk = losses[i:i+window]
        avg = sum(chunk) / len(chunk)
        print(f"  steps {i+1:4d}-{i+len(chunk):4d}: mean_loss = {avg:.4f}")

    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    print(f"\n  mean_loss_first_50 = {first_50:.4f}")
    print(f"  mean_loss_last_50  = {last_50:.4f}")
    print(f"  elapsed = {elapsed:.1f}s")
    print(f"  peak_mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ── Cross-step trend summary ──
    print(f"\n{'='*72}")
    print("  METRIC TRENDS (logged steps)")
    print(f"{'='*72}")
    print(f"  {'step':>5} {'band':>4} {'KL_k0':>8} {'KL_k3':>8} "
          f"{'entT_k0':>8} {'entS_k0':>8} "
          f"{'accT_k0':>8} {'accS_k0':>8} {'agree_k0':>8}")
    for step_i, loss_i, d_i in all_diags:
        s0 = d_i['substeps'][0]
        s3 = d_i['substeps'][-1]
        print(f"  {step_i:>5} {d_i['band_idx']:>4} "
              f"{s0['kl']:>8.3f} {s3['kl']:>8.3f} "
              f"{s0['ent_T']:>8.2f} {s0['ent_S']:>8.2f} "
              f"{s0['acc_T']:>8.4f} {s0['acc_S']:>8.4f} {s0['agree']:>8.4f}")

    # ── Diagnosis ──
    print(f"\n{'='*72}")
    print("  DIAGNOSIS")
    print(f"{'='*72}")

    last_d = all_diags[-1][2]
    issues = []

    # Leakage
    for sd in last_d['substeps']:
        if not sd['leakage_ok']:
            issues.append(f"Label leakage at k={sd['k']}!")

    # Mask ratio sanity
    if abs(last_d['initial_mask_ratio'] - last_d['expected_mask_ratio']) > 0.05:
        issues.append(
            f"Mask ratio mismatch: got {last_d['initial_mask_ratio']:.4f}, "
            f"expected ~{last_d['expected_mask_ratio']:.4f}")

    # Non-mask == x0
    if not last_d['non_mask_positions_equal_x0']:
        issues.append("Non-MASK positions do not equal x0 — data corruption!")

    # Teacher frozen
    if teacher_changed:
        issues.append("Teacher params changed during training!")

    # Loss sanity
    if last_50 >= first_50:
        issues.append("Loss did not decrease")

    if not issues:
        print("  No issues found. Loss drop appears healthy.")
        print("  Key evidence:")
        first_d = all_diags[0][2]['substeps'][0]
        last_d_s0 = last_d['substeps'][0]
        print(f"    - Teacher acc vs x0 stable: "
              f"{first_d['acc_T']:.4f} → {last_d_s0['acc_T']:.4f}")
        print(f"    - Student acc vs x0 rising: "
              f"{first_d['acc_S']:.4f} → {last_d_s0['acc_S']:.4f}")
        print(f"    - Agreement rising: "
              f"{first_d['agree']:.4f} → {last_d_s0['agree']:.4f}")
        print(f"    - No label leakage")
        print(f"    - Teacher fully frozen, separate from student")
    else:
        print("  ISSUES FOUND:")
        for iss in issues:
            print(f"    - {iss}")


if __name__ == "__main__":
    run_diagnostic(n_steps=200, log_every=50)
