#!/usr/bin/env python3
"""STM (Scheduled Trajectory Mixing) training for MDLM D-Head Accelerator.

Combines Residual D-Head with STM to address teacher-forced / free-running
mismatch. Loss: forward KL only. Output layer frozen.
No warm-start from residual checkpoint.
"""

import os
import time
import math
import argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from config import Config
from data import get_dataloader
from model import build_student, build_teacher
from diffusion_utils import forward_noise, absorbing_reverse_step
from train import KL_loss, get_cosine_schedule_with_warmup
from inference import generate_samples


# ── STM Utilities ───────────────────────────────────────────────────

def random_partition(total, N, min_frac=0.05):
    """Partition total into N random segments via Dirichlet.

    Args:
        total: [B] total amount to partition
        N: number of segments
        min_frac: minimum fraction — each segment >= min_frac/N * total[b]

    Returns:
        intervals: [B, N], each row sums to total[b]
    """
    B = total.shape[0]
    device = total.device
    dtype = total.dtype

    # Dirichlet(1,...,1) via exponential
    exp_samples = torch.empty(B, N, device=device, dtype=dtype).exponential_(1.0)
    weights = exp_samples / exp_samples.sum(dim=1, keepdim=True)  # [B, N]

    # Enforce minimum fraction per segment
    floor = min_frac / N
    weights = floor + (1.0 - min_frac) * weights  # still sums to 1.0

    # Scale by total
    intervals = weights * total[:, None]

    # total=0 → all zeros
    intervals = intervals.masked_fill((total == 0)[:, None], 0.0)

    return intervals


def get_tau(step, tau_min, warmup_frac, decay_frac, tau_total_steps):
    """Compute tau at given step.

    Schedule:
        [0, warmup_end):  tau = 1.0
        [warmup_end, decay_end):  cosine decay  1.0 → tau_min
        [decay_end, ...):  tau = tau_min
    """
    warmup_end = int(warmup_frac * tau_total_steps)
    decay_end = warmup_end + int(decay_frac * tau_total_steps)

    if step < warmup_end:
        return 1.0
    elif step < decay_end:
        progress = (step - warmup_end) / max(1, decay_end - warmup_end)
        return tau_min + 0.5 * (1.0 - tau_min) * (1.0 + math.cos(math.pi * progress))
    else:
        return tau_min


# ── STM Train Step ──────────────────────────────────────────────────

def train_step_stm(student, teacher, x0, config, optimizer, scheduler,
                   step, tau):
    """One STM training step.

    Returns dict: loss, band_idx, tau, student_total, teacher_total,
                  mean_student_interval, mean_teacher_interval
    """
    B = x0.shape[0]
    device = x0.device
    N = config.num_intermediate_states
    MASK_ID = config.mask_token_id
    chunk_size = config.kl_chunk_size
    n_bands = config.n_bands

    # 1. Sample band
    band_idx = torch.randint(0, n_bands, (1,)).item()
    t_band_high = 1.0 - band_idx / n_bands
    t_band_low = 1.0 - (band_idx + 1) / n_bands

    # t_src with endpoint bias
    endpoint_prob = 0.5 if band_idx == 0 else 0.3
    use_endpoint = torch.rand(B, device=device) < endpoint_prob
    t_uniform = (torch.rand(B, device=device)
                 * (t_band_high - t_band_low) + t_band_low)
    t_endpoint = torch.full((B,), t_band_high, device=device)
    t_src = torch.where(use_endpoint, t_endpoint, t_uniform)
    t_dst = torch.full((B,), t_band_low, device=device)

    # 2. Forward noise
    z = forward_noise(x0, t_src, MASK_ID)

    # 3. Backbone forward once
    hidden_src, c_src = student.forward_backbone(z, t_src)

    # 4. Compute intervals
    Delta = t_src - t_dst                         # [B]
    student_total = (1.0 - tau) * Delta           # [B]
    teacher_total = tau * Delta                   # [B]

    student_intervals = random_partition(
        student_total, N, config.partition_min_frac)   # [B, N]
    teacher_intervals = random_partition(
        teacher_total, N, config.partition_min_frac)   # [B, N]

    # 5. STM loop
    optimizer.zero_grad(set_to_none=True)

    raw_t = t_src.clone()
    total_loss_value = 0.0

    for i in range(N):
        # A. Student rollout to anchor (no_grad)
        t_a = (raw_t - student_intervals[:, i]).clamp(min=0)
        t_a = torch.minimum(t_a, raw_t)

        with torch.no_grad():
            logits_stu = student.heads.compute_one_head(
                hidden_src=hidden_src, z=z, c=c_src,
                t_cur=raw_t, band_idx=band_idx)
            log_p_stu = F.log_softmax(logits_stu.float(), dim=-1)
            z_anchor = absorbing_reverse_step(
                z=z, log_p=log_p_stu,
                t_curr=raw_t, t_next=t_a,
                mask_token_id=MASK_ID)
            del logits_stu, log_p_stu

        # B. Teacher forward at anchor (no_grad)
        with torch.no_grad():
            log_pT = teacher.forward_log_probs(z_anchor, t_a)

        # C. Student query at anchor (WITH grad)
        logits_S = student.heads.compute_one_head(
            hidden_src=hidden_src, z=z_anchor, c=c_src,
            t_cur=t_a, band_idx=band_idx)
        log_pS = F.log_softmax(logits_S.float(), dim=-1)

        # D. Loss: KL(pT || pS) on MASK positions, immediate backward
        mask = (z_anchor == MASK_ID)
        loss_i = KL_loss(log_pT, log_pS, mask, chunk_size=chunk_size)
        (loss_i / N).backward(retain_graph=(i < N - 1))

        total_loss_value += loss_i.detach().item()

        # E. Teacher rollout (no_grad)
        t_b = (t_a - teacher_intervals[:, i]).clamp(min=0)
        t_b = torch.minimum(t_b, t_a)
        with torch.no_grad():
            z = absorbing_reverse_step(
                z=z_anchor, log_p=log_pT,
                t_curr=t_a, t_next=t_b,
                mask_token_id=MASK_ID)

        raw_t = t_b
        del logits_S, log_pS, log_pT, loss_i

    # Final time error: raw_t should equal t_dst
    final_t_error = (raw_t - t_dst).abs().max().item()

    clip_grad_norm_(student.get_trainable_parameters(), config.grad_clip)
    optimizer.step()
    scheduler.step()

    return {
        'loss': total_loss_value / N,
        'band_idx': band_idx,
        'tau': tau,
        'student_total': student_total.mean().item(),
        'teacher_total': teacher_total.mean().item(),
        'mean_student_interval': student_intervals.mean().item(),
        'mean_teacher_interval': teacher_intervals.mean().item(),
        'final_t_error': final_t_error,
    }


# ── Diagnostics ─────────────────────────────────────────────────────

def _entropy(log_p, mask):
    if mask.sum() == 0:
        return 0.0
    lp = log_p[mask].float()
    p = lp.exp()
    e = -(p * lp)
    e = torch.where(p > 0, e, torch.zeros_like(e))
    return e.sum(-1).mean().item()


def _acc(log_p, x0, mask):
    if mask.sum() == 0:
        return 0.0
    return (log_p[mask].argmax(-1) == x0[mask]).float().mean().item()


def _agree(log_pT, log_pS, mask):
    if mask.sum() == 0:
        return 0.0
    return (log_pT[mask].argmax(-1)
            == log_pS[mask].argmax(-1)).float().mean().item()


def diag_all_bands(student, teacher, x0, config):
    """Per-band KL / entropy / accuracy / agreement."""
    MASK_ID = config.mask_token_id
    device = x0.device
    was_training = student.training
    student.eval()
    rows = []
    for b in range(4):
        t_mid = 1.0 - (b + 0.5) / 4
        t = torch.full((x0.shape[0],), t_mid, device=device)
        z = forward_noise(x0, t, MASK_ID)
        mask = (z == MASK_ID)
        with torch.no_grad():
            h, c = student.forward_backbone(z, t)
            lpT = teacher.forward_log_probs(z, t)
            lgS = student.heads.compute_one_head(
                hidden_src=h, z=z, c=c, t_cur=t, band_idx=b)
            lpS = F.log_softmax(lgS.float(), dim=-1)
            kl = KL_loss(lpT, lpS, mask, chunk_size=512).item()
        rows.append(dict(
            band=b, kl=kl,
            ent_T=_entropy(lpT, mask), ent_S=_entropy(lpS, mask),
            acc_T=_acc(lpT, x0, mask), acc_S=_acc(lpS, x0, mask),
            agree=_agree(lpT, lpS, mask),
        ))
    if was_training:
        student.train()
    return rows


def print_diag_table(diags, header=""):
    if header:
        print(f"\n  {header}")
    print(f"  {'band':>4} {'KL':>8} {'ent_T':>7} {'ent_S':>7} "
          f"{'acc_T':>7} {'acc_S':>7} {'agree':>7}")
    for d in diags:
        print(f"  {d['band']:>4} {d['kl']:>8.3f} {d['ent_T']:>7.2f} "
              f"{d['ent_S']:>7.2f} {d['acc_T']:>7.4f} "
              f"{d['acc_S']:>7.4f} {d['agree']:>7.4f}")


# ── On-policy STM diagnostics ───────────────────────────────────────

def _entropy_chunked(log_p, mask, chunk_size=512):
    """Memory-efficient entropy: process masked positions in chunks."""
    n = mask.sum().item()
    if n == 0:
        return 0.0
    lp_m = log_p[mask].float()  # [M, V]
    total = 0.0
    for s in range(0, n, chunk_size):
        lp_c = lp_m[s:s + chunk_size]
        p_c = lp_c.exp()
        e_c = -(p_c * lp_c)
        e_c = torch.where(p_c > 0, e_c, torch.zeros_like(e_c))
        total += e_c.sum(-1).sum().item()
        del p_c, e_c
    del lp_m
    return total / n


def on_policy_stm_diag(student, teacher, x0, config, tau):
    """Run full STM loop (no_grad) per band, report per-substep metrics.

    Memory-optimized: chunked entropy, aggressive del, empty_cache between bands.
    """
    MASK_ID = config.mask_token_id
    N = config.num_intermediate_states
    device = x0.device
    was_training = student.training
    student.eval()

    torch.cuda.empty_cache()

    results = []
    for band_idx in range(4):
        t_band_high = 1.0 - band_idx / 4
        t_band_low = 1.0 - (band_idx + 1) / 4
        B = x0.shape[0]
        t_src = torch.full((B,), t_band_high, device=device)
        t_dst = torch.full((B,), t_band_low, device=device)

        z = forward_noise(x0, t_src, MASK_ID)

        with torch.no_grad():
            hidden_src, c_src = student.forward_backbone(z, t_src)

            Delta = t_src - t_dst
            stu_total = (1.0 - tau) * Delta
            tea_total = tau * Delta
            stu_iv = random_partition(stu_total, N, config.partition_min_frac)
            tea_iv = random_partition(tea_total, N, config.partition_min_frac)

            raw_t = t_src.clone()
            substeps = []

            for i in range(N):
                t_a = (raw_t - stu_iv[:, i]).clamp(min=0)
                t_a = torch.minimum(t_a, raw_t)

                # Student rollout → z_anchor, free logits immediately
                lgs = student.heads.compute_one_head(
                    hidden_src=hidden_src, z=z, c=c_src,
                    t_cur=raw_t, band_idx=band_idx)
                lps = F.log_softmax(lgs.float(), dim=-1)
                del lgs
                z_anchor = absorbing_reverse_step(
                    z, lps, raw_t, t_a, MASK_ID)
                del lps

                # Teacher at anchor
                lpT = teacher.forward_log_probs(z_anchor, t_a)

                # Student at anchor
                lgS = student.heads.compute_one_head(
                    hidden_src=hidden_src, z=z_anchor, c=c_src,
                    t_cur=t_a, band_idx=band_idx)
                lpS = F.log_softmax(lgS.float(), dim=-1)
                del lgS

                mask = (z_anchor == MASK_ID)
                mr = mask.float().mean().item()

                # Metrics that need both lpT and lpS
                kl = KL_loss(lpT, lpS, mask, chunk_size=512).item()
                ag = _agree(lpT, lpS, mask)

                # Student-only metrics, then free lpS
                eS = _entropy_chunked(lpS, mask)
                aS = _acc(lpS, x0, mask)
                del lpS

                # Teacher-only metrics (lpT still needed for rollout)
                eT = _entropy_chunked(lpT, mask)
                aT = _acc(lpT, x0, mask)

                substeps.append(dict(
                    i=i, kl=kl, mask_ratio=mr,
                    ent_T=eT, ent_S=eS,
                    acc_T=aT, acc_S=aS,
                    agree=ag, t_a=t_a.mean().item()))

                # Teacher rollout, then free lpT
                t_b = (t_a - tea_iv[:, i]).clamp(min=0)
                t_b = torch.minimum(t_b, t_a)
                z = absorbing_reverse_step(
                    z_anchor, lpT, t_a, t_b, MASK_ID)
                del lpT
                raw_t = t_b

            del hidden_src, c_src

        ft_err = (raw_t - t_dst).abs().max().item()
        fm_ratio = (z == MASK_ID).float().mean().item()
        results.append(dict(
            band=band_idx, substeps=substeps,
            final_t_error=ft_err, final_mask_ratio=fm_ratio))

        torch.cuda.empty_cache()

    if was_training:
        student.train()
    return results


def print_on_policy_diag(results, header=""):
    if header:
        print(f"\n  {header}")
    for r in results:
        b = r['band']
        print(f"  band {b}: final_mask={r['final_mask_ratio']:.4f}  "
              f"t_error={r['final_t_error']:.2e}")
        print(f"    {'sub':>3} {'KL':>7} {'ent_T':>6} {'ent_S':>6} "
              f"{'accT':>6} {'accS':>6} {'agree':>6} "
              f"{'mask':>6} {'t_a':>6}")
        for s in r['substeps']:
            print(f"    {s['i']:>3} {s['kl']:>7.3f} "
                  f"{s['ent_T']:>6.2f} {s['ent_S']:>6.2f} "
                  f"{s['acc_T']:>6.3f} {s['acc_S']:>6.3f} "
                  f"{s['agree']:>6.3f} "
                  f"{s['mask_ratio']:>6.3f} {s['t_a']:>6.3f}")


# ── Sanity check (inline, quick) ───────────────────────────────────

def sanity_check_residual(student, config, device):
    """At init, delta_proj=0 → DHead logits == output_layer(hidden_src, c)."""
    print(f"\n{'='*72}")
    print(f"  SANITY CHECK: residual init")
    print(f"{'='*72}")

    student.eval()
    B, L = 2, config.max_length
    z = torch.randint(0, config.vocab_size, (B, L), device=device)
    t = torch.full((B,), 0.5, device=device)

    with torch.no_grad():
        h, c = student.forward_backbone(z, t)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits_ref = student.heads._output_layer(h, c)
        logits_ref = logits_ref.float()

        logits_dh = student.heads.compute_one_head(
            hidden_src=h, z=z, c=c, t_cur=t, band_idx=0)

        mc = config.mask_token_id
        ref = torch.cat([logits_ref[..., :mc],
                         logits_ref[..., mc + 1:]], dim=-1)
        dh = torch.cat([logits_dh[..., :mc],
                        logits_dh[..., mc + 1:]], dim=-1)

    max_err = (ref - dh).abs().max().item()
    ds = student.heads.delta_scale.data.tolist()
    print(f"  delta_scale init: {ds}")
    print(f"  max_abs(logits_ref - logits_dhead) = {max_err:.2e}")
    ok = max_err < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'} (threshold 1e-3)")
    return ok


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="STM training")
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--tau_total_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs_stm")
    args = parser.parse_args()

    config = Config()
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.seed = args.seed
    config.max_length = 1024
    config.n_bands = 4
    config.K = 4
    config.num_intermediate_states = 4
    config.warmup_steps = 1000
    config.total_steps = args.max_steps
    config.tau_total_steps = args.tau_total_steps
    config.tau_min = 0.1
    config.tau_warmup_frac = 0.05
    config.tau_decay_frac = 0.45
    config.partition_min_frac = 0.05
    config.beta1 = 0.9
    config.beta2 = 0.95
    config.weight_decay = 0.0
    config.grad_clip = 1.0
    config.backbone_lora_rank = 128
    config.kl_chunk_size = 512
    config.dropout = 0.0
    config.log_every = 50
    config.save_every = 500
    config.sample_every = 500
    config.num_sample_texts = 4
    config.output_dir_stm = args.output_dir

    max_steps = args.max_steps
    device = config.device
    DIAG_EVERY = 500
    SAMPLE_CHARS = 500

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"{'='*72}")
    print(f"  STM TRAINING — {max_steps} steps, "
          f"tau_total={config.tau_total_steps}")
    print(f"{'='*72}")

    print("Building student...")
    student = build_student(config, device)

    print("Building teacher...")
    teacher = build_teacher(config, device)
    teacher.eval()

    # ── Sanity check ──
    ok = sanity_check_residual(student, config, device)
    if not ok:
        print("SANITY CHECK FAILED. Aborting.")
        return

    # ── Parameter count ──
    trainable_params = student.get_trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n  Total trainable: {n_trainable:,}")

    # ── Data & optimizer ──
    print("\nLoading data...")
    train_loader = get_dataloader(config, split="train")
    train_iter = iter(train_loader)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, max_steps)

    os.makedirs(config.output_dir_stm, exist_ok=True)

    # ── Training ──
    student.train()
    losses = []
    step_metrics = []
    t0 = time.time()
    step_t0 = time.time()

    print(f"\nStarting STM training ({max_steps} steps)...\n")

    abort = False
    for step in range(1, max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x0 = batch['input_ids'].to(device)

        tau = get_tau(step, config.tau_min, config.tau_warmup_frac,
                      config.tau_decay_frac, config.tau_total_steps)

        # ── Train step with OOM guard ──
        try:
            metrics = train_step_stm(
                student, teacher, x0, config, optimizer, scheduler,
                step, tau)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"FATAL: OOM at step {step}. Aborting.")
                torch.cuda.empty_cache()
                abort = True
                break
            raise

        losses.append(metrics['loss'])
        step_metrics.append(metrics)

        # ── Auto-stop: NaN / Inf ──
        if metrics['loss'] != metrics['loss']:
            print(f"FATAL: NaN loss at step {step}. Aborting.")
            abort = True
            break
        if abs(metrics['loss']) > 1e10:
            print(f"FATAL: Inf-like loss={metrics['loss']:.2e} "
                  f"at step {step}. Aborting.")
            abort = True
            break

        # ── Auto-stop: final_t_error ──
        if metrics['final_t_error'] > 1e-3:
            print(f"FATAL: final_t_error={metrics['final_t_error']:.4e} "
                  f"> 1e-3 at step {step}. Aborting.")
            abort = True
            break

        # ── Regular log (every 50 steps) ──
        if step % config.log_every == 0:
            w = min(config.log_every, len(losses))
            avg_loss = sum(losses[-w:]) / w
            lr = scheduler.get_last_lr()[0]
            mem = torch.cuda.max_memory_allocated() / 1e9
            elapsed = time.time() - step_t0
            sec_per_step = elapsed / config.log_every
            step_t0 = time.time()
            ds = student.heads.delta_scale.data.tolist()
            ds_str = " ".join(f"{v:.3f}" for v in ds)
            print(f"step {step:5d} | loss {avg_loss:.4f} | "
                  f"lr {lr:.2e} | tau {tau:.3f} | "
                  f"band {metrics['band_idx']} | "
                  f"mem {mem:.2f}GB | {sec_per_step:.2f}s/step | "
                  f"ds=[{ds_str}]")

        # ── Diagnostics (every 500 steps) ──
        if step % DIAG_EVERY == 0:
            batch_diag = next(iter(train_loader))
            x0_diag = batch_diag['input_ids'].to(device)

            # STATIC DIAG
            diags = diag_all_bands(student, teacher, x0_diag, config)
            print_diag_table(diags,
                             f"STATIC DIAG @ step {step}")

            # ON-POLICY STM DIAG
            op_diag = on_policy_stm_diag(
                student, teacher, x0_diag, config, tau)
            print_on_policy_diag(op_diag,
                                 f"ON-POLICY STM DIAG @ step {step} "
                                 f"(tau={tau:.3f})")

            # STM-specific averages over last DIAG_EVERY steps
            recent = step_metrics[-DIAG_EVERY:]
            avg_stu_t = sum(m['student_total'] for m in recent) / len(recent)
            avg_tea_t = sum(m['teacher_total'] for m in recent) / len(recent)
            avg_stu_i = sum(m['mean_student_interval']
                           for m in recent) / len(recent)
            avg_tea_i = sum(m['mean_teacher_interval']
                           for m in recent) / len(recent)
            avg_ft_err = sum(m['final_t_error']
                            for m in recent) / len(recent)
            print(f"\n  delta_scale: "
                  f"{student.heads.delta_scale.data.tolist()}")
            print(f"  tau: {tau:.4f}")
            print(f"  mean student_total: {avg_stu_t:.4f}")
            print(f"  mean teacher_total: {avg_tea_t:.4f}")
            print(f"  mean student interval: {avg_stu_i:.4f}")
            print(f"  mean teacher interval: {avg_tea_i:.4f}")
            print(f"  mean final_t_error: {avg_ft_err:.2e}")

            student.train()

        # ── Checkpoint ──
        if step % config.save_every == 0:
            path = os.path.join(config.output_dir_stm,
                                f"ckpt_step{step}.pt")
            try:
                torch.save({
                    'step': step,
                    'student_state_dict': {
                        'backbone_loras':
                            student.backbone_loras.state_dict(),
                        'heads':
                            student.heads.trainable_state_dict(),
                    },
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'config': config,
                }, path)
                print(f"  Saved: {path}")
            except Exception as e:
                print(f"FATAL: checkpoint save failed at step {step}: "
                      f"{e}. Aborting.")
                abort = True
                break

        # ── Samples ──
        if step % config.sample_every == 0:
            print(f"\n  --- Samples @ step {step} ---")
            student.eval()
            with torch.no_grad():
                samples = generate_samples(
                    student, config,
                    num_samples=config.num_sample_texts, device=device)
            for i in range(samples.shape[0]):
                text = tokenizer.decode(samples[i].tolist(),
                                        skip_special_tokens=True)
                n_mask = (samples[i] == config.mask_token_id).sum().item()
                print(f"  Sample {i} (mask={n_mask}): "
                      f"{text[:SAMPLE_CHARS]}")
            student.train()
            print()

        if abort:
            break

    # ── Final summary ──
    total_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{'='*72}")
    print(f"  FINAL SUMMARY — STM, {len(losses)} steps")
    print(f"{'='*72}")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Peak memory: {peak_mem:.2f} GB")
    print(f"  delta_scale final: "
          f"{student.heads.delta_scale.data.tolist()}")

    # Loss trajectory
    print(f"\n  Loss trajectory:")
    for w in range(0, len(losses), 500):
        chunk = losses[w:w + 500]
        if chunk:
            print(f"    steps {w + 1:5d}-{w + len(chunk):5d}: "
                  f"mean={sum(chunk) / len(chunk):.4f}")

    # Final diagnostics
    student.eval()
    batch_final = next(iter(train_loader))
    x0_final = batch_final['input_ids'].to(device)
    final_diags = diag_all_bands(student, teacher, x0_final, config)
    print_diag_table(final_diags, "Final diagnostics")

    # Final STM stats
    if step_metrics:
        all_stu = sum(m['student_total'] for m in step_metrics) / len(step_metrics)
        all_tea = sum(m['teacher_total'] for m in step_metrics) / len(step_metrics)
        print(f"\n  Overall mean student_total: {all_stu:.4f}")
        print(f"  Overall mean teacher_total: {all_tea:.4f}")


if __name__ == "__main__":
    main()
