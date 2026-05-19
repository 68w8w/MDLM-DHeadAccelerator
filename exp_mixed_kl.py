#!/usr/bin/env python3
"""Experiment: mixed KL (forward + reverse), 500 steps.

Three variants in one run (separate processes would be cleaner but
batch=4 fits fine):
  A. forward KL only       (baseline)
  B. 0.7 fwd + 0.3 rev
  C. JSD = 0.5 * KL(M||S) + 0.5 * KL(M||T), M = 0.5*(pT+pS)
"""

import sys
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
from inference import generate_samples


# ── Loss functions ───────────────────────────────────────────────────

def forward_KL(log_pT, log_pS, mask, chunk_size=0):
    """KL(pT || pS): same as train.py KL_loss."""
    return KL_loss(log_pT, log_pS, mask, chunk_size=chunk_size)


def reverse_KL(log_pT, log_pS, mask, chunk_size=0):
    """KL(pS || pT) on MASK positions."""
    if mask.sum() == 0:
        return log_pS.sum() * 0.0
    log_pT_m = log_pT[mask].float()
    log_pS_m = log_pS[mask].float()

    if chunk_size <= 0:
        pS = log_pS_m.exp()
        term = pS * (log_pS_m - log_pT_m)
        term = torch.where(pS > 0, term, torch.zeros_like(term))
        term = term.clamp(max=100.0)
        return term.sum(dim=-1).mean()

    losses = []
    for start in range(0, log_pS_m.shape[0], chunk_size):
        end = start + chunk_size
        lpT = log_pT_m[start:end]
        lpS = log_pS_m[start:end]
        pS = lpS.exp()
        term = pS * (lpS - lpT)
        term = torch.where(pS > 0, term, torch.zeros_like(term))
        term = term.clamp(max=100.0)
        losses.append(term.sum(dim=-1))
    return torch.cat(losses, dim=0).mean()


def mixed_KL(log_pT, log_pS, mask, chunk_size=0, alpha=0.7):
    """alpha * KL(pT||pS) + (1-alpha) * KL(pS||pT)."""
    fwd = forward_KL(log_pT, log_pS, mask, chunk_size=chunk_size)
    rev = reverse_KL(log_pT, log_pS, mask, chunk_size=chunk_size)
    return alpha * fwd + (1 - alpha) * rev


def JSD_loss(log_pT, log_pS, mask, chunk_size=0):
    """Jensen-Shannon divergence: 0.5*KL(pT||M) + 0.5*KL(pS||M), M=0.5*(pT+pS).

    Symmetric, bounded [0, ln2], no mode collapse.
    """
    if mask.sum() == 0:
        return log_pS.sum() * 0.0

    log_pT_m = log_pT[mask].float()  # [M, V]
    log_pS_m = log_pS[mask].float()  # [M, V]

    pT = log_pT_m.exp()
    pS = log_pS_m.exp()
    M = 0.5 * (pT + pS)
    log_M = M.clamp(min=1e-30).log()

    # KL(pT || M)
    t1 = pT * (log_pT_m - log_M)
    t1 = torch.where(pT > 0, t1, torch.zeros_like(t1))

    # KL(pS || M)
    t2 = pS * (log_pS_m - log_M)
    t2 = torch.where(pS > 0, t2, torch.zeros_like(t2))

    jsd = 0.5 * t1.sum(dim=-1) + 0.5 * t2.sum(dim=-1)
    return jsd.mean()


# ── Diagnostics ──────────────────────────────────────────────────────

def _entropy_on_mask(log_p, mask):
    if mask.sum() == 0:
        return 0.0
    lp = log_p[mask].float()
    p = lp.exp()
    ent = -(p * lp)
    ent = torch.where(p > 0, ent, torch.zeros_like(ent))
    return ent.sum(dim=-1).mean().item()


def _top1_acc(log_p, x0, mask):
    if mask.sum() == 0:
        return 0.0
    return (log_p[mask].argmax(-1) == x0[mask]).float().mean().item()


def _agreement(log_pT, log_pS, mask):
    if mask.sum() == 0:
        return 0.0
    return (log_pT[mask].argmax(-1) == log_pS[mask].argmax(-1)).float().mean().item()


def diag_all_bands(student, teacher, x0, config):
    MASK_ID = config.mask_token_id
    device = x0.device
    student.eval()
    results = []
    for b in range(4):
        t_mid = 1.0 - (b + 0.5) / 4
        t = torch.full((x0.shape[0],), t_mid, device=device)
        z = forward_noise(x0, t, MASK_ID)
        mask = (z == MASK_ID)
        with torch.no_grad():
            hidden, c = student.forward_backbone(z, t)
            log_pT = teacher.forward_log_probs(z, t)
            logits_S = student.heads.compute_one_head(
                hidden_src=hidden, z=z, c=c, t_cur=t, band_idx=b)
            log_pS = F.log_softmax(logits_S.float(), dim=-1)
        results.append(dict(
            band=b,
            fwd_kl=forward_KL(log_pT, log_pS, mask, 512).item(),
            rev_kl=reverse_KL(log_pT, log_pS, mask, 512).item(),
            jsd=JSD_loss(log_pT, log_pS, mask).item(),
            ent_T=_entropy_on_mask(log_pT, mask),
            ent_S=_entropy_on_mask(log_pS, mask),
            acc_T=_top1_acc(log_pT, x0, mask),
            acc_S=_top1_acc(log_pS, x0, mask),
            agree=_agreement(log_pT, log_pS, mask),
        ))
    return results


# ── Train step ───────────────────────────────────────────────────────

def train_step_generic(student, teacher, x0, config, optimizer, scheduler,
                       loss_fn, chunk_size):
    B = x0.shape[0]
    device = x0.device
    K = config.K
    MASK_ID = config.mask_token_id

    band_idx = torch.randint(0, config.n_bands, (1,)).item()
    t_band_high = 1.0 - band_idx / config.n_bands
    t_band_low = 1.0 - (band_idx + 1) / config.n_bands

    endpoint_prob = 0.5 if band_idx == 0 else 0.3
    use_ep = torch.rand(B, device=device) < endpoint_prob
    t_uni = torch.rand(B, device=device) * (t_band_high - t_band_low) + t_band_low
    t_ep = torch.full((B,), t_band_high, device=device)
    t_src = torch.where(use_ep, t_ep, t_uni)
    t_dst = torch.full((B,), t_band_low, device=device)

    z_t = forward_noise(x0, t_src, MASK_ID)
    hidden_src, c_src = student.forward_backbone(z_t, t_src)

    optimizer.zero_grad(set_to_none=True)
    z = z_t.clone()
    Delta = (t_src - t_dst) / K
    total_loss = 0.0

    for k in range(K):
        t_cur = t_src - k * Delta
        t_next = t_src - (k + 1) * Delta

        with torch.no_grad():
            log_pT = teacher.forward_log_probs(z, t_cur)

        logits_S = student.heads.compute_one_head(
            hidden_src=hidden_src, z=z, c=c_src,
            t_cur=t_cur, band_idx=band_idx)
        log_pS = F.log_softmax(logits_S.float(), dim=-1)
        mask = (z == MASK_ID)

        loss_k = loss_fn(log_pT, log_pS, mask, chunk_size=chunk_size)
        (loss_k / K).backward(retain_graph=(k < K - 1))
        total_loss += loss_k.detach().item()

        with torch.no_grad():
            z = absorbing_reverse_step(
                z=z, log_p=log_pT, t_curr=t_cur, t_next=t_next,
                mask_token_id=MASK_ID)
        del log_pT, logits_S, log_pS, loss_k

    clip_grad_norm_(student.get_trainable_parameters(), config.grad_clip)
    optimizer.step()
    scheduler.step()
    return total_loss / K


# ── Run one experiment ───────────────────────────────────────────────

def run_one(name, loss_fn, config, teacher, train_loader, tokenizer,
            n_steps=500):
    device = config.device
    MASK_ID = config.mask_token_id

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print(f"\n{'='*72}")
    print(f"  {name}  ({n_steps} steps, batch={config.batch_size})")
    print(f"{'='*72}")

    student = build_student(config, device)
    student.train()

    params = student.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        params, lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, n_steps)

    data_iter = iter(train_loader)
    def get_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            return next(data_iter)

    losses = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        x0 = get_batch()['input_ids'].to(device)
        loss = train_step_generic(
            student, teacher, x0, config, optimizer, scheduler,
            loss_fn=loss_fn, chunk_size=config.kl_chunk_size)
        losses.append(loss)

        if loss != loss:
            print(f"  NaN at step {step}!")
            break

        if step % 50 == 0:
            avg = sum(losses[-50:]) / 50
            print(f"  step {step:4d} | loss {avg:.6f} | "
                  f"time {time.time()-t0:.0f}s")

    # Diagnostics
    print(f"\n  --- Diagnostics ---")
    x0 = get_batch()['input_ids'].to(device)
    diags = diag_all_bands(student, teacher, x0, config)

    print(f"  {'band':>4} {'fwd_KL':>8} {'rev_KL':>8} {'JSD':>8} "
          f"{'ent_T':>7} {'ent_S':>7} {'acc_S':>7} {'agree':>7}")
    for d in diags:
        print(f"  {d['band']:>4} {d['fwd_kl']:>8.3f} {d['rev_kl']:>8.3f} "
              f"{d['jsd']:>8.4f} {d['ent_T']:>7.2f} {d['ent_S']:>7.2f} "
              f"{d['acc_S']:>7.4f} {d['agree']:>7.4f}")

    # Loss summary
    print(f"\n  Loss: first_50={sum(losses[:50])/50:.4f}  "
          f"last_50={sum(losses[-50:])/50:.4f}")

    # Samples
    student.eval()
    with torch.no_grad():
        samples = generate_samples(student, config, num_samples=2, device=device)
    for i in range(samples.shape[0]):
        text = tokenizer.decode(samples[i].tolist(), skip_special_tokens=True)
        print(f"  Sample {i}: {text[:200]}")

    print(f"  Time: {time.time()-t0:.0f}s  Peak: "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    # Cleanup
    del student, optimizer, scheduler, params
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return losses, diags


# ── Main ─────────────────────────────────────────────────────────────

def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "all"

    config = Config()
    config.kl_chunk_size = 512
    config.batch_size = 4
    config.max_length = 1024
    config.K = 4
    config.n_bands = 4
    config.backbone_lora_rank = 128
    config.dropout = 0.0
    config.lr = 1e-4
    config.warmup_steps = 100
    config.grad_clip = 1.0
    device = config.device

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("Building teacher (shared)...")
    teacher = build_teacher(config, device)
    teacher.eval()

    print("Loading data...")
    train_loader = get_dataloader(config, split="train")

    experiments = {
        "fwd": ("FORWARD_KL",
                lambda lpT, lpS, m, chunk_size=0: forward_KL(lpT, lpS, m, chunk_size)),
        "mix": ("MIXED_KL (0.7*fwd + 0.3*rev)",
                lambda lpT, lpS, m, chunk_size=0: mixed_KL(lpT, lpS, m, chunk_size, alpha=0.7)),
        "jsd": ("JSD",
                lambda lpT, lpS, m, chunk_size=0: JSD_loss(lpT, lpS, m, chunk_size)),
    }

    if variant == "all":
        to_run = ["fwd", "mix", "jsd"]
    else:
        to_run = [variant]

    all_results = {}
    for key in to_run:
        name, loss_fn = experiments[key]
        losses, diags = run_one(
            name, loss_fn, config, teacher, train_loader, tokenizer,
            n_steps=500)
        all_results[key] = (losses, diags)

    # ── Comparison table ──
    if len(all_results) > 1:
        print(f"\n{'='*72}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*72}")

        header = f"  {'metric':>15}"
        for key in all_results:
            header += f"  {key:>12}"
        print(header)

        # Last 50 loss
        row = f"  {'loss(last50)':>15}"
        for key in all_results:
            ls = all_results[key][0]
            row += f"  {sum(ls[-50:])/50:>12.4f}"
        print(row)

        # Per-band metrics (band 1 as representative)
        for metric in ['ent_S', 'acc_S', 'agree', 'fwd_kl']:
            row = f"  {f'{metric}(b1)':>15}"
            for key in all_results:
                d = all_results[key][1][1]  # band 1
                row += f"  {d[metric]:>12.4f}"
            print(row)


if __name__ == "__main__":
    main()