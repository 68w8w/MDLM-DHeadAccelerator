#!/usr/bin/env python3
"""Experiment: safe reverse KL variants, 500 steps each.

Usage:
    python exp_safe_rkl.py fwd        # forward KL baseline
    python exp_safe_rkl.py rev        # safe reverse KL only
    python exp_safe_rkl.py mix        # 0.7*fwd + 0.3*safe_rev
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
    return KL_loss(log_pT, log_pS, mask, chunk_size=chunk_size)


def safe_reverse_KL(log_pT, log_pS, mask, chunk_size=512,
                    mask_token_id=50257):
    """Numerically safe reverse KL: KL(pS || pT) on MASK positions.

    Key fixes:
      - Remove MASK token column (avoid -inf in difference)
      - Re-normalize student on valid vocab
      - Floor teacher log-probs at -30 (prevent infinite penalty)
    """
    if mask.sum() == 0:
        return log_pS.sum() * 0.0

    lpT = log_pT[mask].float().clone()
    lpS = log_pS[mask].float().clone()

    # Zero out MASK token column
    lpT[:, mask_token_id] = -torch.inf
    lpS[:, mask_token_id] = -torch.inf

    # Re-normalize student over non-MASK vocab
    lpS = F.log_softmax(lpS, dim=-1)

    # Floor teacher to avoid reverse KL infinite penalty
    lpT_safe = lpT.clamp_min(-30.0)

    pS = lpS.exp()

    # Only compute where pS > 0
    valid = pS > 0
    term = torch.zeros_like(lpS)
    term[valid] = pS[valid] * (lpS[valid] - lpT_safe[valid])

    return term.sum(dim=-1).mean()


def mixed_KL(log_pT, log_pS, mask, chunk_size=512, alpha=0.7):
    """alpha * forward_KL + (1-alpha) * safe_reverse_KL."""
    fwd = forward_KL(log_pT, log_pS, mask, chunk_size=chunk_size)
    rev = safe_reverse_KL(log_pT, log_pS, mask, chunk_size=chunk_size)
    return alpha * fwd + (1 - alpha) * rev


# ── Diagnostics ──────────────────────────────────────────────────────

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
    return (log_pT[mask].argmax(-1) == log_pS[mask].argmax(-1)).float().mean().item()


def diag_all(student, teacher, x0, config):
    MASK_ID = config.mask_token_id
    device = x0.device
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
        rows.append(dict(
            band=b,
            fwd=forward_KL(lpT, lpS, mask, 512).item(),
            rev=safe_reverse_KL(lpT, lpS, mask, 512).item(),
            ent_T=_entropy(lpT, mask), ent_S=_entropy(lpS, mask),
            acc_T=_acc(lpT, x0, mask), acc_S=_acc(lpS, x0, mask),
            agree=_agree(lpT, lpS, mask),
        ))
    return rows


# ── Train step ───────────────────────────────────────────────────────

def train_step(student, teacher, x0, config, optimizer, scheduler, loss_fn):
    B = x0.shape[0]
    device = x0.device
    K = config.K
    MASK_ID = config.mask_token_id

    band_idx = torch.randint(0, config.n_bands, (1,)).item()
    t_hi = 1.0 - band_idx / config.n_bands
    t_lo = 1.0 - (band_idx + 1) / config.n_bands

    ep = 0.5 if band_idx == 0 else 0.3
    use_ep = torch.rand(B, device=device) < ep
    t_src = torch.where(use_ep,
                        torch.full((B,), t_hi, device=device),
                        torch.rand(B, device=device) * (t_hi - t_lo) + t_lo)
    t_dst = torch.full((B,), t_lo, device=device)

    z_t = forward_noise(x0, t_src, MASK_ID)
    hidden, c_src = student.forward_backbone(z_t, t_src)

    optimizer.zero_grad(set_to_none=True)
    z = z_t.clone()
    D = (t_src - t_dst) / K
    total = 0.0

    for k in range(K):
        t_cur = t_src - k * D
        t_nxt = t_src - (k + 1) * D

        with torch.no_grad():
            lpT = teacher.forward_log_probs(z, t_cur)

        lgS = student.heads.compute_one_head(
            hidden_src=hidden, z=z, c=c_src, t_cur=t_cur, band_idx=band_idx)
        lpS = F.log_softmax(lgS.float(), dim=-1)
        mask = (z == MASK_ID)

        lk = loss_fn(lpT, lpS, mask, chunk_size=512)
        (lk / K).backward(retain_graph=(k < K - 1))
        total += lk.detach().item()

        with torch.no_grad():
            z = absorbing_reverse_step(z=z, log_p=lpT, t_curr=t_cur,
                                       t_next=t_nxt, mask_token_id=MASK_ID)
        del lpT, lgS, lpS, lk

    clip_grad_norm_(student.get_trainable_parameters(), config.grad_clip)
    optimizer.step()
    scheduler.step()
    return total / K


# ── Main ─────────────────────────────────────────────────────────────

def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "rev"
    n_steps = 500

    loss_fns = {
        "fwd": ("FORWARD_KL", forward_KL),
        "rev": ("SAFE_REVERSE_KL", safe_reverse_KL),
        "mix": ("MIXED (0.7*fwd + 0.3*safe_rev)", mixed_KL),
    }
    name, loss_fn = loss_fns[variant]

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

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"{'='*72}")
    print(f"  {name}  ({n_steps} steps, batch={config.batch_size})")
    print(f"{'='*72}")

    print("Building student...")
    student = build_student(config, device)
    student.train()

    print("Building teacher...")
    teacher = build_teacher(config, device)
    teacher.eval()

    print("Loading data...")
    loader = get_dataloader(config, split="train")
    it = iter(loader)

    def batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    params = student.get_trainable_parameters()
    opt = torch.optim.AdamW(params, lr=config.lr,
                            betas=(config.beta1, config.beta2),
                            weight_decay=config.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, config.warmup_steps, n_steps)

    losses = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        x0 = batch()['input_ids'].to(device)
        loss = train_step(student, teacher, x0, config, opt, sched, loss_fn)
        losses.append(loss)

        if loss != loss:
            print(f"  NaN at step {step}!")
            break

        if step % 50 == 0:
            avg = sum(losses[-50:]) / 50
            print(f"  step {step:4d} | loss {avg:.4f} | "
                  f"time {time.time()-t0:.0f}s")

    # ── Diagnostics ──
    print(f"\n  --- Diagnostics after {len(losses)} steps ---")
    student.eval()
    x0 = batch()['input_ids'].to(device)
    diags = diag_all(student, teacher, x0, config)

    print(f"  {'band':>4} {'fwd':>8} {'s_rev':>8} "
          f"{'ent_T':>7} {'ent_S':>7} {'acc_T':>7} "
          f"{'acc_S':>7} {'agree':>7}")
    for d in diags:
        print(f"  {d['band']:>4} {d['fwd']:>8.3f} {d['rev']:>8.3f} "
              f"{d['ent_T']:>7.2f} {d['ent_S']:>7.2f} "
              f"{d['acc_T']:>7.4f} {d['acc_S']:>7.4f} {d['agree']:>7.4f}")

    print(f"\n  Loss: first_50={sum(losses[:50])/min(50,len(losses)):.4f}  "
          f"last_50={sum(losses[-50:])/min(50,len(losses)):.4f}")

    # ── Samples ──
    with torch.no_grad():
        samples = generate_samples(student, config, num_samples=2, device=device)
    for i in range(2):
        text = tokenizer.decode(samples[i].tolist(), skip_special_tokens=True)
        n_mask = (samples[i] == config.mask_token_id).sum().item()
        print(f"  Sample {i} (mask={n_mask}): {text[:250]}")

    print(f"\n  Time: {time.time()-t0:.0f}s  "
          f"Peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()
