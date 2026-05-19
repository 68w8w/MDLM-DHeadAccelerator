#!/usr/bin/env python3
"""Experiment: reverse KL vs forward KL, 500 steps side-by-side comparison.

Reverse KL = KL(p_S || p_T) = sum(p_S * (log_pS - log_pT))
  - mode-seeking: penalizes student for putting mass where teacher has low prob
  - should produce more peaked student distributions

Forward KL = KL(p_T || p_S) = sum(p_T * (log_pT - log_pS))  [current]
  - mean-seeking: penalizes student for NOT covering teacher's modes
  - produces flatter student distributions
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
from inference import generate_samples


# ── Reverse KL ───────────────────────────────────────────────────────

def reverse_KL_loss(log_pT, log_pS, mask, chunk_size=0):
    """Reverse KL: KL(p_S || p_T) on MASK positions.

    sum(p_S * (log_pS - log_pT)), averaged over MASK positions.
    """
    if mask.sum() == 0:
        return log_pS.sum() * 0.0

    log_pT_m = log_pT[mask].float()  # [M, V]
    log_pS_m = log_pS[mask].float()  # [M, V]

    if chunk_size <= 0:
        pS = log_pS_m.exp()
        term = pS * (log_pS_m - log_pT_m)
        # 0 * (-inf) = 0 by convention
        term = torch.where(pS > 0, term, torch.zeros_like(term))
        # Clamp to prevent explosion from log_pT ≈ -inf where pS > 0
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


# ── Diagnostics ──────────────────────────────────────────────────────

def _entropy_on_mask(log_p, mask):
    if mask.sum() == 0:
        return 0.0
    log_p_m = log_p[mask].float()
    p_m = log_p_m.exp()
    ent = -(p_m * log_p_m)
    ent = torch.where(p_m > 0, ent, torch.zeros_like(ent))
    return ent.sum(dim=-1).mean().item()


def _top1_acc(log_p, x0, mask):
    if mask.sum() == 0:
        return 0.0
    return (log_p[mask].argmax(-1) == x0[mask]).float().mean().item()


def _agreement(log_pT, log_pS, mask):
    if mask.sum() == 0:
        return 0.0
    return (log_pT[mask].argmax(-1) == log_pS[mask].argmax(-1)).float().mean().item()


# ── Train step (parameterized loss) ─────────────────────────────────

def train_step_exp(
    student, teacher, x0, config, optimizer, scheduler,
    loss_fn, chunk_size,
):
    B = x0.shape[0]
    device = x0.device
    K = config.K
    MASK_ID = config.mask_token_id

    band_idx = torch.randint(0, config.n_bands, (1,)).item()
    t_band_high = 1.0 - band_idx / config.n_bands
    t_band_low = 1.0 - (band_idx + 1) / config.n_bands

    endpoint_prob = 0.5 if band_idx == 0 else 0.3
    use_endpoint = torch.rand(B, device=device) < endpoint_prob
    t_uniform = torch.rand(B, device=device) * (t_band_high - t_band_low) + t_band_low
    t_endpoint = torch.full((B,), t_band_high, device=device)
    t_src = torch.where(use_endpoint, t_endpoint, t_uniform)
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


def diag_step(student, teacher, x0, config, band_idx=None):
    """Run one diagnostic pass (no grad, no backward)."""
    B = x0.shape[0]
    device = x0.device
    MASK_ID = config.mask_token_id

    if band_idx is None:
        band_idx = torch.randint(0, config.n_bands, (1,)).item()

    t_band_high = 1.0 - band_idx / config.n_bands
    t_band_low = 1.0 - (band_idx + 1) / config.n_bands
    t_mid = (t_band_high + t_band_low) / 2
    t = torch.full((B,), t_mid, device=device)
    z = forward_noise(x0, t, MASK_ID)
    mask = (z == MASK_ID)

    with torch.no_grad():
        hidden, c = student.forward_backbone(z, t)
        log_pT = teacher.forward_log_probs(z, t)
        logits_S = student.heads.compute_one_head(
            hidden_src=hidden, z=z, c=c, t_cur=t, band_idx=band_idx)
        log_pS = F.log_softmax(logits_S.float(), dim=-1)

        fwd_kl = KL_loss(log_pT, log_pS, mask, chunk_size=512).item()
        rev_kl = reverse_KL_loss(log_pT, log_pS, mask, chunk_size=512).item()

    return dict(
        band=band_idx,
        fwd_kl=fwd_kl,
        rev_kl=rev_kl,
        ent_T=_entropy_on_mask(log_pT, mask),
        ent_S=_entropy_on_mask(log_pS, mask),
        acc_T=_top1_acc(log_pT, x0, mask),
        acc_S=_top1_acc(log_pS, x0, mask),
        agree=_agreement(log_pT, log_pS, mask),
        mask_ratio=mask.float().mean().item(),
    )


# ── Main ─────────────────────────────────────────────────────────────

def run_experiment(n_steps=500):
    config = Config()
    config.kl_chunk_size = 512
    config.batch_size = 8
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

    print("Loading data...")
    train_loader = get_dataloader(config, split="train")
    data_iter = iter(train_loader)

    def get_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            return next(data_iter)

    # ── Run both experiments ──
    for exp_name, loss_fn in [
        ("FORWARD_KL", KL_loss),
        ("REVERSE_KL", reverse_KL_loss),
    ]:
        print(f"\n{'='*72}")
        print(f"  EXPERIMENT: {exp_name}  ({n_steps} steps)")
        print(f"{'='*72}")

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.empty_cache()
        import gc; gc.collect()

        print("  Building student...")
        student = build_student(config, device)
        student.train()

        print("  Building teacher...")
        teacher = build_teacher(config, device)
        teacher.eval()

        params = student.get_trainable_parameters()
        optimizer = torch.optim.AdamW(
            params, lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, n_steps)

        losses = []
        t0 = time.time()

        for step in range(1, n_steps + 1):
            batch = get_batch()
            x0 = batch['input_ids'].to(device)

            loss = train_step_exp(
                student, teacher, x0, config, optimizer, scheduler,
                loss_fn=loss_fn, chunk_size=config.kl_chunk_size)
            losses.append(loss)

            if step % 50 == 0:
                avg = sum(losses[-50:]) / 50
                elapsed = time.time() - t0
                print(f"  step {step:4d} | loss {avg:.4f} | "
                      f"time {elapsed:.0f}s")

        # ── Post-training diagnostics (all 4 bands) ──
        print(f"\n  --- Diagnostics after {n_steps} steps ---")
        student.eval()
        batch = get_batch()
        x0 = batch['input_ids'].to(device)

        print(f"  {'band':>4} {'fwd_KL':>8} {'rev_KL':>8} "
              f"{'ent_T':>7} {'ent_S':>7} {'acc_T':>7} "
              f"{'acc_S':>7} {'agree':>7}")
        for b in range(4):
            d = diag_step(student, teacher, x0, config, band_idx=b)
            print(f"  {d['band']:>4} {d['fwd_kl']:>8.3f} {d['rev_kl']:>8.3f} "
                  f"{d['ent_T']:>7.2f} {d['ent_S']:>7.2f} {d['acc_T']:>7.4f} "
                  f"{d['acc_S']:>7.4f} {d['agree']:>7.4f}")

        # ── Loss trajectory summary ──
        print(f"\n  Loss trajectory:")
        for w in range(0, n_steps, 100):
            chunk = losses[w:w+100]
            if chunk:
                print(f"    steps {w+1:4d}-{w+len(chunk):4d}: "
                      f"mean={sum(chunk)/len(chunk):.4f}")

        # ── Generate samples ──
        print(f"\n  --- Samples ---")
        with torch.no_grad():
            samples = generate_samples(student, config, num_samples=2, device=device)
        for i in range(samples.shape[0]):
            text = tokenizer.decode(samples[i].tolist(), skip_special_tokens=True)
            print(f"  Sample {i}: {text[:300]}")

        print(f"\n  Total time: {time.time() - t0:.0f}s")
        print(f"  Peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

        # Free everything to make room for next experiment
        del student, teacher, optimizer, scheduler, params
        import gc; gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*72}")
    print(f"  DONE")
    print(f"{'='*72}")


if __name__ == "__main__":
    run_experiment(n_steps=500)
