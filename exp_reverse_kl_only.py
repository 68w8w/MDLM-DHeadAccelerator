#!/usr/bin/env python3
"""Experiment: reverse KL only, 500 steps. Compare against forward KL baseline."""

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


def reverse_KL_loss(log_pT, log_pS, mask, chunk_size=0):
    """Reverse KL: KL(p_S || p_T) on MASK positions."""
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


def main():
    config = Config()
    config.kl_chunk_size = 512
    config.batch_size = 4   # reduced from 8: reverse KL graph is larger
    config.max_length = 1024
    config.K = 4
    config.n_bands = 4
    config.backbone_lora_rank = 128
    config.dropout = 0.0
    config.lr = 1e-4
    config.warmup_steps = 100
    config.grad_clip = 1.0
    device = config.device
    n_steps = 500
    MASK_ID = config.mask_token_id

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"{'='*72}")
    print(f"  REVERSE KL EXPERIMENT  ({n_steps} steps)")
    print(f"{'='*72}")

    print("Building student...")
    student = build_student(config, device)
    student.train()

    print("Building teacher...")
    teacher = build_teacher(config, device)
    teacher.eval()

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
        B = x0.shape[0]
        K = config.K

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

            loss_k = reverse_KL_loss(log_pT, log_pS, mask, chunk_size=512)
            (loss_k / K).backward(retain_graph=(k < K - 1))
            total_loss += loss_k.detach().item()

            with torch.no_grad():
                z = absorbing_reverse_step(
                    z=z, log_p=log_pT, t_curr=t_cur, t_next=t_next,
                    mask_token_id=MASK_ID)

            del log_pT, logits_S, log_pS, loss_k

        clip_grad_norm_(params, config.grad_clip)
        optimizer.step()
        scheduler.step()

        avg_loss = total_loss / K
        losses.append(avg_loss)

        if step % 50 == 0:
            avg = sum(losses[-50:]) / 50
            print(f"  step {step:4d} | rev_kl {avg:.4f} | "
                  f"time {time.time()-t0:.0f}s")

    # ── Diagnostics ──
    print(f"\n  --- Diagnostics after {n_steps} steps ---")
    student.eval()
    batch = get_batch()
    x0 = batch['input_ids'].to(device)

    print(f"  {'band':>4} {'fwd_KL':>8} {'rev_KL':>8} "
          f"{'ent_T':>7} {'ent_S':>7} {'acc_T':>7} "
          f"{'acc_S':>7} {'agree':>7}")

    for b in range(4):
        t_mid = 1.0 - (b + 0.5) / 4
        t = torch.full((config.batch_size,), t_mid, device=device)
        z = forward_noise(x0, t, MASK_ID)
        mask = (z == MASK_ID)

        with torch.no_grad():
            hidden, c = student.forward_backbone(z, t)
            log_pT = teacher.forward_log_probs(z, t)
            logits_S = student.heads.compute_one_head(
                hidden_src=hidden, z=z, c=c, t_cur=t, band_idx=b)
            log_pS = F.log_softmax(logits_S.float(), dim=-1)

            fwd = KL_loss(log_pT, log_pS, mask, chunk_size=512).item()
            rev = reverse_KL_loss(log_pT, log_pS, mask, chunk_size=512).item()

        print(f"  {b:>4} {fwd:>8.3f} {rev:>8.3f} "
              f"{_entropy_on_mask(log_pT, mask):>7.2f} "
              f"{_entropy_on_mask(log_pS, mask):>7.2f} "
              f"{_top1_acc(log_pT, x0, mask):>7.4f} "
              f"{_top1_acc(log_pS, x0, mask):>7.4f} "
              f"{_agreement(log_pT, log_pS, mask):>7.4f}")

    # ── Loss trajectory ──
    print(f"\n  Loss trajectory:")
    for w in range(0, n_steps, 100):
        chunk = losses[w:w+100]
        if chunk:
            print(f"    steps {w+1:4d}-{w+len(chunk):4d}: "
                  f"mean={sum(chunk)/len(chunk):.4f}")

    # ── Samples ──
    print(f"\n  --- Samples ---")
    with torch.no_grad():
        samples = generate_samples(student, config, num_samples=2, device=device)
    for i in range(samples.shape[0]):
        text = tokenizer.decode(samples[i].tolist(), skip_special_tokens=True)
        print(f"  Sample {i}: {text[:300]}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")
    print(f"  Peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ── Forward KL baseline (from previous run, same seed) ──
    print(f"\n{'='*72}")
    print(f"  COMPARISON (same seed=42, same 500 steps)")
    print(f"{'='*72}")
    print(f"  Reverse KL loss @500 (last 50 avg): {sum(losses[-50:])/50:.4f}")
    print(f"  (Compare with forward KL diagnostics above — focus on ent_S and agree)")


if __name__ == "__main__":
    main()
