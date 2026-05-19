#!/usr/bin/env python3
"""Dinner run: full 30k-step training with enhanced logging.

Usage:
    CUDA_VISIBLE_DEVICES=6 python dinner_run.py

Stops only on: NaN/Inf, OOM, checkpoint save failure, or user interrupt.
"""

import os
import sys
import time
import math
import traceback

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer

from config import Config
from data import get_dataloader
from model import build_student, build_teacher
from diffusion_utils import forward_noise, absorbing_reverse_step
from inference import generate_samples
from train import KL_loss, get_cosine_schedule_with_warmup, train_step


# ── diagnostic helpers (from diagnostic_run.py) ─────────────────────

def _entropy_on_mask(log_p, mask):
    if mask.sum() == 0:
        return 0.0
    log_p_m = log_p[mask].float()
    p_m = log_p_m.exp()
    ent = -(p_m * log_p_m)
    ent = torch.where(p_m > 0, ent, torch.zeros_like(ent))
    return ent.sum(dim=-1).mean().item()


def _top1_acc_vs_x0(log_p, x0, mask):
    if mask.sum() == 0:
        return 0.0
    pred = log_p[mask].argmax(dim=-1)
    gt = x0[mask]
    return (pred == gt).float().mean().item()


def _top1_agreement(log_pT, log_pS, mask):
    if mask.sum() == 0:
        return 0.0
    return (log_pT[mask].argmax(-1) == log_pS[mask].argmax(-1)).float().mean().item()


def train_step_with_diag(
    student, teacher, x0, config, optimizer, scheduler, step,
):
    """Same as train_step but also returns per-substep diagnostics.

    Returns (avg_loss, diag_dict).
    diag_dict has keys: band_idx, substeps (list of per-k dicts with
        kl, ent_S, acc_S, agree, mask_count, mask_ratio).
    """
    B = x0.shape[0]
    device = x0.device
    K = config.K
    MASK_ID = config.mask_token_id
    chunk_size = config.kl_chunk_size

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
    substeps = []

    for k in range(K):
        t_cur = t_src - k * Delta
        t_next = t_src - (k + 1) * Delta

        with torch.no_grad():
            log_pT = teacher.forward_log_probs(z, t_cur)

        c_cur = c_src
        logits_S = student.heads.compute_one_head(
            hidden_src=hidden_src, z=z, c=c_cur,
            t_cur=t_cur, band_idx=band_idx,
        )
        log_pS = F.log_softmax(logits_S.float(), dim=-1)
        mask = (z == MASK_ID)

        # Collect diagnostics (detached, no grad)
        with torch.no_grad():
            kl_k_val = KL_loss(log_pT, log_pS.detach(), mask, chunk_size=chunk_size).item()
            ent_S = _entropy_on_mask(log_pS.detach(), mask)
            acc_S = _top1_acc_vs_x0(log_pS.detach(), x0, mask)
            agree = _top1_agreement(log_pT, log_pS.detach(), mask)
            mc = mask.sum().item()

        substeps.append(dict(
            k=k, kl=kl_k_val, ent_S=ent_S, acc_S=acc_S, agree=agree,
            mask_count=mc, mask_ratio=mc / (B * config.max_length),
        ))

        # Backward
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

    diag = dict(band_idx=band_idx, substeps=substeps)
    return total_loss / K, diag


# ── main ─────────────────────────────────────────────────────────────

def main():
    # ── Explicit config (no defaults relied on) ──
    config = Config()
    config.total_steps = 30000
    config.batch_size = 8
    config.max_length = 1024
    config.K = 4
    config.n_bands = 4
    config.lr = 1e-4
    config.warmup_steps = 1000
    config.beta1 = 0.9
    config.beta2 = 0.95
    config.weight_decay = 0.0
    config.grad_clip = 1.0
    config.precision = "bf16"
    config.seed = 42
    config.backbone_lora_rank = 128
    config.kl_chunk_size = 512
    config.dropout = 0.0
    config.log_every = 50
    config.save_every = 500
    config.sample_every = 500
    config.num_sample_texts = 4
    config.output_dir = "./outputs"

    SAMPLE_CHARS = 300
    DIAG_EVERY = 500   # detailed diagnostics interval

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    device = config.device

    print(f"{'='*72}")
    print(f"  DINNER RUN — {config.total_steps} steps")
    print(f"{'='*72}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  batch={config.batch_size}  L={config.max_length}  K={config.K}  "
          f"bands={config.n_bands}")
    print(f"  lr={config.lr}  warmup={config.warmup_steps}  "
          f"lora_rank={config.backbone_lora_rank}")
    print(f"  kl_chunk_size={config.kl_chunk_size}  dropout={config.dropout}")
    print(f"  save_every={config.save_every}  sample_every={config.sample_every}")
    print()

    # ── Build models ──
    print("Building student...")
    student = build_student(config, device)
    student.train()

    print("Building teacher...")
    teacher = build_teacher(config, device)
    teacher.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ── Data ──
    print("Loading data...")
    train_loader = get_dataloader(config, split="train")
    train_iter = iter(train_loader)

    # ── Optimizer ──
    trainable_params = student.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, config.total_steps)

    os.makedirs(config.output_dir, exist_ok=True)

    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_params:,}")
    print()

    # ── Phase 1: 50-step warmup to estimate speed ──
    print("Phase 1: 50-step warmup for speed estimation...")
    losses = []
    t_warmup_start = time.time()

    for step in range(1, 51):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x0 = batch['input_ids'].to(device)
        loss = train_step(student, teacher, x0, config, optimizer, scheduler, step)
        losses.append(loss)

        if loss != loss:  # NaN
            print(f"FATAL: NaN at warmup step {step}. Aborting.")
            sys.exit(1)

    t_warmup_end = time.time()
    sec_per_step = (t_warmup_end - t_warmup_start) / 50
    steps_in_90min = int(90 * 60 / sec_per_step)
    avg50 = sum(losses) / len(losses)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"  50-step warmup done in {t_warmup_end - t_warmup_start:.1f}s")
    print(f"  sec/step = {sec_per_step:.2f}")
    print(f"  avg_loss (first 50) = {avg50:.4f}")
    print(f"  peak_mem = {peak_mem:.2f} GB")
    print(f"  estimated steps in 1.5 hours: ~{steps_in_90min}")
    print(f"  estimated time for 30000 steps: "
          f"~{30000 * sec_per_step / 3600:.1f} hours")
    print()

    # ── Phase 2: continue training (steps 51 → total_steps) ──
    print(f"Phase 2: continuing to step {config.total_steps}...")
    print(f"  (will not auto-stop at 1500; only NaN/Inf/OOM/ckpt-fail)")
    print()

    t_train_start = time.time()

    for step in range(51, config.total_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x0 = batch['input_ids'].to(device)

        # Use diagnostic step at DIAG_EVERY intervals; normal step otherwise
        is_diag_step = (step % DIAG_EVERY == 0)

        try:
            if is_diag_step:
                loss, diag = train_step_with_diag(
                    student, teacher, x0, config, optimizer, scheduler, step)
            else:
                loss = train_step(
                    student, teacher, x0, config, optimizer, scheduler, step)
        except torch.cuda.OutOfMemoryError:
            print(f"\nFATAL: OOM at step {step}. Peak mem: "
                  f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            print("Saving emergency checkpoint...")
            _save_ckpt(student, optimizer, step, config, tag="emergency")
            sys.exit(1)

        losses.append(loss)

        # ── NaN / Inf check ──
        if loss != loss or abs(loss) > 1e8:
            print(f"\nFATAL: {'NaN' if loss != loss else 'Inf'} loss "
                  f"at step {step}. Aborting.")
            _save_ckpt(student, optimizer, step, config, tag="emergency")
            sys.exit(1)

        # ── Regular log (every 50 steps) ──
        if step % config.log_every == 0:
            window = min(config.log_every, len(losses))
            avg = sum(losses[-window:]) / window
            lr_now = scheduler.get_last_lr()[0]
            mem = torch.cuda.max_memory_allocated() / 1e9
            elapsed = time.time() - t_train_start
            eta_h = (config.total_steps - step) * sec_per_step / 3600
            print(f"step {step:6d} | loss {avg:.4f} | "
                  f"lr {lr_now:.2e} | mem {mem:.2f}GB | "
                  f"elapsed {elapsed/60:.0f}m | eta {eta_h:.1f}h")

        # ── Diagnostics (every 500 steps) ──
        if is_diag_step:
            print(f"\n  [diag@{step}] band={diag['band_idx']}")
            print(f"  {'k':>3} {'KL':>8} {'ent_S':>7} {'acc_S':>7} "
                  f"{'agree':>7} {'mask_r':>7}")
            for sd in diag['substeps']:
                print(f"  {sd['k']:>3} {sd['kl']:>8.3f} {sd['ent_S']:>7.2f} "
                      f"{sd['acc_S']:>7.4f} {sd['agree']:>7.4f} "
                      f"{sd['mask_ratio']:>7.4f}")
            print()

        # ── Checkpoint (every 500 steps) ──
        if step % config.save_every == 0:
            ok = _save_ckpt(student, optimizer, step, config)
            if not ok:
                print(f"FATAL: checkpoint save failed at step {step}. Aborting.")
                sys.exit(1)

        # ── Sample (every 500 steps) ──
        if step % config.sample_every == 0:
            print(f"  --- Generating {config.num_sample_texts} samples "
                  f"at step {step} ---")
            student.eval()
            with torch.no_grad():
                samples = generate_samples(
                    student, config,
                    num_samples=config.num_sample_texts, device=device)
            for i in range(samples.shape[0]):
                text = tokenizer.decode(
                    samples[i].tolist(), skip_special_tokens=True)
                print(f"  Sample {i}: {text[:SAMPLE_CHARS]}")
            student.train()
            print()

    # ── Done ──
    total_time = time.time() - t_warmup_start
    print(f"\n{'='*72}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*72}")
    print(f"  Total steps: {config.total_steps}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Final avg loss (last 500): "
          f"{sum(losses[-500:])/len(losses[-500:]):.4f}")
    print(f"  Peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


def _save_ckpt(student, optimizer, step, config, tag=None):
    """Save checkpoint. Returns True on success."""
    name = f"ckpt_step{step}.pt" if tag is None else f"ckpt_{tag}_step{step}.pt"
    path = os.path.join(config.output_dir, name)
    try:
        torch.save({
            'step': step,
            'student_state_dict': {
                'backbone_loras': student.backbone_loras.state_dict(),
                'heads': student.heads.trainable_state_dict(),
            },
            'optimizer': optimizer.state_dict(),
            'config': config,
        }, path)
        print(f"  Saved checkpoint: {path}")
        return True
    except Exception as e:
        print(f"  CHECKPOINT SAVE FAILED: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
