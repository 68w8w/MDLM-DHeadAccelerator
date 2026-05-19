#!/usr/bin/env python3
"""Post-stop evaluation: load latest checkpoint, generate samples, diagnostics."""

import os
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import Config
from model import build_student, build_teacher
from inference import generate_samples
from train import KL_loss
from diffusion_utils import forward_noise

SAMPLE_CHARS = 500
CKPT_DIR = "./outputs"


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
    return (log_p[mask].argmax(-1) == x0[mask]).float().mean().item()


def _top1_agreement(log_pT, log_pS, mask):
    if mask.sum() == 0:
        return 0.0
    return (log_pT[mask].argmax(-1) == log_pS[mask].argmax(-1)).float().mean().item()


def find_latest_ckpt():
    pts = [f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_step") and f.endswith(".pt")]
    if not pts:
        return None, 0
    steps = []
    for f in pts:
        m = re.search(r"step(\d+)", f)
        if m:
            steps.append((int(m.group(1)), f))
    steps.sort()
    last_step, last_file = steps[-1]
    return os.path.join(CKPT_DIR, last_file), last_step


def parse_log_losses(log_path="dinner_run.log"):
    """Parse step-level losses from log."""
    losses = []
    with open(log_path) as f:
        for line in f:
            m = re.match(r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)", line)
            if m:
                losses.append((int(m.group(1)), float(m.group(2))))
    return losses


def parse_last_diag(log_path="dinner_run.log"):
    """Parse the last diagnostic block from log."""
    lines = open(log_path).readlines()
    last_diag_idx = None
    for i, line in enumerate(lines):
        if "[diag@" in line:
            last_diag_idx = i
    if last_diag_idx is None:
        return None
    block = []
    for line in lines[last_diag_idx:last_diag_idx+10]:
        block.append(line.rstrip())
    return "\n".join(block)


def main():
    config = Config()
    config.kl_chunk_size = 512
    config.batch_size = 8
    config.max_length = 1024
    config.K = 4
    config.n_bands = 4
    config.num_sample_texts = 4
    device = config.device

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # ── Find latest checkpoint ──
    ckpt_path, ckpt_step = find_latest_ckpt()
    print(f"{'='*72}")
    print(f"  POST-STOP EVALUATION")
    print(f"{'='*72}")
    print(f"  Latest checkpoint: {ckpt_path} (step {ckpt_step})")

    # ── List all checkpoints ──
    pts = sorted([f for f in os.listdir(CKPT_DIR)
                  if f.startswith("ckpt_step") and f.endswith(".pt")])
    print(f"\n  Saved checkpoints ({len(pts)}):")
    for f in pts:
        size = os.path.getsize(os.path.join(CKPT_DIR, f)) / 1e6
        print(f"    {f}  ({size:.0f} MB)")

    # ── Parse log for loss trajectory ──
    losses = parse_log_losses()
    print(f"\n{'='*72}")
    print(f"  LOSS TRAJECTORY (from log)")
    print(f"{'='*72}")
    if losses:
        for step_i, loss_i in losses:
            marker = " <-- last"  if step_i == losses[-1][0] else ""
            print(f"  step {step_i:6d} | loss {loss_i:.4f}{marker}")

        last_n = losses[-min(10, len(losses)):]
        recent_avg = sum(l for _, l in last_n) / len(last_n)
        print(f"\n  Recent {len(last_n)}-point mean loss: {recent_avg:.4f}")
        print(f"  Stopped at step: ~{losses[-1][0]}")

    # ── Parse last diagnostic ──
    last_diag = parse_last_diag()
    if last_diag:
        print(f"\n{'='*72}")
        print(f"  LAST DIAGNOSTIC BLOCK (from log)")
        print(f"{'='*72}")
        print(last_diag)

    # ── Parse speed ──
    with open("dinner_run.log") as f:
        log_text = f.read()
    m = re.search(r"sec/step = ([\d.]+)", log_text)
    sec_per_step = float(m.group(1)) if m else None
    m = re.search(r"peak_mem = ([\d.]+)", log_text)
    warmup_peak = float(m.group(1)) if m else None

    # ── Load model from checkpoint ──
    print(f"\n{'='*72}")
    print(f"  LOADING MODEL FROM CHECKPOINT")
    print(f"{'='*72}")

    print("  Building student...")
    student = build_student(config, device)
    print("  Building teacher...")
    teacher = build_teacher(config, device)
    teacher.eval()

    # Load checkpoint weights
    ckpt = torch.load(ckpt_path, map_location=device)
    student.backbone_loras.load_state_dict(ckpt['student_state_dict']['backbone_loras'])
    student.heads.load_trainable_state_dict(ckpt['student_state_dict']['heads'])
    print(f"  Loaded checkpoint step {ckpt['step']}")

    # ── Save a latest checkpoint ──
    latest_path = os.path.join(CKPT_DIR, f"ckpt_latest_step{ckpt_step}.pt")
    torch.save(ckpt, latest_path)
    print(f"  Saved: {latest_path}")

    # ── Generate samples ──
    print(f"\n{'='*72}")
    print(f"  GENERATING 4 SAMPLES")
    print(f"{'='*72}")
    student.eval()
    with torch.no_grad():
        samples = generate_samples(student, config, num_samples=4, device=device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    MASK_ID = config.mask_token_id

    for i in range(samples.shape[0]):
        n_mask = (samples[i] == MASK_ID).sum().item()
        text = tokenizer.decode(samples[i].tolist(), skip_special_tokens=True)
        print(f"\n  Sample {i}:")
        print(f"    remaining MASK: {n_mask}")
        print(f"    text (first {SAMPLE_CHARS} chars):")
        print(f"    {text[:SAMPLE_CHARS]}")

    # ── Fresh diagnostics on a real batch ──
    print(f"\n{'='*72}")
    print(f"  FRESH DIAGNOSTICS (1 batch, all 4 bands)")
    print(f"{'='*72}")

    from data import get_dataloader
    loader = get_dataloader(config, split="train")
    batch = next(iter(loader))
    x0 = batch['input_ids'].to(device)

    student.eval()
    print(f"\n  {'band':>4} {'ent_T':>7} {'ent_S':>7} {'acc_T':>7} "
          f"{'acc_S':>7} {'agree':>7} {'KL':>8}")

    for band_idx in range(4):
        t_band_high = 1.0 - band_idx / 4
        t_band_low = 1.0 - (band_idx + 1) / 4
        t_mid = (t_band_high + t_band_low) / 2
        t = torch.full((config.batch_size,), t_mid, device=device)

        z = forward_noise(x0, t, MASK_ID)
        mask = (z == MASK_ID)

        with torch.no_grad():
            hidden, c = student.forward_backbone(z, t)
            log_pT = teacher.forward_log_probs(z, t)
            logits_S = student.heads.compute_one_head(
                hidden_src=hidden, z=z, c=c, t_cur=t, band_idx=band_idx)
            log_pS = F.log_softmax(logits_S.float(), dim=-1)

            ent_T = _entropy_on_mask(log_pT, mask)
            ent_S = _entropy_on_mask(log_pS, mask)
            acc_T = _top1_acc_vs_x0(log_pT, x0, mask)
            acc_S = _top1_acc_vs_x0(log_pS, x0, mask)
            agree = _top1_agreement(log_pT, log_pS, mask)
            kl = KL_loss(log_pT, log_pS, mask, chunk_size=512).item()

        print(f"  {band_idx:>4} {ent_T:>7.2f} {ent_S:>7.2f} {acc_T:>7.4f} "
              f"{acc_S:>7.4f} {agree:>7.4f} {kl:>8.3f}")

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  Stopped at step:      ~{losses[-1][0] if losses else 'unknown'}")
    print(f"  Checkpoint loaded:    step {ckpt_step}")
    print(f"  Recent mean loss:     {recent_avg:.4f}" if losses else "")
    if sec_per_step:
        print(f"  sec/step:             {sec_per_step}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak GPU memory:      {peak:.2f} GB")
    print(f"  Checkpoints saved:    {len(pts)}")
    total_mask = (samples == MASK_ID).sum().item()
    print(f"  Samples MASK remain:  {total_mask}")


if __name__ == "__main__":
    main()
