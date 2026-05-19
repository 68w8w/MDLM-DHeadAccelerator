#!/usr/bin/env python3
"""Variant R: Residual D-Head — sanity check + 3000-step training.

Changes from baseline:
    out = hidden_src + delta_scale * delta_proj(dhead_output)
    logits = output_layer(out, c)

Loss: forward KL only (no change).
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer

from config import Config
from data import get_dataloader
from model import build_student, build_teacher
from diffusion_utils import forward_noise, absorbing_reverse_step
from train import KL_loss, get_cosine_schedule_with_warmup, train_step
from inference import generate_samples


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


def diag_all_bands(student, teacher, x0, config):
    """Run diagnostics on all 4 bands. Returns list of dicts."""
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


# ── Sanity check: zero-init residual ─────────────────────────────────

def sanity_check_residual(student, teacher, config, device):
    """At init, delta_proj=0 → out should equal hidden_src exactly."""
    print(f"\n{'='*72}")
    print(f"  SANITY CHECK: zero-init residual")
    print(f"{'='*72}")

    student.eval()
    B, L = 2, config.max_length
    z = torch.randint(0, config.vocab_size, (B, L), device=device)
    t = torch.full((B,), 0.5, device=device)

    with torch.no_grad():
        hidden_src, c = student.forward_backbone(z, t)

        # Get logits from direct output_layer(hidden_src, c) — the "identity" path
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits_ref = student.heads._output_layer(hidden_src, c)
        logits_ref = logits_ref.float()

        # Get logits from D-Head (should match at init since delta_proj=0)
        logits_dhead = student.heads.compute_one_head(
            hidden_src=hidden_src, z=z, c=c, t_cur=t, band_idx=0)
        # Compare on non-MASK columns only (both have -inf at MASK_ID)
        mask_col = config.mask_token_id
        logits_ref_cmp = torch.cat([logits_ref[..., :mask_col],
                                     logits_ref[..., mask_col+1:]], dim=-1)
        logits_dh_cmp = torch.cat([logits_dhead[..., :mask_col],
                                    logits_dhead[..., mask_col+1:]], dim=-1)

    max_err = (logits_ref_cmp - logits_dh_cmp).abs().max().item()
    # Also check delta_scale values
    scales = student.heads.delta_scale.data.tolist()

    print(f"  delta_scale init values: {scales}")
    print(f"  max_abs(logits_ref - logits_dhead) = {max_err:.2e}")

    passed = max_err < 1e-3
    print(f"  {'PASS' if passed else 'FAIL'} (threshold 1e-3)")

    # Check delta_proj is truly zero
    for i, proj in enumerate(student.heads.delta_proj):
        w_max = proj.weight.abs().max().item()
        print(f"  delta_proj[{i}] weight max = {w_max:.2e}")

    return passed


# ── Main ─────────────────────────────────────────────────────────────

def main():
    n_steps = 3000

    config = Config()
    config.total_steps = n_steps
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
    config.seed = 42
    config.backbone_lora_rank = 128
    config.kl_chunk_size = 512
    config.dropout = 0.0
    config.log_every = 50
    config.save_every = 500
    config.sample_every = 500
    config.num_sample_texts = 4
    config.output_dir = "./outputs_residual"
    device = config.device

    SAMPLE_CHARS = 500
    DIAG_EVERY = 500

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"{'='*72}")
    print(f"  RESIDUAL D-HEAD EXPERIMENT — {n_steps} steps")
    print(f"{'='*72}")

    print("Building student...")
    student = build_student(config, device)

    print("Building teacher...")
    teacher = build_teacher(config, device)
    teacher.eval()

    # ── Sanity check ──
    ok = sanity_check_residual(student, teacher, config, device)
    if not ok:
        print("SANITY CHECK FAILED. Aborting.")
        return

    # ── Parameter count ──
    trainable_params = student.get_trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    delta_params = (sum(p.numel() for p in student.heads.delta_proj.parameters())
                    + student.heads.delta_scale.numel())
    print(f"\n  Total trainable: {n_trainable:,}")
    print(f"  delta_proj + delta_scale: {delta_params:,}")

    # ── Data & optimizer ──
    print("\nLoading data...")
    train_loader = get_dataloader(config, split="train")
    train_iter = iter(train_loader)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, n_steps)

    os.makedirs(config.output_dir, exist_ok=True)

    # ── Training loop ──
    student.train()
    losses = []
    t0 = time.time()

    print(f"\nStarting training ({n_steps} steps)...\n")

    for step in range(1, n_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x0 = batch['input_ids'].to(device)
        loss = train_step(student, teacher, x0, config, optimizer, scheduler, step)
        losses.append(loss)

        if loss != loss:
            print(f"FATAL: NaN at step {step}")
            break

        # Regular log
        if step % config.log_every == 0:
            w = min(config.log_every, len(losses))
            avg = sum(losses[-w:]) / w
            lr = scheduler.get_last_lr()[0]
            mem = torch.cuda.max_memory_allocated() / 1e9
            elapsed = time.time() - t0
            # Also log delta_scale
            ds = student.heads.delta_scale.data.tolist()
            ds_str = " ".join(f"{v:.3f}" for v in ds)
            print(f"step {step:5d} | loss {avg:.4f} | lr {lr:.2e} | "
                  f"mem {mem:.2f}GB | ds=[{ds_str}]")

        # Diagnostics
        if step % DIAG_EVERY == 0:
            batch_diag = next(iter(train_loader))
            x0_diag = batch_diag['input_ids'].to(device)
            diags = diag_all_bands(student, teacher, x0_diag, config)
            print_diag_table(diags, f"Diagnostics @ step {step}")
            student.train()

        # Checkpoint
        if step % config.save_every == 0:
            path = os.path.join(config.output_dir, f"ckpt_step{step}.pt")
            torch.save({
                'step': step,
                'student_state_dict': {
                    'backbone_loras': student.backbone_loras.state_dict(),
                    'heads': student.heads.trainable_state_dict(),
                },
                'optimizer': optimizer.state_dict(),
                'config': config,
            }, path)
            print(f"  Saved: {path}")

        # Samples
        if step % config.sample_every == 0:
            print(f"\n  --- Samples @ step {step} ---")
            student.eval()
            with torch.no_grad():
                samples = generate_samples(
                    student, config, num_samples=config.num_sample_texts,
                    device=device)
            for i in range(samples.shape[0]):
                text = tokenizer.decode(samples[i].tolist(),
                                        skip_special_tokens=True)
                n_mask = (samples[i] == config.mask_token_id).sum().item()
                print(f"  Sample {i} (mask={n_mask}): {text[:SAMPLE_CHARS]}")
            student.train()
            print()

    # ── Final summary ──
    total_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{'='*72}")
    print(f"  FINAL SUMMARY — Residual D-Head, {len(losses)} steps")
    print(f"{'='*72}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Peak memory: {peak_mem:.2f} GB")
    print(f"  delta_scale final: {student.heads.delta_scale.data.tolist()}")

    # Loss trajectory
    print(f"\n  Loss trajectory:")
    for w in range(0, len(losses), 500):
        chunk = losses[w:w+500]
        if chunk:
            print(f"    steps {w+1:5d}-{w+len(chunk):5d}: "
                  f"mean={sum(chunk)/len(chunk):.4f}")

    # Final diagnostics
    student.eval()
    batch_final = next(iter(train_loader))
    x0_final = batch_final['input_ids'].to(device)
    final_diags = diag_all_bands(student, teacher, x0_final, config)
    print_diag_table(final_diags, "Final diagnostics")

    # ── Comparison with baseline ──
    print(f"\n{'='*72}")
    print(f"  COMPARISON: Residual D-Head vs Baseline (old forward KL)")
    print(f"{'='*72}")
    print(f"  Baseline @3000 steps (from dinner_run.log):")
    print(f"    loss=2.35  ent_S≈7.4  acc_S≈0.046  agree≈0.144")
    print(f"  Baseline @6500 steps:")
    print(f"    loss=2.37  ent_S≈7.3  acc_S≈0.046  agree≈0.185")
    print(f"")
    print(f"  Residual D-Head @{len(losses)} steps:")
    last500 = losses[-500:] if len(losses) >= 500 else losses
    print(f"    loss={sum(last500)/len(last500):.4f}")
    if final_diags:
        avg_ent_S = sum(d['ent_S'] for d in final_diags) / 4
        avg_acc_S = sum(d['acc_S'] for d in final_diags) / 4
        avg_agree = sum(d['agree'] for d in final_diags) / 4
        print(f"    avg ent_S={avg_ent_S:.2f}  "
              f"avg acc_S={avg_acc_S:.4f}  "
              f"avg agree={avg_agree:.4f}")


if __name__ == "__main__":
    main()
