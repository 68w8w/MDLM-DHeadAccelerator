"""Ablation: stale backbone (1x per band) vs refresh backbone (every substep).

If refresh >> stale in Gen-PPL, stale hidden is confirmed as main bottleneck.
"""

import sys
import os
import argparse
import time

MDLM_DIR = "/data1/wulingdan/data/diffusion/mdlm"
sys.path.insert(0, MDLM_DIR)

import numpy as np
import torch
import omegaconf
import transformers
import diffusion as mdlm_diffusion

from config import Config
from model import build_student
from inference import generate_samples_official, generate_samples_refresh


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_student(ckpt_path, config, device):
    student = build_student(config, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    student.backbone_loras.load_state_dict(
        ckpt['student_state_dict']['backbone_loras'])
    student.heads.load_trainable_state_dict(
        ckpt['student_state_dict']['heads'])
    student.eval()
    step = ckpt.get('step', 'unknown')
    print(f"Loaded step {step}")
    return student, step


def build_evaluator(device):
    cfg = omegaconf.OmegaConf.create({
        "backbone": "hf_dit",
        "model": {"name": "kuleshov-group/mdlm-owt", "length": 1024},
        "parameterization": "subs", "subs_masking": False,
        "time_conditioning": False, "T": 0,
        "sampling": {"predictor": "ddpm_cache", "steps": 16,
                     "noise_removal": True, "semi_ar": False,
                     "stride_length": 1, "num_strides": 1,
                     "num_sample_batches": 1, "num_sample_log": 2},
        "training": {"ema": 0.0, "antithetic_sampling": True,
                     "importance_sampling": False, "sampling_eps": 1e-3,
                     "change_of_variables": False},
        "eval": {"checkpoint_path": "kuleshov-group/mdlm-owt",
                 "disable_ema": True, "compute_generative_perplexity": True,
                 "perplexity_batch_size": 8,
                 "gen_ppl_eval_model_name_or_path": "gpt2-large",
                 "generate_samples": True},
        "loader": {"eval_batch_size": 8},
        "noise": {"type": "loglinear"},
        "optim": {"lr": 3e-4, "weight_decay": 0, "beta1": 0.9,
                  "beta2": 0.999, "eps": 1e-8},
    })
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    evaluator = mdlm_diffusion.Diffusion(cfg, tokenizer=tokenizer)
    evaluator = evaluator.to(device).eval()
    evaluator.ema = None
    return evaluator, tokenizer


def gen_batched(student, config, num_samples, batch_size, device, gen_fn):
    all_samples = []
    remaining = num_samples
    generated = 0
    while remaining > 0:
        bs = min(batch_size, remaining)
        samples = gen_fn(student, config, num_samples=bs, device=device)
        all_samples.append(samples.cpu())
        generated += bs
        remaining -= bs
        if generated % 64 == 0 or remaining == 0:
            print(f"    {generated}/{num_samples}")
    return torch.cat(all_samples, dim=0)


def eval_ppl(samples, evaluator, tokenizer):
    texts = tokenizer.batch_decode(samples.tolist())
    evaluator.gen_ppl_metric.reset()
    evaluator.compute_generative_perplexity(texts)
    return evaluator.gen_ppl_metric.compute().item()


def main():
    args = parse_args()
    config = Config()
    config.device = args.device

    student, step = load_student(args.ckpt, config, args.device)
    evaluator, tokenizer = build_evaluator(args.device)

    # ── Path A: stale backbone (1x per band) ──
    print(f"\n{'='*60}")
    print(f"  A: STALE backbone (1x per band, NFE=16)")
    print(f"{'='*60}")
    torch.manual_seed(args.seed)
    t0 = time.time()
    samples_stale = gen_batched(student, config, args.num_samples,
                                args.batch_size, args.device,
                                generate_samples_official)
    time_stale = time.time() - t0
    ppl_stale = eval_ppl(samples_stale, evaluator, tokenizer)
    for i in range(3):
        print(f"  Sample {i}: {tokenizer.decode(samples_stale[i].tolist())[:200]}")
    print(f"  Gen-PPL: {ppl_stale:.2f}  ({time_stale:.1f}s)")

    # ── Path B: refresh backbone (every substep, NFE=32) ──
    print(f"\n{'='*60}")
    print(f"  B: REFRESH backbone (every substep, NFE=32)")
    print(f"{'='*60}")
    torch.manual_seed(args.seed)
    t0 = time.time()
    samples_refresh = gen_batched(student, config, args.num_samples,
                                  args.batch_size, args.device,
                                  generate_samples_refresh)
    time_refresh = time.time() - t0
    ppl_refresh = eval_ppl(samples_refresh, evaluator, tokenizer)
    for i in range(3):
        print(f"  Sample {i}: {tokenizer.decode(samples_refresh[i].tolist())[:200]}")
    print(f"  Gen-PPL: {ppl_refresh:.2f}  ({time_refresh:.1f}s)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  STALE vs REFRESH — step {step}")
    print(f"{'='*60}")
    print(f"  Stale (1x/band):    Gen-PPL = {ppl_stale:.2f}")
    print(f"  Refresh (every sub): Gen-PPL = {ppl_refresh:.2f}")
    ratio = ppl_stale / max(ppl_refresh, 1)
    print(f"  Ratio: {ratio:.2f}x")
    if ppl_refresh < ppl_stale * 0.5:
        print(f"  → Stale hidden is the PRIMARY bottleneck.")
    elif ppl_refresh < ppl_stale * 0.8:
        print(f"  → Stale hidden matters, but not the only issue.")
    else:
        print(f"  → Stale hidden is NOT the main issue.")


if __name__ == "__main__":
    main()
