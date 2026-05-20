"""Evaluate student using official MDLM Gen-PPL protocol.

Uses official Diffusion.compute_generative_perplexity() for apples-to-apples
comparison with teacher baseline.
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
from inference import generate_samples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./eval_results")
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
    print(f"Loaded student from {ckpt_path} (step {step})")
    return student, step


def build_gen_ppl_evaluator(device):
    """Build a minimal Diffusion object just for compute_generative_perplexity."""
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
    return evaluator


@torch.no_grad()
def generate_all_samples(student, config, num_samples, batch_size, device):
    all_samples = []
    remaining = num_samples
    generated = 0
    while remaining > 0:
        bs = min(batch_size, remaining)
        samples = generate_samples(student, config, num_samples=bs,
                                   device=device)
        all_samples.append(samples.cpu())
        generated += bs
        remaining -= bs
        if generated % 64 == 0 or remaining == 0:
            print(f"  Generated {generated}/{num_samples}")
    return torch.cat(all_samples, dim=0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config = Config()
    config.device = args.device

    # Load student
    student, step = load_student(args.ckpt, config, args.device)

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    t0 = time.time()
    samples = generate_all_samples(
        student, config, args.num_samples, args.batch_size, args.device)
    gen_time = time.time() - t0
    print(f"Generation done in {gen_time:.1f}s")

    # Safety checks
    n_mask = (samples == config.mask_token_id).sum().item()
    print(f"Remaining MASK: {n_mask}/{samples.numel()}")
    assert n_mask == 0, f"Residual MASK tokens: {n_mask}"

    # Free student
    del student
    torch.cuda.empty_cache()

    # Decode to text
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    text_samples = tokenizer.batch_decode(samples.tolist())

    # Show samples
    for i in range(min(5, len(text_samples))):
        print(f"  Sample {i}: {text_samples[i][:200]}")

    # Official Gen-PPL
    print(f"\nComputing official Gen-PPL...")
    evaluator = build_gen_ppl_evaluator(args.device)
    evaluator.gen_ppl_metric.reset()
    evaluator.compute_generative_perplexity(text_samples)
    gen_ppl = evaluator.gen_ppl_metric.compute().item()

    print(f"\n{'='*60}")
    print(f"  STUDENT EVALUATION — step {step}")
    print(f"{'='*60}")
    print(f"  Checkpoint:      {args.ckpt}")
    print(f"  Num samples:     {args.num_samples}")
    print(f"  Gen time:        {gen_time:.1f}s")
    print(f"  Official Gen-PPL: {gen_ppl:.2f}")
    print(f"{'='*60}")

    # Save
    np.savez(os.path.join(args.output_dir, f"student_official_step{step}.npz"),
             samples=samples.numpy(),
             gen_ppl=gen_ppl, gen_time=gen_time, step=step)


if __name__ == "__main__":
    main()
