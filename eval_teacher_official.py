"""Teacher baseline using official MDLM Diffusion._sample() + compute_generative_perplexity().

No hand-written sampling logic — directly calls the official codebase.
"""

import sys
import os
import argparse
import time

# Add MDLM repo to path
MDLM_DIR = "/data1/wulingdan/data/diffusion/mdlm"
sys.path.insert(0, MDLM_DIR)

import numpy as np
import torch
import omegaconf
import transformers

# Official MDLM imports
import diffusion as mdlm_diffusion
import noise_schedule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./eval_results")
    p.add_argument("--steps", type=str, default="16,64,256,1024",
                   help="Comma-separated total denoising steps")
    return p.parse_args()


def build_official_config(num_steps=128):
    """Build a minimal OmegaConf config for Diffusion."""
    cfg = omegaconf.OmegaConf.create({
        "backbone": "hf_dit",
        "model": {
            "name": "kuleshov-group/mdlm-owt",
            "length": 1024,
        },
        "parameterization": "subs",
        "subs_masking": False,
        "time_conditioning": False,
        "T": 0,
        "sampling": {
            "predictor": "ddpm_cache",
            "steps": num_steps,
            "noise_removal": True,
            "semi_ar": False,
            "stride_length": 1,
            "num_strides": 1,
            "num_sample_batches": 1,
            "num_sample_log": 2,
        },
        "training": {
            "ema": 0.0,
            "antithetic_sampling": True,
            "importance_sampling": False,
            "sampling_eps": 1e-3,
            "change_of_variables": False,
        },
        "eval": {
            "checkpoint_path": "kuleshov-group/mdlm-owt",
            "disable_ema": True,
            "compute_generative_perplexity": True,
            "perplexity_batch_size": 8,
            "gen_ppl_eval_model_name_or_path": "gpt2-large",
            "generate_samples": True,
        },
        "loader": {
            "eval_batch_size": 8,
        },
        "noise": {
            "type": "loglinear",
        },
        "optim": {
            "lr": 3e-4,
            "weight_decay": 0,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
    })
    return cfg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    steps_list = [int(x) for x in args.steps.split(",")]

    # Build config and load official Diffusion model
    config = build_official_config()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    print("Building official MDLM Diffusion model...")
    model = mdlm_diffusion.Diffusion(config, tokenizer=tokenizer)
    model = model.to(args.device)
    model.eval()

    # Disable EMA (we don't have a checkpoint with EMA weights)
    model.ema = None

    print(f"  backbone: {config.backbone}")
    print(f"  parameterization: {config.parameterization}")
    print(f"  time_conditioning: {config.time_conditioning}")
    print(f"  noise_removal: {config.sampling.noise_removal}")
    print(f"  sampler: {config.sampling.predictor}")
    print(f"  mask_index: {model.mask_index}")
    print(f"  vocab_size: {model.vocab_size}")

    print(f"\n{'='*72}")
    print(f"  OFFICIAL MDLM TEACHER BASELINE")
    print(f"  {args.num_samples} samples, L={config.model.length}")
    print(f"  steps: {steps_list}")
    print(f"{'='*72}\n")

    all_results = []

    for num_steps in steps_list:
        print(f"\n{'─'*60}")
        print(f"  Official MDLM NFE={num_steps}")
        print(f"{'─'*60}")

        # Update config sampling steps
        config.sampling.steps = num_steps

        # Generate in batches
        all_samples_ids = []
        all_text_samples = []
        t0 = time.time()
        num_batches = args.num_samples // config.loader.eval_batch_size

        for b in range(num_batches):
            with torch.no_grad():
                samples = model._sample(num_steps=num_steps)
            all_samples_ids.append(samples.cpu())
            texts = tokenizer.batch_decode(samples)
            all_text_samples.extend(texts)

            if (b + 1) % 16 == 0 or b == num_batches - 1:
                generated = (b + 1) * config.loader.eval_batch_size
                n_mask = (samples == model.mask_index).float().mean().item()
                print(f"  Generated {generated}/{args.num_samples}, "
                      f"mask={n_mask:.4f}")

        gen_time = time.time() - t0
        all_ids = torch.cat(all_samples_ids, dim=0)

        # Safety checks
        n_mask = (all_ids == model.mask_index).sum().item()
        print(f"  Gen time: {gen_time:.1f}s | "
              f"total mask: {n_mask}/{all_ids.numel()}")

        # Show samples
        for i in range(min(3, len(all_text_samples))):
            print(f"  Sample {i}: {all_text_samples[i][:200]}")

        # Official Gen-PPL using model.compute_generative_perplexity
        model.gen_ppl_metric.reset()
        model.compute_generative_perplexity(all_text_samples)
        gen_ppl = model.gen_ppl_metric.compute().item()
        print(f"  Official Gen-PPL: {gen_ppl:.2f}")

        all_results.append({
            'num_steps': num_steps,
            'gen_ppl': gen_ppl,
            'gen_time': gen_time,
            'mask_remaining': n_mask,
        })

        # Save samples
        np.savez(os.path.join(args.output_dir,
                              f"official_teacher_nfe{num_steps}.npz"),
                 samples=all_ids.numpy())

    # Summary
    print(f"\n{'='*72}")
    print(f"  OFFICIAL MDLM TEACHER BASELINE SUMMARY")
    print(f"{'='*72}")
    print(f"  {'NFE':>6}  {'Gen-PPL':>10}  {'mask':>8}  {'time':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}")
    for r in all_results:
        print(f"  {r['num_steps']:6d}  {r['gen_ppl']:10.2f}  "
              f"{r['mask_remaining']:8d}  {r['gen_time']:7.1f}s")
    print(f"{'='*72}")

    np.savez(os.path.join(args.output_dir, "official_teacher_baseline.npz"),
             steps=[r['num_steps'] for r in all_results],
             gen_ppl=[r['gen_ppl'] for r in all_results],
             gen_time=[r['gen_time'] for r in all_results])


if __name__ == "__main__":
    main()
