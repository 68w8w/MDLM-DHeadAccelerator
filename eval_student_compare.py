"""Compare student Gen-PPL: old inference vs official-compatible inference.

Both use official MDLM compute_generative_perplexity for PPL.
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
from inference import generate_samples, generate_samples_official


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
            print(f"    Generated {generated}/{num_samples}")
    return torch.cat(all_samples, dim=0)


def eval_samples(samples, evaluator, tokenizer, config, label):
    """Evaluate samples and print detailed stats."""
    n_mask = (samples == config.mask_token_id).sum().item()
    tok_max = samples.max().item()
    tok_min = samples.min().item()

    texts = tokenizer.batch_decode(samples.tolist())

    # EOS stats
    eos_id = tokenizer.eos_token_id
    eos_counts = [(s == eos_id).sum().item() for s in samples]
    text_lens = [len(t) for t in texts]

    print(f"\n  [{label}] Stats:")
    print(f"    mask_remaining: {n_mask}/{samples.numel()}")
    print(f"    token range: [{tok_min}, {tok_max}]")
    print(f"    EOS per sample: mean={np.mean(eos_counts):.1f}, "
          f"min={min(eos_counts)}, max={max(eos_counts)}")
    print(f"    text length: mean={np.mean(text_lens):.0f}, "
          f"min={min(text_lens)}, max={max(text_lens)}")

    # Show 3 samples
    for i in range(min(3, len(texts))):
        print(f"    Sample {i}: {texts[i][:200]}")

    # Official Gen-PPL
    evaluator.gen_ppl_metric.reset()
    evaluator.compute_generative_perplexity(texts)
    gen_ppl = evaluator.gen_ppl_metric.compute().item()
    print(f"    Official Gen-PPL: {gen_ppl:.2f}")

    return gen_ppl


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config = Config()
    config.device = args.device

    student, step = load_student(args.ckpt, config, args.device)
    evaluator, tokenizer = build_evaluator(args.device)

    results = {}

    # ── Path A: old inference (absorbing_reverse_step + noise_removal) ──
    print(f"\n{'='*60}")
    print(f"  PATH A: old inference (absorbing_reverse_step)")
    print(f"{'='*60}")
    torch.manual_seed(args.seed)
    t0 = time.time()
    samples_old = gen_batched(student, config, args.num_samples,
                              args.batch_size, args.device,
                              generate_samples)
    time_old = time.time() - t0
    ppl_old = eval_samples(samples_old, evaluator, tokenizer, config,
                           "old inference")
    results['old'] = {'ppl': ppl_old, 'time': time_old}

    # ── Path B: official-compatible inference (_ddpm_update) ──
    print(f"\n{'='*60}")
    print(f"  PATH B: official-compatible inference (_ddpm_update)")
    print(f"{'='*60}")
    torch.manual_seed(args.seed)
    t0 = time.time()
    samples_new = gen_batched(student, config, args.num_samples,
                              args.batch_size, args.device,
                              generate_samples_official)
    time_new = time.time() - t0
    ppl_new = eval_samples(samples_new, evaluator, tokenizer, config,
                           "official-compatible inference")
    results['official'] = {'ppl': ppl_new, 'time': time_new}

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"  COMPARISON TABLE — student step {step}")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Sampling':<30} {'NFE':>4} {'Gen-PPL':>10}")
    print(f"  {'─'*25} {'─'*30} {'─'*4} {'─'*10}")
    print(f"  {'Teacher':<25} {'official Diffusion._sample':<30} {'16':>4} {'305.79':>10}")
    print(f"  {'Teacher':<25} {'official Diffusion._sample':<30} {'64':>4} {'107.58':>10}")
    print(f"  {'Student (step '+str(step)+')':<25} {'old absorbing_reverse_step':<30} {'16':>4} {ppl_old:>10.2f}")
    print(f"  {'Student (step '+str(step)+')':<25} {'official-compatible':<30} {'16':>4} {ppl_new:>10.2f}")
    print(f"{'='*60}")

    # ── Diagnosis ──
    print(f"\n  DIAGNOSIS:")
    if ppl_new > 1000:
        print(f"  → Gen-PPL still >1000 with official sampler.")
        print(f"  → Issue is D-Head quality, not sampling path.")
        print(f"  → Consider: larger D-Head, more bands, LoRA rank, or fine-tune.")
    elif ppl_new < 800:
        print(f"  → Gen-PPL dropped significantly with official sampler!")
        print(f"  → Old training is valid; sampler mismatch was the issue.")
    diff = abs(ppl_old - ppl_new) / max(ppl_old, 1)
    if diff > 0.2:
        print(f"  → Old vs official sampler diff: {diff*100:.1f}% — sampler matters.")
    else:
        print(f"  → Old vs official sampler diff: {diff*100:.1f}% — sampler not the issue.")

    # Save
    np.savez(os.path.join(args.output_dir, f"compare_step{step}.npz"),
             ppl_old=ppl_old, ppl_new=ppl_new,
             time_old=time_old, time_new=time_new)


if __name__ == "__main__":
    main()
