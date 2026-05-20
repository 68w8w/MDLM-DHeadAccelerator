"""Sweep inference substeps_per_band and compare PPL / MAUVE.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u eval_substeps_sweep.py \
        --ckpt ./outputs_stm_30k/ckpt_step18500.pt
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from tqdm import trange

import mauve

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
    p.add_argument("--substeps", type=str, default="16,32,64,128,256",
                   help="Comma-separated substeps_per_band values")
    # MAUVE params
    p.add_argument("--mauve_max_tokens", type=int, default=100)
    p.add_argument("--mauve_scaling_factor", type=float, default=5.0)
    p.add_argument("--mauve_num_rounds", type=int, default=5)
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


@torch.no_grad()
def generate_all_samples(student, config, num_samples, batch_size, device,
                         substeps_per_band):
    all_samples = []
    remaining = num_samples
    generated = 0
    while remaining > 0:
        bs = min(batch_size, remaining)
        samples = generate_samples(student, config, num_samples=bs,
                                   device=device,
                                   substeps_per_band=substeps_per_band)
        all_samples.append(samples.cpu())
        generated += bs
        remaining -= bs
    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def compute_ppl(samples, ar_model, batch_size, device):
    all_losses = []
    n = samples.shape[0]
    for idx in range(0, n, batch_size):
        batch = samples[idx:idx + batch_size].to(device)
        logits = ar_model(batch).logits[:, :-1]
        logits = torch.log_softmax(logits.float(), dim=-1)
        targets = batch[:, 1:]
        nll = -torch.gather(logits, dim=-1,
                            index=targets.unsqueeze(-1))[..., 0]
        per_sample_loss = nll.mean(dim=-1)
        all_losses.extend(per_sample_loss.cpu().tolist())
    all_losses = torch.tensor(all_losses)
    ppl = all_losses.mean().exp().item()
    return ppl


@torch.no_grad()
def compute_mauve_score(samples, references, feature_extractor,
                        max_tokens, scaling_factor, num_rounds, batch_size):
    samples_trunc = samples[:, :max_tokens]
    references_trunc = references[:, :max_tokens]

    q_features = mauve.utils.featurize_tokens_from_model(
        model=feature_extractor,
        tokenized_texts=samples_trunc,
        batch_size=batch_size,
        name="generated",
    ).numpy()

    p_features = mauve.utils.featurize_tokens_from_model(
        model=feature_extractor,
        tokenized_texts=references_trunc,
        batch_size=batch_size,
        name="reference",
    ).numpy()

    results = []
    for i in range(num_rounds):
        res = mauve.compute_mauve(
            p_features=p_features, q_features=q_features,
            seed=1 + i, device_id=0, verbose=False,
            batch_size=batch_size,
            mauve_scaling_factor=scaling_factor,
        ).mauve
        results.append(float(res))

    return np.mean(results), np.std(results), results


def load_references(config, num_samples):
    import datasets
    hf_ds = datasets.load_from_disk(config.valid_data)
    refs = []
    for idx in range(min(num_samples, len(hf_ds))):
        ids = hf_ds[idx]['input_ids'][:config.max_length]
        refs.append(torch.tensor(ids, dtype=torch.long))
    return torch.stack(refs, dim=0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    substeps_list = [int(x) for x in args.substeps.split(",")]
    config = Config()
    config.device = args.device

    # Load student once
    student, step = load_student(args.ckpt, config, args.device)

    # Load eval models once
    print("Loading gpt2-large for PPL...")
    ar_model = AutoModelForCausalLM.from_pretrained("gpt2-large").eval()
    ar_model = ar_model.to(args.device)

    print("Loading gpt2-large for MAUVE features...")
    feature_extractor = AutoModel.from_pretrained("gpt2-large").eval()
    feature_extractor = feature_extractor.to(args.device)

    # Load references once
    references = load_references(config, args.num_samples)

    # Also show a few sample texts
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"\n{'='*72}")
    print(f"  SUBSTEPS SWEEP — ckpt step {step}, {args.num_samples} samples")
    print(f"  substeps: {substeps_list}")
    print(f"{'='*72}\n")

    all_results = []

    for K in substeps_list:
        total_steps = 4 * K  # 4 bands × K substeps
        print(f"\n{'─'*60}")
        print(f"  substeps_per_band={K}  (total denoising steps={total_steps})")
        print(f"{'─'*60}")

        # Generate
        t0 = time.time()
        samples = generate_all_samples(
            student, config, args.num_samples, args.batch_size,
            args.device, substeps_per_band=K)
        gen_time = time.time() - t0

        n_mask = (samples == config.mask_token_id).sum().item()
        print(f"  Gen time: {gen_time:.1f}s | "
              f"mask remaining: {n_mask}/{samples.numel()}")

        # Show 3 samples
        for i in range(min(3, samples.shape[0])):
            text = tokenizer.decode(samples[i].tolist(),
                                    skip_special_tokens=True)
            print(f"  Sample {i}: {text[:200]}")

        # PPL
        ppl = compute_ppl(samples, ar_model, args.batch_size, args.device)
        print(f"  PPL: {ppl:.2f}")

        # MAUVE
        mauve_mean, mauve_std, mauve_rounds = compute_mauve_score(
            samples, references, feature_extractor,
            args.mauve_max_tokens, args.mauve_scaling_factor,
            args.mauve_num_rounds, args.batch_size)
        print(f"  MAUVE: {mauve_mean:.4f} ± {mauve_std:.4f}")

        all_results.append({
            'substeps_per_band': K,
            'total_steps': total_steps,
            'ppl': ppl,
            'mauve_mean': mauve_mean,
            'mauve_std': mauve_std,
            'gen_time': gen_time,
            'mask_remaining': n_mask,
        })

        # Save samples
        np.savez(os.path.join(args.output_dir,
                              f"samples_step{step}_K{K}.npz"),
                 samples=samples.numpy())

    # Final summary table
    print(f"\n{'='*72}")
    print(f"  SUMMARY — step {step}")
    print(f"{'='*72}")
    print(f"  {'K':>5}  {'total':>6}  {'PPL':>10}  {'MAUVE':>12}  {'time':>8}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*8}")
    for r in all_results:
        print(f"  {r['substeps_per_band']:5d}  {r['total_steps']:6d}  "
              f"{r['ppl']:10.2f}  "
              f"{r['mauve_mean']:.4f}±{r['mauve_std']:.4f}  "
              f"{r['gen_time']:7.1f}s")
    print(f"{'='*72}")

    # Save all results
    results_path = os.path.join(args.output_dir,
                                f"sweep_step{step}.npz")
    np.savez(results_path,
             substeps=[r['substeps_per_band'] for r in all_results],
             ppl=[r['ppl'] for r in all_results],
             mauve_mean=[r['mauve_mean'] for r in all_results],
             mauve_std=[r['mauve_std'] for r in all_results],
             gen_time=[r['gen_time'] for r in all_results])
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
