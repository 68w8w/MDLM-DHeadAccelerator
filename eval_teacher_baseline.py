"""Teacher baseline: generate with MDLM teacher at various NFE, compute Gen-PPL.

Aligned with official MDLM implementation:
- _ddpm_update transition kernel (not simplified absorbing_reverse_step)
- noise_removal with argmax at the end
- Gen-PPL: decode → retokenize → attention_mask → cross_entropy (official protocol)
- _subs_parameterization for teacher forward
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from tqdm import trange

from config import Config
from model import build_teacher


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./eval_results")
    p.add_argument("--steps", type=str, default="16,64,256,1024",
                   help="Comma-separated total denoising steps")
    return p.parse_args()


# ── Official MDLM sampling primitives ─────────────────────────────────

def _mdlm_sample_categorical(categorical_probs):
    """Official MDLM sampling: Gumbel trick on probabilities (not logits)."""
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def subs_log_probs(logits, xt, mask_id):
    """Official MDLM _subs_parameterization:
    mask token → -inf, normalize; unmasked positions → one-hot."""
    logits = logits.float()
    logits[:, :, mask_id] = -1e9
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    unmasked = (xt != mask_id)
    logits[unmasked] = -1e9
    logits[unmasked, xt[unmasked]] = 0.0
    return logits


@torch.no_grad()
def teacher_forward_subs(teacher, z, t_tensor):
    """Teacher forward with official _subs_parameterization."""
    dit = teacher.hf_model.backbone
    if not dit.config.time_conditioning:
        sigma = torch.zeros_like(t_tensor)
    else:
        sigma = t_tensor

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = dit(z, sigma)
        logits = out[0] if isinstance(out, tuple) else out

    return subs_log_probs(logits, z, teacher.mask_token_id)


# ── Generation (official MDLM _ddpm_update + noise_removal) ──────────

@torch.no_grad()
def generate_teacher(teacher, config, num_samples, total_steps,
                     batch_size, device):
    """Generate samples matching official MDLM _ddpm_update + noise_removal."""
    MASK_ID = config.mask_token_id
    L = config.max_length
    eps = 1e-3  # MDLM default sampling_eps

    all_samples = []
    remaining = num_samples
    generated = 0

    while remaining > 0:
        bs = min(batch_size, remaining)
        z = torch.full((bs, L), MASK_ID, device=device, dtype=torch.long)

        # Official MDLM schedule: linspace(1, eps, num_steps+1)
        timesteps = torch.linspace(1, eps, total_steps + 1, device=device)
        dt = (1 - eps) / total_steps

        for i in range(total_steps):
            t = timesteps[i]
            t_tensor = t * torch.ones(bs, device=device)

            # move_chance ≈ t for loglinear noise (verified: diff < 0.001)
            move_chance_t = t
            move_chance_s = (t - dt).clamp(min=0)

            # Teacher forward with subs parameterization
            log_p_x0 = teacher_forward_subs(teacher, z, t_tensor)
            p_x0 = log_p_x0.exp()  # [B, L, V]

            # Official _ddpm_update: full transition kernel q(x_s | x_t)
            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[:, :, MASK_ID] = move_chance_s

            _x = _mdlm_sample_categorical(q_xs)

            copy_flag = (z != MASK_ID).to(z.dtype)
            z = (copy_flag * z + (1 - copy_flag) * _x).long()

        # Official noise_removal: final argmax denoise
        t_eps = timesteps[-1] * torch.ones(bs, device=device)
        log_p_final = teacher_forward_subs(teacher, z, t_eps)
        pred = log_p_final.argmax(dim=-1)
        is_mask = (z == MASK_ID)
        z = torch.where(is_mask, pred, z)

        all_samples.append(z.cpu())
        generated += bs
        remaining -= bs

        if generated % 64 == 0 or remaining == 0:
            n_mask = (z == MASK_ID).float().mean().item()
            print(f"  Generated {generated}/{num_samples}, "
                  f"mask={n_mask:.4f}")

    return torch.cat(all_samples, dim=0)


# ── Gen-PPL (official protocol: decode → retokenize → attn_mask) ─────

@torch.no_grad()
def compute_gen_ppl(samples, sample_tokenizer, eval_model, eval_tokenizer,
                    batch_size, device, max_length=1024):
    """Official MDLM Gen-PPL: decode samples to text, retokenize with eval
    model tokenizer, compute NLL with attention mask."""
    # Decode generated token IDs to text
    texts = sample_tokenizer.batch_decode(samples.tolist(),
                                          skip_special_tokens=True)

    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_tokenizer.pad_token_id = eval_tokenizer.eos_token_id

    all_nll = []

    for i in trange(0, len(texts), batch_size, desc="Gen-PPL"):
        enc = eval_tokenizer(
            texts[i:i + batch_size],
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        logits = eval_model(input_ids, attention_mask=attn_mask).logits[:, :-1]
        targets = input_ids[:, 1:]
        mask = attn_mask[:, 1:].bool()

        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)

        # Only count non-padding tokens
        all_nll.append(nll[mask].cpu())

    total_nll = torch.cat(all_nll)
    mean_nll = total_nll.mean()
    ppl = torch.exp(mean_nll).item()
    return ppl, mean_nll.item()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    steps_list = [int(x) for x in args.steps.split(",")]
    config = Config()
    config.device = args.device

    # Load teacher
    print("Building teacher...")
    teacher = build_teacher(config, args.device)
    print(f"  time_conditioning: "
          f"{teacher.hf_model.backbone.config.time_conditioning}")

    # Load gpt2-large for PPL eval
    print("Loading gpt2-large for Gen-PPL...")
    eval_model = AutoModelForCausalLM.from_pretrained("gpt2-large").eval()
    eval_model = eval_model.to(args.device)

    # Tokenizers
    sample_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eval_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

    print(f"\n{'='*72}")
    print(f"  TEACHER BASELINE (official protocol)")
    print(f"  {args.num_samples} samples, L={config.max_length}")
    print(f"  steps: {steps_list}")
    print(f"  Gen-PPL: decode → retokenize → attn_mask → CE")
    print(f"  noise_removal: argmax")
    print(f"{'='*72}\n")

    all_results = []

    for total_steps in steps_list:
        print(f"\n{'─'*60}")
        print(f"  Teacher NFE={total_steps} (+1 noise_removal)")
        print(f"{'─'*60}")

        t0 = time.time()
        samples = generate_teacher(
            teacher, config, args.num_samples, total_steps,
            args.batch_size, args.device)
        gen_time = time.time() - t0

        # Safety checks
        n_mask = (samples == config.mask_token_id).sum().item()
        assert n_mask == 0, f"Residual MASK tokens: {n_mask}"
        assert samples.max().item() < eval_model.config.vocab_size, (
            f"Token ID {samples.max().item()} >= "
            f"eval model vocab {eval_model.config.vocab_size}")

        print(f"  Gen time: {gen_time:.1f}s | "
              f"mask: {n_mask}/{samples.numel()}")

        # Show samples
        for i in range(min(3, samples.shape[0])):
            text = sample_tokenizer.decode(samples[i].tolist(),
                                           skip_special_tokens=True)
            print(f"  Sample {i}: {text[:200]}")

        # Gen-PPL (official protocol)
        ppl, mean_nll = compute_gen_ppl(
            samples, sample_tokenizer, eval_model, eval_tokenizer,
            args.batch_size, args.device)
        print(f"  Gen-PPL: {ppl:.2f}  (mean NLL: {mean_nll:.4f})")

        all_results.append({
            'total_steps': total_steps,
            'ppl': ppl,
            'mean_nll': mean_nll,
            'gen_time': gen_time,
            'mask_remaining': n_mask,
        })

        # Save samples
        np.savez(os.path.join(args.output_dir,
                              f"teacher_samples_nfe{total_steps}.npz"),
                 samples=samples.numpy())

    # Summary
    print(f"\n{'='*72}")
    print(f"  TEACHER BASELINE SUMMARY (official Gen-PPL)")
    print(f"{'='*72}")
    print(f"  {'NFE':>6}  {'Gen-PPL':>10}  {'NLL':>8}  {'mask':>8}  {'time':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
    for r in all_results:
        print(f"  {r['total_steps']:6d}  {r['ppl']:10.2f}  "
              f"{r['mean_nll']:8.4f}  "
              f"{r['mask_remaining']:8d}  {r['gen_time']:7.1f}s")
    print(f"{'='*72}")

    np.savez(os.path.join(args.output_dir, "teacher_baseline.npz"),
             steps=[r['total_steps'] for r in all_results],
             ppl=[r['ppl'] for r in all_results],
             mean_nll=[r['mean_nll'] for r in all_results],
             gen_time=[r['gen_time'] for r in all_results])


if __name__ == "__main__":
    main()
