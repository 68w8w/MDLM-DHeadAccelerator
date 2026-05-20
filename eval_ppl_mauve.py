"""Evaluate D-Head student: generate 1024 unconditional samples, compute PPL and MAUVE.

PPL: gpt2-large teacher-forced NLL, following SDTT protocol.
MAUVE: gpt2-large feature extractor, token-level, scaling_factor=5, 5 seeds.
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


# ── Argument parsing ──────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to student checkpoint (.pt)")
    p.add_argument("--num_samples", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for generation and eval")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./eval_results")
    # MAUVE params (SDTT defaults)
    p.add_argument("--mauve_max_tokens", type=int, default=100)
    p.add_argument("--mauve_scaling_factor", type=float, default=5.0)
    p.add_argument("--mauve_num_rounds", type=int, default=5)
    return p.parse_args()


# ── Load student from checkpoint ──────────────────────────────────────

def load_student(ckpt_path, config, device):
    """Build student and load trained weights from checkpoint."""
    student = build_student(config, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Load LoRA weights
    student.backbone_loras.load_state_dict(
        ckpt['student_state_dict']['backbone_loras'])
    # Load D-Head weights
    student.heads.load_trainable_state_dict(
        ckpt['student_state_dict']['heads'])

    student.eval()
    step = ckpt.get('step', 'unknown')
    print(f"Loaded student from {ckpt_path} (step {step})")
    return student, step


# ── Generate samples ──────────────────────────────────────────────────

@torch.no_grad()
def generate_all_samples(student, config, num_samples, batch_size, device):
    """Generate num_samples sequences in batches."""
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
        n_mask = (samples == config.mask_token_id).float().mean().item()
        print(f"  Generated {generated}/{num_samples}, "
              f"remaining_mask={n_mask:.4f}")

    return torch.cat(all_samples, dim=0)  # [num_samples, L]


# ── Gen-PPL (official: decode → retokenize → attn_mask → CE) ─────────

@torch.no_grad()
def compute_ppl(samples, batch_size, device):
    """Compute Gen-PPL using gpt2-large, following official MDLM protocol.

    decode samples → retokenize with eval tokenizer → NLL with attn_mask.
    """
    print("\n" + "=" * 60)
    print("  Computing Gen-PPL with gpt2-large (official protocol)")
    print("=" * 60)

    ar_model = AutoModelForCausalLM.from_pretrained("gpt2-large").eval()
    ar_model = ar_model.to(device)

    sample_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eval_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_tokenizer.pad_token_id = eval_tokenizer.eos_token_id

    # Safety: no MASK tokens should remain
    from config import Config
    cfg = Config()
    assert (samples == cfg.mask_token_id).sum().item() == 0, \
        "Residual MASK tokens in samples!"
    assert samples.max().item() < ar_model.config.vocab_size, \
        f"Token ID {samples.max().item()} >= vocab {ar_model.config.vocab_size}"

    # Decode to text
    texts = sample_tokenizer.batch_decode(samples.tolist(),
                                          skip_special_tokens=True)

    all_nll = []
    n = len(texts)

    for i in trange(0, n, batch_size, desc="Gen-PPL"):
        enc = eval_tokenizer(
            texts[i:i + batch_size],
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        logits = ar_model(input_ids, attention_mask=attn_mask).logits[:, :-1]
        targets = input_ids[:, 1:]
        mask = attn_mask[:, 1:].bool()

        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)

        all_nll.append(nll[mask].cpu())

    total_nll = torch.cat(all_nll)
    avg_nll = total_nll.mean()
    ppl = torch.exp(avg_nll).item()

    print(f"  Mean NLL: {avg_nll.item():.4f}")
    print(f"  Gen-PPL: {ppl:.2f}")

    del ar_model, sample_tokenizer, eval_tokenizer
    torch.cuda.empty_cache()

    return ppl, total_nll.numpy()


# ── MAUVE (SDTT protocol: gpt2-large features, token-level) ──────────

@torch.no_grad()
def compute_mauve_score(samples, references, max_tokens, scaling_factor,
                        num_rounds, batch_size, device):
    """Compute MAUVE following SDTT protocol.

    Uses gpt2-large base model (no LM head) as feature extractor.
    Token-level featurization, first max_tokens tokens only.
    """
    print("\n" + "=" * 60)
    print("  Computing MAUVE with gpt2-large features")
    print("=" * 60)

    feature_extractor = AutoModel.from_pretrained("gpt2-large").eval()
    feature_extractor = feature_extractor.to(device)

    # Truncate to first max_tokens tokens
    samples_trunc = samples[:, :max_tokens]
    references_trunc = references[:, :max_tokens]

    print(f"  Samples shape: {samples_trunc.shape}")
    print(f"  References shape: {references_trunc.shape}")

    # Token-level featurization via mauve utility
    q_features = mauve.utils.featurize_tokens_from_model(
        model=feature_extractor,
        tokenized_texts=samples_trunc,
        batch_size=batch_size,
        name="generated samples",
    ).numpy()

    p_features = mauve.utils.featurize_tokens_from_model(
        model=feature_extractor,
        tokenized_texts=references_trunc,
        batch_size=batch_size,
        name="references",
    ).numpy()

    # Run MAUVE with multiple seeds
    mauve_results = []
    for run_idx in range(num_rounds):
        res = mauve.compute_mauve(
            p_features=p_features,
            q_features=q_features,
            seed=1 + run_idx,
            device_id=0,
            verbose=False,
            batch_size=batch_size,
            mauve_scaling_factor=scaling_factor,
        ).mauve
        mauve_results.append(float(res))
        print(f"  Round {run_idx + 1}: {res:.4f}")

    mauve_mean = np.mean(mauve_results)
    mauve_std = np.std(mauve_results)

    print(f"  MAUVE: {mauve_mean:.4f} +/- {mauve_std:.4f}")

    del feature_extractor
    torch.cuda.empty_cache()

    return mauve_mean, mauve_std, mauve_results


# ── Load reference data ───────────────────────────────────────────────

def load_references(config, num_samples):
    """Load reference sequences from validation set."""
    import datasets
    path = config.valid_data
    print(f"Loading references from {path}")
    hf_ds = datasets.load_from_disk(path)

    indices = list(range(min(num_samples, len(hf_ds))))
    refs = []
    for idx in indices:
        ids = hf_ds[idx]['input_ids'][:config.max_length]
        refs.append(torch.tensor(ids, dtype=torch.long))

    refs = torch.stack(refs, dim=0)  # [num_samples, L]
    print(f"  References: {refs.shape}")
    return refs


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    config = Config()
    config.device = args.device

    # ── Load student ──
    student, step = load_student(args.ckpt, config, args.device)

    # ── Generate samples ──
    print(f"\nGenerating {args.num_samples} samples...")
    t0 = time.time()
    samples = generate_all_samples(
        student, config, args.num_samples, args.batch_size, args.device)
    gen_time = time.time() - t0
    print(f"Generation done in {gen_time:.1f}s "
          f"({gen_time / args.num_samples:.2f}s per sample)")

    # Check for remaining masks
    n_mask = (samples == config.mask_token_id).sum().item()
    n_total = samples.numel()
    print(f"Remaining MASK tokens: {n_mask}/{n_total} "
          f"({n_mask / n_total * 100:.2f}%)")

    # Save generated samples
    samples_path = os.path.join(args.output_dir,
                                f"samples_step{step}.npz")
    np.savez(samples_path, samples=samples.numpy(), step=step)
    print(f"Saved samples to {samples_path}")

    # Free student model memory
    del student
    torch.cuda.empty_cache()

    # ── PPL ──
    ppl, per_sample_nll = compute_ppl(samples, args.batch_size, args.device)

    # ── MAUVE ──
    references = load_references(config, args.num_samples)
    mauve_mean, mauve_std, mauve_rounds = compute_mauve_score(
        samples=samples,
        references=references,
        max_tokens=args.mauve_max_tokens,
        scaling_factor=args.mauve_scaling_factor,
        num_rounds=args.mauve_num_rounds,
        batch_size=args.batch_size,
        device=args.device,
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS — step {step}")
    print("=" * 60)
    print(f"  Checkpoint:  {args.ckpt}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Gen time:    {gen_time:.1f}s ({gen_time / args.num_samples:.2f}s/sample)")
    print(f"  PPL:         {ppl:.2f}")
    print(f"  MAUVE:       {mauve_mean:.4f} +/- {mauve_std:.4f}")
    print(f"  MAUVE rounds: {mauve_rounds}")
    print("=" * 60)

    # Save results
    results = {
        'step': step,
        'ckpt': args.ckpt,
        'num_samples': args.num_samples,
        'ppl': ppl,
        'per_sample_nll': per_sample_nll,
        'mauve_mean': mauve_mean,
        'mauve_std': mauve_std,
        'mauve_rounds': mauve_rounds,
        'gen_time': gen_time,
    }
    results_path = os.path.join(args.output_dir, f"results_step{step}.npz")
    np.savez(results_path, **{k: np.array(v) for k, v in results.items()})
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
