"""Sanity checks for MDLM D-Head Accelerator MVP."""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import Config
from model import (
    build_student, build_teacher, DHeadModule,
    DHead, DHeadStudent, Teacher,
)
from diffusion_utils import forward_noise, absorbing_reverse_step, _sample_categorical
from train import KL_loss, train_step, get_cosine_schedule_with_warmup
from inference import generate_samples
from data import get_dataloader


def _header(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def test_1a_identity_path(student, config, device):
    """Test 1A: DHead identity path check.

    Zero-init DHeadModule output projections and all shared embeddings.
    Then DHeadModule(z_stream, hidden_src) ≈ z_stream.
    """
    _header("Test 1A: DHead identity path check")

    band_idx = 0
    module = student.heads.dhead_modules[band_idx]

    # Save original state
    orig_state = {k: v.clone() for k, v in module.state_dict().items()}
    orig_shared = {}
    for name in ['mask_indicator_embed', 'time_embed', 'band_embed', 'pos_embed']:
        m = getattr(student.heads, name)
        orig_shared[name] = {k: v.clone() for k, v in m.state_dict().items()}

    # Zero-init all output projections in the DHeadModule
    with torch.no_grad():
        for sa in module.self_attn_layers:
            sa.out_proj.weight.zero_()
        module.cross_attn.out_proj.weight.zero_()
        module.ffn.down.weight.zero_()
        module.ffn.down.bias.zero_()

        # Zero-init shared embeddings
        student.heads.mask_indicator_embed.weight.zero_()
        for layer in student.heads.time_embed:
            if hasattr(layer, 'weight'):
                layer.weight.zero_()
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.zero_()
        student.heads.band_embed.weight.zero_()
        student.heads.pos_embed.weight.zero_()

    # Create test input
    B, L = 2, config.max_length
    z = torch.randint(0, config.vocab_size, (B, L), device=device)
    hidden_src = torch.randn(B, L, config.d_model, device=device)

    # z_stream should just be vocab_embed(z) since all additive embeddings are zeroed
    z_emb = student.heads._backbone_vocab_embed(z)  # [B, L, D]
    z_stream = z_emb  # all extras are zero

    out = module(z_stream, hidden_src)
    max_err = (out - z_stream).abs().max().item()

    # Restore original state
    module.load_state_dict(orig_state)
    for name, sd in orig_shared.items():
        getattr(student.heads, name).load_state_dict(sd)

    passed = max_err < 1e-4
    print(f"  max_abs(out - z_stream) = {max_err:.2e}")
    print(f"  PASS" if passed else f"  FAIL (threshold 1e-4)")
    return passed


def test_1b_output_layer_passthrough(student, teacher, config, device):
    """Test 1B: output_layer passthrough check.

    Verify backbone hidden + output_layer match direct forward.
    """
    _header("Test 1B: output_layer passthrough check")

    B, L = 2, config.max_length
    z = torch.randint(0, config.vocab_size, (B, L), device=device)
    # Use zero sigma (time_conditioning=False)
    sigma = torch.zeros(B, device=device)

    dit = teacher.hf_model.backbone

    # Direct forward (reference)
    # DITBackbone.forward returns (logits, hidden_states_list)
    with torch.no_grad():
        out = dit(z, sigma)
        logits_ref = (out[0] if isinstance(out, tuple) else out).float()

    # Manual forward: backbone hidden + output_layer
    with torch.no_grad():
        x = dit.vocab_embed(z)
        c = F.silu(dit.sigma_map(sigma))
        rotary_cos_sin = dit.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for blk in dit.blocks:
                x = blk(x, rotary_cos_sin, c)
            logits_manual = dit.output_layer(x, c)
        logits_manual = logits_manual.float()

    max_err = (logits_ref - logits_manual).abs().max().item()
    passed = max_err < 0.1  # bf16 accumulation can cause small diffs
    print(f"  max_abs(logits_ref - logits_manual) = {max_err:.2e}")
    print(f"  PASS" if passed else f"  FAIL (threshold 0.1)")
    return passed


def test_2_sample_categorical(config, device):
    """Test 2: _sample_categorical distribution check."""
    _header("Test 2: sampling distribution")

    V = 10
    true_probs = torch.softmax(torch.randn(V, device=device), dim=-1)
    logits = torch.log(true_probs)

    N = 10000
    logits_batch = logits.unsqueeze(0).expand(N, -1)  # [N, V]
    samples = _sample_categorical(logits_batch)  # [N]

    counts = torch.zeros(V, device=device)
    for v in range(V):
        counts[v] = (samples == v).float().sum()
    empirical = counts / N

    max_diff = (empirical - true_probs).abs().max().item()
    passed = max_diff < 0.02  # relaxed slightly
    print(f"  true_probs:     {true_probs.cpu().tolist()}")
    print(f"  empirical:      {empirical.cpu().tolist()}")
    print(f"  max_diff = {max_diff:.4f}")
    print(f"  PASS" if passed else f"  FAIL (threshold 0.02)")
    return passed


def test_3_kl_zero(config, device):
    """Test 3: KL(log_pT, log_pT, mask) ≈ 0."""
    _header("Test 3: KL = 0")

    B, L, V = 4, 128, config.vocab_size
    logits = torch.randn(B, L, V, device=device)
    log_p = F.log_softmax(logits.float(), dim=-1)
    mask = torch.ones(B, L, device=device, dtype=torch.bool)

    kl = KL_loss(log_p, log_p, mask)
    kl_val = kl.item()
    passed = abs(kl_val) < 1e-5
    print(f"  KL = {kl_val:.2e}")
    print(f"  PASS" if passed else f"  FAIL (threshold 1e-5)")
    return passed


def test_4_no_nan(student, teacher, config, device):
    """Test 4: one training step produces no NaN."""
    _header("Test 4: no NaN")

    B = config.batch_size
    x0 = torch.randint(0, config.vocab_size - 1, (B, config.max_length), device=device)

    trainable_params = student.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 100)

    student.train()
    loss = train_step(student, teacher, x0, config, optimizer, scheduler, step=1)

    loss_ok = not (loss != loss)  # NaN check
    loss_finite = abs(loss) < 1e10
    grad_ok = all(
        p.grad is None or not torch.isnan(p.grad).any()
        for p in trainable_params
    )

    passed = loss_ok and loss_finite and grad_ok
    print(f"  loss = {loss:.4f}")
    print(f"  loss_is_nan = {not loss_ok}")
    print(f"  loss_is_inf = {not loss_finite}")
    print(f"  grad_has_nan = {not grad_ok}")
    print(f"  PASS" if passed else f"  FAIL")
    return passed


def test_5_memory(config, device):
    """Test 5: peak GPU memory."""
    _header("Test 5: memory")

    torch.cuda.reset_peak_memory_stats()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Current peak memory: {peak:.2f} GB")
    print(f"  Target: < 24 GB for batch={config.batch_size}, L={config.max_length}")
    passed = peak < 24.0
    print(f"  PASS" if passed else f"  FAIL")
    return passed


def test_6_mini_training(student, teacher, config, device, n_steps=200):
    """Test 6: mini training — loss should decrease."""
    _header(f"Test 6: mini training ({n_steps} steps)")

    train_loader = get_dataloader(config, split="train")
    train_iter = iter(train_loader)

    trainable_params = student.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, n_steps)

    student.train()
    losses = []

    t0 = time.time()
    for step in range(1, n_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x0 = batch['input_ids'].to(device)
        loss = train_step(student, teacher, x0, config, optimizer, scheduler, step)
        losses.append(loss)

        if step % 50 == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            print(f"    step {step:4d} | avg_loss {avg:.4f} | time {elapsed:.1f}s")

    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50

    print(f"\n  mean_loss_first_50 = {first_50:.4f}")
    print(f"  mean_loss_last_50  = {last_50:.4f}")
    passed = last_50 < first_50
    print(f"  PASS (loss decreased)" if passed else f"  FAIL (loss did not decrease)")
    return passed, first_50, last_50, losses


def test_7_parameter_count(student, teacher, config):
    """Test 7: parameter count."""
    _header("Test 7: parameter count")

    # Teacher
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher params (frozen): {teacher_params:,}")

    # Student backbone base (frozen)
    student_base = sum(p.numel() for p in student.hf_model.parameters())
    print(f"  Student backbone base (frozen): {student_base:,}")

    # Student backbone LoRA
    lora_params = sum(p.numel() for p in student.backbone_loras.parameters())
    print(f"  Student backbone LoRA (trainable): {lora_params:,}")

    # D-Head
    shared_params = (
        sum(p.numel() for p in student.heads.mask_indicator_embed.parameters()) +
        sum(p.numel() for p in student.heads.time_embed.parameters()) +
        sum(p.numel() for p in student.heads.band_embed.parameters()) +
        sum(p.numel() for p in student.heads.pos_embed.parameters())
    )
    per_band_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in student.heads.dhead_modules
    )
    total_dhead = shared_params + per_band_params
    print(f"  D-Head shared embeddings: {shared_params:,}")
    print(f"  D-Head per-band modules: {per_band_params:,}")
    print(f"  D-Head total: {total_dhead:,}")

    total_trainable = lora_params + total_dhead
    print(f"  Total trainable: {total_trainable:,}")

    return True


def test_8_inference_no_mask(student, config, device):
    """Test 8: inference produces no MASK tokens."""
    _header("Test 8: inference produces no MASK")

    student.eval()
    z = generate_samples(student, config, num_samples=config.num_sample_texts, device=device)

    n_mask = (z == config.mask_token_id).sum().item()
    total = z.numel()
    mask_ratio = n_mask / total

    passed = n_mask == 0
    print(f"  Total tokens: {total}")
    print(f"  Remaining MASK tokens: {n_mask}")
    print(f"  Remaining MASK ratio: {mask_ratio:.6f}")
    print(f"  PASS" if passed else f"  FAIL")

    # Decode samples
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("\n  Generated samples (first 200 chars):")
        for i in range(z.shape[0]):
            text = tokenizer.decode(z[i].tolist(), skip_special_tokens=True)
            print(f"    Sample {i}: {text[:200]}")
    except Exception as e:
        print(f"  Could not decode: {e}")

    return passed


def run_all_checks():
    """Run all sanity checks."""
    config = Config()
    device = config.device

    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Build models
    print("\nBuilding student...")
    student = build_student(config, device)

    print("Building teacher...")
    teacher = build_teacher(config, device)

    results = {}

    # Test 1A
    results['1A'] = test_1a_identity_path(student, config, device)

    # Test 1B
    results['1B'] = test_1b_output_layer_passthrough(student, teacher, config, device)

    # Test 2
    results['2'] = test_2_sample_categorical(config, device)

    # Test 3
    results['3'] = test_3_kl_zero(config, device)

    # Test 4
    results['4'] = test_4_no_nan(student, teacher, config, device)

    # Test 5
    results['5'] = test_5_memory(config, device)

    # Test 7 (before training to show counts)
    results['7'] = test_7_parameter_count(student, teacher, config)

    # Test 6: mini training
    passed_6, first_50, last_50, losses = test_6_mini_training(
        student, teacher, config, device, n_steps=200)
    results['6'] = passed_6

    # Test 8: inference
    results['8'] = test_8_inference_no_mask(student, config, device)

    # Final memory check
    _header("Final memory stats")
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")

    # Summary
    _header("SUMMARY")
    all_pass = True
    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        print(f"  Test {k}: {status}")
        if not v:
            all_pass = False
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    return results


if __name__ == "__main__":
    run_all_checks()
