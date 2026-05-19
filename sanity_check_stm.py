#!/usr/bin/env python3
"""Sanity checks for STM (Scheduled Trajectory Mixing) training.

Tests:
  1. random_partition — row sums, min interval, total=0
  2. tau schedule — warmup, decay, floor
  3. residual init — delta_proj=0 ⇒ logits match
  4. zero-step — tau=1 student noop, tau=0 teacher noop
  5. one train_step_stm — loss/grad finite
  6. teacher frozen — 0 trainable params, unchanged after step
  7. inference no MASK — 4 bands × K substeps clears all masks
"""

import sys
import torch
import torch.nn.functional as F

from config import Config
from model import build_student, build_teacher
from diffusion_utils import forward_noise, absorbing_reverse_step
from train import KL_loss, get_cosine_schedule_with_warmup
from train_stm import random_partition, get_tau, train_step_stm
from inference import generate_samples


def _header(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ────────────────────────────────────────────────────────────────────
# Test 1: random_partition
# ────────────────────────────────────────────────────────────────────

def test_random_partition(device):
    _header("Test 1: random_partition")
    passed = True
    B, N = 8, 4
    min_frac = 0.05

    total = torch.rand(B, device=device) * 0.25

    intervals = random_partition(total, N, min_frac=min_frac)

    # 1a — row sums equal total
    row_sums = intervals.sum(dim=1)
    max_sum_err = (row_sums - total).abs().max().item()
    ok = max_sum_err < 1e-5
    print(f"  1a. Row-sum error: {max_sum_err:.2e} — "
          f"{'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # 1b — each segment >= min_frac / N * total[b]
    ok_b = True
    for b in range(B):
        if total[b] > 0:
            expected_min = min_frac / N * total[b].item()
            actual_min = intervals[b].min().item()
            if actual_min < expected_min - 1e-7:
                print(f"  1b. FAIL b={b}: min={actual_min:.6f} "
                      f"< expected={expected_min:.6f}")
                ok_b = False
    print(f"  1b. Min-interval constraint: "
          f"{'PASS' if ok_b else 'FAIL'}")
    passed = passed and ok_b

    # 1c — total=0 → all zeros
    total_zero = torch.zeros(4, device=device)
    intervals_zero = random_partition(total_zero, N)
    ok_c = (intervals_zero == 0).all().item()
    print(f"  1c. total=0 all zeros: {'PASS' if ok_c else 'FAIL'}")
    passed = passed and ok_c

    print(f"  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ────────────────────────────────────────────────────────────────────
# Test 2: tau schedule
# ────────────────────────────────────────────────────────────────────

def test_tau_schedule():
    _header("Test 2: tau schedule")
    passed = True
    tau_min = 0.1
    warmup_frac = 0.05
    decay_frac = 0.45
    total_steps = 3000

    warmup_end = int(warmup_frac * total_steps)   # 150
    decay_end = warmup_end + int(decay_frac * total_steps)  # 1500

    # 2a — step=0 ⇒ tau=1
    tau_0 = get_tau(0, tau_min, warmup_frac, decay_frac, total_steps)
    ok = abs(tau_0 - 1.0) < 1e-6
    print(f"  2a. tau(0) = {tau_0:.4f} — {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # 2b — after warmup, starts decreasing
    tau_w = get_tau(warmup_end, tau_min, warmup_frac, decay_frac,
                    total_steps)
    tau_w50 = get_tau(warmup_end + 50, tau_min, warmup_frac, decay_frac,
                      total_steps)
    ok = tau_w50 < tau_w
    print(f"  2b. tau({warmup_end})={tau_w:.4f} > "
          f"tau({warmup_end + 50})={tau_w50:.4f} — "
          f"{'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # 2c — decay end ⇒ tau_min
    tau_de = get_tau(decay_end, tau_min, warmup_frac, decay_frac,
                     total_steps)
    ok = abs(tau_de - tau_min) < 1e-4
    print(f"  2c. tau({decay_end}) = {tau_de:.4f} ≈ {tau_min} — "
          f"{'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # 2d — far past decay ⇒ tau_min
    tau_late = get_tau(total_steps, tau_min, warmup_frac, decay_frac,
                       total_steps)
    ok = abs(tau_late - tau_min) < 1e-6
    print(f"  2d. tau({total_steps}) = {tau_late:.4f} — "
          f"{'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    print(f"  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ────────────────────────────────────────────────────────────────────
# Test 3: residual init (delta_proj=0 → logits ≈ output_layer)
# ────────────────────────────────────────────────────────────────────

def test_residual_init(student, config, device):
    _header("Test 3: residual init")

    student.eval()
    B, L = 2, config.max_length
    z = torch.randint(0, config.vocab_size, (B, L), device=device)
    t = torch.full((B,), 0.5, device=device)

    with torch.no_grad():
        h, c = student.forward_backbone(z, t)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits_ref = student.heads._output_layer(h, c)
        logits_ref = logits_ref.float()

        logits_dh = student.heads.compute_one_head(
            hidden_src=h, z=z, c=c, t_cur=t, band_idx=0)

        mc = config.mask_token_id
        ref = torch.cat([logits_ref[..., :mc],
                         logits_ref[..., mc + 1:]], dim=-1)
        dh = torch.cat([logits_dh[..., :mc],
                        logits_dh[..., mc + 1:]], dim=-1)

    max_err = (ref - dh).abs().max().item()
    passed = max_err < 1e-3
    print(f"  max_abs = {max_err:.2e} — {'PASS' if passed else 'FAIL'}")
    return passed


# ────────────────────────────────────────────────────────────────────
# Test 4: zero-step
# ────────────────────────────────────────────────────────────────────

def test_zero_step(device):
    _header("Test 4: zero-step (tau=1 / tau=0)")
    passed = True
    B, L, N = 4, 64, 4
    MASK_ID = 50257

    # 4a — tau=1 ⇒ student_total=0, all student intervals=0
    total_zero = torch.zeros(B, device=device)
    intervals = random_partition(total_zero, N)
    ok = (intervals == 0).all().item()
    print(f"  4a. tau=1 ⇒ student intervals all zero: "
          f"{'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # 4b — t_curr == t_next ⇒ z unchanged (teacher zero-step)
    z = torch.randint(0, MASK_ID, (B, L), device=device)
    z[:, :32] = MASK_ID
    log_p = F.log_softmax(torch.randn(B, L, MASK_ID + 1, device=device),
                          dim=-1)
    t_curr = torch.full((B,), 0.5, device=device)
    t_next = t_curr.clone()
    z_after = absorbing_reverse_step(z, log_p, t_curr, t_next, MASK_ID)
    ok = (z_after == z).all().item()
    print(f"  4b. t_curr==t_next ⇒ z unchanged: "
          f"{'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    print(f"  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ────────────────────────────────────────────────────────────────────
# Test 5: one train_step_stm — finite loss & grad
# ────────────────────────────────────────────────────────────────────

def test_one_stm_step(student, teacher, config, device):
    _header("Test 5: one train_step_stm")

    B = config.batch_size
    x0 = torch.randint(0, config.vocab_size - 1,
                       (B, config.max_length), device=device)

    trainable_params = student.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 100)

    student.train()
    metrics = train_step_stm(
        student, teacher, x0, config, optimizer, scheduler,
        step=1, tau=0.5)

    loss = metrics['loss']
    loss_ok = not (loss != loss)
    loss_finite = abs(loss) < 1e10
    grad_ok = all(
        p.grad is None or (not torch.isnan(p.grad).any()
                           and not torch.isinf(p.grad).any())
        for p in trainable_params
    )

    passed = loss_ok and loss_finite and grad_ok
    print(f"  loss = {loss:.4f}")
    print(f"  loss_nan = {not loss_ok}  loss_inf = {not loss_finite}  "
          f"grad_bad = {not grad_ok}")
    print(f"  tau={metrics['tau']:.2f}  "
          f"stu_total={metrics['student_total']:.4f}  "
          f"tea_total={metrics['teacher_total']:.4f}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


# ────────────────────────────────────────────────────────────────────
# Test 6: teacher frozen
# ────────────────────────────────────────────────────────────────────

def test_teacher_frozen(student, teacher, config, device):
    _header("Test 6: teacher frozen")
    passed = True

    # 6a — 0 trainable params
    n_trainable = sum(1 for p in teacher.parameters() if p.requires_grad)
    ok = n_trainable == 0
    print(f"  6a. Teacher trainable params: {n_trainable} — "
          f"{'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # 6b — params unchanged after step
    snap = {k: v.clone() for k, v in teacher.state_dict().items()}

    B = config.batch_size
    x0 = torch.randint(0, config.vocab_size - 1,
                       (B, config.max_length), device=device)
    params = student.get_trainable_parameters()
    opt = torch.optim.AdamW(params, lr=1e-4)
    sched = get_cosine_schedule_with_warmup(opt, 10, 100)
    student.train()
    train_step_stm(student, teacher, x0, config, opt, sched,
                   step=1, tau=0.5)

    changed = False
    for k, v in teacher.state_dict().items():
        if not torch.equal(v, snap[k]):
            print(f"  6b. CHANGED: {k}")
            changed = True
    ok = not changed
    print(f"  6b. Teacher params unchanged: "
          f"{'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    print(f"  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ────────────────────────────────────────────────────────────────────
# Test 7: inference no MASK
# ────────────────────────────────────────────────────────────────────

def test_inference_no_mask(student, config, device):
    _header("Test 7: inference no MASK")

    student.eval()
    with torch.no_grad():
        z = generate_samples(student, config, num_samples=2, device=device)

    n_mask = (z == config.mask_token_id).sum().item()
    passed = n_mask == 0
    print(f"  Tokens: {z.numel()}  remaining MASK: {n_mask}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


# ────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────

def run_all():
    config = Config()
    config.kl_chunk_size = 512
    device = config.device

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    results = {}

    # Pure-logic tests (no model)
    results['1'] = test_random_partition(device)
    results['2'] = test_tau_schedule()
    results['4'] = test_zero_step(device)

    # Model tests
    print("\nBuilding student...")
    student = build_student(config, device)
    print("Building teacher...")
    teacher = build_teacher(config, device)
    teacher.eval()

    results['3'] = test_residual_init(student, config, device)
    results['5'] = test_one_stm_step(student, teacher, config, device)
    results['6'] = test_teacher_frozen(student, teacher, config, device)
    results['7'] = test_inference_no_mask(student, config, device)

    # Summary
    _header("SUMMARY")
    all_pass = True
    for k in sorted(results.keys(), key=int):
        status = "PASS" if results[k] else "FAIL"
        print(f"  Test {k}: {status}")
        if not results[k]:
            all_pass = False

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n  Peak GPU memory: {peak:.2f} GB")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
