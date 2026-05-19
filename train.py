"""Training loop for MDLM D-Head Accelerator."""

import os
import time
import math
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from config import Config
from data import get_dataloader
from model import build_student, build_teacher
from diffusion_utils import forward_noise, absorbing_reverse_step
from inference import generate_samples


def KL_loss(log_pT: torch.Tensor, log_pS: torch.Tensor,
            mask: torch.Tensor, chunk_size: int = 0) -> torch.Tensor:
    """KL divergence on MASK positions only.

    Args:
        log_pT: [B, L, V] teacher log-probs
        log_pS: [B, L, V] student log-probs
        mask:   [B, L] bool — True where z == MASK
        chunk_size: if > 0, process in chunks to save memory

    Returns:
        scalar loss
    """
    if mask.sum() == 0:
        return log_pS.sum() * 0.0

    log_pT_m = log_pT[mask].float()  # [M, V]
    log_pS_m = log_pS[mask].float()  # [M, V]

    if chunk_size <= 0:
        pT = log_pT_m.exp()
        term = pT * (log_pT_m - log_pS_m)
        term = torch.where(pT > 0, term, torch.zeros_like(term))
        return term.sum(dim=-1).mean()

    losses = []
    for start in range(0, log_pT_m.shape[0], chunk_size):
        end = start + chunk_size
        lpT = log_pT_m[start:end]
        lpS = log_pS_m[start:end]
        pT = lpT.exp()
        term = pT * (lpT - lpS)
        term = torch.where(pT > 0, term, torch.zeros_like(term))
        losses.append(term.sum(dim=-1))

    return torch.cat(losses, dim=0).mean()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine decay with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(
    student, teacher, x0, config, optimizer, scheduler, step,
):
    """One training step: sample band, noise, backbone, K substeps with immediate backward.

    Returns:
        total_loss: float (detached average over K substeps)
    """
    B = x0.shape[0]
    device = x0.device
    K = config.K
    MASK_ID = config.mask_token_id
    chunk_size = config.kl_chunk_size

    # 1. Sample random band
    band_idx = torch.randint(0, config.n_bands, (1,)).item()
    t_band_high = 1.0 - band_idx / config.n_bands
    t_band_low = 1.0 - (band_idx + 1) / config.n_bands

    # t_src: mix of endpoint (t_band_high) and Uniform(t_band_low, t_band_high)
    # Band 0 needs more t=1.0 full-mask starts; other bands use 30% endpoint.
    endpoint_prob = 0.5 if band_idx == 0 else 0.3
    use_endpoint = torch.rand(B, device=device) < endpoint_prob
    t_uniform = torch.rand(B, device=device) * (t_band_high - t_band_low) + t_band_low
    t_endpoint = torch.full((B,), t_band_high, device=device)
    t_src = torch.where(use_endpoint, t_endpoint, t_uniform)
    t_dst = torch.full((B,), t_band_low, device=device)

    # 2. Forward noising
    z_t = forward_noise(x0, t_src, MASK_ID)

    # 3. Student backbone forward once
    hidden_src, c_src = student.forward_backbone(z_t, t_src)

    # 4. K teacher-forced substeps with immediate backward
    optimizer.zero_grad(set_to_none=True)

    z = z_t.clone()
    Delta = (t_src - t_dst) / K  # [B]
    total_loss_value = 0.0

    for k in range(K):
        t_cur = t_src - k * Delta
        t_next = t_src - (k + 1) * Delta

        # Teacher target
        with torch.no_grad():
            log_pT = teacher.forward_log_probs(z, t_cur)  # [B, L, V]

        # Student D-Head prediction
        c_cur = c_src  # time_conditioning=False → reuse c_src

        logits_S = student.heads.compute_one_head(
            hidden_src=hidden_src,
            z=z,
            c=c_cur,
            t_cur=t_cur,
            band_idx=band_idx,
        )  # [B, L, V]

        log_pS = F.log_softmax(logits_S.float(), dim=-1)

        # KL loss only on MASK positions
        mask = (z == MASK_ID)
        loss_k = KL_loss(log_pT, log_pS, mask, chunk_size=chunk_size)

        # Backward immediately
        (loss_k / K).backward(retain_graph=(k < K - 1))

        total_loss_value += loss_k.detach().item()

        # Teacher-forced rollout
        with torch.no_grad():
            z = absorbing_reverse_step(
                z=z,
                log_p=log_pT,
                t_curr=t_cur,
                t_next=t_next,
                mask_token_id=MASK_ID,
            )

        del log_pT, logits_S, log_pS, loss_k

    clip_grad_norm_(student.get_trainable_parameters(), config.grad_clip)
    optimizer.step()
    scheduler.step()

    return total_loss_value / K


def train(config: Config = None, max_steps: int = None):
    """Main training loop."""
    if config is None:
        config = Config()

    if max_steps is None:
        max_steps = config.total_steps

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    device = config.device

    print("Building student...")
    student = build_student(config, device)
    student.train()

    print("Building teacher...")
    teacher = build_teacher(config, device)
    teacher.eval()

    print("Loading data...")
    train_loader = get_dataloader(config, split="train")
    train_iter = iter(train_loader)

    # Optimizer
    trainable_params = student.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, max_steps)

    # Tokenizer for sampling
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        tokenizer = None

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Starting training for {max_steps} steps...")
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")

    losses = []
    for step in range(1, max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x0 = batch['input_ids'].to(device)

        loss = train_step(student, teacher, x0, config, optimizer, scheduler, step)
        losses.append(loss)

        if step % config.log_every == 0:
            avg_loss = sum(losses[-config.log_every:]) / len(losses[-config.log_every:])
            lr_now = scheduler.get_last_lr()[0]
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"step {step:6d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | mem {mem:.2f}GB")

        if step % config.save_every == 0:
            ckpt_path = os.path.join(config.output_dir, f"ckpt_step{step}.pt")
            torch.save({
                'step': step,
                'student_state_dict': {
                    'backbone_loras': student.backbone_loras.state_dict(),
                    'heads': student.heads.trainable_state_dict(),
                },
                'optimizer': optimizer.state_dict(),
                'config': config,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        if step % config.sample_every == 0 and tokenizer is not None:
            print(f"--- Generating samples at step {step} ---")
            student.eval()
            with torch.no_grad():
                samples = generate_samples(
                    student, config, num_samples=config.num_sample_texts)
            for i, s in enumerate(samples):
                text = tokenizer.decode(s.tolist(), skip_special_tokens=True)
                print(f"  Sample {i}: {text[:200]}")
            student.train()

    return student, teacher, losses


if __name__ == "__main__":
    config = Config()
    train(config)
