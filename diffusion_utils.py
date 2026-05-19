"""Diffusion utilities: noising, reverse step, sampling."""

import torch
import torch.nn.functional as F


def forward_noise(x0: torch.Tensor, t: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Absorbing forward noise: each token independently masked with prob t.

    Args:
        x0: [B, L] clean token ids
        t:  [B] noise level in [0, 1]
        mask_id: MASK token id

    Returns:
        z_t: [B, L] noised tokens
    """
    B, L = x0.shape
    # Each position masked independently with probability t
    mask_prob = t[:, None].expand(B, L)  # [B, L]
    rand = torch.rand(B, L, device=x0.device)
    z_t = torch.where(rand < mask_prob, mask_id, x0)
    return z_t


def _sample_categorical(logits: torch.Tensor) -> torch.Tensor:
    """Sample from categorical distribution defined by logits.

    Args:
        logits: [..., V] unnormalized log-probs

    Returns:
        samples: [...] sampled indices
    """
    # Gumbel-max trick for efficient categorical sampling
    uniform = torch.rand_like(logits.float())
    # Clamp to avoid log(0)
    uniform = uniform.clamp(min=1e-30)
    gumbel = -torch.log(-torch.log(uniform))
    return (logits.float() + gumbel).argmax(dim=-1)


def absorbing_reverse_step(
    z: torch.Tensor,
    log_p: torch.Tensor,
    t_curr: torch.Tensor,
    t_next: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """One reverse step in absorbing diffusion.

    For each MASK position, unmask with probability (t_curr - t_next) / t_curr.
    Non-MASK positions stay unchanged.

    Args:
        z:       [B, L] current tokens (may contain MASK)
        log_p:   [B, L, V] log-probs over vocab (MASK col should be -inf)
        t_curr:  [B] current noise level
        t_next:  [B] next noise level (t_next <= t_curr)
        mask_token_id: MASK token id

    Returns:
        z_next: [B, L] updated tokens
    """
    B, L = z.shape
    device = z.device

    if not torch.is_tensor(t_curr):
        t_curr = torch.full((B,), float(t_curr), device=device)
    if not torch.is_tensor(t_next):
        t_next = torch.full((B,), float(t_next), device=device)

    t_curr = t_curr.to(device=device, dtype=torch.float32)
    t_next = t_next.to(device=device, dtype=torch.float32)

    # Probability of unmasking a currently-masked token
    # unmask_prob = (t_curr - t_next) / t_curr, but avoid 0/0
    # If t_curr == 0, no tokens should be masked anyway → unmask_prob = 0
    # If t_curr == t_next (zero-step), unmask_prob = 0
    safe_t_curr = t_curr.clamp(min=1e-20)
    unmask_prob = ((t_curr - t_next) / safe_t_curr).clamp(0, 1)  # [B]
    unmask_prob = unmask_prob[:, None].expand(B, L)  # [B, L]

    is_mask = (z == mask_token_id)  # [B, L]

    # Sample candidate tokens from p(x0 | z_t)
    candidates = _sample_categorical(log_p)  # [B, L]

    # Decide which MASK positions to unmask
    rand = torch.rand(B, L, device=device)
    do_unmask = is_mask & (rand < unmask_prob)

    z_next = torch.where(do_unmask, candidates, z)
    return z_next
