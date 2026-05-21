"""Inference for MDLM D-Head Accelerator.

Two sampling paths:
  generate_samples()          — old simplified absorbing_reverse_step
  generate_samples_official() — official-compatible _ddpm_update transition
"""

import torch
import torch.nn.functional as F

from config import Config
from diffusion_utils import absorbing_reverse_step


# ── Official MDLM primitives ─────────────────────────────────────────

def _mdlm_sample_categorical(categorical_probs):
    """Official MDLM Gumbel-trick sampling on probabilities."""
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _loglinear_total_noise(t, eps=1e-3):
    """LogLinear noise schedule: sigma(t) = -log(1 - (1-eps)*t)."""
    return -torch.log1p(-(1 - eps) * t)


def _move_chance(t, eps=1e-3):
    """Mask probability at time t under loglinear schedule."""
    sigma = _loglinear_total_noise(t, eps)
    return 1.0 - torch.exp(-sigma)


@torch.no_grad()
def generate_samples(
    student,
    config: Config,
    num_samples: int = 4,
    device: str = None,
    substeps_per_band: int = None,
) -> torch.Tensor:
    """Full inference: 4 bands × K substeps.

    Args:
        student: DHeadStudent model (eval mode)
        config: Config
        num_samples: number of sequences to generate
        device: device override
        substeps_per_band: override config.K for inference

    Returns:
        z: [num_samples, L] generated token ids
    """
    if device is None:
        device = config.device

    L = config.max_length
    K = substeps_per_band if substeps_per_band is not None else config.K
    MASK_ID = config.mask_token_id
    n_bands = config.n_bands

    # Start from all MASK
    z = torch.full((num_samples, L), MASK_ID, device=device, dtype=torch.long)

    for band_idx in range(n_bands):
        t_band_high = 1.0 - band_idx / n_bands
        t_band_low = 1.0 - (band_idx + 1) / n_bands

        t_src = torch.full((num_samples,), t_band_high, device=device)
        t_dst = torch.full((num_samples,), t_band_low, device=device)

        # Backbone refresh once per band
        hidden_src, c_src = student.forward_backbone(z, t_src)

        Delta = (t_src - t_dst) / K

        for k in range(K):
            t_cur = t_src - k * Delta
            t_next = t_src - (k + 1) * Delta

            c_cur = c_src  # time_conditioning=False

            logits = student.heads.compute_one_head(
                hidden_src=hidden_src,
                z=z,
                c=c_cur,
                t_cur=t_cur,
                band_idx=band_idx,
            )

            log_p = F.log_softmax(logits.float(), dim=-1)

            z = absorbing_reverse_step(
                z=z,
                log_p=log_p,
                t_curr=t_cur,
                t_next=t_next,
                mask_token_id=MASK_ID,
            )

        remaining_mask_ratio = (z == MASK_ID).float().mean().item()
        print(f"after band {band_idx}, remaining_mask_ratio={remaining_mask_ratio:.6f}")

    # noise_removal: final argmax denoise (matches official MDLM protocol)
    is_mask = (z == MASK_ID)
    if is_mask.any():
        # Re-run last band's head on remaining masks
        t_eps = torch.full((num_samples,), 1e-3, device=device)
        hidden_final, c_final = student.forward_backbone(z, t_eps)
        logits_final = student.heads.compute_one_head(
            hidden_src=hidden_final, z=z, c=c_final,
            t_cur=t_eps, band_idx=n_bands - 1)
        pred = logits_final.argmax(dim=-1)
        z = torch.where(is_mask, pred, z)
        print(f"noise_removal: cleared {is_mask.sum().item()} remaining masks")

    return z


# ── Official-compatible sampling ──────────────────────────────────────

@torch.no_grad()
def generate_samples_official(
    student,
    config: Config,
    num_samples: int = 4,
    device: str = None,
    substeps_per_band: int = None,
) -> torch.Tensor:
    """Official-compatible inference: _ddpm_update transition + noise_removal.

    Uses the same transition kernel as official MDLM _ddpm_update:
      q_xs = p_x0 * (move_chance_t - move_chance_s)
      q_xs[:, :, MASK] = move_chance_s

    Instead of simplified absorbing_reverse_step.
    """
    if device is None:
        device = config.device

    L = config.max_length
    K = substeps_per_band if substeps_per_band is not None else config.K
    MASK_ID = config.mask_token_id
    n_bands = config.n_bands
    eps = 1e-3  # loglinear eps

    z = torch.full((num_samples, L), MASK_ID, device=device, dtype=torch.long)

    for band_idx in range(n_bands):
        t_band_high = 1.0 - band_idx / n_bands
        t_band_low = 1.0 - (band_idx + 1) / n_bands

        t_src = torch.full((num_samples,), t_band_high, device=device)
        t_dst = torch.full((num_samples,), t_band_low, device=device)

        # Backbone refresh once per band
        hidden_src, c_src = student.forward_backbone(z, t_src)

        # Uniform timesteps within band
        band_timesteps = torch.linspace(
            t_band_high, t_band_low + (t_band_high - t_band_low) / K * 0,
            K + 1, device=device)
        # Actually: evenly spaced from t_band_high to t_band_low
        band_timesteps = torch.linspace(t_band_high, t_band_low, K + 1,
                                        device=device)
        band_dt = (t_band_high - t_band_low) / K

        for k in range(K):
            t_cur = band_timesteps[k]
            t_next = band_timesteps[k + 1]
            t_cur_batch = torch.full((num_samples,), t_cur.item(),
                                     device=device)

            # Student D-Head forward
            logits = student.heads.compute_one_head(
                hidden_src=hidden_src, z=z, c=c_src,
                t_cur=t_cur_batch, band_idx=band_idx)

            # subs parameterization: MASK col = -inf, log_softmax
            logits = logits.float()
            logits[:, :, MASK_ID] = -1e9
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

            # Lock in unmasked positions (official _subs_parameterization)
            unmasked = (z != MASK_ID)
            logits[unmasked] = -1e9
            logits[unmasked, z[unmasked]] = 0.0

            p_x0 = logits.exp()  # [B, L, V]

            # Official _ddpm_update transition kernel
            mc_t = _move_chance(t_cur, eps)
            mc_s = _move_chance(t_next, eps)

            q_xs = p_x0 * (mc_t - mc_s)
            q_xs[:, :, MASK_ID] = mc_s

            _x = _mdlm_sample_categorical(q_xs)

            # Non-MASK positions stay unchanged
            copy_flag = (z != MASK_ID).to(z.dtype)
            z = (copy_flag * z + (1 - copy_flag) * _x).long()

        remaining_mask_ratio = (z == MASK_ID).float().mean().item()
        print(f"after band {band_idx}, remaining_mask_ratio="
              f"{remaining_mask_ratio:.6f}")

    # noise_removal: final argmax
    is_mask = (z == MASK_ID)
    if is_mask.any():
        t_eps = torch.full((num_samples,), eps, device=device)
        hidden_final, c_final = student.forward_backbone(z, t_eps)
        logits_final = student.heads.compute_one_head(
            hidden_src=hidden_final, z=z, c=c_final,
            t_cur=t_eps, band_idx=n_bands - 1)
        pred = logits_final.argmax(dim=-1)
        z = torch.where(is_mask, pred, z)
        print(f"noise_removal: cleared {is_mask.sum().item()} masks")

    return z


# ── Upper bound: refresh backbone every substep ──────────────────────

@torch.no_grad()
def generate_samples_refresh(
    student,
    config: Config,
    num_samples: int = 4,
    device: str = None,
    substeps_per_band: int = None,
) -> torch.Tensor:
    """Same as official-compatible but refreshes backbone EVERY substep.

    This is the upper bound — if this produces much better Gen-PPL,
    stale hidden_src is confirmed as the main bottleneck.
    NFE = n_bands * K * 2 (backbone + D-Head per step).
    """
    if device is None:
        device = config.device

    L = config.max_length
    K = substeps_per_band if substeps_per_band is not None else config.K
    MASK_ID = config.mask_token_id
    n_bands = config.n_bands
    eps = 1e-3

    z = torch.full((num_samples, L), MASK_ID, device=device, dtype=torch.long)

    for band_idx in range(n_bands):
        t_band_high = 1.0 - band_idx / n_bands
        t_band_low = 1.0 - (band_idx + 1) / n_bands

        band_timesteps = torch.linspace(t_band_high, t_band_low, K + 1,
                                        device=device)

        for k in range(K):
            t_cur = band_timesteps[k]
            t_next = band_timesteps[k + 1]
            t_cur_batch = torch.full((num_samples,), t_cur.item(),
                                     device=device)

            # REFRESH backbone every substep
            hidden_src, c_src = student.forward_backbone(z, t_cur_batch)

            logits = student.heads.compute_one_head(
                hidden_src=hidden_src, z=z, c=c_src,
                t_cur=t_cur_batch, band_idx=band_idx)

            logits = logits.float()
            logits[:, :, MASK_ID] = -1e9
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

            unmasked = (z != MASK_ID)
            logits[unmasked] = -1e9
            logits[unmasked, z[unmasked]] = 0.0

            p_x0 = logits.exp()

            mc_t = _move_chance(t_cur, eps)
            mc_s = _move_chance(t_next, eps)

            q_xs = p_x0 * (mc_t - mc_s)
            q_xs[:, :, MASK_ID] = mc_s

            _x = _mdlm_sample_categorical(q_xs)

            copy_flag = (z != MASK_ID).to(z.dtype)
            z = (copy_flag * z + (1 - copy_flag) * _x).long()

        remaining_mask_ratio = (z == MASK_ID).float().mean().item()
        print(f"after band {band_idx}, remaining_mask_ratio="
              f"{remaining_mask_ratio:.6f}")

    # noise_removal
    is_mask = (z == MASK_ID)
    if is_mask.any():
        t_eps = torch.full((num_samples,), eps, device=device)
        hidden_final, c_final = student.forward_backbone(z, t_eps)
        logits_final = student.heads.compute_one_head(
            hidden_src=hidden_final, z=z, c=c_final,
            t_cur=t_eps, band_idx=n_bands - 1)
        pred = logits_final.argmax(dim=-1)
        z = torch.where(is_mask, pred, z)
        print(f"noise_removal: cleared {is_mask.sum().item()} masks")

    return z
