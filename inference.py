"""Inference for MDLM D-Head Accelerator."""

import torch
import torch.nn.functional as F

from config import Config
from diffusion_utils import absorbing_reverse_step


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

    return z
