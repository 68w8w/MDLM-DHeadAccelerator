# MDLM D-Head Accelerator MVP Spec

## Goal
Validate that a D-Head module can correctly decode from cached backbone hidden states, reducing backbone NFE from 16 to 4.

## Architecture
- **Backbone**: MDLM 169M DiT (kuleshov-group/mdlm-owt), frozen base + LoRA (rank 128)
- **Teacher**: Independently loaded frozen MDLM (no LoRA contamination)
- **D-Head**: 4 per-band modules, each with 2 self-attn + 1 cross-attn + 1 FFN
- **Position encoding**: Absolute (MVP), upgradeable to RoPE

## Noise bands
| Band | t range       | Head   |
|------|---------------|--------|
| 0    | [0.75, 1.00]  | head_0 |
| 1    | [0.50, 0.75]  | head_1 |
| 2    | [0.25, 0.50]  | head_2 |
| 3    | [0.00, 0.25]  | head_3 |

- K = 4 substeps per band → 16 total reverse steps
- Backbone forward = 1 per band → 4 total NFE

## Training
- Teacher-forced substeps with immediate backward (memory efficient)
- KL loss on MASK positions only
- AdamW, cosine decay with warmup, bf16 mixed precision

## Inference
- 4 bands × 4 substeps, backbone cached once per band
- Start from all-MASK, end with no MASK tokens
