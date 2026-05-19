"""Configuration for MDLM D-Head Accelerator MVP."""

from dataclasses import dataclass


@dataclass
class Config:
    # ── Model ──
    d_model: int = 768
    cond_dim: int = 128          # backbone conditioning dim
    n_heads: int = 12
    head_dim: int = 64
    ffn_dim: int = 2048
    dropout: float = 0.0
    vocab_size: int = 50258
    mask_token_id: int = 50257
    max_length: int = 1024       # L

    # ── D-Head ──
    n_bands: int = 4
    K: int = 4                   # substeps per band
    dhead_self_attn_layers: int = 2
    dhead_cross_attn_layers: int = 1
    dhead_ffn_layers: int = 1

    # ── Backbone LoRA ──
    backbone_lora_rank: int = 128

    # ── Training ──
    batch_size: int = 8
    lr: float = 1e-4
    warmup_steps: int = 1000
    total_steps: int = 30000
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42

    # ── Numerical ──
    neg_infinity: float = -1e9
    kl_chunk_size: int = 0       # 0 = no chunking; set e.g. 512 if OOM

    # ── Logging / checkpointing ──
    log_every: int = 50
    save_every: int = 5000
    sample_every: int = 5000
    num_sample_texts: int = 4

    # ── Paths ──
    hf_model_id: str = "kuleshov-group/mdlm-owt"
    train_data: str = "/data1/wulingdan/data/diffusion/mdlm/cache/openwebtext-train_train_bs1024_wrapped.dat"
    valid_data: str = "/data1/wulingdan/data/diffusion/mdlm/cache/openwebtext-valid_validation_bs1024_wrapped.dat"
    output_dir: str = "./outputs"

    # ── Hardware ──
    device: str = "cuda"
    precision: str = "bf16"
