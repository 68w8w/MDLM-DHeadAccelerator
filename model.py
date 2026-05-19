"""MDLM D-Head Accelerator: Student, Teacher, DHead modules."""

import math
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from einops import rearrange
import flash_attn.flash_attn_interface
import flash_attn.layers.rotary

from config import Config


# ======================================================================
# Utilities from MDLM backbone (needed for forward_backbone)
# ======================================================================

@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=True)
    else:
        out = scale * F.dropout(x, p=prob, training=True)
    if residual is not None:
        out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
    if bias is not None:
        out = scale * (x + bias)
    else:
        out = scale * x
    if residual is not None:
        out = residual + out
    return out


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


# ======================================================================
# LoRA
# ======================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation. B is zero-initialized → output is 0 at init."""

    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


# ======================================================================
# D-Head attention blocks (MVP: absolute position embedding, no RoPE)
# ======================================================================

class SelfAttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention block."""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]  (caller does residual: x = x + sa(x))
        """
        B, L, D = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = rearrange(qkv, 'b s (three h d) -> (b s) three h d',
                         three=3, h=self.n_heads)

        cu_seqlens = torch.arange(
            0, (B + 1) * L, step=L,
            dtype=torch.int32, device=x.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            attn_out = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16), cu_seqlens, L, 0.0, causal=False)

        attn_out = rearrange(attn_out.to(x.dtype), '(b s) h d -> b s (h d)', b=B)
        out = self.out_proj(attn_out)
        out = self.dropout(out)
        return out


class CrossAttentionBlock(nn.Module):
    """Pre-norm multi-head cross-attention block.

    Q from query stream, K/V from key_value (hidden_src).
    """

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.kv_proj = nn.Linear(d, 2 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:     [B, L, D]
            key_value: [B, L, D]  (hidden_src from backbone)
        Returns:
            out: [B, L, D]
        """
        B, L, D = query.shape

        q = self.q_proj(self.norm_q(query))
        kv = self.kv_proj(self.norm_kv(key_value))

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
        k, v = rearrange(kv, 'b s (two h d) -> two b h s d',
                         two=2, h=self.n_heads)

        # Scaled dot-product attention
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            attn_out = F.scaled_dot_product_attention(
                q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        attn_out = attn_out.to(query.dtype)

        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')
        out = self.out_proj(attn_out)
        out = self.dropout(out)
        return out


class FFN(nn.Module):
    """Pre-norm feed-forward network."""

    def __init__(self, d: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, ffn_dim)
        self.act = nn.GELU()
        self.down = nn.Linear(ffn_dim, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]  (caller does residual)
        """
        h = self.norm(x)
        h = self.down(self.act(self.up(h)))
        h = self.dropout(h)
        return h


# ======================================================================
# DHeadModule (per-band)
# ======================================================================

class DHeadModule(nn.Module):
    """Per-band module: 2 self-attn + 1 cross-attn + 1 FFN."""

    def __init__(self, d: int = 768, n_heads: int = 12,
                 ffn_dim: int = 2048, dropout: float = 0.0):
        super().__init__()

        self.self_attn_layers = nn.ModuleList([
            SelfAttentionBlock(d=d, n_heads=n_heads, dropout=dropout)
            for _ in range(2)
        ])

        self.cross_attn = CrossAttentionBlock(
            d=d, n_heads=n_heads, dropout=dropout)

        self.ffn = FFN(d=d, ffn_dim=ffn_dim, dropout=dropout)

    def forward(self, z_stream: torch.Tensor,
                hidden_src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_stream:   [B, L, D]  (embedded current z)
            hidden_src: [B, L, D]  (cached backbone hidden)
        Returns:
            out: [B, L, D]
        """
        x = z_stream

        for sa in self.self_attn_layers:
            x = x + sa(x)

        x = x + self.cross_attn(query=x, key_value=hidden_src)

        x = x + self.ffn(x)

        return x


# ======================================================================
# DHead (all 4 bands + shared embeddings)
# ======================================================================

class DHead(nn.Module):
    """4 independent DHeadModules + shared embeddings + frozen output_layer."""

    def __init__(self, config: Config, output_layer: nn.Module,
                 backbone_vocab_embed: nn.Module):
        super().__init__()
        self.config = config
        self.d = config.d_model
        self.vocab_size = config.vocab_size
        self.mask_token_id = config.mask_token_id
        self.max_length = config.max_length

        # Frozen modules from backbone (references, not re-registered as parameters)
        self._output_layer = output_layer
        self._backbone_vocab_embed = backbone_vocab_embed

        for p in self._output_layer.parameters():
            p.requires_grad_(False)
        for p in self._backbone_vocab_embed.parameters():
            p.requires_grad_(False)

        # Shared trainable embeddings
        self.mask_indicator_embed = nn.Embedding(2, self.d)

        self.time_embed = nn.Sequential(
            nn.Linear(1, self.d),
            nn.GELU(),
            nn.Linear(self.d, self.d),
        )

        self.band_embed = nn.Embedding(4, self.d)

        # MVP: absolute position embedding
        self.pos_embed = nn.Embedding(self.max_length, self.d)

        # 4 independent per-band modules
        self.dhead_modules = nn.ModuleList([
            DHeadModule(
                d=self.d,
                n_heads=config.n_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
            )
            for _ in range(4)
        ])

    def compute_one_head(self, hidden_src: torch.Tensor, z: torch.Tensor,
                         c: torch.Tensor, t_cur, band_idx: int) -> torch.Tensor:
        """
        Args:
            hidden_src: [B, L, D]
            z:          [B, L] current token ids
            c:          conditioning from backbone (for output_layer)
            t_cur:      [B] or scalar
            band_idx:   int (0..3)
        Returns:
            logits: [B, L, V]
        """
        B, L = z.shape
        device = z.device

        if not torch.is_tensor(t_cur):
            t_cur = torch.full((B,), float(t_cur), device=device)
        elif t_cur.dim() == 0:
            t_cur = t_cur.expand(B)

        t_cur = t_cur.to(device=device, dtype=hidden_src.dtype)

        # Vocab embedding (frozen)
        z_emb = self._backbone_vocab_embed(z)  # [B, L, D]

        # Position embedding
        pos_ids = torch.arange(L, device=device)
        pos_e = self.pos_embed(pos_ids)[None, :, :]  # [1, L, D]

        # Mask indicator
        is_mask = (z == self.mask_token_id).long()
        mask_ind = self.mask_indicator_embed(is_mask)  # [B, L, D]

        # Time embedding
        time_e = self.time_embed(t_cur[:, None])[:, None, :]  # [B, 1, D]

        # Band embedding
        band_idx_t = torch.full(
            (B,), int(band_idx), device=device, dtype=torch.long)
        band_e = self.band_embed(band_idx_t)[:, None, :]  # [B, 1, D]

        z_stream = z_emb + pos_e + mask_ind + time_e + band_e

        # Per-band module
        out = self.dhead_modules[band_idx](z_stream, hidden_src)  # [B, L, D]

        # Frozen output layer
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self._output_layer(out, c)
        logits = logits.float()
        logits[..., self.mask_token_id] = -torch.inf

        return logits


# ======================================================================
# DHeadStudent
# ======================================================================

class DHeadStudent(nn.Module):
    """Frozen MDLM backbone + LoRA + DHead."""

    def __init__(self, config: Config, hf_model):
        super().__init__()
        self.config = config
        self.hf_model = hf_model
        self.dit = hf_model.backbone

        # Freeze all original parameters
        for p in self.hf_model.parameters():
            p.requires_grad = False
        self.hf_model.eval()

        # Backbone LoRA
        self.backbone_loras = nn.ModuleDict()
        self._inject_backbone_lora(config.backbone_lora_rank)

        # D-Head (shares dit.output_layer and dit.vocab_embed)
        self.heads = DHead(
            config,
            output_layer=self.dit.output_layer,
            backbone_vocab_embed=self.dit.vocab_embed,
        )

    def train(self, mode=True):
        """Keep frozen backbone in eval mode always."""
        super().train(mode)
        self.hf_model.eval()
        return self

    def _inject_backbone_lora(self, rank: int):
        for i, blk in enumerate(self.dit.blocks):
            self.backbone_loras[f'b{i}_qkv'] = LoRALayer(
                blk.attn_qkv.in_features, blk.attn_qkv.out_features, rank)
            self.backbone_loras[f'b{i}_out'] = LoRALayer(
                blk.attn_out.in_features, blk.attn_out.out_features, rank)
            self.backbone_loras[f'b{i}_up'] = LoRALayer(
                blk.mlp[0].in_features, blk.mlp[0].out_features, rank)
            self.backbone_loras[f'b{i}_down'] = LoRALayer(
                blk.mlp[2].in_features, blk.mlp[2].out_features, rank)

    def forward_backbone(self, indices: torch.Tensor,
                         sigma: torch.Tensor):
        """Forward through backbone + LoRA → (hidden, c).

        Args:
            indices: [B, L] token ids
            sigma:   [B] time values

        Returns:
            hidden: [B, L, D] — before output_layer
            c:      [B, cond_dim] — conditioning vector
        """
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)

        dit = self.dit

        # time_conditioning=False → zero out sigma
        if not dit.config.time_conditioning:
            sigma = torch.zeros_like(sigma)

        x = dit.vocab_embed(indices)
        c = F.silu(dit.sigma_map(sigma))
        rotary_cos_sin = dit.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i, blk in enumerate(dit.blocks):
                B, S = x.shape[0], x.shape[1]
                if self.training:
                    bds = bias_dropout_add_scale_fused_train
                else:
                    bds = bias_dropout_add_scale_fused_inference

                (shift_msa, scale_msa, gate_msa,
                 shift_mlp, scale_mlp, gate_mlp) = (
                    blk.adaLN_modulation(c)[:, None].chunk(6, dim=2))

                # --- Attention ---
                x_skip = x
                x_norm = modulate_fused(blk.norm1(x), shift_msa, scale_msa)
                qkv = blk.attn_qkv(x_norm) + self.backbone_loras[f'b{i}_qkv'](x_norm)
                qkv = rearrange(qkv, 'b s (three h d) -> b s three h d',
                                three=3, h=blk.n_heads)
                with torch.cuda.amp.autocast(enabled=False):
                    cos, sin = rotary_cos_sin
                    qkv = apply_rotary_pos_emb(
                        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                cu = torch.arange(0, (B + 1) * S, step=S,
                                  dtype=torch.int32, device=qkv.device)
                attn = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                    qkv, cu, S, 0., causal=False)
                attn = rearrange(attn, '(b s) h d -> b s (h d)', b=B)
                proj = blk.attn_out(attn) + self.backbone_loras[f'b{i}_out'](attn)
                x = bds(proj, None, gate_msa, x_skip, blk.dropout)

                # --- MLP ---
                mlp_in = modulate_fused(blk.norm2(x), shift_mlp, scale_mlp)
                up = blk.mlp[0](mlp_in) + self.backbone_loras[f'b{i}_up'](mlp_in)
                mid = blk.mlp[1](up)       # GELU
                down = blk.mlp[2](mid) + self.backbone_loras[f'b{i}_down'](mid)
                x = bds(down, None, gate_mlp, x, blk.dropout)

        return x, c

    def get_conditioning(self, t: torch.Tensor):
        """Get conditioning from time (for future use with time_conditioning=True)."""
        dit = self.dit
        if not dit.config.time_conditioning:
            t = torch.zeros_like(t)
        return F.silu(dit.sigma_map(t))

    def get_trainable_parameters(self) -> list:
        """Return list of all trainable parameters."""
        params = list(self.backbone_loras.parameters())
        params += list(self.heads.mask_indicator_embed.parameters())
        params += list(self.heads.time_embed.parameters())
        params += list(self.heads.band_embed.parameters())
        params += list(self.heads.pos_embed.parameters())
        params += list(self.heads.dhead_modules.parameters())
        return params


# ======================================================================
# Teacher
# ======================================================================

class Teacher(nn.Module):
    """Frozen teacher: independently loaded MDLM checkpoint, eval mode."""

    def __init__(self, config: Config, hf_model):
        super().__init__()
        self.config = config
        self.hf_model = hf_model
        self.mask_token_id = config.mask_token_id
        for p in self.hf_model.parameters():
            p.requires_grad = False
        self.hf_model.eval()

    @torch.no_grad()
    def forward_log_probs(self, z: torch.Tensor, t) -> torch.Tensor:
        """Compute [B, L, V] log-probs (MASK col = -inf)."""
        if isinstance(t, (int, float)):
            t = torch.full((z.shape[0],), t, device=z.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(z.shape[0])
        if t.ndim > 1:
            t = t.squeeze(-1)

        # Teacher forward: zero-out sigma if time_conditioning=False
        dit = self.hf_model.backbone
        if not dit.config.time_conditioning:
            sigma = torch.zeros_like(t)
        else:
            sigma = t

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = dit(z, sigma)
            # DITBackbone.forward returns (logits, hidden_states_list)
            logits = out[0] if isinstance(out, tuple) else out

        logits = logits.float()
        logits[:, :, self.mask_token_id] = -torch.inf
        return F.log_softmax(logits, dim=-1)


# ======================================================================
# Factory functions
# ======================================================================

def _load_hf_model(config: Config, device: str = "cuda"):
    """Load the MDLM HuggingFace model."""
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        config.hf_model_id, trust_remote_code=True)
    return model.to(device)


def build_student(config: Config, device: str = "cuda") -> DHeadStudent:
    hf_model = _load_hf_model(config, device)
    return DHeadStudent(config, hf_model).to(device)


def build_teacher(config: Config, device: str = "cuda") -> Teacher:
    hf_model = _load_hf_model(config, device)
    teacher = Teacher(config, hf_model).to(device)
    teacher.eval()
    return teacher
