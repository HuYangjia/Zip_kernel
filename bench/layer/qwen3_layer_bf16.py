"""BF16 reference Qwen3 transformer layer for the r79 replacement bench.

Structural fidelity
-------------------
Mirrors ``Zip/model.py::Qwen3DecoderLayer`` 1:1 **except**:
  * weights are random (we only time, never verify logits)
  * dtype is BF16 (Qwen3 official training dtype; per MODEL_SELECTION.md §2)
  * attention uses ``F.scaled_dot_product_attention`` (SDPA) for the realistic
    prefill/decode cost, matching HF's _supports_sdpa=True default path
  * KV cache is a flat tensor pair, NOT a DynamicCache, so the decode path
    mirrors "steady state" (past_kv_len=2048 already present).

Key Qwen3 quirks preserved
--------------------------
  1. head_dim = 128 regardless of num_heads (``HEAD_DIM`` in configs)
  2. q_norm / k_norm act on the head_dim axis (QK-norm variant)
  3. RMSNorm casts input to fp32 internally then casts back

This module exposes TWO useful callables per layer instance:
  * ``run_full_layer(x, ...)``     -- the whole decoder block (sanity check)
  * per-op bound methods exposed via ``layer.ops`` for per-op timing

The per-op form returns zero-arg lambdas suitable for
``bench.layer.timing.measure``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel.bench.configs.qwen3_shapes import HEAD_DIM, Qwen3Config


# -----------------------------------------------------------------------------
# RMSNorm (fp32 internal, matches Qwen3RMSNorm in Zip/model.py)
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (self.weight.to(torch.float32) * xf).to(in_dtype)


# -----------------------------------------------------------------------------
# RoPE helpers — we pre-compute cos/sin for the full (past+cur) positions and
# cache them, so the per-call cost is only the rotate_half + mul.
# -----------------------------------------------------------------------------
def _build_cos_sin(
    head_dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 1_000_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) of shape [seq_len, head_dim]."""
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                        # [S, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)                 # [S, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _apply_rope(
    q: torch.Tensor,  # [B, Hq, S, D]
    k: torch.Tensor,  # [B, Hk, S, D]
    cos: torch.Tensor,  # [S, D]
    sin: torch.Tensor,  # [S, D]
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    q_out = q * cos_b + _rotate_half(q) * sin_b
    k_out = k * cos_b + _rotate_half(k) * sin_b
    return q_out, k_out


# -----------------------------------------------------------------------------
# Per-op callables exposed for the bench harness
# -----------------------------------------------------------------------------
@dataclass
class PerOpCallables:
    """Zero-arg callables, each launches a single op on fresh tensors.

    The harness is expected to call ``measure(fn, ...)`` on each entry.
    Callables are stable across iterations (no in-place drift) because we
    always read from pre-allocated input tensors.
    """
    # 7 un-fused Linears (FP16 baseline side)
    q_proj: Callable[[], torch.Tensor]
    k_proj: Callable[[], torch.Tensor]
    v_proj: Callable[[], torch.Tensor]
    o_proj: Callable[[], torch.Tensor]
    gate_proj: Callable[[], torch.Tensor]
    up_proj: Callable[[], torch.Tensor]
    down_proj: Callable[[], torch.Tensor]
    # non-replaced pieces (also stay BF16 in the mixed layer)
    input_rmsnorm: Callable[[], torch.Tensor]
    post_rmsnorm:  Callable[[], torch.Tensor]
    q_norm:        Callable[[], torch.Tensor]
    k_norm:        Callable[[], torch.Tensor]
    rope:          Callable[[], tuple[torch.Tensor, torch.Tensor]]
    attention:     Callable[[], torch.Tensor]
    # whole-layer sanity check
    full_layer:    Callable[[], torch.Tensor]


# -----------------------------------------------------------------------------
# Qwen3 single layer in BF16
# -----------------------------------------------------------------------------
class Qwen3LayerBF16(nn.Module):
    """A faithful-enough Qwen3 decoder layer in BF16 with random weights.

    Construction parameters mirror ``Qwen3Config`` from HuggingFace.  The layer
    is pre-allocated against a specific (batch, seqlen, past_kv_len) so that
    per-op timing callables never re-allocate.
    """

    def __init__(
        self,
        cfg: Qwen3Config,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        rms_eps: float = 1e-6,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype

        h = cfg.hidden
        im = cfg.intermediate
        qd = cfg.q_out
        kvd = cfg.kv_out

        # ---- projections (bias=False, Qwen3 attention_bias default) -------
        self.q_proj = nn.Linear(h, qd, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(h, kvd, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(h, kvd, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(qd, h, bias=False, device=device, dtype=dtype)
        self.gate_proj = nn.Linear(h, im, bias=False, device=device, dtype=dtype)
        self.up_proj   = nn.Linear(h, im, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(im, h, bias=False, device=device, dtype=dtype)

        # ---- norms --------------------------------------------------------
        self.input_layernorm         = RMSNorm(h, eps=rms_eps, dtype=dtype).to(device)
        self.post_attention_layernorm = RMSNorm(h, eps=rms_eps, dtype=dtype).to(device)
        # QK-norm on head_dim only (Qwen3 quirk)
        self.q_norm = RMSNorm(cfg.head_dim, eps=rms_eps, dtype=dtype).to(device)
        self.k_norm = RMSNorm(cfg.head_dim, eps=rms_eps, dtype=dtype).to(device)

    # -------------------------------------------------------------------
    # Full forward — used as a sanity check against the per-op sum.
    # -------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                                    # [B, S, H]
        cos: torch.Tensor, sin: torch.Tensor,               # [S, D] for current positions
        k_cache: torch.Tensor | None = None,                # [B, Hk, P, D]
        v_cache: torch.Tensor | None = None,                # [B, Hk, P, D]
    ) -> torch.Tensor:
        cfg = self.cfg
        B, S, H = x.shape
        Hq, Hk, D = cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim

        # ---- pre-attn residual + norm -------------------------------------
        residual = x
        x_ln = self.input_layernorm(x)

        # ---- q/k/v projections --------------------------------------------
        q = self.q_proj(x_ln).view(B, S, Hq, D).transpose(1, 2)  # [B, Hq, S, D]
        k = self.k_proj(x_ln).view(B, S, Hk, D).transpose(1, 2)
        v = self.v_proj(x_ln).view(B, S, Hk, D).transpose(1, 2)

        # ---- QK-norm on head_dim ------------------------------------------
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ---- RoPE ---------------------------------------------------------
        q, k = _apply_rope(q, k, cos, sin)

        # ---- KV concat (decode path) --------------------------------------
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # ---- attention ----------------------------------------------------
        # SDPA expects [B, H, S, D]; internal GQA broadcast handled by
        # enable_gqa=True in newer torch, else repeat_kv.  We use repeat_kv
        # for portability.
        n_rep = cfg.gqa_group
        if n_rep > 1:
            k_ex = k[:, :, None, :, :].expand(B, Hk, n_rep, k.shape[-2], D)
            v_ex = v[:, :, None, :, :].expand(B, Hk, n_rep, v.shape[-2], D)
            k_ex = k_ex.reshape(B, Hq, k.shape[-2], D)
            v_ex = v_ex.reshape(B, Hq, v.shape[-2], D)
        else:
            k_ex, v_ex = k, v

        # is_causal when no KV cache (prefill), non-causal when decode
        # (decode S=1, mask trivial).
        is_causal = (k_cache is None) and (S > 1)
        attn_out = F.scaled_dot_product_attention(
            q, k_ex, v_ex,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )  # [B, Hq, S, D]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, Hq * D)
        attn_out = self.o_proj(attn_out)

        x = residual + attn_out

        # ---- MLP ----------------------------------------------------------
        residual = x
        x_ln2 = self.post_attention_layernorm(x)
        gate = self.gate_proj(x_ln2)
        up   = self.up_proj(x_ln2)
        mlp_mid = F.silu(gate) * up
        mlp_out = self.down_proj(mlp_mid)
        return residual + mlp_out


# -----------------------------------------------------------------------------
# Per-op callable factory — produces zero-arg lambdas suited for timing.measure
# -----------------------------------------------------------------------------
def build_per_op_callables(
    layer: Qwen3LayerBF16,
    *,
    batch: int,
    seqlen: int,
    past_kv_len: int,
) -> PerOpCallables:
    """Pre-allocate every intermediate tensor once, hand out zero-arg
    callables that re-use them (no allocator churn inside the timing loop).
    """
    cfg = layer.cfg
    device = layer.device
    dtype = layer.dtype

    H = cfg.hidden
    Qd = cfg.q_out
    Kvd = cfg.kv_out
    Im = cfg.intermediate
    D = cfg.head_dim
    Hq, Hk = cfg.num_q_heads, cfg.num_kv_heads

    B, S, P = batch, seqlen, past_kv_len
    Stot = S + P

    # Raw activations.  We scale by 0.4 to roughly match hidden-state range.
    x_hidden = (torch.randn(B, S, H, device=device, dtype=dtype) * 0.4)
    x_im     = (torch.randn(B, S, Im, device=device, dtype=dtype) * 0.4)
    x_qd     = (torch.randn(B, S, Qd, device=device, dtype=dtype) * 0.4)

    # For the 4-D per-head tensors the norms / rope operate on.
    q_4d = torch.randn(B, Hq, S, D, device=device, dtype=dtype) * 0.4
    k_4d = torch.randn(B, Hk, S, D, device=device, dtype=dtype) * 0.4

    cos, sin = _build_cos_sin(D, S, device, dtype)

    # KV cache for the attention sub-op (includes past + current)
    k_cache = torch.randn(B, Hk, P, D, device=device, dtype=dtype) * 0.1 if P > 0 else None
    v_cache = torch.randn(B, Hk, P, D, device=device, dtype=dtype) * 0.1 if P > 0 else None

    # Pre-compute k/v for the attention-only measurement so we don't recount
    # the qkv_proj cost inside the attention callable.
    q_for_attn = torch.randn(B, Hq, S, D, device=device, dtype=dtype) * 0.4
    k_for_attn_cur = torch.randn(B, Hk, S, D, device=device, dtype=dtype) * 0.4
    v_for_attn_cur = torch.randn(B, Hk, S, D, device=device, dtype=dtype) * 0.4
    if P > 0:
        k_full = torch.cat([k_cache, k_for_attn_cur], dim=2)
        v_full = torch.cat([v_cache, v_for_attn_cur], dim=2)
    else:
        k_full = k_for_attn_cur
        v_full = v_for_attn_cur

    n_rep = cfg.gqa_group
    if n_rep > 1:
        k_attn = k_full[:, :, None, :, :].expand(B, Hk, n_rep, k_full.shape[-2], D)
        v_attn = v_full[:, :, None, :, :].expand(B, Hk, n_rep, v_full.shape[-2], D)
        k_attn = k_attn.reshape(B, Hq, k_full.shape[-2], D).contiguous()
        v_attn = v_attn.reshape(B, Hq, v_full.shape[-2], D).contiguous()
    else:
        k_attn = k_full.contiguous()
        v_attn = v_full.contiguous()

    is_causal_attn = (P == 0) and (S > 1)

    # For the full-layer sanity call we need the pre-RoPE cos/sin for current
    # positions and a matching KV cache pair (positions [P : P+S)).
    cos_cur, sin_cur = _build_cos_sin(D, S, device, dtype)

    # ---------------------------------------------------------------------
    # Zero-arg callable definitions
    # ---------------------------------------------------------------------
    def f_q_proj():     return layer.q_proj(x_hidden)
    def f_k_proj():     return layer.k_proj(x_hidden)
    def f_v_proj():     return layer.v_proj(x_hidden)
    def f_o_proj():     return layer.o_proj(x_qd)
    def f_gate_proj():  return layer.gate_proj(x_hidden)
    def f_up_proj():    return layer.up_proj(x_hidden)
    def f_down_proj():  return layer.down_proj(x_im)

    def f_input_rmsnorm(): return layer.input_layernorm(x_hidden)
    def f_post_rmsnorm():  return layer.post_attention_layernorm(x_hidden)
    def f_q_norm(): return layer.q_norm(q_4d)
    def f_k_norm(): return layer.k_norm(k_4d)

    def f_rope():
        return _apply_rope(q_4d, k_4d, cos, sin)

    def f_attention():
        return F.scaled_dot_product_attention(
            q_for_attn, k_attn, v_attn,
            attn_mask=None, dropout_p=0.0, is_causal=is_causal_attn,
        )

    def f_full_layer():
        return layer(x_hidden, cos_cur, sin_cur, k_cache, v_cache)

    return PerOpCallables(
        q_proj=f_q_proj,
        k_proj=f_k_proj,
        v_proj=f_v_proj,
        o_proj=f_o_proj,
        gate_proj=f_gate_proj,
        up_proj=f_up_proj,
        down_proj=f_down_proj,
        input_rmsnorm=f_input_rmsnorm,
        post_rmsnorm=f_post_rmsnorm,
        q_norm=f_q_norm,
        k_norm=f_k_norm,
        rope=f_rope,
        attention=f_attention,
        full_layer=f_full_layer,
    )


__all__ = [
    "RMSNorm",
    "Qwen3LayerBF16",
    "PerOpCallables",
    "build_per_op_callables",
]
