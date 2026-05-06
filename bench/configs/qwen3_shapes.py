"""Qwen3 model shape constants for the r79 replacement bench.

Source: HuggingFace official configs + cross-check against
`kernel/cuda_kernel/benchmarks/bench_qwen3_shapes.py`.

Rationale: we keep the 3 selected models (Qwen3-4B / 8B / 14B) in a single,
authoritative place so every driver script produces the same d_in / d_out
numbers.  Per MODEL_SELECTION.md §3 the replacement bench only substitutes
the 4 fused-GEMM kernels (qkv_fused, o_proj, gate_up_fused, down_proj);
the FP16 baseline however measures the **un-fused** 7 Linears
(q / k / v / o / gate / up / down) per user decision 2026-05-06.
"""

from __future__ import annotations

from dataclasses import dataclass


HEAD_DIM: int = 128  # Qwen3 fixes head_dim=128 regardless of num_heads


@dataclass(frozen=True)
class Qwen3Config:
    name: str
    hidden: int                 # hidden_size / d_model
    intermediate: int           # MLP intermediate_size
    num_q_heads: int            # num_attention_heads
    num_kv_heads: int           # num_key_value_heads (GQA)
    head_dim: int = HEAD_DIM

    @property
    def q_out(self) -> int:
        """d_out of q_proj = num_q_heads * head_dim."""
        return self.num_q_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        """d_out of one (k_proj or v_proj) = num_kv_heads * head_dim."""
        return self.num_kv_heads * self.head_dim

    @property
    def gqa_group(self) -> int:
        """num_q_heads // num_kv_heads, used for repeat_kv."""
        return self.num_q_heads // self.num_kv_heads


# Canonical list of the 3 selected models (MODEL_SELECTION.md §1).
# Values cross-verified against bench_qwen3_shapes.py.
QWEN3_MODELS: tuple[Qwen3Config, ...] = (
    Qwen3Config("Qwen3-4B",  hidden=2560, intermediate=9728,  num_q_heads=32, num_kv_heads=8),
    Qwen3Config("Qwen3-8B",  hidden=4096, intermediate=12288, num_q_heads=32, num_kv_heads=8),
    Qwen3Config("Qwen3-14B", hidden=5120, intermediate=17408, num_q_heads=40, num_kv_heads=8),
)

QWEN3_BY_NAME: dict[str, Qwen3Config] = {c.name: c for c in QWEN3_MODELS}


# -----------------------------------------------------------------------------
# Phase / batch sweep  (MODEL_SELECTION.md §2)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PhaseConfig:
    name: str           # "prefill" / "decode"
    seqlen: int         # tokens per sample (prefill=2048, decode=1)
    past_kv_len: int    # past KV cache length (0 for prefill, 2048 for decode)

PREFILL = PhaseConfig(name="prefill", seqlen=2048, past_kv_len=0)
DECODE  = PhaseConfig(name="decode",  seqlen=1,    past_kv_len=2048)

PHASES: tuple[PhaseConfig, ...] = (PREFILL, DECODE)
BATCH_SIZES: tuple[int, ...] = (4, 8, 16, 32)


# -----------------------------------------------------------------------------
# Un-fused projection enumeration (FP16 baseline side)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjShape:
    """One un-fused Linear in a Qwen3 transformer layer."""
    proj: str       # q / k / v / o / gate / up / down
    d_in: int
    d_out: int

def enumerate_unfused_projs(cfg: Qwen3Config) -> list[ProjShape]:
    """The 7 un-fused Linears measured on the FP16 side."""
    return [
        ProjShape("q",    d_in=cfg.hidden,        d_out=cfg.q_out),
        ProjShape("k",    d_in=cfg.hidden,        d_out=cfg.kv_out),
        ProjShape("v",    d_in=cfg.hidden,        d_out=cfg.kv_out),
        ProjShape("o",    d_in=cfg.q_out,         d_out=cfg.hidden),
        ProjShape("gate", d_in=cfg.hidden,        d_out=cfg.intermediate),
        ProjShape("up",   d_in=cfg.hidden,        d_out=cfg.intermediate),
        ProjShape("down", d_in=cfg.intermediate,  d_out=cfg.hidden),
    ]

def enumerate_fused_projs(cfg: Qwen3Config) -> list[ProjShape]:
    """The 4 fused Linears used on the CUDA (W4A4) replacement side.

    Not consumed by the BF16 bench; kept here so both sides read shapes
    from the same source of truth.
    """
    return [
        ProjShape("qkv_fused",     d_in=cfg.hidden,        d_out=cfg.q_out + 2 * cfg.kv_out),
        ProjShape("o",             d_in=cfg.q_out,         d_out=cfg.hidden),
        ProjShape("gate_up_fused", d_in=cfg.hidden,        d_out=2 * cfg.intermediate),
        ProjShape("down",          d_in=cfg.intermediate,  d_out=cfg.hidden),
    ]


__all__ = [
    "HEAD_DIM",
    "Qwen3Config",
    "QWEN3_MODELS",
    "QWEN3_BY_NAME",
    "PhaseConfig",
    "PREFILL",
    "DECODE",
    "PHASES",
    "BATCH_SIZES",
    "ProjShape",
    "enumerate_unfused_projs",
    "enumerate_fused_projs",
]
