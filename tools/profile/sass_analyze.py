"""SASS + resource-usage static analyser for the compiled CUDA extension.

Task-item.md step 9.  Given the path to ``hkust_v9_cuda.so`` this script
runs ``cuobjdump --dump-sass`` and ``cuobjdump --dump-resource-usage``,
classifies SASS instructions into operational families, and emits three
static verdicts per kernel:

  * ``tc_underutil``      - **MAC-weighted** TC share < 50 %.  Meaning:
                            the tensor pipeline is emitted (IMMA/HMMA
                            present) but does not dominate the MAC
                            budget, or it dominates MACs but its issue
                            slots are starved by epilogue FMA / address
                            IMAD / shared-memory roundtrip.  The label
                            name is retained for downstream taxonomy
                            stability; the *meaning* is "MMA pipeline
                            starvation".  See
                            ``cuda_kernel/logs/phase2_microscope/
                            phase2_tc_rediagnosis.md`` for the
                             2026-04-28 rediagnosis that swapped the
                            trigger from ``tc_fraction<5 %`` (issue-slot
                            share) to ``mac_tc_share<50 %`` (MAC share).
  * ``epilogue_fma_bound``- CUDA_FMA issue-slot fraction > 40 %
                            (FMA dominant) — typically the dequant
                            epilogue dominates the non-tensor-pipeline
                            budget.
  * ``register_spill``    - resource-usage reports STACK > 0 or LOCAL > 0
                            (indicates local-memory usage i.e. register
                            spilling).

The script does **not** require a GPU, it's a pure offline static pass;
we rely on cuobjdump being on ``$PATH`` (typically ``/usr/local/cuda/bin``).

MAC-weight reference (sm_89)
----------------------------
One ``mma.m16n8k64.s4.s4.s32`` does ``16 * 8 * 64 = 8192`` integer MACs;
one ``HFMA2`` does 2 MACs; one ``FFMA`` / ``IMAD`` does 1 MAC; one
``IDP4A`` does 4 MACs.  A naive "TC-instruction-count / total-insts"
ratio therefore *underestimates* TC compute share by several thousand
times whenever the kernel is IMMA-based: 64 IMMAs looks like ~1.3 %
of a 5000-instruction histogram but delivers 99 % of the MACs.  The
``mac_tc_share`` property below fixes this by weighing each family
by its per-instruction MAC throughput.

Usage
-----
    python -m kernel.tools.profile.sass_analyze \
        --so /root/.cache/hkust_v9_cuda/hkust_v9_cuda.so \
        --out logs/phase2_microscope/_sass/sass_profile.txt

If ``--so`` is not given, the module auto-loads ``kernel.cuda_kernel.ops``
and inspects the ``_ext.__file__``.  The output directory is created if
needed.  A per-kernel summary is also emitted as ``sass_profile.json``.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Instruction families
# ---------------------------------------------------------------------------
# SASS mnemonic prefixes (left-most token of the instruction word).
# These regexes are anchored to the start of the mnemonic to avoid false
# hits on operand names like ``HMMA.16816.F16.F16``.
FAMILY_PATTERNS: Dict[str, re.Pattern[str]] = {
    # Tensor Core family (SM89).  Both HMMA (fp16 TC) and IMMA (int TC)
    # count toward "TC utilisation" for our W4A4 kernel.
    "TC_MMA":   re.compile(r"^\s*(?:HMMA|IMMA|BMMA|DMMA)\b"),
    # CUDA core ALU-FMA family.
    "CUDA_FMA": re.compile(r"^\s*(?:FFMA|DFMA|IMAD|IMAD\.)"),
    # Dot-product-accumulate (INT8x4 -> INT32, 4 MAC / inst on sm_89).
    # Used by dense_gemv_decode / fused_gemv_decode to compute the W4
    # GEMV lane on CUDA cores (dp4a path, not MMA path).
    "DP4A":     re.compile(r"^\s*(?:IDP|IDP\.4A|IDP4A)\b"),
    # Integer ALU (not FMA): adds, shifts, logic.
    "INT_ALU":  re.compile(r"^\s*(?:IADD|IADD3|ISETP|ISCADD|LOP3|LOP|SHF|SHL|SHR|IMUL|XMAD)\b"),
    # Load/store to global memory.
    "LDG":      re.compile(r"^\s*(?:LDG|LDGSTS|LDGDEPBAR)\b"),
    "STG":      re.compile(r"^\s*(?:STG|STGDEPBAR)\b"),
    # Shared memory ops.
    "LDS":      re.compile(r"^\s*(?:LDS|LDSM)\b"),
    "STS":      re.compile(r"^\s*(?:STS)\b"),
    # Warp-sync barriers & control flow of interest.
    "BAR":      re.compile(r"^\s*(?:BAR|BARRIER|DEPBAR|MEMBAR)\b"),
    "SYNC":     re.compile(r"^\s*(?:BSSY|BSYNC|WARPSYNC)\b"),
    # Branch / predicate.
    "BRANCH":   re.compile(r"^\s*(?:BRA|JMP|BRX|EXIT|RET|NANOSLEEP|KILL)\b"),
    # Misc SFU / type-conversion.
    "CVT":      re.compile(r"^\s*(?:I2I|I2F|F2I|F2F|FRND|CVT)\b"),
    # Predicate / move.
    "MOV":      re.compile(r"^\s*(?:MOV|SEL|ISET|PSET|S2R|R2P|P2R)\b"),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class KernelStats:
    """Per-kernel summary extracted from SASS + resource usage."""

    name: str
    regs: int
    stack: int
    shared: int
    local: int
    total_insts: int
    counts: Dict[str, int] = field(default_factory=dict)

    # Per-instruction MAC throughput on sm_89.  These constants underpin
    # ``mac_tc_share`` and fix the 2026-04-28 rediagnosis (see
    # ``phase2_tc_rediagnosis.md``).  An IMMA m16n8k64.s4.s4.s32 does
    # 16*8*64=8192 MACs; HMMA m16n8k16.f16 does 16*8*16=2048 MACs; we
    # pick the dominant s4 variant since our W4A4 kernels use s4 IMMA
    # throughout.  For HFMA2 nvcc emits FFMA-buckets as scalar FMA so we
    # keep the conservative 1 MAC/inst (pack-2 underestimate is OK: it
    # biases MAC share *down*, so tc_underutil triggering is still
    # sound as an upper bound).
    _MAC_PER_INST = {
        "TC_MMA": 8192,  # s4 m16n8k64 on sm_89 (our hot path)
        "CUDA_FMA": 1,
        "DP4A": 4,
    }

    @property
    def tc_fraction(self) -> float:
        """Issue-slot share of TC instructions (not MAC share).

        This is the *count-weighted* fraction of IMMA/HMMA in the SASS
        histogram.  It is useful as a pipeline-scheduling indicator
        ("how often does the warp scheduler hand off to the tensor
        pipeline?") but is **not** a direct proxy for compute share.
        See :attr:`mac_tc_share` for the compute-weighted equivalent.
        """
        return self.counts.get("TC_MMA", 0) / self.total_insts if self.total_insts else 0.0

    @property
    def mac_tc_share(self) -> float:
        """MAC-weighted share of work delivered by TC instructions.

        ``mac_tc_share = MAC(TC) / MAC(TC + CUDA_FMA + DP4A)``.  This is
        the correct denominator for claims of the form "is TC carrying
        the compute?".  See module docstring and
        ``phase2_tc_rediagnosis.md`` for why this matters.
        """
        mac_tc = self.counts.get("TC_MMA", 0) * self._MAC_PER_INST["TC_MMA"]
        mac_cuda = (
            self.counts.get("CUDA_FMA", 0) * self._MAC_PER_INST["CUDA_FMA"]
            + self.counts.get("DP4A", 0) * self._MAC_PER_INST["DP4A"]
        )
        total = mac_tc + mac_cuda
        if total == 0:
            return 0.0
        return mac_tc / total

    @property
    def fma_fraction(self) -> float:
        return self.counts.get("CUDA_FMA", 0) / self.total_insts if self.total_insts else 0.0

    @property
    def ld_fraction(self) -> float:
        if not self.total_insts:
            return 0.0
        return (self.counts.get("LDG", 0) + self.counts.get("LDS", 0)) / self.total_insts

    def verdicts(self) -> List[str]:
        v: List[str] = []
        # Gate ``tc_underutil`` on MAC-weighted share, not issue-slot
        # share.  A kernel with 64 IMMAs + 1408 FFMAs has tc_fraction=
        # 1.28 % but mac_tc_share=99.5 %; the old rule flagged it as
        # "TC not emitted", which is wrong.  We keep the 50 % threshold
        # deliberately loose so that only kernels whose MAC budget
        # genuinely does not reach the tensor pipeline (e.g.
        # ``activation_quant``, ``gemv_decode`` which are dp4a / pure
        # dequant) still trigger; that is the correct semantics.
        if self.mac_tc_share < 0.50:
            v.append("tc_underutil")
        if self.fma_fraction > 0.40:
            v.append("epilogue_fma_bound")
        if self.stack > 0 or self.local > 0:
            v.append("register_spill")
        if self.regs >= 128:
            v.append("high_reg_pressure")
        return v


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
_RESOURCE_FUNC_RE = re.compile(
    r"^\s*Function\s+(?P<name>\S+):\s*$"
)
_RESOURCE_STATS_RE = re.compile(
    r"REG:(?P<reg>\d+)\s+STACK:(?P<stack>\d+)\s+SHARED:(?P<shared>\d+)\s+LOCAL:(?P<local>\d+)"
)


def parse_resource_usage(text: str) -> Dict[str, Dict[str, int]]:
    """Extract {kernel_mangled_name: {reg, stack, shared, local}} map."""
    out: Dict[str, Dict[str, int]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = _RESOURCE_FUNC_RE.match(line)
        if m:
            current = m.group("name")
            continue
        if current is None:
            continue
        m = _RESOURCE_STATS_RE.search(line)
        if m:
            # Take the *first* occurrence per function — there's exactly one.
            if current not in out:
                out[current] = {
                    "regs": int(m.group("reg")),
                    "stack": int(m.group("stack")),
                    "shared": int(m.group("shared")),
                    "local": int(m.group("local")),
                }
            current = None
    return out


# SASS dump format varies by cuobjdump version.  For sm89 / CUDA 12.x the
# per-function header is:
#   Function : _ZN...
#   .headerflags ...
# and the instruction lines are:
#   /*0008*/  IADD3 R0, ...
# We split on "Function : " to get per-kernel blocks.
_SASS_BLOCK_RE = re.compile(
    r"^\s*Function\s*:\s*(?P<name>\S+)\s*$", re.MULTILINE
)


def parse_sass(text: str) -> Dict[str, Dict[str, int]]:
    """Split cuobjdump --dump-sass text by kernel, count instruction families."""
    out: Dict[str, Dict[str, int]] = {}
    matches = list(_SASS_BLOCK_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group("name")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        counts: Dict[str, int] = {k: 0 for k in FAMILY_PATTERNS}
        total = 0
        for line in block.splitlines():
            # SASS instruction lines begin with ``/*offset*/``.  We only
            # count actual instructions, not labels / section headers.
            stripped = line.strip()
            if not stripped.startswith("/*"):
                continue
            # Strip "/*0008*/" prefix.
            try:
                body = stripped.split("*/", 1)[1].strip()
            except IndexError:
                continue
            if not body or body.startswith(".") or body.startswith("//"):
                continue
            total += 1
            for fam, pat in FAMILY_PATTERNS.items():
                if pat.match(body):
                    counts[fam] += 1
                    break
        out[name] = {"total": total, **counts}
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _default_so_path() -> Path:
    # Lazy import — this module should be usable even if torch is absent
    # on the host doing the analysis (the ``.so`` was produced elsewhere).
    try:
        from kernel.cuda_kernel.ops import _ext  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(
            f"could not import kernel.cuda_kernel.ops to locate .so: {exc}"
        ) from exc
    return Path(_ext.__file__)


def _run_cuobjdump(so: Path, flag: str) -> str:
    exe = shutil.which("cuobjdump")
    if not exe:
        raise SystemExit(
            "cuobjdump not found on $PATH — set PATH=/usr/local/cuda/bin:$PATH"
        )
    res = subprocess.run(
        [exe, flag, str(so)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise SystemExit(
            f"cuobjdump {flag} failed ({res.returncode}):\n{res.stderr[:2000]}"
        )
    return res.stdout


# Cheap, best-effort demangler.  We strip the symbol prefix
# ``_ZN8hkust_v9<len><name>E...`` to get the first namespace segment
# (the kernel *family*).  If we can't demangle cleanly, fall back to
# the mangled name.
_MANGLED_FAMILY_RE = re.compile(r"_ZN8hkust_v9\d+(?P<fam>[a-zA-Z0-9_]+)\d+")


def _short_name(mangled: str) -> str:
    m = _MANGLED_FAMILY_RE.match(mangled)
    return m.group("fam") if m else mangled


def analyse(so: Path) -> Tuple[List[KernelStats], str, str]:
    res_text = _run_cuobjdump(so, "--dump-resource-usage")
    sass_text = _run_cuobjdump(so, "--dump-sass")

    res_by_name = parse_resource_usage(res_text)
    sass_by_name = parse_sass(sass_text)

    # Merge on mangled name.  Some kernels appear only in resource-usage
    # (small device helpers) and some only in SASS (rare).  Keep union.
    all_names = set(res_by_name) | set(sass_by_name)
    stats: List[KernelStats] = []
    for name in sorted(all_names):
        res = res_by_name.get(name, {"regs": 0, "stack": 0, "shared": 0, "local": 0})
        sass = sass_by_name.get(name, {"total": 0})
        counts = {k: int(sass.get(k, 0)) for k in FAMILY_PATTERNS}
        stats.append(
            KernelStats(
                name=name,
                regs=res["regs"],
                stack=res["stack"],
                shared=res["shared"],
                local=res["local"],
                total_insts=int(sass.get("total", 0)),
                counts=counts,
            )
        )
    return stats, res_text, sass_text


def render_report(stats: List[KernelStats]) -> str:
    lines: List[str] = []
    lines.append("# Phase 2 SASS Static Profile\n")
    lines.append(
        "One row per kernel instantiation compiled into `hkust_v9_cuda.so`.\n"
        "`TC%` = issue-slot share of HMMA+IMMA (useful as a scheduler\n"
        "indicator but **not** a compute-share proxy); `MAC_TC%` =\n"
        "MAC-weighted compute share of TC under sm_89 s4 IMMA throughput\n"
        "(8192 MAC/IMMA vs 1 MAC/FFMA, 4 MAC/DP4A) — this is the field\n"
        "that now drives the ``tc_underutil`` verdict; `FMA%` counts\n"
        "FFMA+IMAD; `LD%` counts LDG+LDS.  Verdicts follow task-item.md\n"
        "§9 as amended by ``phase2_tc_rediagnosis.md`` (2026-04-28).\n"
    )
    lines.append(
        "| # | family | regs | shared | stack | insts | TC% | MAC_TC% | FMA% | LD% | verdicts |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, s in enumerate(stats, 1):
        fam = _short_name(s.name)
        verd = ",".join(s.verdicts()) or "-"
        lines.append(
            f"| {i} | `{fam}` | {s.regs} | {s.shared} | {s.stack} | "
            f"{s.total_insts} | {s.tc_fraction*100:.1f} | "
            f"{s.mac_tc_share*100:.1f} | "
            f"{s.fma_fraction*100:.1f} | {s.ld_fraction*100:.1f} | {verd} |"
        )
    lines.append("")

    # Aggregate verdict counts.
    agg: Dict[str, int] = {}
    for s in stats:
        for v in s.verdicts():
            agg[v] = agg.get(v, 0) + 1
    lines.append("## Aggregate verdict counts")
    lines.append("")
    if not agg:
        lines.append("_no verdicts triggered across all kernels._")
    else:
        for v, c in sorted(agg.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{v}`: **{c}** / {len(stats)} kernels")
    lines.append("")
    return "\n".join(lines)


def _repo_root() -> Path:
    """Resolve the repository root from this file's location.

    ``tools/profile/sass_analyze.py`` lives 2 directories below the repo
    root (``tools/`` is a sibling of ``cuda_kernel/``).
    """
    return Path(__file__).resolve().parents[2]


def main() -> None:
    default_out = _repo_root() / "cuda_kernel/logs/phase2_microscope/_sass/sass_profile.md"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--so", type=Path, default=None, help="Path to hkust_v9_cuda.so")
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Markdown report destination.",
    )
    args = ap.parse_args()

    so = args.so or _default_so_path()
    print(f"[sass_analyze] analysing {so}")
    stats, res_text, sass_text = analyse(so)
    print(f"[sass_analyze] found {len(stats)} kernels")

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    args.out.write_text(render_report(stats))
    print(f"[sass_analyze] wrote {args.out}")

    # Per-kernel JSON for downstream joins
    json_path = args.out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "so_path": str(so),
                "kernels": [
                    {
                        **asdict(s),
                        "tc_fraction": s.tc_fraction,
                        "mac_tc_share": s.mac_tc_share,
                        "fma_fraction": s.fma_fraction,
                        "ld_fraction": s.ld_fraction,
                        "verdicts": s.verdicts(),
                    }
                    for s in stats
                ],
            },
            indent=2,
        )
    )
    print(f"[sass_analyze] wrote {json_path}")

    # Raw dumps for debugging — trimmed to avoid repo bloat.
    (out_dir / "resource_usage.txt").write_text(res_text)
    (out_dir / "sass.txt").write_text(sass_text[:5_000_000])  # cap at 5 MB
    print(f"[sass_analyze] wrote raw dumps to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
