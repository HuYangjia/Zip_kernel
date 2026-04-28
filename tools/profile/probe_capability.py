"""Capability probe for kernel efficiency diagnosis plan (Step 0).

Purpose
-------
Before kicking off Phase 1/2 data collection, this script verifies on the
actual GPU host whether every tool we plan to depend on is usable.

It writes two artefacts under ``logs/_probe/<timestamp>/``:

* ``probe_report.json`` — machine-readable capability matrix.
* ``probe_report.md``   — human-readable summary.

The script is strictly **read-only / non-invasive**:
    - does not edit any source file
    - does not import the production hot path
    - runs trivial CUDA work only to test graph capture & metric sampling
    - tolerates missing tools and records them as capability=False

It is safe to run on either the local laptop (which will mostly report
"no CUDA") or the AutoDL 4090 host.

Run
---
    python tools/profile/probe_capability.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# ``probe_capability.py`` lives at ``<kernel>/tools/profile/probe_capability.py``
# so parents[2] resolves to the kernel submodule root (local: ``kernel/``,
# autodl: ``/root/Zip_kernel``).
KERNEL_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = KERNEL_ROOT / "cuda_kernel" / "logs" / "_profile"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 20) -> Dict[str, Any]:
    """Run a shell command, capture stdout/stderr/rc, never raise."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": " ".join(cmd),
            "rc": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "ok": proc.returncode == 0,
        }
    except FileNotFoundError:
        return {
            "cmd": " ".join(cmd),
            "rc": -1,
            "stdout": "",
            "stderr": "executable not found",
            "ok": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "cmd": " ".join(cmd),
            "rc": -2,
            "stdout": "",
            "stderr": f"timeout after {timeout}s",
            "ok": False,
        }


def _which(binary: str) -> str | None:
    path = shutil.which(binary)
    return path if path else None


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_host() -> Dict[str, Any]:
    """Basic host/env info: os, user, cwd, python, git sha."""
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "user": os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown",
        "hostname": os.uname().nodename,
        "platform": sys.platform,
    }
    sha = _run(["git", "rev-parse", "HEAD"])
    info["git_sha"] = sha["stdout"][:12] if sha["ok"] else "n/a"
    return info


def probe_gpu() -> Dict[str, Any]:
    """Detect GPU name / driver via nvidia-smi (best effort)."""
    out: Dict[str, Any] = {"available": False}
    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if smi["ok"] and smi["stdout"]:
        parts = [p.strip() for p in smi["stdout"].split(",")]
        out.update(
            {
                "available": True,
                "name": parts[0] if len(parts) > 0 else "",
                "driver": parts[1] if len(parts) > 1 else "",
                "memory": parts[2] if len(parts) > 2 else "",
                "compute_capability": parts[3] if len(parts) > 3 else "",
            }
        )
    else:
        out["error"] = smi["stderr"] or "nvidia-smi unavailable"
    return out


def probe_tool_versions() -> Dict[str, Any]:
    return {
        "nsys_path": _which("nsys"),
        "nsys": _run(["nsys", "--version"]),
        "ncu": _run(["ncu", "--version"]),
        "cuobjdump": _run(["cuobjdump", "--version"]),
        "nvcc": _run(["nvcc", "--version"]),
        "ptxas": _run(["ptxas", "--version"]),
        "nvidia_smi": _run(["nvidia-smi", "-L"]),
    }


def probe_torch() -> Dict[str, Any]:
    """Import torch & report CUDA availability. No production hot-path import."""
    try:
        import torch  # noqa: WPS433

        info: Dict[str, Any] = {
            "import_ok": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_cap"] = torch.cuda.get_device_capability(0)
            try:
                prop = torch.cuda.get_device_properties(0)
                info["sm_count"] = prop.multi_processor_count
                info["total_memory_gb"] = round(
                    prop.total_memory / (1024 ** 3), 2
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                info["prop_error"] = repr(exc)
        return info
    except Exception as exc:  # pragma: no cover
        return {"import_ok": False, "error": repr(exc)}


def probe_cuda_graph() -> Dict[str, Any]:
    """Capture + replay a trivial graph to prove CUDA Graph is usable."""
    try:
        import torch  # noqa: WPS433
    except Exception as exc:
        return {"ok": False, "error": f"torch import failed: {exc!r}"}
    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA not available"}

    try:
        dev = torch.device("cuda")
        a = torch.randn(1024, device=dev)
        b = torch.randn(1024, device=dev)

        # Warm-up on a side stream before capturing (requirement of CUDA Graph).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                c = a + b
                c = c * 2
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            c = a + b
            c = c * 2

        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()

        return {"ok": True, "replayed": 5, "result_sample": float(c[0].item())}
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"ok": False, "error": repr(exc)}


def probe_nsys_metrics(log_dir: Path) -> Dict[str, Any]:
    """Run a trivial CUDA script under nsys with --gpu-metrics-device=all.

    We cannot know if the AutoDL container grants GPU metric access
    without trying; a successful nsys-rep output is the proof.
    """
    result: Dict[str, Any] = {"ok": False}
    if _which("nsys") is None:
        result["error"] = "nsys not found on PATH"
        return result

    trivial = log_dir / "trivial.py"
    trivial.write_text(
        (
            "import torch\n"
            "x = torch.randn(4096, 4096, device='cuda')\n"
            "for _ in range(50):\n"
            "    x = x @ x\n"
            "torch.cuda.synchronize()\n"
        ),
        encoding="utf-8",
    )
    rep_base = log_dir / "nsys_probe"
    # NOTE: --gpu-metrics-device=all is NOT used here because autodl containers
    # block PMU access (ERR_NVGPUCTRPERM).  We only verify that nsys can capture
    # a basic CUDA + NVTX timeline, which is all Phase 1 needs.
    cmd = [
        "nsys",
        "profile",
        "-t",
        "cuda,nvtx,osrt",
        "-f",
        "true",
        "-o",
        str(rep_base),
        sys.executable,
        str(trivial),
    ]
    probe = _run(cmd, timeout=120)
    result["cmd"] = " ".join(cmd)
    result["rc"] = probe["rc"]
    result["stderr_tail"] = probe["stderr"][-500:]
    result["stdout_tail"] = probe["stdout"][-500:]

    rep_file = log_dir / "nsys_probe.nsys-rep"
    result["report_exists"] = rep_file.exists()
    if rep_file.exists():
        result["report_size_bytes"] = rep_file.stat().st_size
    # Detect permission errors commonly seen in container environments.
    lowered = (probe["stderr"] + " " + probe["stdout"]).lower()
    result["permission_denied"] = any(
        tag in lowered
        for tag in (
            "ERR_NVGPUCTRPERM",
            "access denied",
            "requires running as root",
            "administrator privileges",
        )
    )
    # Phase 1 only needs basic CUDA timeline (no GPU metrics).
    result["ok"] = probe["rc"] == 0 and rep_file.exists()
    # Separately record whether GPU metrics sampling is available.
    result["gpu_metrics_blocked"] = "gpu-metrics" in lowered or "gpu_metrics" in lowered
    return result


def probe_cuda_kernel_import() -> Dict[str, Any]:
    """Check that the production cuda_kernel module can be imported.

    Follows run_server.md: must ``cd /root`` and use ``from kernel....``.
    On the local laptop this will almost certainly fail; that is expected.
    """
    info: Dict[str, Any] = {"attempted": True}
    try:
        import importlib  # noqa: WPS433

        # Ensure '/root' (or kernel parent) is on sys.path so 'kernel' resolves.
        candidate_roots = [
            "/root",
            str(KERNEL_ROOT.parent),
        ]
        for root in candidate_roots:
            if root not in sys.path:
                sys.path.insert(0, root)

        m = importlib.import_module("kernel.cuda_kernel.ops")
        info["ok"] = True
        info["module_file"] = getattr(m, "__file__", "n/a")
        info["has_activation_quant_cuda"] = hasattr(m, "activation_quant_cuda")
        info["has_fused_dense_sparse_cuda"] = hasattr(
            m, "fused_dense_sparse_cuda"
        )
    except Exception as exc:  # pragma: no cover
        info["ok"] = False
        info["error"] = repr(exc)
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write_markdown(report: Dict[str, Any], out: Path) -> None:
    """Human-readable summary of the capability matrix."""
    lines: list[str] = []
    lines.append("# Capability Probe Report")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {report['timestamp_utc']}")
    lines.append(f"- Host: {report['host']['hostname']} ({report['host']['platform']})")
    lines.append(f"- Git SHA: {report['host']['git_sha']}")
    lines.append(f"- Python: {report['host']['python_version']}")
    lines.append("")

    gpu = report["gpu"]
    lines.append("## GPU")
    if gpu.get("available"):
        lines.append(f"- Name: {gpu.get('name')}")
        lines.append(f"- Driver: {gpu.get('driver')}")
        lines.append(f"- Memory: {gpu.get('memory')}")
        lines.append(f"- Compute capability: {gpu.get('compute_capability')}")
    else:
        lines.append(f"- not available ({gpu.get('error', 'no nvidia-smi')})")
    lines.append("")

    lines.append("## Tooling")
    tools = report["tools"]
    for key in ("nsys", "ncu", "cuobjdump", "nvcc", "ptxas"):
        entry = tools.get(key, {})
        ok = "OK" if entry.get("ok") else "MISSING"
        head = (entry.get("stdout") or entry.get("stderr") or "").splitlines()
        version_line = head[0] if head else ""
        lines.append(f"- {key}: {ok}  `{version_line}`")
    lines.append("")

    torch_info = report["torch"]
    lines.append("## Torch")
    if torch_info.get("import_ok"):
        lines.append(f"- torch {torch_info.get('version')} (CUDA {torch_info.get('cuda_version', 'n/a')})")
        if torch_info.get("cuda_available"):
            lines.append(f"- device: {torch_info.get('device_name')} (SM {torch_info.get('device_cap')})")
            lines.append(f"- SM count: {torch_info.get('sm_count')}")
    else:
        lines.append(f"- import failed: {torch_info.get('error')}")
    lines.append("")

    cg = report["cuda_graph"]
    lines.append("## CUDA Graph capture + replay")
    lines.append(f"- ok: {cg.get('ok')}")
    if not cg.get("ok"):
        lines.append(f"- error: {cg.get('error')}")
    lines.append("")

    nsys = report["nsys_metrics"]
    lines.append("## nsys GPU Metrics Sampling")
    lines.append(f"- overall ok: {nsys.get('ok')}")
    lines.append(f"- report file exists: {nsys.get('report_exists')}")
    lines.append(f"- rc: {nsys.get('rc')}")
    lines.append(f"- permission_denied: {nsys.get('permission_denied')}")
    if not nsys.get("ok"):
        lines.append("- stderr tail:")
        lines.append("```")
        lines.append(nsys.get("stderr_tail", ""))
        lines.append("```")
    lines.append("")

    imp = report["cuda_kernel_import"]
    lines.append("## cuda_kernel import")
    lines.append(f"- ok: {imp.get('ok')}")
    if imp.get("ok"):
        lines.append(f"- module file: {imp.get('module_file')}")
        lines.append(f"- activation_quant_cuda present: {imp.get('has_activation_quant_cuda')}")
        lines.append(f"- fused_dense_sparse_cuda present: {imp.get('has_fused_dense_sparse_cuda')}")
    else:
        lines.append(f"- error: {imp.get('error')}")
    lines.append("")

    lines.append("## Decision matrix")
    verdict = report["verdict"]
    for key, value in verdict.items():
        lines.append(f"- {key}: {value}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    """Apply decision rules that later steps of the plan depend on."""
    nsys = report["nsys_metrics"]
    tools = report["tools"]
    # Phase 1 timeline: nsys basic CUDA trace (no GPU metrics needed).
    # Phase 2 GPU metrics: blocked in autodl containers; use microbench bisection.
    # SASS: cuobjdump lives in /usr/local/cuda/bin, needs PATH export.
    # ncu SM counters: also blocked (ERR_NVGPUCTRPERM); microbench is primary.
    return {
        "phase1_timeline_available": bool(
            tools.get("nsys", {}).get("ok") and nsys.get("report_exists")
        ),
        "phase2_gpu_metrics_available": False,  # PMU blocked in container
        "phase2_microbench_bisection_available": bool(
            report["cuda_kernel_import"].get("ok")
        ),
        "sass_static_analysis_available": bool(tools.get("cuobjdump", {}).get("ok")),
        "sass_available_with_cuda_path": True,  # /usr/local/cuda/bin/cuobjdump exists
        "cuda_graph_replay_available": bool(report["cuda_graph"].get("ok")),
        "ncu_available": bool(tools.get("ncu", {}).get("ok")),
        "ncu_sm_counters_available": False,  # ERR_NVGPUCTRPERM in container
        "cuda_kernel_importable": bool(report["cuda_kernel_import"].get("ok")),
    }


def main() -> int:
    ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = LOG_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "timestamp_utc": ts,
        "host": probe_host(),
        "gpu": probe_gpu(),
        "tools": probe_tool_versions(),
        "torch": probe_torch(),
        "cuda_graph": probe_cuda_graph(),
        "nsys_metrics": probe_nsys_metrics(out_dir),
        "cuda_kernel_import": probe_cuda_kernel_import(),
    }
    report["verdict"] = _compute_verdict(report)

    (out_dir / "probe_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _write_markdown(report, out_dir / "probe_report.md")

    print("=" * 60)
    print("Capability probe complete.")
    print(f"Artefacts: {out_dir}")
    print("Verdict:")
    for k, v in report["verdict"].items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Non-zero rc if the critical capabilities are all missing.
    verdict = report["verdict"]
    critical_missing = (
        not verdict["phase1_timeline_available"]
        and not verdict["sass_static_analysis_available"]
    )
    return 2 if critical_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
