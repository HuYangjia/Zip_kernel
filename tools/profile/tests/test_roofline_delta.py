"""Unit tests for :mod:`kernel.tools.profile.roofline_delta`.

These tests run on CPU in <0.1 s; they lock down the two roofline
formulas against hand-computed references and verify the markdown
renderer produces a table with the expected column/header shape.
"""
from __future__ import annotations

import io
import json
import math

import pytest

from kernel.tools.profile.roofline_delta import (
    EFF_FP16,
    EFF_HBM,
    EFF_INT4,
    GROUP,
    ShapeSpec,
    cuda_roof_us,
    fp16_roof_us,
    parse_tag,
    render_delta_markdown,
)


class TestParseTag:
    def test_canonical_audit_tag(self) -> None:
        spec = parse_tag("audit_0p6B_kv_T8_1024_2048")
        assert spec == ShapeSpec(T=8, d_in=1024, d_out=2048)

    def test_legacy_four_field_tag(self) -> None:
        # Fallback: tag with a T<n> marker at some position other than idx=3
        spec = parse_tag("benchmark_T16_4096_4096")
        assert spec == ShapeSpec(T=16, d_in=4096, d_out=4096)

    @pytest.mark.parametrize("bad", ["no_t_field_here", "missing_T8", "audit_T_nodin_2048"])
    def test_malformed_tags_raise(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_tag(bad)


class TestCudaRoof:
    def test_t1_gemv_path_returns_max_of_compute_and_bytes(self) -> None:
        # For T=1 small-d_in the bandwidth side should dominate
        got = cuda_roof_us(1, 1024, 2048)
        # Hand computation reconstruction:
        ng = 1024 // GROUP
        bytes_gv = 2 * 1024 + 0.5 * 1024 * 2048 + 4 * 2048 * ng + 2 * 2048
        compute = 2 * 1 * 1024 * 2048 / (EFF_INT4 * 1e6)
        expected = max(compute, bytes_gv / (EFF_HBM * 1e3))
        assert math.isclose(got, expected, rel_tol=1e-9)

    def test_t_gt_1_splits_quant_and_gemm(self) -> None:
        got = cuda_roof_us(128, 2560, 2048)
        ng = 2560 // GROUP
        bytes_q = 2 * 128 * 2560 + 0.5 * 128 * 2560 + 2 * 128 + 4 * 128 * ng
        bytes_g = 0.5 * 2560 * 2048 + 0.5 * 128 * 2560 + 4 * 2048 * ng + 2 * 128 * 2048
        compute = 2 * 128 * 2560 * 2048 / (EFF_INT4 * 1e6)
        expected = bytes_q / (EFF_HBM * 1e3) + max(compute, bytes_g / (EFF_HBM * 1e3))
        assert math.isclose(got, expected, rel_tol=1e-9)

    def test_monotonic_in_shape(self) -> None:
        # Larger problem must take longer under roofline
        small = cuda_roof_us(8, 1024, 2048)
        large = cuda_roof_us(8, 2048, 2048)
        assert large > small


class TestFp16Roof:
    def test_compute_bound_regime(self) -> None:
        # Large square matmul is compute-bound on sm_89
        got = fp16_roof_us(1024, 4096, 4096)
        flops = 2 * 1024 * 4096 * 4096
        bytes_ = 2 * (4096 * 4096 + 1024 * 4096 + 1024 * 4096)
        assert got == max(flops / (EFF_FP16 * 1e6), bytes_ / (EFF_HBM * 1e3))
        # And specifically compute side should win here
        assert flops / (EFF_FP16 * 1e6) > bytes_ / (EFF_HBM * 1e3)

    def test_memory_bound_regime(self) -> None:
        # T=1 GEMV is memory-bound
        got = fp16_roof_us(1, 2048, 2048)
        flops = 2 * 1 * 2048 * 2048
        bytes_ = 2 * (2048 * 2048 + 1 * 2048 + 1 * 2048)
        assert got == max(flops / (EFF_FP16 * 1e6), bytes_ / (EFF_HBM * 1e3))
        assert bytes_ / (EFF_HBM * 1e3) > flops / (EFF_FP16 * 1e6)


class TestRenderDeltaMarkdown:
    @pytest.fixture
    def fake_bench(self) -> dict:
        return {
            "meta": {
                "run_id": "unit_20260429_000000",
                "device": "FakeGPU",
                "warmup": 200,
                "outer": 10,
                "inner": 100,
                "K": 5,
            },
            "records": [
                {
                    "tag": "audit_0p6B_kv_T8_1024_2048",
                    "t_eager_med_us": 100.0,
                    "t_graph_med_us": 50.0,
                },
                {
                    "tag": "audit_0p6B_q_T1_1024_2048",
                    "t_eager_med_us": 60.0,
                    "t_graph_med_us": 30.0,
                },
            ],
        }

    def test_title_header_and_meta_line_present(self, fake_bench: dict) -> None:
        buf = io.StringIO()
        out = render_delta_markdown(fake_bench, title="Pytest R50", out=buf)
        assert buf.getvalue() == out
        assert "## Pytest R50 — roofline delta" in out
        assert "bench run: `unit_20260429_000000`" in out
        assert "warmup=200" in out

    def test_table_has_one_row_per_record(self, fake_bench: dict) -> None:
        out = render_delta_markdown(fake_bench, title="T")
        table_rows = [l for l in out.splitlines() if l.startswith("| audit_")]
        assert len(table_rows) == 2

    def test_summary_includes_median_delta(self, fake_bench: dict) -> None:
        out = render_delta_markdown(fake_bench, title="T")
        # Both records have eager_us = 2× graph_us → graph is 2× faster →
        # Δ eff is always positive.
        assert "median Δ cuda_eff" in out
        assert "aggregate wall-time saved" in out

    def test_custom_labels_propagate(self, fake_bench: dict) -> None:
        out = render_delta_markdown(
            fake_bench, title="T", eager_label="old", opt_label="new"
        )
        # Header row of the table
        assert "| old_us | old_eff |" in out
        assert "| new_us | new_eff |" in out

    def test_empty_records_yields_no_summary(self) -> None:
        bench = {"meta": {}, "records": []}
        out = render_delta_markdown(bench, title="Empty")
        # Header + table headers present, no summary bullets
        assert "## Empty — roofline delta" in out
        assert "aggregate wall-time saved" not in out

    def test_returned_string_is_json_serialisable_when_wrapped(self, fake_bench: dict) -> None:
        out = render_delta_markdown(fake_bench, title="T")
        # Sanity: no exotic chars that break json.dumps (used by some
        # downstream report pipelines)
        json.dumps({"body": out})
