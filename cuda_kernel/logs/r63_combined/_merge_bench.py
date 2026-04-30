"""Merge r62_f2_final (0.6-8B) + r63_large_models (14-70B) bench.json into one
combined bench.json so the existing qwen3_roofline_report.py can emit a
single unified report covering all 7 models × 140 shapes.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # cuda_kernel/
SOURCES = [
    ROOT / 'logs/r62_f2_final/qwen3_20260430_122555/bench.json',
    ROOT / 'logs/r63_large_models/qwen3_20260430_124225/bench.json',
]
OUT = ROOT / 'logs/r63_combined/bench.json'

merged_records = []
merged_meta = None
for src in SOURCES:
    data = json.loads(src.read_text())
    merged_records.extend(data['records'])
    if merged_meta is None:
        merged_meta = dict(data.get('meta', {}))
        merged_meta['source'] = 'combined r62_f2_final + r63_large_models'
        merged_meta['source_files'] = [str(s) for s in SOURCES]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(
    {'meta': merged_meta, 'records': merged_records},
    indent=2,
))
print(f"wrote {OUT} ({len(merged_records)} records, "
      f"{sum(1 for r in merged_records if r.get('kernel') == 'end_to_end')} e2e)")
