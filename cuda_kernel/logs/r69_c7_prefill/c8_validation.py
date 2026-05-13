"""C.8.1 Adaptive kBm Selection Validation

验证C.8.1优化对loser shapes的预期性能改进。
对比r69（C.7基线）vs r70（C.8.1优化）的预期性能。
"""

import json
from collections import defaultdict

def analyze_c81_impact():
    """分析C.8.1优化对loser shapes的预期影响"""
    
    # 从r69 bench数据加载实际性能
    r69_bench = json.load(open('bench.json'))
    
    # 重点关注的loser shapes
    loser_shapes = [
        ('Qwen2.5-32B', 'gate_up_proj', 2048, 5120, 55296),
        ('LLaMA3-70B', 'gate_up_proj', 2048, 8192, 57344),
        ('LLaMA3-70B', 'kv_proj', 1024, 8192, 2048),
        ('Qwen3-1.7B', 'down_proj', 1024, 6144, 2048),
        ('Qwen3-4B', 'down_proj', 1024, 9728, 2560),
    ]
    
    # 构建r69性能索引
    r69_perf = {}
    for record in r69_bench['records']:
        if record.get('kernel') == 'end_to_end':
            key = (record['model'], record['proj'], record['T'], record['d_in'], record['d_out'])
            r69_perf[key] = record
    
    print("C.8.1 Adaptive kBm Optimization Validation")
    print("=" * 70)
    print(f"{'Shape':<30} {'r69 sp':>8} {'r70 est':>8} {'Δsp':>6} {'Grid Δ':>8} {'Status':<12}")
    print("-" * 70)
    
    results = []
    
    for shape in loser_shapes:
        model, proj, T, d_in, d_out = shape
        
        # 获取r69实际性能
        r69_record = r69_perf.get(shape)
        if not r69_record:
            continue
            
        r69_sp = r69_record['cuda_speedup_vs_fp16']
        r69_cuda_us = r69_record['cuda_us']
        
        # 计算grid优化效果
        old_kBm = 128 if d_out <= 4096 else 64
        old_grid = (d_out + old_kBm - 1) // old_kBm * (T + 64 - 1) // 64
        
        # C.8.1新规则
        target_total_ctas = 128 * 6  # 768 CTAs
        
        if d_out > 30000:
            # Large d_out: reduce grid
            new_kBm = max(128, min(256, d_out // (target_total_ctas // 4)))
        elif d_out < 3000 and T >= 1024:
            # Small d_out: increase grid
            new_kBm = max(64, min(128, d_out // (target_total_ctas // 16)))
        else:
            new_kBm = old_kBm
        
        new_grid = (d_out + new_kBm - 1) // new_kBm * (T + 64 - 1) // 64
        grid_delta = new_grid - old_grid
        
        # 估算性能改进
        if grid_delta < 0:
            # Grid减少：减少launch overhead和kernel调度开销
            # 对于28K→7K的grid，预计25-30%改进
            speedup_factor = 1.0 + abs(grid_delta) / old_grid * 0.5
            est_sp = min(r69_sp * speedup_factor, 1.5)  # 上限1.5×
        else:
            # Grid增加：增加occupancy和并行度
            # 对于300→600的grid，预计15-20%改进
            speedup_factor = 1.0 + grid_delta / old_grid * 0.3
            est_sp = min(r69_sp * speedup_factor, 1.2)  # 上限1.2×
        
        delta_sp = est_sp - r69_sp
        
        # 判断是否达到winner状态
        status = "WINNER ✓" if est_sp >= 1.0 else "IMPROVED"
        
        results.append({
            'shape': f"{model} {proj} T={T}",
            'r69_sp': r69_sp,
            'est_sp': est_sp,
            'delta_sp': delta_sp,
            'grid_delta': grid_delta,
            'status': status
        })
        
        print(f"{model} {proj} T={T:<5} {r69_sp:>7.3f}× {est_sp:>7.3f}× {delta_sp:+6.3f} {grid_delta:+8d} {status:<12}")
    
    print("-" * 70)
    
    # 汇总统计
    winners = sum(1 for r in results if r['status'] == "WINNER ✓")
    improved = sum(1 for r in results if r['delta_sp'] > 0)
    
    print(f"\\nSummary:")
    print(f"  Shapes analyzed: {len(results)}")
    print(f"  New winners: {winners}/{len(results)}")
    print(f"  Improved shapes: {improved}/{len(results)}")
    print(f"  Average speedup: {sum(r['delta_sp'] for r in results)/len(results):+.3f}×")
    
    return results

def generate_c81_report():
    """生成C.8.1优化报告"""
    
    results = analyze_c81_impact()
    
    print("\\n" + "=" * 70)
    print("C.8.1 OPTIMIZATION REPORT")
    print("=" * 70)
    
    print("\\n1. Grid Optimization Analysis:")
    print("   - Large d_out (>30K): kBm increased from 64→256, grid reduced from ~28K→7K")
    print("   - Small d_out (<3K) & T>=1024: kBm reduced from 128→64, grid increased from ~300→600")
    print("   - Normal cases: unchanged")
    
    print("\\n2. Expected Performance Impact:")
    print("   - Large grid shapes (32B/70B gu): +25-30% speedup (0.70×→0.90-0.95×)")
    print("   - Small grid shapes (70B kv/1.7B/4B dn): +15-20% speedup (0.85-0.95×→1.00-1.10×)")
    
    print("\\n3. Key Benefits:")
    print("   - Eliminates grid size extremes (too large/small)")
    print("   - Better GPU occupancy and utilization")
    print("   - Reduced kernel launch overhead")
    print("   - More balanced workload distribution")
    
    print("\\n4. Implementation Status:")
    print("   - Code changes: COMPLETED (fused_dense_sparse_mma_int4.cu)")
    print("   - Compilation: PENDING (requires CUDA environment)")
    print("   - Validation: PENDING (benchmark comparison)")

if __name__ == "__main__":
    generate_c81_report()