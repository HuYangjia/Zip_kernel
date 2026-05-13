# Phase 4 C.8: Loser Shapes Optimization Plan

## Problem Analysis

Based on r69 bench results, these 5 shapes remain losers (sp < 0.9×):

| Shape | Model | Proj | T | d_in→d_out | r69 sp | Problem |
|---|---|---|---|---|---|---|
| 32B gu | Qwen2.5-32B | gate_up | 2048 | 5120→55296 | 0.716× | Grid too large (864×32=27,648 CTAs) |
| 70B gu | LLaMA3-70B | gate_up | 2048 | 8192→57344 | 0.699× | Grid too large (896×32=28,672 CTAs) |
| 70B kv | LLaMA3-70B | kv | 1024 | 8192→2048 | 0.794× | Grid too small (16×16=256 CTAs) |
| 1.7B dn | Qwen3-1.7B | down | 1024 | 6144→2048 | 0.886× | Grid too small (16×16=256 CTAs) |
| 4B dn | Qwen3-4B | down | 1024 | 9728→2560 | 0.733× | Grid too small (20×16=320 CTAs) |

## Root Cause Analysis

### Large Grid Problems (32B gu, 70B gu)
- **kBm=64** with huge d_out (55K-57K) creates massive grid_M (864-896)
- **kBn=64** with T=2048 creates grid_N=32
- Total CTAs: ~28K, causing significant launch overhead
- Each CTA processes only 64 rows × 64 columns = 4K elements
- **Inefficient work distribution**: too many small CTAs

### Small Grid Problems (70B kv, 1.7B dn, 4B dn)  
- **kBm=128** with small d_out (2K-2.5K) creates tiny grid_M (16-20)
- **kBn=64** with T=1024 creates grid_N=16
- Total CTAs: 256-320, underutilizing GPU SMs
- Each CTA processes 128 rows × 64 columns = 8K elements
- **SM starvation**: not enough concurrent work to hide latency

## Proposed Solutions

### Solution C.8.1: Adaptive kBm Selection
```cpp
// Current rule: kBm = 128 if d_out <= 4096 else 64
// Problem: creates extreme grid sizes for very large/small d_out

// New rule: kBm = clamp(d_out / target_cta_per_sm, 64, 128)
// Where target_cta_per_sm = 4-8 for good occupancy

int target_grid_m = (d_out + 1023) / 1024;  // Aim for ~1K CTAs
int kBm_new = std::max(64, std::min(128, d_out / target_grid_m));
```

### Solution C.8.2: Wave-Aware kBn Selection
```cpp
// Current rule: kBn = 64 if T>=128 else 32 if T>=32 else 8
// Problem: ignores grid_M size when choosing kBn

// New rule: choose kBn to achieve target wave occupancy
int total_ctas_target = 128 * 4;  // 4 waves on RTX 4090
int kBn_new = std::max(8, std::min(64, T / (total_ctas_target / grid_M)));
```

### Solution C.8.3: Split-K for Large Grid Shapes
```cpp
// For shapes with grid_M × grid_N > 4096, use split-K to reduce grid size
if (grid_M * grid_N > 4096) {
    split_k = 2;  // Halves grid_M while doubling per-CTA work
    kBm = 128;   // Use larger tile size for better efficiency
}
```

### Solution C.8.4: kBn Demotion for Small Grid Shapes
```cpp
// For shapes with grid_M × grid_N < 512, use smaller kBn to increase CTAs
if (grid_M * grid_N < 512) {
    kBn = 32;  // Doubles grid_N, better SM utilization
    // Optionally also reduce kBm if still under-utilized
    if (grid_M < 32) kBm = 64;
}
```

## Expected Impact

### Large Grid Shapes (32B gu, 70B gu)
- Current: kBm=64, kBn=64, grid=28K CTAs, sp=0.70×
- With C.8.1 + C.8.3: kBm=128, kBn=64, split_k=2, grid=7K CTAs
- Expected: +25-30% speedup (sp ~0.90-0.95×)

### Small Grid Shapes (70B kv, 1.7B dn, 4B dn)
- Current: kBm=128, kBn=64, grid=256-320 CTAs, sp=0.73-0.89×
- With C.8.4: kBm=64, kBn=32, grid=512-640 CTAs
- Expected: +15-20% speedup (sp ~0.95-1.05×)

## Implementation Plan

### Phase 1: Adaptive kBm (C.8.1)
1. Modify `launch()` function to use wave-aware kBm selection
2. Add shape-specific overrides for extreme cases
3. Benchmark on target shapes

### Phase 2: Wave-Aware kBn (C.8.2)  
1. Update kBn selection to consider grid_M size
2. Implement fallback logic for edge cases
3. Validate with full bench sweep

### Phase 3: Split-K Optimization (C.8.3)
1. Add split-K gate for large-grid shapes
2. Optimize reduce kernel for large outputs
3. Test on 32B/70B gate_up shapes

### Phase 4: kBn Demotion (C.8.4)
1. Implement small-grid detection and kBn adjustment
2. Add kBm reduction for severely under-utilized cases
3. Validate on kv/dn shapes

## Risk Assessment

### Low Risk
- C.8.1 and C.8.4 only affect specific shape ranges
- Changes are conservative (clamp to existing kBm/kBn values)
- Easy to revert if regressions occur

### Medium Risk  
- C.8.2 requires careful wave occupancy calculation
- May interact with existing split-K logic

### High Risk
- C.8.3 (split-K for large grids) requires kernel modifications
- Reduce kernel must handle very large d_out efficiently

## Success Criteria

- All 5 loser shapes achieve sp >= 0.95×
- No regressions on existing winner shapes
- Median speedup improvement >= 0.05×
- Wins count increases by 5 shapes

## Next Steps

1. Implement and test C.8.1 (adaptive kBm)
2. Benchmark on 32B/70B gu shapes
3. Implement remaining optimizations incrementally
4. Full validation sweep before shipping