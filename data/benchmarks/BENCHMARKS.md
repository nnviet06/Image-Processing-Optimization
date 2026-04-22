# CPU Optimization Benchmarks

## Phase 1.1: Naive Gaussian Blur

### Test Image: test1.png (3840x2160 - 4K)
- **Time (avg 5 runs):** 2856.76 ms
- **Notes:** Baseline scalar implementation

### Test Image: test2.png (3000x2000)
- **Time (avg 5 runs):** 2035.74 ms
- **Notes:** Real image

---

## Phase 1.2: SIMD Gaussian (AVX2)

### Test Image: test1.png (3840x2160 - 4K)
- **Time (avg 5 runs):** 497.363 ms
- **Speedup:** 5.7x

---

## Phase 1.3: Threading Gaussian

### Test Image: test1.png (3840x2160 - 4K)
- **Time (avg 5 runs):** 352.238 ms
- **Speedup:** 8.1x

---

## Phase 1.4: Cache-aware Gaussian (BROKEN)

### Test Image: test1.png (3840x2160 - 4K)
- **Time (avg 5 runs):** 2135.79 ms
- **Notes:** Slower than naive - tiling overhead issue. Skip for now.

---