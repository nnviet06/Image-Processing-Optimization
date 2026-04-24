# IMAGE PROCESSING OPTIMIZATION

High-performance image processing library demonstrating CPU optimization (SIMD, threading, cache-awareness) and GPU acceleration (CUDA) with intelligent scheduler for heterogeneous computing trade-offs.

**Focus:** Understanding when to use CPU vs GPU through layered optimization, benchmarking, and empirical validation.

## Project Overview

Three-phase optimization of Gaussian blur:
1. **Phase 1: CPU Optimization** — SIMD, threading, cache techniques
2. **Phase 2: GPU Acceleration** — CUDA kernels, batch processing
3. **Phase 3: Smart Scheduler** — Automated CPU/GPU selection

---

## Phase 1: CPU Optimization

Gaussian blur optimization from naive scalar to vectorized + threaded:

| Layer | Technique | Time (4K) | Speedup | Status |
|---|---|---|---|---|
| **1.1** | Naive scalar | 2856.76ms | 1.0x | ✓ |
| **1.2** | SIMD AVX2 | 497.36ms | 5.7x | ✓ |
| **1.3** | Multi-threading | 352.24ms | 8.1x | ✓ |
| **1.4** | Cache-aware tiling | 2135.79ms | 0.75x | ✗ Broken |

**Key insight:** Threading dominates on large images (4K), but SIMD wins on smaller images (3K).

---

## Phase 2: GPU Acceleration (CUDA)

Three GPU optimization layers with batch processing:

| Layer | Technique | Single Image (4K) | Batch 10x (4K) | Speedup |
|---|---|---|---|---|
| **2.1** | Basic CUDA kernel | 20.6ms | — | 138.7x |
| **2.2** | Pinned memory + streams | ~40ms | — | 71.4x |
| **2.3** | Batch pipeline | — | 29.3ms/img | 97.4x |

**Key insight:** GPU dominates even single image (20ms vs 352ms CPU). Batch processing improves throughput but single-image latency already GPU-optimal due to PCIe amortization.

---

## Phase 3: Smart Scheduler (In Progress)

Intelligent CPU/GPU selection based on workload:
- Single image? Use GPU (20ms beats 352ms CPU)
- Batch of 10+? GPU batch pipeline (34 img/sec throughput)
- Decision logic: Profile image size + batch size, choose optimal path

---

## Benchmarks

Full results: `data/benchmarks/BENCHMARKS.md`

Test images:
- **test1.png**: 3840×2160 (4K grayscale)
- **test2.png**: 3000×2000 (color, converted to grayscale)

---

## Build & Run

```bash
# Build
cd build
cmake -G Ninja ..
ninja

# Phase 1 CPU
./gaussian_naive ../data/test1.png
./gaussian_simd ../data/test1.png
./gaussian_threaded ../data/test1.png
./gaussian_cache_aware ../data/test1.png

# Phase 2 GPU (requires CUDA)
./gaussian_gpu_kernel ../data/test1.png
./gaussian_gpu_optimized ../data/test1.png
./gaussian_gpu_batch ../data/test1.png
```

---

## Known Issues

- **#4**: Threading slower than SIMD on smaller images (overhead > work)
- **#5**: Cache-aware tiling slower than naive (tiling overhead destroying benefit)
- Phase 3 scheduler not yet implemented

---