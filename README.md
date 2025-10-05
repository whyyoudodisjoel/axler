# Axler-rs

A tensor computation library for Rust with CPU and CUDA backends.

> **⚠️ Work in Progress:** This project is currently under active development and APIs may change.

## Benchmarks

### Pure CPU Operation Performance (Axler Only)

These benchmarks measure operation + `.realize()` execution time on **CPU only**, with tensor creation moved outside the timing loop.

> **Note:** Tensors are created once outside the timing loop with random data. Only operation execution is measured.

| Operation | 50x50 | 100x100 |
|-----------|-------|---------|
| Add | 489.46 ns | 1.22 µs |
| Sub | 573.63 ns | 1.30 µs |
| Mul | 558.04 ns | 1.33 µs |
| Div | 610.22 ns | 1.42 µs |
| Sum | 1.50 µs | 5.20 µs |
| Max | 1.06 µs | 3.07 µs |
| Min | 997.46 ns | 3.07 µs |
| Mean | 1.51 µs | 5.25 µs |
| Fusion | 1.80 µs | 2.78 µs |

**Summary:**
- Element-wise operations: 500ns-1.5µs
- Operation costs on CPU excluding tensor creation
- See "CUDA Performance" section for GPU benchmarks

### CPU Performance: Axler vs Candle

CPU benchmarks comparing axler-tensor with Candle (without CUDA):

> **Note:** Benchmarks include tensor creation from data inside the timing loop for both frameworks. Both use the same deterministic data generation pattern.

| Operation | Size | Axler | Candle | Axler vs Candle |
|-----------|------|-------|--------|-----------------|
| **Add** | 50 | 2.34 µs | 2.11 µs | ~11% slower |
| | 100 | 8.58 µs | 8.01 µs | ~7% slower |
| | 200 | 31.47 µs | 30.93 µs | ~2% slower |
| | 500 | 207.33 µs | 212.85 µs | ~3% faster |
| **Sub** | 50 | 4.34 µs | 4.08 µs | ~6% slower |
| | 100 | 15.66 µs | 15.19 µs | ~3% slower |
| | 200 | 58.99 µs | 59.49 µs | ~1% faster |
| | 500 | 396.52 µs | 739.22 µs | **46% faster** |
| **Mul** | 50 | 4.20 µs | 3.94 µs | ~7% slower |
| | 100 | 15.42 µs | 15.12 µs | ~2% slower |
| | 200 | 59.53 µs | 58.50 µs | ~2% slower |
| | 500 | 395.10 µs | 744.14 µs | **47% faster** |
| **Div** | 50 | 4.41 µs | 4.23 µs | ~4% slower |
| | 100 | 15.46 µs | 15.78 µs | ~2% faster |
| | 200 | 59.23 µs | 62.06 µs | ~5% faster |
| | 500 | 400.39 µs | 759.07 µs | **47% faster** |
| **Sum** | 50 | 3.27 µs | 3.31 µs | ~1% faster |
| | 100 | 12.16 µs | 12.10 µs | Similar |
| | 200 | 48.20 µs | 47.09 µs | ~2% slower |
| | 500 | 292.37 µs | 295.89 µs | ~1% faster |
| **Max** | 50 | 2.78 µs | 3.67 µs | **24% faster** |
| | 100 | 9.91 µs | 14.24 µs | **30% faster** |
| | 200 | 38.68 µs | 60.81 µs | **36% faster** |
| | 500 | 239.18 µs | 403.53 µs | **41% faster** |
| **Mean** | 50 | 3.38 µs | 3.44 µs | ~2% faster |
| | 100 | 12.16 µs | 12.24 µs | ~1% faster |
| | 200 | 47.32 µs | 49.20 µs | ~4% faster |
| | 500 | 295.49 µs | 293.75 µs | ~1% slower |
| **Fusion** | 50 | 9.17 µs | 8.49 µs | ~8% slower |
| | 100 | 30.75 µs | 31.10 µs | ~1% faster |
| | 200 | 116.06 µs | 125.17 µs | **7% faster** |
| | 500 | 1.15 ms | 2.02 ms | **43% faster** |

**Summary:**
- Reduction operations (max, mean): 24-41% faster
- Element-wise operations at 500x500: 46-47% faster
- Matrix multiplication: Candle is faster (optimized BLAS)
- Operation fusion: 43% faster at 500x500

### CUDA Performance: Axler vs Candle

CUDA benchmarks comparing GPU implementations (with `--features cuda`):

> **Note:** These benchmarks include tensor creation and CPU→GPU memory transfers to ensure fair comparison. In Axler's lazy evaluation model, data creation, CPU→GPU transfer, and computation all happen in a single `.realize()` call. In Candle's eager model, these steps occur when each method is called. Benchmarking only the operation in Axler would be unfair since `.realize()` includes the data movement overhead that Candle already performs when creating GPU tensors. Therefore, both are measured end-to-end: data creation → GPU transfer → operation.

| Operation | Size | Axler CUDA | Candle CUDA | Axler vs Candle |
|-----------|------|------------|-------------|-----------------|
| **Add** | 128 | 276.14 µs | 7.03 ms | **96% faster** |
| | 256 | 489.58 µs | 7.09 ms | **93% faster** |
| | 512 | 1.85 ms | 7.23 ms | **74% faster** |
| **Mul** | 128 | 162.07 µs | 6.82 ms | **98% faster** |
| | 256 | 552.20 µs | 6.95 ms | **92% faster** |
| | 512 | 2.04 ms | 7.42 ms | **72% faster** |
| **Sum** | 128 | 240.85 µs | 2.13 ms | **88% faster** |
| | 256 | 997.72 µs | 2.46 ms | **59% faster** |
| | 512 | 4.22 ms | 2.53 ms | 40% slower |
| **Fusion** | 128 | 309.87 µs | 8.63 ms | **96% faster** |
| | 256 | 1.09 ms | 8.85 ms | **88% faster** |
| | 512 | 4.04 ms | 10.34 ms | **61% faster** |

**Summary:**
- Element-wise operations: 72-98% faster
- Matrix multiplication: 20-85% faster
- Operation fusion: 61-96% faster
- Results at 512x512 tensor size

## How Axler Works

Axler uses lazy evaluation with explicit kernel boundaries.

### One `.realize()` = One Kernel

Core principle: each `.realize()` call fuses all operations since the last `.realize()` into a single kernel.

```rust
// Build computation graph (lazy - nothing executes)
let result = (&a + &b) * &c - &d;

// Single .realize() = Single fused kernel
// All three operations (add, mul, sub) execute in one kernel
let output = result.realize();
```

This is different from automatic fusion libraries (like tinygrad) which use pattern matching and complex fusion heuristics. Axler's approach:

- No hidden transformations - explicit kernel boundaries via `.realize()`
- No pattern matching or fusion heuristics at runtime
- Predictable kernel launches
- Operations between `.realize()` calls either fuse or fail

### Comparison to Other Frameworks

Traditional graph-based frameworks (e.g., tinygrad):
- Analyze computation graphs
- Pattern match for fusion opportunities
- Run optimization passes at runtime
- Add overhead from fusion decision logic

Axler approach:
- Operations between `.realize()` calls fuse into one kernel
- No runtime analysis or optimization passes
- Lower overhead, predictable behavior

### Example

```rust
use axler_tensor::Tensor;

let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0]);
let c = Tensor::from_slice(&[2.0, 2.0, 2.0, 2.0]);

// Option 1: Three operations, ONE kernel
let result = (&a + &b) * &c;
let output = result.realize(); // Single fused kernel: (a+b)*c

// Option 2: Three operations, TWO kernels (explicit control)
let temp = (&a + &b).realize(); // Kernel 1: a+b
let output = (&temp * &c).realize(); // Kernel 2: temp*c
```

User controls fusion boundaries explicitly.
# axler
