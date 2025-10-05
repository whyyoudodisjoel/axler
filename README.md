# Axler-rs

A tensor computation library for Rust with CPU and CUDA backends.

> **⚠️ Work in Progress:** This project is currently under active development and APIs may change.

## Benchmarks

### Pure CPU Operation Performance (Axler Only)

These benchmarks measure operation + `.realize()` execution time on **CPU only**, with tensor creation moved outside the timing loop.

> **Note:** Tensors are created once outside the timing loop with random data. Only operation execution is measured.

| Operation | 50x50 | 100x100 |
|-----------|-------|---------|
| Add | 287.25 ns | 963.19 ns |
| Sub | 368.64 ns | 1.15 µs |
| Mul | 381.18 ns | 1.10 µs |
| Div | 409.97 ns | 1.32 µs |
| Sum | 1.46 µs | 5.13 µs |
| Max | 904.77 ns | 2.95 µs |
| Min | 899.81 ns | 2.93 µs |
| Mean | 1.41 µs | 5.14 µs |
| Fusion | 738.07 ns | 1.70 µs |

**Summary:**
- Element-wise operations: 287ns-1.5µs
- Operation costs on CPU excluding tensor creation
- See "CUDA Performance" section for GPU benchmarks

### CPU Performance: Axler vs Candle

CPU benchmarks comparing axler-tensor with Candle (without CUDA):

> **Note:** Benchmarks include tensor creation from data inside the timing loop for both frameworks. Both use the same deterministic data generation pattern.

| Operation | Size | Axler | Candle | Axler vs Candle |
|-----------|------|-------|--------|-----------------|
| **Add** | 50 | 2.11 µs | 2.14 µs | ~1% faster |
| | 100 | 8.05 µs | 8.12 µs | ~1% faster |
| | 200 | 32.77 µs | 31.17 µs | ~5% slower |
| | 500 | 207.34 µs | 209.99 µs | ~1% faster |
| **Sub** | 50 | 3.89 µs | 3.96 µs | ~2% faster |
| | 100 | 15.14 µs | 15.25 µs | ~1% faster |
| | 200 | 58.73 µs | 59.09 µs | ~1% faster |
| | 500 | 420.82 µs | 756.06 µs | **44% faster** |
| **Mul** | 50 | 3.90 µs | 3.98 µs | ~2% faster |
| | 100 | 15.85 µs | 16.41 µs | ~3% faster |
| | 200 | 58.35 µs | 59.41 µs | ~2% faster |
| | 500 | 400.33 µs | 757.41 µs | **47% faster** |
| **Div** | 50 | 3.98 µs | 4.34 µs | ~8% faster |
| | 100 | 15.26 µs | 15.88 µs | ~4% faster |
| | 200 | 59.12 µs | 65.98 µs | **10% faster** |
| | 500 | 399.85 µs | 832.84 µs | **52% faster** |
| **Sum** | 50 | 3.19 µs | 3.35 µs | ~5% faster |
| | 100 | 12.07 µs | 12.14 µs | ~1% faster |
| | 200 | 47.22 µs | 47.36 µs | Similar |
| | 500 | 293.88 µs | 293.76 µs | Similar |
| **Max** | 50 | 2.66 µs | 3.69 µs | **28% faster** |
| | 100 | 10.06 µs | 14.40 µs | **30% faster** |
| | 200 | 38.35 µs | 60.96 µs | **37% faster** |
| | 500 | 239.92 µs | 415.97 µs | **42% faster** |
| **Mean** | 50 | 3.20 µs | 3.53 µs | **9% faster** |
| | 100 | 12.06 µs | 12.30 µs | ~2% faster |
| | 200 | 47.50 µs | 47.66 µs | Similar |
| | 500 | 293.81 µs | 297.54 µs | ~1% faster |
| **Fusion** | 50 | 7.73 µs | 8.52 µs | **9% faster** |
| | 100 | 29.58 µs | 31.38 µs | **6% faster** |
| | 200 | 119.32 µs | 127.76 µs | **7% faster** |
| | 500 | 1.20 ms | 2.09 ms | **43% faster** |

**Summary:**
- Reduction operations (max, mean): 28-42% faster
- Element-wise operations at 500x500: 44-52% faster
- Operation fusion: 6-43% faster across all sizes

### CUDA Performance: Axler vs Candle

CUDA benchmarks comparing GPU implementations (with `--features cuda`):

> **Note:** These benchmarks include tensor creation and CPU→GPU memory transfers to ensure fair comparison. In Axler's lazy evaluation model, data creation, CPU→GPU transfer, and computation all happen in a single `.realize()` call. In Candle's eager model, these steps occur when each method is called. Benchmarking only the operation in Axler would be unfair since `.realize()` includes the data movement overhead that Candle already performs when creating GPU tensors. Therefore, both are measured end-to-end: data creation → GPU transfer → operation.

| Operation | Size | Axler CUDA | Candle CUDA | Axler vs Candle |
|-----------|------|------------|-------------|-----------------|
| **Add** | 128 | 138.82 µs | 6.95 ms | **98% faster** |
| | 256 | 309.08 µs | 8.51 ms | **96% faster** |
| | 512 | 1.07 ms | 9.06 ms | **88% faster** |
| **Mul** | 128 | 182.29 µs | 9.12 ms | **98% faster** |
| | 256 | 586.53 µs | 8.99 ms | **93% faster** |
| | 512 | 2.21 ms | 9.87 ms | **78% faster** |
| **Sum** | 128 | 265.89 µs | 2.25 ms | **88% faster** |
| | 256 | 1.28 ms | 2.34 ms | **45% faster** |
| | 512 | 4.75 ms | 2.66 ms | 44% slower |
| **Fusion** | 128 | 341.91 µs | 8.86 ms | **96% faster** |
| | 256 | 1.14 ms | 9.59 ms | **88% faster** |
| | 512 | 4.30 ms | 11.23 ms | **62% faster** |

**Summary:**
- Element-wise operations: 78-98% faster
- Operation fusion: 62-96% faster
- Reduction operations (sum): 45-88% faster (slower at 512x512 due to GPU overhead)

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
