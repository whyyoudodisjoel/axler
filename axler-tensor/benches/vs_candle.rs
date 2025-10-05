use candle_core::{Device, Tensor as CandleTensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use axler_tensor::Tensor as TinyTensor;

#[cfg(feature = "cuda")]
use axler_uop::DeviceType;

fn benchmark_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    group.sample_size(100);

    for size in &[50, 100, 200, 500] {
        let sz = *size;

        // axler benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let t = TinyTensor::from_slice(&data);
                let t_reshaped = t.reshape(&[sz, sz]);
                let result = &t_reshaped + &t_reshaped;
                let realized = result.realize();
                black_box(&realized);
            });
        });

        // Candle benchmark
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let t = CandleTensor::from_vec(data, (sz, sz), &Device::Cpu).unwrap();
                let result = (&t + &t).unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

fn benchmark_sub(c: &mut Criterion) {
    let mut group = c.benchmark_group("sub");
    group.sample_size(100);

    for size in &[50, 100, 200, 500] {
        let sz = *size;

        // axler benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data1 = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let data2 = (0..sz * sz).map(|i| (i + 1) as f32).collect::<Vec<_>>();

                let t1 = TinyTensor::from_slice(&data1);
                let t1_reshaped = t1.reshape(&[sz, sz]);
                let t2 = TinyTensor::from_slice(&data2);
                let t2_reshaped = t2.reshape(&[sz, sz]);
                let result = &t1_reshaped - &t2_reshaped;
                let realized = result.realize();
                black_box(&realized);
            });
        });

        // Candle benchmark
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data1 = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let data2 = (0..sz * sz).map(|i| (i + 1) as f32).collect::<Vec<_>>();

                let t1 = CandleTensor::from_vec(data1, (sz, sz), &Device::Cpu).unwrap();
                let t2 = CandleTensor::from_vec(data2, (sz, sz), &Device::Cpu).unwrap();

                let result = (&t1 - &t2).unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

fn benchmark_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");
    group.sample_size(100);

    for size in &[50, 100, 200, 500] {
        let sz = *size;

        // axler benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data1 = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let data2 = (0..sz * sz).map(|i| (i + 1) as f32).collect::<Vec<_>>();

                let t1 = TinyTensor::from_slice(&data1);
                let t1_reshaped = t1.reshape(&[sz, sz]);
                let t2 = TinyTensor::from_slice(&data2);
                let t2_reshaped = t2.reshape(&[sz, sz]);
                let result = &t1_reshaped * &t2_reshaped;
                let realized = result.realize();
                black_box(&realized);
            });
        });

        // Candle benchmark
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data1 = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let data2 = (0..sz * sz).map(|i| (i + 1) as f32).collect::<Vec<_>>();

                let t1 = CandleTensor::from_vec(data1, (sz, sz), &Device::Cpu).unwrap();
                let t2 = CandleTensor::from_vec(data2, (sz, sz), &Device::Cpu).unwrap();

                let result = (&t1 * &t2).unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

fn benchmark_div(c: &mut Criterion) {
    let mut group = c.benchmark_group("div");
    group.sample_size(100);

    for size in &[50, 100, 200, 500] {
        let sz = *size;

        // axler benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data1 = (0..sz * sz).map(|i| i as f32 + 1.0).collect::<Vec<_>>();
                let data2 = (0..sz * sz)
                    .map(|i| (i + 1) as f32 + 0.1)
                    .collect::<Vec<_>>();

                let t1 = TinyTensor::from_slice(&data1);
                let t1_reshaped = t1.reshape(&[sz, sz]);
                let t2 = TinyTensor::from_slice(&data2);
                let t2_reshaped = t2.reshape(&[sz, sz]);
                let result = &t1_reshaped / &t2_reshaped;
                let realized = result.realize();
                black_box(&realized);
            });
        });

        // Candle benchmark
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data1 = (0..sz * sz).map(|i| i as f32 + 1.0).collect::<Vec<_>>();
                let data2 = (0..sz * sz)
                    .map(|i| (i + 1) as f32 + 0.1)
                    .collect::<Vec<_>>();

                let t1 = CandleTensor::from_vec(data1, (sz, sz), &Device::Cpu).unwrap();
                let t2 = CandleTensor::from_vec(data2, (sz, sz), &Device::Cpu).unwrap();

                let result = (&t1 / &t2).unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

fn benchmark_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");
    group.sample_size(100);

    for size in &[50, 100, 200, 500] {
        let sz = *size;

        // axler benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let t = TinyTensor::from_slice(&data);
                let t_reshaped = t.reshape(&[sz, sz]);
                let result = t_reshaped.sum(None);
                let realized = result.realize();
                black_box(&realized);
            });
        });

        // Candle benchmark
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let t = CandleTensor::from_vec(data, (sz, sz), &Device::Cpu).unwrap();

                let result = t.sum_all().unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

fn benchmark_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("max");
    group.sample_size(100);

    for size in &[50, 100, 200, 500] {
        let sz = *size;

        // axler benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let t = TinyTensor::from_slice(&data);
                let t_reshaped = t.reshape(&[sz, sz]);
                let result = t_reshaped.max(None);
                let realized = result.realize();
                black_box(&realized);
            });
        });

        // Candle benchmark
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let t = CandleTensor::from_vec(data, (sz, sz), &Device::Cpu).unwrap();

                let result = t.max_keepdim(0).unwrap().max_keepdim(1).unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

fn benchmark_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean");
    group.sample_size(100);

    for size in &[50, 100, 200, 500] {
        let sz = *size;

        // axler benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let t = TinyTensor::from_slice(&data);
                let t_reshaped = t.reshape(&[sz, sz]);
                let result = t_reshaped.mean(None);
                let realized = result.realize();
                black_box(&realized);
            });
        });

        // Candle benchmark
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let t = CandleTensor::from_vec(data, (sz, sz), &Device::Cpu).unwrap();

                let result = t.mean_all().unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

fn benchmark_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion");
    group.sample_size(100);

    for size in &[50, 100, 200, 500] {
        let sz = *size;

        // axler benchmark - complex fused operation: ((a + b) * c) - d
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data_a = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let data_b = (0..sz * sz).map(|i| (i + 1) as f32).collect::<Vec<_>>();
                let data_c = (0..sz * sz).map(|i| (i + 2) as f32).collect::<Vec<_>>();
                let data_d = (0..sz * sz).map(|i| (i + 3) as f32).collect::<Vec<_>>();

                let a = TinyTensor::from_slice(&data_a);
                let a_reshaped = a.reshape(&[sz, sz]);
                let b_tensor = TinyTensor::from_slice(&data_b);
                let b_reshaped = b_tensor.reshape(&[sz, sz]);
                let c = TinyTensor::from_slice(&data_c);
                let c_reshaped = c.reshape(&[sz, sz]);
                let d = TinyTensor::from_slice(&data_d);
                let d_reshaped = d.reshape(&[sz, sz]);
                let add_result = &a_reshaped + &b_reshaped;
                let mul_result = &add_result * &c_reshaped;
                let final_result = &mul_result - &d_reshaped;
                let realized = final_result.realize();
                black_box(&realized);
            });
        });

        // Candle benchmark - same operation
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data_a = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let data_b = (0..sz * sz).map(|i| (i + 1) as f32).collect::<Vec<_>>();
                let data_c = (0..sz * sz).map(|i| (i + 2) as f32).collect::<Vec<_>>();
                let data_d = (0..sz * sz).map(|i| (i + 3) as f32).collect::<Vec<_>>();

                let a = CandleTensor::from_vec(data_a, (sz, sz), &Device::Cpu).unwrap();
                let b_tensor = CandleTensor::from_vec(data_b, (sz, sz), &Device::Cpu).unwrap();
                let c = CandleTensor::from_vec(data_c, (sz, sz), &Device::Cpu).unwrap();
                let d = CandleTensor::from_vec(data_d, (sz, sz), &Device::Cpu).unwrap();

                let add_result = (&a + &b_tensor).unwrap();
                let mul_result = (&add_result * &c).unwrap();
                let final_result = (&mul_result - &d).unwrap();
                black_box(&final_result);
            });
        });
    }

    group.finish();
}

// ============================================================================
// CUDA Benchmarks
// ============================================================================

#[cfg(feature = "cuda")]
fn benchmark_add_cuda(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_cuda");
    group.sample_size(10); // Reduced from 100 to avoid GPU OOM

    for size in &[128, 256, 512] {
        // Removed 1024 to reduce memory usage
        let sz = *size;

        // axler CUDA benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let t = TinyTensor::from_slice(&data);
                let t_reshaped = t.reshape(&[sz, sz]);
                let t_cuda = t_reshaped.to_device(DeviceType::CUDA);
                let result = &t_cuda + &t_cuda;
                let realized = result.realize();
                black_box(&realized);
            });
        });

        // Candle CUDA benchmark
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let device = Device::new_cuda(0).unwrap();
                let t = CandleTensor::from_vec(data, (sz, sz), &device).unwrap();

                let result = (&t + &t).unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_mul_cuda(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_cuda");
    group.sample_size(10);

    for size in &[128, 256, 512] {
        let sz = *size;

        // axler CUDA benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data1 = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let data2 = (0..sz * sz).map(|i| (i + 1) as f32).collect::<Vec<_>>();

                let t1 = TinyTensor::from_slice(&data1);
                let t1_reshaped = t1.reshape(&[sz, sz]);
                let t1_cuda = t1_reshaped.to_device(DeviceType::CUDA);

                let t2 = TinyTensor::from_slice(&data2);
                let t2_reshaped = t2.reshape(&[sz, sz]);
                let t2_cuda = t2_reshaped.to_device(DeviceType::CUDA);
                let result = &t1_cuda * &t2_cuda;
                let realized = result.realize();
                black_box(&realized);
            });
        });

        // Candle CUDA benchmark
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data1 = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let data2 = (0..sz * sz).map(|i| (i + 1) as f32).collect::<Vec<_>>();

                let device = Device::new_cuda(0).unwrap();
                let t1 = CandleTensor::from_vec(data1, (sz, sz), &device).unwrap();
                let t2 = CandleTensor::from_vec(data2, (sz, sz), &device).unwrap();

                let result = (&t1 * &t2).unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_sum_cuda(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_cuda");
    group.sample_size(10);

    for size in &[128, 256, 512] {
        let sz = *size;

        // axler CUDA benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let t = TinyTensor::from_slice(&data);
                let t_reshaped = t.reshape(&[sz, sz]);
                let t_cuda = t_reshaped.to_device(DeviceType::CUDA);
                let result = t_cuda.sum(None);
                let realized = result.realize();
                black_box(&realized);
            });
        });

        // Candle CUDA benchmark
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let device = Device::new_cuda(0).unwrap();
                let t = CandleTensor::from_vec(data, (sz, sz), &device).unwrap();
                let result = t.sum_all().unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_fusion_cuda(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_cuda");
    group.sample_size(10);

    for size in &[128, 256, 512] {
        let sz = *size;

        // axler CUDA benchmark - complex fused operation: ((a + b) * c) - d
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data_a = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let data_b = (0..sz * sz).map(|i| (i + 1) as f32).collect::<Vec<_>>();
                let data_c = (0..sz * sz).map(|i| (i + 2) as f32).collect::<Vec<_>>();
                let data_d = (0..sz * sz).map(|i| (i + 3) as f32).collect::<Vec<_>>();

                let a_cpu = TinyTensor::from_slice(&data_a);
                let a_reshaped = a_cpu.reshape(&[sz, sz]);
                let a = a_reshaped.to_device(DeviceType::CUDA);

                let b_cpu = TinyTensor::from_slice(&data_b);
                let b_reshaped = b_cpu.reshape(&[sz, sz]);
                let b_tensor = b_reshaped.to_device(DeviceType::CUDA);

                let c_cpu = TinyTensor::from_slice(&data_c);
                let c_reshaped = c_cpu.reshape(&[sz, sz]);
                let c = c_reshaped.to_device(DeviceType::CUDA);

                let d_cpu = TinyTensor::from_slice(&data_d);
                let d_reshaped = d_cpu.reshape(&[sz, sz]);
                let d = d_reshaped.to_device(DeviceType::CUDA);
                let add_result = &a + &b_tensor;
                let mul_result = &add_result * &c;
                let final_result = &mul_result - &d;
                let realized = final_result.realize();
                black_box(&realized);
            });
        });

        // Candle CUDA benchmark - same operation
        group.bench_with_input(BenchmarkId::new("candle", sz), &sz, |b, &sz| {
            b.iter(|| {
                let data_a = (0..sz * sz).map(|i| i as f32).collect::<Vec<_>>();
                let data_b = (0..sz * sz).map(|i| (i + 1) as f32).collect::<Vec<_>>();
                let data_c = (0..sz * sz).map(|i| (i + 2) as f32).collect::<Vec<_>>();
                let data_d = (0..sz * sz).map(|i| (i + 3) as f32).collect::<Vec<_>>();

                let device = Device::new_cuda(0).unwrap();
                let a = CandleTensor::from_vec(data_a, (sz, sz), &device).unwrap();
                let b_tensor = CandleTensor::from_vec(data_b, (sz, sz), &device).unwrap();
                let c = CandleTensor::from_vec(data_c, (sz, sz), &device).unwrap();
                let d = CandleTensor::from_vec(data_d, (sz, sz), &device).unwrap();
                let add_result = (&a + &b_tensor).unwrap();
                let mul_result = (&add_result * &c).unwrap();
                let final_result = (&mul_result - &d).unwrap();
                black_box(&final_result);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

// CPU benchmarks group
criterion_group!(
    cpu_benches,
    benchmark_add,
    benchmark_sub,
    benchmark_mul,
    benchmark_div,
    benchmark_sum,
    benchmark_max,
    benchmark_mean,
    benchmark_fusion,
);

// CUDA benchmarks group (only compiled with cuda feature)
#[cfg(feature = "cuda")]
criterion_group!(
    cuda_benches,
    benchmark_add_cuda,
    benchmark_mul_cuda,
    benchmark_sum_cuda,
    benchmark_fusion_cuda,
);

// Main entry point - conditionally includes CUDA benchmarks
#[cfg(feature = "cuda")]
criterion_main!(cuda_benches);

#[cfg(not(feature = "cuda"))]
criterion_main!(cpu_benches);
