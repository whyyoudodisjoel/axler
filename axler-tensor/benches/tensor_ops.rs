use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use axler_tensor::Tensor;

fn benchmark_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for size in &[50, 100] {
        let sz = *size;

        // axler benchmark
        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            let data = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();
            let t = Tensor::from_slice(&data);
            let t_reshaped = t.reshape(&[sz, sz]);

            b.iter(|| {
                let result = &t_reshaped + &t_reshaped;
                let realized = result.realize();
                black_box(&realized);
            });
        });
    }

    group.finish();
}

fn benchmark_sub(c: &mut Criterion) {
    let mut group = c.benchmark_group("sub");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for size in &[50, 100] {
        let sz = *size;

        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            let data1 = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();
            let data2 = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();

            let t1 = Tensor::from_slice(&data1);
            let t1_reshaped = t1.reshape(&[sz, sz]);
            let t2 = Tensor::from_slice(&data2);
            let t2_reshaped = t2.reshape(&[sz, sz]);

            b.iter(|| {
                let result = &t1_reshaped - &t2_reshaped;
                let realized = result.realize();
                black_box(&realized);
            });
        });
    }

    group.finish();
}

fn benchmark_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for size in &[50, 100] {
        let sz = *size;

        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            let data1 = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();
            let data2 = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();

            let t1 = Tensor::from_slice(&data1);
            let t1_reshaped = t1.reshape(&[sz, sz]);
            let t2 = Tensor::from_slice(&data2);
            let t2_reshaped = t2.reshape(&[sz, sz]);

            b.iter(|| {
                let result = &t1_reshaped * &t2_reshaped;
                let realized = result.realize();
                black_box(&realized);
            });
        });
    }

    group.finish();
}

fn benchmark_div(c: &mut Criterion) {
    let mut group = c.benchmark_group("div");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for size in &[50, 100] {
        let sz = *size;

        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            let data1 = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();
            let data2 = rand::random_iter::<f32>()
                .take(sz * sz)
                .map(|x| x + 0.1)
                .collect::<Vec<_>>();

            let t1 = Tensor::from_slice(&data1);
            let t1_reshaped = t1.reshape(&[sz, sz]);
            let t2 = Tensor::from_slice(&data2);
            let t2_reshaped = t2.reshape(&[sz, sz]);

            b.iter(|| {
                let result = &t1_reshaped / &t2_reshaped;
                let realized = result.realize();
                black_box(&realized);
            });
        });
    }

    group.finish();
}

fn benchmark_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for size in &[50, 100] {
        let sz = *size;

        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            let data = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();

            let t = Tensor::from_slice(&data);
            let t_reshaped = t.reshape(&[sz, sz]);

            b.iter(|| {
                let result = t_reshaped.sum(None);
                let realized = result.realize();
                black_box(&realized);
            });
        });
    }

    group.finish();
}

fn benchmark_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("max");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for size in &[50, 100] {
        let sz = *size;

        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            let data = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();

            let t = Tensor::from_slice(&data);
            let t_reshaped = t.reshape(&[sz, sz]);

            b.iter(|| {
                let result = t_reshaped.max(None);
                let realized = result.realize();
                black_box(&realized);
            });
        });
    }

    group.finish();
}

fn benchmark_min(c: &mut Criterion) {
    let mut group = c.benchmark_group("min");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for size in &[50, 100] {
        let sz = *size;

        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            let data = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();

            let t = Tensor::from_slice(&data);
            let t_reshaped = t.reshape(&[sz, sz]);

            b.iter(|| {
                let result = t_reshaped.min(None);
                let realized = result.realize();
                black_box(&realized);
            });
        });
    }

    group.finish();
}

fn benchmark_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for size in &[50, 100] {
        let sz = *size;

        group.bench_with_input(BenchmarkId::new("axler", sz), &sz, |b, &sz| {
            let data = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();

            let t = Tensor::from_slice(&data);
            let t_reshaped = t.reshape(&[sz, sz]);

            b.iter(|| {
                let result = t_reshaped.mean(None);
                let realized = result.realize();
                black_box(&realized);
            });
        });
    }

    group.finish();
}

fn benchmark_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for size in &[50, 100] {
        let sz = *size;

        // Complex fused operation: ((a + b) * c) - d
        group.bench_with_input(BenchmarkId::new("axler_fused", sz), &sz, |b, &sz| {
            let data_a = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();
            let data_b = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();
            let data_c = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();
            let data_d = rand::random_iter::<f32>().take(sz * sz).collect::<Vec<_>>();

            let a = Tensor::from_slice(&data_a);
            let a_reshaped = a.reshape(&[sz, sz]);
            let b_tensor = Tensor::from_slice(&data_b);
            let b_reshaped = b_tensor.reshape(&[sz, sz]);
            let c = Tensor::from_slice(&data_c);
            let c_reshaped = c.reshape(&[sz, sz]);
            let d = Tensor::from_slice(&data_d);
            let d_reshaped = d.reshape(&[sz, sz]);

            b.iter(|| {
                let add_result = &a_reshaped + &b_reshaped;
                let mul_result = &add_result * &c_reshaped;
                let final_result = &mul_result - &d_reshaped;
                let realized = final_result.realize();
                black_box(&realized);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_add,
    benchmark_sub,
    benchmark_mul,
    benchmark_div,
    benchmark_sum,
    benchmark_max,
    benchmark_min,
    benchmark_mean,
    benchmark_fusion
);

criterion_main!(benches);
