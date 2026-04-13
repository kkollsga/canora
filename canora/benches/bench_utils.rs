use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;

use canora::types::Float;
use canora::util::utils;

fn bench_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame");
    for n_samples in [4_096, 22_050, 110_250, 441_000] {
        let signal = Array1::from_shape_fn(n_samples, |i| (i as Float * 0.1).sin());
        group.bench_with_input(
            BenchmarkId::new("n_fft=2048,hop=512", n_samples),
            &signal,
            |b, sig| {
                b.iter(|| utils::frame(sig.view(), 2048, 512).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");
    for size in [1_000, 10_000, 100_000] {
        let data = Array1::from_shape_fn(size, |i| (i as Float * 0.1).sin());
        group.bench_with_input(BenchmarkId::new("l2", size), &data, |b, d| {
            b.iter(|| utils::normalize(d.view(), "l2", None).unwrap());
        });
    }
    group.finish();
}

fn bench_pad_center(c: &mut Criterion) {
    let mut group = c.benchmark_group("pad_center");
    for size in [512, 2048, 8192] {
        let data = Array1::from_shape_fn(size, |i| i as Float);
        group.bench_with_input(BenchmarkId::new("2x", size), &data, |b, d| {
            b.iter(|| utils::pad_center(d.view(), size * 2).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_frame, bench_normalize, bench_pad_center);
criterion_main!(benches);
