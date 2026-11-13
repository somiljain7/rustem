//! Throughput benchmark for embed-service.
//!
//! Uses manual timing (harness = false) to test various batch sizes and
//! session × thread configurations. Outputs a markdown table.

use embed_service::embedder::{Embedder, EmbedderConfig};
use hdrhistogram::Histogram;
use std::sync::Arc;
use std::time::Instant;

/// Test texts: multilingual mix for realistic benchmarking.
const TEST_TEXTS: &[&str] = &[
    "query: I want to order pizza",
    "query: मुझे पिज़्ज़ा ऑर्डर करना है",
    "query: mujhe pizza order karna hai",
    "query: What is the weather today?",
    "query: आज मौसम कैसा है?",
    "query: play some music",
    "query: set an alarm for 7am",
    "query: book a cab to airport",
];

/// Batch sizes to benchmark.
const BATCH_SIZES: &[usize] = &[1, 8, 16, 64];

/// Session × thread configurations to test.
/// (num_sessions, intra_threads_per_session)  — product ≤ 8 to fit i5-9300H.
const SESSION_THREAD_CONFIGS: &[(usize, usize)] = &[
    (1, 8),
    (2, 4),
    (4, 2),
    (8, 1),
];

const WARMUP_ITERATIONS: usize = 3;
const TIMED_ITERATIONS: usize = 20;

fn build_embedder(sessions: usize, threads: usize) -> Embedder {
    let config = EmbedderConfig {
        num_sessions: sessions,
        intra_threads: threads,
        ..Default::default()
    };
    Embedder::new(config).expect("Failed to create embedder")
}

fn bench_config(sessions: usize, threads: usize) {
    let embedder = Arc::new(build_embedder(sessions, threads));

    println!("\n### {}s × {}t (total {} threads)", sessions, threads, sessions * threads);
    println!();
    println!("| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (texts/s) |");
    println!("|------:|---------:|---------:|---------:|---------------------:|");

    for &batch_size in BATCH_SIZES {
        let batch: Vec<&str> = TEST_TEXTS
            .iter()
            .cycle()
            .take(batch_size)
            .copied()
            .collect();

        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let _ = embedder.embed_batch(&batch, true);
        }

        // Timed
        let mut hist =
            Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).expect("histogram creation");
        let mut total_texts = 0usize;
        let overall_start = Instant::now();

        for _ in 0..TIMED_ITERATIONS {
            let start = Instant::now();
            let _ = embedder.embed_batch(&batch, true);
            let elapsed_us = start.elapsed().as_micros() as u64;
            hist.record(elapsed_us).ok();
            total_texts += batch_size;
        }

        let overall_s = overall_start.elapsed().as_secs_f64();
        let throughput = total_texts as f64 / overall_s;

        let p50 = hist.value_at_percentile(50.0) as f64 / 1000.0;
        let p95 = hist.value_at_percentile(95.0) as f64 / 1000.0;
        let p99 = hist.value_at_percentile(99.0) as f64 / 1000.0;

        println!(
            "| {:>5} | {:>8.2} | {:>8.2} | {:>8.2} | {:>20.1} |",
            batch_size, p50, p95, p99, throughput
        );
    }
}

fn main() {
    // Initialize ONNX Runtime once before all benchmarks.
    ort::init()
        .with_name("bench-throughput")
        .commit()
        .expect("Failed to init ONNX Runtime");

    println!("# embed-service Throughput Benchmark");
    println!();
    println!(
        "Iterations: {} (warmup: {}), Test texts: {}, Batch sizes: {:?}",
        TIMED_ITERATIONS,
        WARMUP_ITERATIONS,
        TEST_TEXTS.len(),
        BATCH_SIZES
    );

    for &(sessions, threads) in SESSION_THREAD_CONFIGS {
        bench_config(sessions, threads);
    }

    println!("\n_Benchmark complete._");
}
