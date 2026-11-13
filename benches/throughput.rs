//! Throughput benchmark for embed-service.
//!
//! Runs `sessions` concurrent client threads per configuration, so the session
//! pool is actually exercised (sequential calls would only ever use one session
//! and make the sessions x threads comparison meaningless).

use embed_service::embedder::{Embedder, EmbedderConfig};
use hdrhistogram::Histogram;
use std::time::Instant;

/// Raw texts: the embedder adds the "query: " prefix itself.
const TEST_TEXTS: &[&str] = &[
    "I want to order pizza",
    "मुझे पिज़्ज़ा ऑर्डर करना है",
    "mujhe pizza order karna hai",
    "What is the weather today?",
    "आज मौसम कैसा है?",
    "play some music",
    "set an alarm for 7am",
    "book a cab to airport",
];

const BATCH_SIZES: &[usize] = &[1, 8, 16, 64];

/// (num_sessions, intra_threads_per_session)
const SESSION_THREAD_CONFIGS: &[(usize, usize)] = &[(1, 8), (2, 4), (4, 2), (8, 1)];

const WARMUP_ITERATIONS: usize = 3;
const TIMED_ITERATIONS: usize = 20;

fn bench_config(sessions: usize, threads: usize) {
    let embedder = Embedder::new(EmbedderConfig {
        num_sessions: sessions,
        intra_threads: threads,
        ..Default::default()
    })
    .expect("Failed to create embedder");

    println!(
        "\n### {}s x {}t (total {} threads, {} concurrent clients)\n",
        sessions,
        threads,
        sessions * threads,
        sessions
    );
    println!("| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (texts/s) |");
    println!("|------:|---------:|---------:|---------:|---------------------:|");

    for &batch_size in BATCH_SIZES {
        let batch: Vec<&str> = TEST_TEXTS.iter().cycle().take(batch_size).copied().collect();

        for _ in 0..WARMUP_ITERATIONS {
            let _ = embedder.embed_batch(&batch, true);
        }

        let start_all = Instant::now();
        let lats: Vec<Vec<u64>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..sessions)
                .map(|_| {
                    let (e, b) = (&embedder, &batch);
                    s.spawn(move || {
                        (0..TIMED_ITERATIONS)
                            .map(|_| {
                                let t = Instant::now();
                                e.embed_batch(b, true).expect("embed failed");
                                t.elapsed().as_micros() as u64
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let secs = start_all.elapsed().as_secs_f64();

        let mut hist = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).unwrap();
        let mut total = 0usize;
        for us in lats.into_iter().flatten() {
            hist.record(us.max(1)).ok();
            total += batch_size;
        }

        println!(
            "| {:>5} | {:>8.2} | {:>8.2} | {:>8.2} | {:>20.1} |",
            batch_size,
            hist.value_at_percentile(50.0) as f64 / 1000.0,
            hist.value_at_percentile(95.0) as f64 / 1000.0,
            hist.value_at_percentile(99.0) as f64 / 1000.0,
            total as f64 / secs
        );
    }
}

fn main() {
    println!("# embed-service Throughput Benchmark\n");
    println!(
        "Iterations/client: {} (warmup: {}), Batch sizes: {:?}",
        TIMED_ITERATIONS, WARMUP_ITERATIONS, BATCH_SIZES
    );
    for &(sessions, threads) in SESSION_THREAD_CONFIGS {
        bench_config(sessions, threads);
    }
    println!("\n_Benchmark complete._");
}
