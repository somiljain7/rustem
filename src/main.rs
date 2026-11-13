use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use hdrhistogram::Histogram;
use std::io::{self, BufRead};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

use embed_service::batcher::BatcherConfig;
use embed_service::cache::EmbeddingCache;
use embed_service::embedder::{cosine_similarity, Embedder, EmbedderConfig};
use embed_service::server::{self, ServerConfig};

#[derive(Parser)]
#[command(name = "embed-service", about = "CPU-optimized multilingual text embedding service")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to ONNX model file
    #[arg(long, default_value = "models/model_quantized.onnx")]
    model: String,

    /// Path to tokenizer.json
    #[arg(long, default_value = "models/tokenizer.json")]
    tokenizer: String,

    /// Number of parallel ONNX sessions
    #[arg(long, default_value_t = 4)]
    sessions: usize,

    /// Intra-op threads per session
    #[arg(long, default_value_t = 2)]
    threads: usize,

    /// Maximum sequence length
    #[arg(long, default_value_t = 128)]
    max_length: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Embed texts from stdin or --texts argument, output JSON
    Embed {
        /// Text to embed; repeat for several. If omitted, reads stdin (one per line).
        #[arg(long = "text")]
        texts: Vec<String>,

        /// Treat texts as queries ("query: " prefix); pass `--is-query false` for passages
        #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
        is_query: bool,
    },

    /// Run benchmark with configurable parameters
    Bench {
        /// Batch sizes to benchmark (comma-separated)
        #[arg(long, default_value = "1,8,16,64")]
        batch_sizes: String,

        /// Number of iterations per batch size
        #[arg(long, default_value_t = 50)]
        iterations: usize,

        /// Number of warmup iterations
        #[arg(long, default_value_t = 5)]
        warmup: usize,

        /// Concurrent client threads (use >= --sessions to exercise the session pool)
        #[arg(long, default_value_t = 1)]
        clients: usize,
    },

    /// Start HTTP server
    Serve {
        /// Bind address
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,

        /// Port
        #[arg(long, default_value_t = 8080)]
        port: u16,

        /// Cache capacity (0 to disable)
        #[arg(long, default_value_t = 10000)]
        cache_capacity: usize,

        /// Max batch size for the dynamic batcher
        #[arg(long, default_value_t = 32)]
        max_batch_size: usize,

        /// Max wait time (ms) for batching
        #[arg(long, default_value_t = 10)]
        max_wait_ms: u64,
    },

    /// Run fidelity evaluation (cosine similarity against reference embeddings)
    Eval {
        /// Path to reference JSON file: {"texts": [...], "embeddings": [[...], ...], "is_query": bool}
        #[arg(long)]
        reference: String,
    },
}

fn build_embedder_config(cli: &Cli) -> EmbedderConfig {
    EmbedderConfig {
        model_path: cli.model.clone(),
        tokenizer_path: cli.tokenizer.clone(),
        num_sessions: cli.sessions,
        intra_threads: cli.threads,
        max_length: cli.max_length,
        ..Default::default()
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Embed { texts, is_query } => cmd_embed(&cli, texts, *is_query),
        Commands::Bench {
            batch_sizes,
            iterations,
            warmup,
            clients,
        } => cmd_bench(&cli, batch_sizes, *iterations, *warmup, *clients),
        Commands::Serve {
            bind,
            port,
            cache_capacity,
            max_batch_size,
            max_wait_ms,
        } => cmd_serve(
            &cli,
            bind,
            *port,
            *cache_capacity,
            *max_batch_size,
            *max_wait_ms,
        ),
        Commands::Eval { reference } => cmd_eval(&cli, reference),
    }
}

// ── embed ─────────────────────────────────────────────────────────────────

fn cmd_embed(cli: &Cli, texts_arg: &[String], is_query: bool) -> Result<()> {
    let config = build_embedder_config(cli);
    let embedder = Embedder::new(config)?;

    let texts: Vec<String> = if !texts_arg.is_empty() {
        texts_arg.to_vec()
    } else {
        info!("Reading texts from stdin (one per line, Ctrl+D / Ctrl+Z to finish)...");
        let stdin = io::stdin();
        stdin
            .lock()
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .collect()
    };

    if texts.is_empty() {
        eprintln!("No texts provided.");
        return Ok(());
    }

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let result = embedder.embed_batch(&text_refs, is_query)?;

    let output = serde_json::json!({
        "embeddings": result.embeddings,
        "dim": embedder.dim(),
        "count": result.embeddings.len(),
        "tokenize_us": result.tokenize_us,
        "inference_us": result.inference_us,
        "total_us": result.total_us,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

// ── bench ─────────────────────────────────────────────────────────────────

fn cmd_bench(
    cli: &Cli,
    batch_sizes_str: &str,
    iterations: usize,
    warmup: usize,
    clients: usize,
) -> Result<()> {
    let config = build_embedder_config(cli);
    let embedder = Arc::new(Embedder::new(config)?);
    let clients = clients.max(1);

    // Raw texts: the embedder adds the "query: " prefix itself.
    let test_texts: Vec<&str> = vec![
        "I want to order pizza",
        "मुझे पिज़्ज़ा ऑर्डर करना है",
        "mujhe pizza order karna hai",
        "What is the weather today?",
        "आज मौसम कैसा है?",
        "play some music",
        "set an alarm for 7am",
        "book a cab to airport",
    ];

    let batch_sizes: Vec<usize> = batch_sizes_str
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("Invalid batch sizes")?;

    // Fixed seed for reproducibility: use deterministic text cycling.
    eprintln!(
        "\n📊 Benchmark: {} iterations + {} warmup per batch size",
        iterations, warmup
    );
    eprintln!(
        "   Model: {}, Sessions: {}, Threads/session: {}",
        cli.model, cli.sessions, cli.threads
    );
    eprintln!();

    // Header
    println!("| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (texts/s) |");
    println!("|------:|---------:|---------:|---------:|---------------------:|");

    for &batch_size in &batch_sizes {
        // Build batch by cycling through test_texts deterministically.
        let batch: Vec<&str> = test_texts
            .iter()
            .cycle()
            .take(batch_size)
            .copied()
            .collect();

        // Warmup
        for _ in 0..warmup {
            let _ = embedder.embed_batch(&batch, true)?;
        }

        // Timed runs: `clients` concurrent threads, each doing `iterations` calls.
        let mut hist =
            Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).expect("histogram creation");
        let overall_start = Instant::now();

        let per_client: Vec<Result<Vec<u64>>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..clients)
                .map(|_| {
                    let embedder = &embedder;
                    let batch = &batch;
                    s.spawn(move || -> Result<Vec<u64>> {
                        let mut lat = Vec::with_capacity(iterations);
                        for _ in 0..iterations {
                            let start = Instant::now();
                            embedder.embed_batch(batch, true)?;
                            lat.push(start.elapsed().as_micros() as u64);
                        }
                        Ok(lat)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("client panicked"))))
                .collect()
        });

        let overall_elapsed_s = overall_start.elapsed().as_secs_f64();
        let mut total_texts = 0usize;
        for lat in per_client {
            for us in lat? {
                hist.record(us.max(1)).ok();
                total_texts += batch_size;
            }
        }
        let throughput = total_texts as f64 / overall_elapsed_s;

        let p50 = hist.value_at_percentile(50.0) as f64 / 1000.0;
        let p95 = hist.value_at_percentile(95.0) as f64 / 1000.0;
        let p99 = hist.value_at_percentile(99.0) as f64 / 1000.0;

        println!(
            "| {:>5} | {:>8.2} | {:>8.2} | {:>8.2} | {:>20.1} |",
            batch_size, p50, p95, p99, throughput
        );
    }

    println!();
    Ok(())
}

// ── serve ─────────────────────────────────────────────────────────────────

fn cmd_serve(
    cli: &Cli,
    bind: &str,
    port: u16,
    cache_capacity: usize,
    max_batch_size: usize,
    max_wait_ms: u64,
) -> Result<()> {
    let config = build_embedder_config(cli);
    let embedder = Arc::new(Embedder::new(config)?);

    let server_config = ServerConfig {
        bind: bind.to_string(),
        port,
        cache_capacity,
        batcher: BatcherConfig {
            max_batch_size,
            max_wait_ms,
            ..Default::default()
        },
    };

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(server::run_server(embedder, server_config))
}

// ── eval ──────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct ReferenceData {
    texts: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    is_query: bool,
}

fn cmd_eval(cli: &Cli, reference_path: &str) -> Result<()> {
    let config = build_embedder_config(cli);
    let embedder = Embedder::new(config)?;

    let ref_json = std::fs::read_to_string(reference_path)
        .with_context(|| format!("Failed to read reference file: {}", reference_path))?;
    let ref_data: ReferenceData = serde_json::from_str(&ref_json)?;

    if ref_data.texts.len() != ref_data.embeddings.len() {
        anyhow::bail!(
            "Reference file mismatch: {} texts vs {} embeddings",
            ref_data.texts.len(),
            ref_data.embeddings.len()
        );
    }

    info!("Evaluating {} texts against reference", ref_data.texts.len());

    let text_refs: Vec<&str> = ref_data.texts.iter().map(|s| s.as_str()).collect();
    let result = embedder.embed_batch(&text_refs, ref_data.is_query)?;

    let mut total_sim = 0.0f64;
    let mut min_sim = f64::MAX;
    let mut max_sim = f64::MIN;

    println!("\n| # | Text (truncated) | Cosine Sim |");
    println!("|--:|:-----------------|:-----------|");

    for (i, (computed, reference)) in result
        .embeddings
        .iter()
        .zip(ref_data.embeddings.iter())
        .enumerate()
    {
        let sim = cosine_similarity(computed, reference) as f64;
        total_sim += sim;
        if sim < min_sim {
            min_sim = sim;
        }
        if sim > max_sim {
            max_sim = sim;
        }

        let text_preview: String = ref_data.texts[i].chars().take(50).collect();
        println!("| {} | {} | {:.6} |", i + 1, text_preview, sim);
    }

    anyhow::ensure!(!ref_data.texts.is_empty(), "Reference file is empty");
    let n = ref_data.texts.len() as f64;
    let avg_sim = total_sim / n;

    println!();
    println!("**Mean cosine similarity:** {:.6}", avg_sim);
    println!("**Min:** {:.6}  |  **Max:** {:.6}", min_sim, max_sim);

    if avg_sim >= 0.999 {
        println!("✅ Fidelity check PASSED (mean sim ≥ 0.999)");
    } else if avg_sim >= 0.99 {
        println!("⚠️  Fidelity check WARNING (mean sim ≥ 0.99 but < 0.999)");
    } else {
        println!("❌ Fidelity check FAILED (mean sim < 0.99)");
    }

    Ok(())
}
