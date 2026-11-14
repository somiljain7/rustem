use anyhow::{Context, Result};
use tokenizers::{Tokenizer, TruncationParams};
use tracing::info;

use crate::pool::{PoolConfig, SessionPool};

/// Configuration for the Embedder
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Path to ONNX model file
    pub model_path: String,
    /// Path to tokenizer.json
    pub tokenizer_path: String,
    /// Number of parallel ONNX sessions
    pub num_sessions: usize,
    /// Intra-op threads per session
    pub intra_threads: usize,
    /// Maximum sequence length (tokens)
    pub max_length: usize,
    /// Expected embedding dimension. 0 = auto-detect from the model output.
    pub embed_dim: usize,
    /// Whether to add E5 prefixes
    pub use_e5_prefix: bool,
    /// Prefix for queries
    pub query_prefix: String,
    /// Prefix for passages/documents
    pub passage_prefix: String,
    /// Whether to normalize embeddings (L2)
    pub normalize: bool,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model_path: "models/model_quantized.onnx".to_string(),
            tokenizer_path: "models/tokenizer.json".to_string(),
            num_sessions: 4,
            intra_threads: 2,
            max_length: 128,
            embed_dim: 0, // auto-detect
            use_e5_prefix: true,
            query_prefix: "query: ".to_string(),
            passage_prefix: "passage: ".to_string(),
            normalize: true,
        }
    }
}

/// Result of embedding operation
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// Embeddings matrix [N x dim]
    pub embeddings: Vec<Vec<f32>>,
    /// Tokenization time in microseconds
    pub tokenize_us: u64,
    /// Inference time in microseconds
    pub inference_us: u64,
    /// Total time in microseconds
    pub total_us: u64,
}

/// The main embedder: a tokenizer plus a pool of ONNX sessions.
pub struct Embedder {
    tokenizer: Tokenizer,
    pool: SessionPool,
    config: EmbedderConfig,
    embed_dim: usize,
    pad_id: i64,
}

impl Embedder {
    /// Create a new Embedder with the given config
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        anyhow::ensure!(config.num_sessions > 0, "num_sessions must be > 0");
        anyhow::ensure!(config.max_length > 0, "max_length must be > 0");
        info!(
            "Initializing embedder: {} sessions x {} intra-threads, max_len={}",
            config.num_sessions, config.intra_threads, config.max_length
        );

        // Tokenizer: configure truncation ONCE (previously cloned per text).
        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_length,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("Failed to configure truncation: {}", e))?;
        tokenizer.with_padding(None);
        let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(0) as i64;

        ort::init()
            .with_name("embed-service")
            .commit()
            .context("Failed to init ONNX Runtime")?;

        let pool = SessionPool::new(&PoolConfig {
            model_path: config.model_path.clone(),
            num_sessions: config.num_sessions,
            intra_threads: config.intra_threads,
        })?;

        let mut embedder = Self {
            tokenizer,
            pool,
            embed_dim: config.embed_dim,
            config,
            pad_id,
        };

        // Detect the real embedding dimension from the model output shape.
        let probe = embedder.embed_batch(&["dim probe"], true)?;
        let detected = probe.embeddings[0].len();
        if embedder.config.embed_dim != 0 && embedder.config.embed_dim != detected {
            anyhow::bail!(
                "Configured embed_dim={} but model produces {}",
                embedder.config.embed_dim,
                detected
            );
        }
        embedder.embed_dim = detected;

        info!("Embedder ready: embed_dim={}", embedder.embed_dim);
        Ok(embedder)
    }

    /// Get the embedding dimension
    pub fn dim(&self) -> usize {
        self.embed_dim
    }

    /// Number of ONNX sessions (max useful inference concurrency).
    pub fn num_sessions(&self) -> usize {
        self.config.num_sessions
    }

    /// Embed a single text (query mode)
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_one(text, true)
    }

    /// Embed a single text (passage mode)
    pub fn embed_passage(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_one(text, false)
    }

    fn embed_one(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
        self.embed_batch(&[text], is_query)?
            .embeddings
            .into_iter()
            .next()
            .context("empty embedding result")
    }

    /// Embed a batch of texts, preserving input order.
    /// is_query: if true, add the query prefix; otherwise the passage prefix.
    pub fn embed_batch(&self, texts: &[&str], is_query: bool) -> Result<EmbedResult> {
        let total_start = std::time::Instant::now();

        if texts.is_empty() {
            return Ok(EmbedResult {
                embeddings: vec![],
                tokenize_us: 0,
                inference_us: 0,
                total_us: 0,
            });
        }

        // Step 1: prefixes
        let prefixed: Vec<String> = if self.config.use_e5_prefix {
            let prefix = if is_query {
                &self.config.query_prefix
            } else {
                &self.config.passage_prefix
            };
            texts.iter().map(|t| format!("{}{}", prefix, t)).collect()
        } else {
            texts.iter().map(|t| t.to_string()).collect()
        };

        // Step 2: batch tokenization (tokenizers parallelises internally); no panics.
        let tok_start = std::time::Instant::now();
        let encodings = self
            .tokenizer
            .encode_batch(prefixed, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let tokenize_us = tok_start.elapsed().as_micros() as u64;

        // Step 3: pad to the longest sequence in the batch.
        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .max(1);

        let mut input_ids = vec![self.pad_id; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];
        let mut token_type_ids = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let base = i * max_len;
            for (j, &id) in enc.get_ids().iter().enumerate() {
                input_ids[base + j] = id as i64;
            }
            for (j, &m) in enc.get_attention_mask().iter().enumerate() {
                attention_mask[base + j] = m as i64;
            }
            for (j, &t) in enc.get_type_ids().iter().enumerate() {
                token_type_ids[base + j] = t as i64;
            }
        }

        // Step 4: inference
        let infer_start = std::time::Instant::now();
        let (hidden, dim) =
            self.run_inference(input_ids, attention_mask.clone(), token_type_ids, batch_size, max_len)?;
        let inference_us = infer_start.elapsed().as_micros() as u64;

        // Step 5: masked mean pooling + L2 normalize
        let embeddings = self.masked_mean_pool(&hidden, &attention_mask, batch_size, max_len, dim);

        Ok(EmbedResult {
            embeddings,
            tokenize_us,
            inference_us,
            total_us: total_start.elapsed().as_micros() as u64,
        })
    }

    /// Run ONNX inference on any free session (blocks until one is available).
    /// Returns the flattened hidden states and the hidden size.
    fn run_inference(
        &self,
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
        token_type_ids: Vec<i64>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let guard = self.pool.get()?;
        let session = guard.session();

        let ids_array = ndarray::Array2::from_shape_vec((batch_size, seq_len), input_ids)?;
        let mask_array = ndarray::Array2::from_shape_vec((batch_size, seq_len), attention_mask)?;
        let types_array = ndarray::Array2::from_shape_vec((batch_size, seq_len), token_type_ids)?;

        let outputs = session.run(ort::inputs![
            "input_ids" => ids_array.view(),
            "attention_mask" => mask_array.view(),
            "token_type_ids" => types_array.view(),
        ]?)?;

        let output_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;
        let shape = output_tensor.shape().to_vec();
        anyhow::ensure!(
            shape.len() == 3 && shape[0] == batch_size && shape[1] == seq_len,
            "Unexpected output shape {:?}, expected [{}, {}, dim]",
            shape,
            batch_size,
            seq_len
        );
        let dim = shape[2];
        Ok((output_tensor.iter().copied().collect(), dim))
    }

    /// Masked mean pooling: average token embeddings weighted by attention mask
    fn masked_mean_pool(
        &self,
        hidden_states: &[f32],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
        dim: usize,
    ) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut embedding = vec![0f32; dim];
            let mut total_weight = 0f32;

            for j in 0..seq_len {
                let mask_val = attention_mask[i * seq_len + j] as f32;
                if mask_val > 0.0 {
                    total_weight += mask_val;
                    let off = (i * seq_len + j) * dim;
                    for (e, h) in embedding.iter_mut().zip(&hidden_states[off..off + dim]) {
                        *e += h * mask_val;
                    }
                }
            }

            let total_weight = total_weight.max(1e-9);
            for e in embedding.iter_mut() {
                *e /= total_weight;
            }

            if self.config.normalize {
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for e in embedding.iter_mut() {
                    *e /= norm;
                }
            }

            result.push(embedding);
        }

        result
    }

    /// Embed a large input by splitting into sub-batches that run concurrently
    /// across the session pool.
    pub fn embed_batch_parallel(
        &self,
        texts: &[&str],
        is_query: bool,
        sub_batch_size: usize,
    ) -> Result<EmbedResult> {
        let total_start = std::time::Instant::now();
        let sub_batch_size = sub_batch_size.max(1);

        if texts.len() <= sub_batch_size {
            return self.embed_batch(texts, is_query);
        }

        // Run in waves of at most `num_sessions` threads (one per ONNX session).
        let chunks: Vec<&[&str]> = texts.chunks(sub_batch_size).collect();
        let mut results: Vec<Result<EmbedResult>> = Vec::with_capacity(chunks.len());
        for wave in chunks.chunks(self.config.num_sessions.max(1)) {
            let wave_results: Vec<Result<EmbedResult>> = std::thread::scope(|s| {
                let handles: Vec<_> = wave
                    .iter()
                    .map(|chunk| s.spawn(move || self.embed_batch(chunk, is_query)))
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("worker panicked"))))
                    .collect()
            });
            results.extend(wave_results);
        }

        let mut all = Vec::with_capacity(texts.len());
        let (mut tok, mut inf) = (0u64, 0u64);
        for r in results {
            let r = r?;
            all.extend(r.embeddings);
            tok += r.tokenize_us;
            inf += r.inference_us;
        }

        Ok(EmbedResult {
            embeddings: all,
            tokenize_us: tok,
            inference_us: inf,
            total_us: total_start.elapsed().as_micros() as u64,
        })
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine_similarity: dimension mismatch");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}
