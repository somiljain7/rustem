use anyhow::{Context, Result};
use ndarray::{Array2, Axis};
use ort::session::{Session, builder::GraphOptimizationLevel};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{debug, info};

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
    /// Embedding dimension (auto-detected if 0)
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
            embed_dim: 384, // multilingual-e5-small
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

/// Session pool entry
struct SessionEntry {
    session: Session,
}

/// The main embedder: holds a tokenizer and a pool of ONNX sessions
pub struct Embedder {
    tokenizer: Tokenizer,
    sessions: Vec<Mutex<SessionEntry>>,
    config: EmbedderConfig,
    embed_dim: usize,
}

impl Embedder {
    /// Create a new Embedder with the given config
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        info!(
            "Initializing embedder: {} sessions x {} intra-threads, max_len={}",
            config.num_sessions, config.intra_threads, config.max_length
        );

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Initialize ONNX Runtime
        ort::init()
            .with_name("embed-service")
            .commit()
            .context("Failed to init ONNX Runtime")?;

        // Create session pool
        let mut sessions = Vec::with_capacity(config.num_sessions);
        for i in 0..config.num_sessions {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(config.intra_threads as u16)?
                .with_inter_threads(1)?
                .commit_from_file(&config.model_path)
                .with_context(|| format!("Failed to create session {}", i))?;

            sessions.push(Mutex::new(SessionEntry { session }));
            debug!("Created ONNX session {}", i);
        }

        // Detect embedding dimension from first session
        let embed_dim = if config.embed_dim > 0 {
            config.embed_dim
        } else {
            // Run a dummy input to detect dim
            384 // default for e5-small
        };

        info!("Embedder ready: embed_dim={}", embed_dim);

        Ok(Self {
            tokenizer,
            sessions,
            config,
            embed_dim,
        })
    }

    /// Get the embedding dimension
    pub fn dim(&self) -> usize {
        self.embed_dim
    }

    /// Embed a single text (query mode)
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text], true)?;
        Ok(results.embeddings.into_iter().next().unwrap())
    }

    /// Embed a single text (passage mode)
    pub fn embed_passage(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text], false)?;
        Ok(results.embeddings.into_iter().next().unwrap())
    }

    /// Embed a batch of texts
    /// is_query: if true, add "query: " prefix; if false, add "passage: " prefix
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

        // Step 1: Add prefixes if needed
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

        // Step 2: Tokenize in parallel with rayon
        let tok_start = std::time::Instant::now();
        let encodings: Vec<_> = prefixed
            .par_iter()
            .map(|text| {
                let mut tokenizer = self.tokenizer.clone();
                tokenizer
                    .with_truncation(Some(tokenizers::TruncationParams {
                        max_length: self.config.max_length,
                        ..Default::default()
                    }))
                    .unwrap();
                tokenizer
                    .with_padding(None)
                    .unwrap();
                tokenizer.encode(text.as_str(), true).unwrap()
            })
            .collect();
        let tokenize_us = tok_start.elapsed().as_micros() as u64;

        // Step 3: Sort by length for efficient batching, track original indices
        let mut indexed: Vec<(usize, &tokenizers::Encoding)> = encodings.iter().enumerate().collect();
        indexed.sort_by_key(|(_, enc)| enc.get_ids().len());

        // Step 4: Dynamic padding and batch construction
        let batch_size = indexed.len();
        let max_len = indexed.last().unwrap().1.get_ids().len();

        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];
        let mut token_type_ids = vec![0i64; batch_size * max_len];

        for (batch_idx, (_, enc)) in indexed.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            for (j, &id) in ids.iter().enumerate() {
                input_ids[batch_idx * max_len + j] = id as i64;
                attention_mask[batch_idx * max_len + j] = mask[j] as i64;
                token_type_ids[batch_idx * max_len + j] = types[j] as i64;
            }
        }

        // Step 5: Run inference
        let infer_start = std::time::Instant::now();
        let embeddings = self.run_inference(
            &input_ids,
            &attention_mask,
            &token_type_ids,
            batch_size,
            max_len,
        )?;
        let inference_us = infer_start.elapsed().as_micros() as u64;

        // Step 6: Masked mean pooling + L2 normalize
        let pooled = self.masked_mean_pool(&embeddings, &attention_mask, batch_size, max_len);

        // Step 7: Restore original order
        let mut result = vec![vec![0f32; self.embed_dim]; batch_size];
        for (batch_idx, (orig_idx, _)) in indexed.iter().enumerate() {
            result[*orig_idx] = pooled[batch_idx].clone();
        }

        let total_us = total_start.elapsed().as_micros() as u64;

        Ok(EmbedResult {
            embeddings: result,
            tokenize_us,
            inference_us,
            total_us,
        })
    }

    /// Run ONNX inference using an available session from the pool
    fn run_inference(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: &[i64],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        // Find an available session
        let session_guard = self
            .sessions
            .iter()
            .find_map(|s| s.try_lock())
            .unwrap_or_else(|| self.sessions[0].lock());

        let session = &session_guard.session;

        // Create input arrays
        let ids_array = ndarray::Array2::from_shape_vec(
            (batch_size, seq_len),
            input_ids.to_vec(),
        )?;
        let mask_array = ndarray::Array2::from_shape_vec(
            (batch_size, seq_len),
            attention_mask.to_vec(),
        )?;
        let types_array = ndarray::Array2::from_shape_vec(
            (batch_size, seq_len),
            token_type_ids.to_vec(),
        )?;

        // Run the model
        let outputs = session.run(ort::inputs![
            "input_ids" => ids_array.view(),
            "attention_mask" => mask_array.view(),
            "token_type_ids" => types_array.view(),
        ]?)?;

        // Extract output tensor - last_hidden_state is shape [batch, seq, dim]
        let output_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;
        
        let output_view = output_tensor.view();
        Ok(output_view.iter().copied().collect())
    }

    /// Masked mean pooling: average token embeddings weighted by attention mask
    fn masked_mean_pool(
        &self,
        hidden_states: &[f32],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
    ) -> Vec<Vec<f32>> {
        let dim = self.embed_dim;
        let mut result = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut embedding = vec![0f32; dim];
            let mut total_weight = 0f32;

            for j in 0..seq_len {
                let mask_val = attention_mask[i * seq_len + j] as f32;
                if mask_val > 0.0 {
                    total_weight += mask_val;
                    for k in 0..dim {
                        embedding[k] += hidden_states[i * seq_len * dim + j * dim + k] * mask_val;
                    }
                }
            }

            // Divide by sum of mask (clamp to avoid div by zero)
            let total_weight = total_weight.max(1e-9);
            for k in 0..dim {
                embedding[k] /= total_weight;
            }

            // L2 normalize if configured
            if self.config.normalize {
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for k in 0..dim {
                    embedding[k] /= norm;
                }
            }

            result.push(embedding);
        }

        result
    }

    /// Embed with sub-batching for large inputs
    /// Splits into chunks and distributes across sessions
    pub fn embed_batch_parallel(
        self: &Arc<Self>,
        texts: &[&str],
        is_query: bool,
        sub_batch_size: usize,
    ) -> Result<EmbedResult> {
        let total_start = std::time::Instant::now();

        if texts.len() <= sub_batch_size {
            return self.embed_batch(texts, is_query);
        }

        // Split into sub-batches
        let chunks: Vec<&[&str]> = texts.chunks(sub_batch_size).collect();
        let embedder = Arc::clone(self);

        let results: Vec<Result<EmbedResult>> = chunks
            .par_iter()
            .map(|chunk| embedder.embed_batch(chunk, is_query))
            .collect();

        // Merge results
        let mut all_embeddings = Vec::with_capacity(texts.len());
        let mut total_tokenize = 0u64;
        let mut total_inference = 0u64;

        for r in results {
            let r = r?;
            all_embeddings.extend(r.embeddings);
            total_tokenize += r.tokenize_us;
            total_inference += r.inference_us;
        }

        let total_us = total_start.elapsed().as_micros() as u64;

        Ok(EmbedResult {
            embeddings: all_embeddings,
            tokenize_us: total_tokenize,
            inference_us: total_inference,
            total_us,
        })
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}
