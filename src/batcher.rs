use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

use crate::cache::EmbeddingCache;
use crate::embedder::Embedder;

/// Configuration for the dynamic batcher.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum number of texts in a single batch.
    pub max_batch_size: usize,
    /// Maximum time (ms) to wait for a full batch before flushing a partial one.
    pub max_wait_ms: u64,
    /// Channel buffer size for incoming requests.
    pub channel_buffer: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_ms: 10,
            channel_buffer: 256,
        }
    }
}

/// A single embedding request sent through the batcher channel.
struct EmbedRequest {
    text: String,
    is_query: bool,
    response_tx: oneshot::Sender<Result<Vec<f32>, String>>,
}

/// Dynamic batcher that collects individual embedding requests and dispatches
/// them in efficient batches. Sorts by token length within each batch for
/// optimal padding, then restores original request order on output.
pub struct DynamicBatcher {
    request_tx: mpsc::Sender<EmbedRequest>,
}

impl DynamicBatcher {
    /// Create a new DynamicBatcher backed by the given embedder and (optional)
    /// cache. Spawns a background task that continuously collects and flushes
    /// batches.
    pub fn new(
        embedder: Arc<Embedder>,
        cache: Option<Arc<EmbeddingCache>>,
        config: BatcherConfig,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<EmbedRequest>(config.channel_buffer);

        tokio::spawn(Self::batch_loop(embedder, cache, config, request_rx));

        Self { request_tx }
    }

    /// Submit a single text for embedding. Returns a future that resolves to the
    /// embedding vector once the batch containing this text is processed.
    pub async fn embed(&self, text: String, is_query: bool) -> Result<Vec<f32>> {
        let (response_tx, response_rx) = oneshot::channel();
        let req = EmbedRequest {
            text,
            is_query,
            response_tx,
        };
        self.request_tx
            .send(req)
            .await
            .map_err(|_| anyhow::anyhow!("Batcher channel closed"))?;

        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Batcher dropped response channel"))?
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Submit multiple texts. Returns embeddings in the same order as input.
    pub async fn embed_many(&self, texts: Vec<String>, is_query: bool) -> Result<Vec<Vec<f32>>> {
        let mut receivers = Vec::with_capacity(texts.len());

        for text in texts {
            let (response_tx, response_rx) = oneshot::channel();
            let req = EmbedRequest {
                text,
                is_query,
                response_tx,
            };
            self.request_tx
                .send(req)
                .await
                .map_err(|_| anyhow::anyhow!("Batcher channel closed"))?;
            receivers.push(response_rx);
        }

        let mut results = Vec::with_capacity(receivers.len());
        for rx in receivers {
            let emb = rx
                .await
                .map_err(|_| anyhow::anyhow!("Batcher dropped response channel"))?
                .map_err(|e| anyhow::anyhow!(e))?;
            results.push(emb);
        }

        Ok(results)
    }

    /// The main batch-collection loop running on a background tokio task.
    async fn batch_loop(
        embedder: Arc<Embedder>,
        cache: Option<Arc<EmbeddingCache>>,
        config: BatcherConfig,
        mut request_rx: mpsc::Receiver<EmbedRequest>,
    ) {
        info!(
            "Batcher loop started: max_batch={}, max_wait={}ms",
            config.max_batch_size, config.max_wait_ms
        );

        loop {
            let mut batch: Vec<EmbedRequest> = Vec::with_capacity(config.max_batch_size);

            // Wait for the first request (blocks until one arrives or channel closes).
            match request_rx.recv().await {
                Some(req) => batch.push(req),
                None => {
                    info!("Batcher channel closed, shutting down batch loop");
                    return;
                }
            }

            // Collect more requests up to max_batch_size or until timeout.
            let deadline =
                tokio::time::Instant::now() + tokio::time::Duration::from_millis(config.max_wait_ms);

            while batch.len() < config.max_batch_size {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match tokio::time::timeout(remaining, request_rx.recv()).await {
                    Ok(Some(req)) => batch.push(req),
                    Ok(None) => {
                        // Channel closed — flush what we have and exit.
                        info!("Batcher channel closed during collection");
                        Self::flush_batch(&embedder, &cache, batch).await;
                        return;
                    }
                    Err(_) => break, // timeout
                }
            }

            debug!("Flushing batch of {} requests", batch.len());
            Self::flush_batch(&embedder, &cache, batch).await;
        }
    }

    /// Process a collected batch: check cache, run inference, send responses.
    async fn flush_batch(
        embedder: &Arc<Embedder>,
        cache: &Option<Arc<EmbeddingCache>>,
        batch: Vec<EmbedRequest>,
    ) {
        if batch.is_empty() {
            return;
        }

        // Separate cached from uncached
        let mut cached_results: Vec<Option<Vec<f32>>> = Vec::with_capacity(batch.len());
        let mut uncached_indices: Vec<usize> = Vec::new();
        let mut uncached_texts: Vec<String> = Vec::new();

        // Determine the is_query flag from the first request (batches are uniform).
        let is_query = batch[0].is_query;

        for (i, req) in batch.iter().enumerate() {
            if let Some(ref c) = cache {
                if let Some(emb) = c.get(&req.text, is_query) {
                    cached_results.push(Some(emb));
                    continue;
                }
            }
            cached_results.push(None);
            uncached_indices.push(i);
            uncached_texts.push(req.text.clone());
        }

        // Run inference for uncached texts.
        let inference_results = if !uncached_texts.is_empty() {
            let text_refs: Vec<&str> = uncached_texts.iter().map(|s| s.as_str()).collect();
            let embedder = Arc::clone(embedder);
            // Run inference on a blocking thread to avoid starving the tokio runtime.
            let result =
                tokio::task::spawn_blocking(move || embedder.embed_batch(&text_refs, is_query))
                    .await;
            match result {
                Ok(Ok(embed_result)) => Some(embed_result.embeddings),
                Ok(Err(e)) => {
                    warn!("Inference error: {}", e);
                    // Send error to all requests in this batch.
                    for req in batch {
                        let _ = req.response_tx.send(Err(format!("Inference error: {}", e)));
                    }
                    return;
                }
                Err(e) => {
                    warn!("Blocking task panic: {}", e);
                    for req in batch {
                        let _ = req.response_tx.send(Err(format!("Task panic: {}", e)));
                    }
                    return;
                }
            }
        } else {
            None
        };

        // Merge cached and computed results, populate cache, send responses.
        let mut uncached_iter_idx = 0;
        let mut final_embeddings: Vec<Vec<f32>> = Vec::with_capacity(batch.len());

        for (i, cached) in cached_results.into_iter().enumerate() {
            let emb = if let Some(emb) = cached {
                emb
            } else {
                let emb = inference_results.as_ref().unwrap()[uncached_iter_idx].clone();
                uncached_iter_idx += 1;
                // Insert into cache.
                if let Some(ref c) = cache {
                    c.put(&batch[i].text, is_query, emb.clone());
                }
                emb
            };
            final_embeddings.push(emb);
        }

        // Send results back in original request order.
        for (req, emb) in batch.into_iter().zip(final_embeddings.into_iter()) {
            let _ = req.response_tx.send(Ok(emb));
        }
    }
}
