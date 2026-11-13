use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore};
use tracing::{debug, info, warn};

use crate::cache::EmbeddingCache;
use crate::embedder::Embedder;

/// Configuration for the dynamic batcher.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum number of texts in a single batch.
    pub max_batch_size: usize,
    /// Maximum time (ms) to wait for a fuller batch, ONLY when other batches are
    /// already in flight. An idle service flushes immediately (no latency floor).
    pub max_wait_ms: u64,
    /// Channel buffer size for incoming requests.
    pub channel_buffer: usize,
    /// Max batches running concurrently (should equal the number of ONNX sessions).
    pub concurrency: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_ms: 10,
            channel_buffer: 256,
            concurrency: 4,
        }
    }
}

/// A single embedding request sent through the batcher channel.
struct EmbedRequest {
    text: String,
    is_query: bool,
    response_tx: oneshot::Sender<Result<Vec<f32>, String>>,
}

/// Dynamic batcher: collects individual requests and dispatches them as batches.
///
/// * Batches run concurrently (bounded by `concurrency`), so all ONNX sessions are used.
/// * While every slot is busy, requests accumulate in the channel, so batching
///   emerges naturally under load without adding idle latency.
/// * Query and passage requests are split into separate inference calls.
/// * Duplicate texts inside a batch are embedded once.
pub struct DynamicBatcher {
    request_tx: mpsc::Sender<EmbedRequest>,
}

impl DynamicBatcher {
    pub fn new(
        embedder: Arc<Embedder>,
        cache: Option<Arc<EmbeddingCache>>,
        config: BatcherConfig,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<EmbedRequest>(config.channel_buffer.max(1));
        tokio::spawn(Self::batch_loop(embedder, cache, config, request_rx));
        Self { request_tx }
    }

    /// Submit a single text for embedding.
    pub async fn embed(&self, text: String, is_query: bool) -> Result<Vec<f32>> {
        let (response_tx, response_rx) = oneshot::channel();
        self.request_tx
            .send(EmbedRequest { text, is_query, response_tx })
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
            self.request_tx
                .send(EmbedRequest { text, is_query, response_tx })
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

    async fn batch_loop(
        embedder: Arc<Embedder>,
        cache: Option<Arc<EmbeddingCache>>,
        config: BatcherConfig,
        mut request_rx: mpsc::Receiver<EmbedRequest>,
    ) {
        let concurrency = config.concurrency.max(1);
        let max_batch = config.max_batch_size.max(1);
        let slots = Arc::new(Semaphore::new(concurrency));
        info!(
            "Batcher started: max_batch={}, max_wait={}ms, concurrency={}",
            max_batch, config.max_wait_ms, concurrency
        );

        loop {
            let first = match request_rx.recv().await {
                Some(r) => r,
                None => {
                    info!("Batcher channel closed, shutting down");
                    break;
                }
            };
            let mut batch = Vec::with_capacity(max_batch);
            batch.push(first);

            // Wait for a free inference slot. Requests pile up in the channel meanwhile.
            let permit = match slots.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => break,
            };

            // Drain whatever is already queued.
            while batch.len() < max_batch {
                match request_rx.try_recv() {
                    Ok(r) => batch.push(r),
                    Err(_) => break,
                }
            }

            // Only wait for more if other batches are running (we're under load anyway).
            let busy = slots.available_permits() < concurrency - 1;
            if busy && batch.len() < max_batch && config.max_wait_ms > 0 {
                let deadline = tokio::time::Instant::now()
                    + tokio::time::Duration::from_millis(config.max_wait_ms);
                while batch.len() < max_batch {
                    match tokio::time::timeout_at(deadline, request_rx.recv()).await {
                        Ok(Some(r)) => batch.push(r),
                        _ => break,
                    }
                }
            }

            debug!("Dispatching batch of {} requests", batch.len());
            let embedder = Arc::clone(&embedder);
            let cache = cache.clone();
            tokio::spawn(Self::process_batch(embedder, cache, batch, permit));
        }
    }

    /// Process one batch: split by mode, dedupe, check cache, infer, respond.
    async fn process_batch(
        embedder: Arc<Embedder>,
        cache: Option<Arc<EmbeddingCache>>,
        batch: Vec<EmbedRequest>,
        _permit: OwnedSemaphorePermit,
    ) {
        let (queries, passages): (Vec<_>, Vec<_>) = batch.into_iter().partition(|r| r.is_query);
        for (group, is_query) in [(queries, true), (passages, false)] {
            if !group.is_empty() {
                Self::process_group(&embedder, &cache, group, is_query).await;
            }
        }
    }

    async fn process_group(
        embedder: &Arc<Embedder>,
        cache: &Option<Arc<EmbeddingCache>>,
        group: Vec<EmbedRequest>,
        is_query: bool,
    ) {
        // unique text -> embedding
        let mut resolved: HashMap<String, Vec<f32>> = HashMap::new();
        let mut to_compute: Vec<String> = Vec::new();

        for req in &group {
            if resolved.contains_key(&req.text) || to_compute.contains(&req.text) {
                continue;
            }
            match cache.as_ref().and_then(|c| c.get(&req.text, is_query)) {
                Some(emb) => {
                    resolved.insert(req.text.clone(), emb);
                }
                None => to_compute.push(req.text.clone()),
            }
        }

        if !to_compute.is_empty() {
            let emb = Arc::clone(embedder);
            let texts = to_compute.clone();
            let result = tokio::task::spawn_blocking(move || {
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                emb.embed_batch(&refs, is_query)
            })
            .await;

            let err = match result {
                Ok(Ok(r)) if r.embeddings.len() == to_compute.len() => {
                    for (text, e) in to_compute.into_iter().zip(r.embeddings) {
                        if let Some(c) = cache {
                            c.put(&text, is_query, e.clone());
                        }
                        resolved.insert(text, e);
                    }
                    None
                }
                Ok(Ok(_)) => Some("Inference returned wrong number of embeddings".to_string()),
                Ok(Err(e)) => Some(format!("Inference error: {}", e)),
                Err(e) => Some(format!("Task panic: {}", e)),
            };

            if let Some(msg) = err {
                warn!("{}", msg);
                for req in group {
                    let _ = req.response_tx.send(Err(msg.clone()));
                }
                return;
            }
        }

        for req in group {
            let out = resolved
                .get(&req.text)
                .cloned()
                .ok_or_else(|| "missing embedding".to_string());
            let _ = req.response_tx.send(out);
        }
    }
}
