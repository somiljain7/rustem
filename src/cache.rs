use lru::LruCache;
use parking_lot::Mutex;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

/// Thread-safe LRU embedding cache.
///
/// Key: hash of `(text, is_query)` — avoids storing full text strings in memory.
/// Value: `Vec<f32>` embedding vector.
pub struct EmbeddingCache {
    inner: Mutex<LruCache<u64, Vec<f32>>>,
    hits: AtomicU64,
    misses: AtomicU64,
    capacity: usize,
}

/// Snapshot of cache statistics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheStats {
    pub capacity: usize,
    pub size: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

impl EmbeddingCache {
    /// Create a new cache with the given maximum capacity (number of embeddings).
    pub fn new(capacity: usize) -> Self {
        info!("Embedding cache initialized: capacity={}", capacity);
        Self {
            inner: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("Cache capacity must be > 0"),
            )),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            capacity,
        }
    }

    /// Compute a deterministic hash key for `(text, is_query)`.
    fn cache_key(text: &str, is_query: bool) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        is_query.hash(&mut hasher);
        hasher.finish()
    }

    /// Look up a cached embedding. Returns `Some(embedding)` on hit, `None` on miss.
    pub fn get(&self, text: &str, is_query: bool) -> Option<Vec<f32>> {
        let key = Self::cache_key(text, is_query);
        let mut cache = self.inner.lock();
        match cache.get(&key) {
            Some(emb) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(emb.clone())
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Insert an embedding into the cache.
    pub fn put(&self, text: &str, is_query: bool, embedding: Vec<f32>) {
        let key = Self::cache_key(text, is_query);
        let mut cache = self.inner.lock();
        cache.put(key, embedding);
    }

    /// Check if a text is cached without updating LRU order.
    pub fn contains(&self, text: &str, is_query: bool) -> bool {
        let key = Self::cache_key(text, is_query);
        let cache = self.inner.lock();
        cache.contains(&key)
    }

    /// Current number of entries in the cache.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.inner.lock().clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Get a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            capacity: self.capacity,
            size: self.len(),
            hits,
            misses,
            hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit_miss() {
        let cache = EmbeddingCache::new(10);
        let emb = vec![1.0, 2.0, 3.0];

        assert!(cache.get("hello", true).is_none());
        cache.put("hello", true, emb.clone());
        assert_eq!(cache.get("hello", true), Some(emb.clone()));

        // Different prefix mode should be a miss.
        assert!(cache.get("hello", false).is_none());

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = EmbeddingCache::new(2);
        cache.put("a", true, vec![1.0]);
        cache.put("b", true, vec![2.0]);
        cache.put("c", true, vec![3.0]); // evicts "a"

        assert!(cache.get("a", true).is_none());
        assert!(cache.get("b", true).is_some());
        assert!(cache.get("c", true).is_some());
    }
}
