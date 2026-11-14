use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, info};

/// A managed ONNX session with tracking metadata.
struct PooledSession {
    session: Session,
    id: usize,
}

/// Configuration for the session pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Path to the ONNX model file.
    pub model_path: String,
    /// Number of independent ONNX sessions.
    pub num_sessions: usize,
    /// Intra-op threads per session (controls parallelism within a single op).
    pub intra_threads: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            model_path: "models/model_quantized.onnx".to_string(),
            num_sessions: 4,
            intra_threads: 2,
        }
    }
}

/// Thread-safe pool of ONNX sessions using a crossbeam channel for
/// work-stealing / lock-free session distribution.
///
/// Sessions are checked out via `get()` and automatically returned
/// when the `SessionGuard` is dropped.
pub struct SessionPool {
    /// Channel-based pool: sessions sit in the channel when idle.
    sender: Sender<PooledSession>,
    receiver: Receiver<PooledSession>,
    /// Total number of sessions in the pool.
    pub num_sessions: usize,
    /// Round-robin counter for diagnostics / fallback.
    counter: AtomicUsize,
}

/// RAII guard that returns the session to the pool on drop.
pub struct SessionGuard<'a> {
    session: Option<PooledSession>,
    pool: &'a SessionPool,
}

impl<'a> SessionGuard<'a> {
    /// Access the underlying ONNX session.
    pub fn session(&self) -> &Session {
        &self.session.as_ref().unwrap().session
    }

    /// Session ID (index in the pool).
    pub fn id(&self) -> usize {
        self.session.as_ref().unwrap().id
    }
}

impl<'a> Drop for SessionGuard<'a> {
    fn drop(&mut self) {
        if let Some(s) = self.session.take() {
            // Return session to pool; ignore error (pool dropped).
            let _ = self.pool.sender.send(s);
        }
    }
}

impl SessionPool {
    /// Create a new session pool. Each session gets its own intra-op thread
    /// allocation. Thread spinning is disabled to avoid busy-wait contention on
    /// CPUs with limited cores.
    pub fn new(config: &PoolConfig) -> Result<Self> {
        info!(
            "Creating session pool: {} sessions × {} intra-threads",
            config.num_sessions, config.intra_threads
        );

        // Bounded channel with capacity == num_sessions ensures we never exceed
        // the pool size, and receivers block only when all sessions are busy.
        let (sender, receiver) = crossbeam_channel::bounded(config.num_sessions);

        for i in 0..config.num_sessions {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(config.intra_threads)?
                .with_inter_threads(1)?
                .commit_from_file(&config.model_path)
                .with_context(|| format!("Failed to create pooled session {}", i))?;

            sender
                .send(PooledSession { session, id: i })
                .map_err(|_| anyhow::anyhow!("Failed to enqueue session {}", i))?;

            debug!("Created pooled session {}", i);
        }

        Ok(Self {
            sender,
            receiver,
            num_sessions: config.num_sessions,
            counter: AtomicUsize::new(0),
        })
    }

    /// Checkout a session from the pool. Blocks if all sessions are busy.
    /// The session is returned automatically when the guard is dropped.
    pub fn get(&self) -> Result<SessionGuard<'_>> {
        let session = self
            .receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("Session pool channel closed"))?;
        self.counter.fetch_add(1, Ordering::Relaxed);
        Ok(SessionGuard {
            session: Some(session),
            pool: self,
        })
    }

    /// Try to checkout a session without blocking. Returns `None` if all sessions are busy.
    pub fn try_get(&self) -> Option<SessionGuard<'_>> {
        self.receiver.try_recv().ok().map(|session| {
            self.counter.fetch_add(1, Ordering::Relaxed);
            SessionGuard {
                session: Some(session),
                pool: self,
            }
        })
    }

    /// Number of sessions currently available (idle) in the pool.
    pub fn available(&self) -> usize {
        self.receiver.len()
    }

    /// Total number of checkouts performed (diagnostic counter).
    pub fn total_checkouts(&self) -> usize {
        self.counter.load(Ordering::Relaxed)
    }
}
