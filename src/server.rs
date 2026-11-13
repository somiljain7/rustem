use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::signal;
use tracing::info;

use crate::batcher::{BatcherConfig, DynamicBatcher};
use crate::cache::{CacheStats, EmbeddingCache};
use crate::embedder::Embedder;

// ── Request / Response types ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
    #[serde(default)]
    pub is_query: bool,
}

#[derive(Debug, Deserialize)]
pub struct EmbedSingleRequest {
    pub text: String,
    #[serde(default)]
    pub is_query: bool,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub dim: usize,
    pub latency_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub dim: usize,
    pub cache: Option<CacheStats>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    error: String,
}

type ApiError = (StatusCode, Json<ErrorResponse>);

fn api_err(code: StatusCode, msg: impl Into<String>) -> ApiError {
    (code, Json(ErrorResponse { error: msg.into() }))
}

// ── Shared application state ──────────────────────────────────────────────

pub struct AppState {
    pub batcher: DynamicBatcher,
    pub cache: Option<Arc<EmbeddingCache>>,
    pub embed_dim: usize,
    pub limits: Limits,
}

/// Request limits protecting the service from oversized inputs.
#[derive(Debug, Clone)]
pub struct Limits {
    pub max_texts: usize,
    pub max_text_chars: usize,
    pub request_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_texts: 256,
            max_text_chars: 8192,
            request_timeout: Duration::from_secs(30),
        }
    }
}

// ── Server configuration ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind: String,
    pub port: u16,
    pub cache_capacity: usize,
    pub batcher: BatcherConfig,
    pub limits: Limits,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            // Loopback by default: there is no auth, so don't expose publicly unasked.
            bind: "127.0.0.1".to_string(),
            port: 8080,
            cache_capacity: 10_000,
            batcher: BatcherConfig::default(),
            limits: Limits::default(),
        }
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────

fn validate(state: &AppState, texts: &[String]) -> Result<(), ApiError> {
    if texts.len() > state.limits.max_texts {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            format!("too many texts: {} > {}", texts.len(), state.limits.max_texts),
        ));
    }
    if let Some(t) = texts.iter().find(|t| t.chars().count() > state.limits.max_text_chars) {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            format!(
                "text too long: {} chars > {}",
                t.chars().count(),
                state.limits.max_text_chars
            ),
        ));
    }
    Ok(())
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cache_stats = state.cache.as_ref().map(|c| c.stats());
    Json(HealthResponse {
        status: "ok".to_string(),
        dim: state.embed_dim,
        cache: cache_stats,
    })
}

async fn embed_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<EmbedRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if payload.texts.is_empty() {
        return Ok(Json(EmbedResponse {
            embeddings: vec![],
            dim: state.embed_dim,
            latency_ms: 0.0,
        }));
    }
    validate(&state, &payload.texts)?;

    let start = Instant::now();
    let embeddings = tokio::time::timeout(
        state.limits.request_timeout,
        state.batcher.embed_many(payload.texts, payload.is_query),
    )
    .await
    .map_err(|_| api_err(StatusCode::GATEWAY_TIMEOUT, "request timed out"))?
    .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(EmbedResponse {
        embeddings,
        dim: state.embed_dim,
        latency_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

async fn embed_single_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<EmbedSingleRequest>,
) -> Result<impl IntoResponse, ApiError> {
    validate(&state, std::slice::from_ref(&payload.text))?;
    let start = Instant::now();

    let embedding = tokio::time::timeout(
        state.limits.request_timeout,
        state.batcher.embed(payload.text, payload.is_query),
    )
    .await
    .map_err(|_| api_err(StatusCode::GATEWAY_TIMEOUT, "request timed out"))?
    .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(EmbedResponse {
        embeddings: vec![embedding],
        dim: state.embed_dim,
        latency_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

// ── Server startup ────────────────────────────────────────────────────────

/// Build the axum router with all routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/embed", post(embed_handler))
        .route("/embed/single", post(embed_single_handler))
        .with_state(state)
}

/// Start the HTTP server with graceful shutdown.
pub async fn run_server(embedder: Arc<Embedder>, config: ServerConfig) -> Result<()> {
    let cache = if config.cache_capacity > 0 {
        Some(Arc::new(EmbeddingCache::new(config.cache_capacity)))
    } else {
        None
    };

    let embed_dim = embedder.dim();

    // Run as many concurrent batches as there are ONNX sessions.
    let mut batcher_cfg = config.batcher.clone();
    batcher_cfg.concurrency = embedder.num_sessions();

    let batcher = DynamicBatcher::new(Arc::clone(&embedder), cache.clone(), batcher_cfg);

    let state = Arc::new(AppState {
        batcher,
        cache,
        embed_dim,
        limits: config.limits.clone(),
    });

    let app = build_router(state);

    let addr = format!("{}:{}", config.bind, config.port);
    info!("Starting embed-service on http://{}", addr);

    let listener = TcpListener::bind(&addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shut down gracefully");
    Ok(())
}

/// Wait for CTRL+C or (on Unix) SIGTERM for graceful shutdown.
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("Failed to install CTRL+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("Received shutdown signal");
}
