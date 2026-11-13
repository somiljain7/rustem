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
use std::time::Instant;
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
struct ErrorResponse {
    error: String,
}

// ── Shared application state ──────────────────────────────────────────────

pub struct AppState {
    pub batcher: DynamicBatcher,
    pub cache: Option<Arc<EmbeddingCache>>,
    pub embed_dim: usize,
}

// ── Server configuration ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind: String,
    pub port: u16,
    pub cache_capacity: usize,
    pub batcher: BatcherConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0".to_string(),
            port: 8080,
            cache_capacity: 10_000,
            batcher: BatcherConfig::default(),
        }
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────

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
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    if payload.texts.is_empty() {
        return Ok(Json(EmbedResponse {
            embeddings: vec![],
            dim: state.embed_dim,
            latency_ms: 0.0,
        }));
    }

    let start = Instant::now();

    let embeddings = state
        .batcher
        .embed_many(payload.texts, payload.is_query)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(EmbedResponse {
        embeddings,
        dim: state.embed_dim,
        latency_ms,
    }))
}

async fn embed_single_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<EmbedSingleRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();

    let embedding = state
        .batcher
        .embed(payload.text, payload.is_query)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(EmbedResponse {
        embeddings: vec![embedding],
        dim: state.embed_dim,
        latency_ms,
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

    let batcher = DynamicBatcher::new(
        Arc::clone(&embedder),
        cache.clone(),
        config.batcher.clone(),
    );

    let state = Arc::new(AppState {
        batcher,
        cache,
        embed_dim,
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

/// Wait for a CTRL+C signal for graceful shutdown.
async fn shutdown_signal() {
    signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    info!("Received shutdown signal");
}
