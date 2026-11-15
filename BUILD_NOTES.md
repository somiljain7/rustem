# Build Notes: changes, issues faced, and pre-run checklist

Read this before building or running on Windows. It records what was changed in the
code review pass, every problem hit while getting the project to build and run, and
the verified results.

## 1. Code changes (review fixes)

| File | Change | Why |
|---|---|---|
| `src/main.rs` | `ServerConfig { .., ..Default::default() }` in `cmd_serve` | `limits` field was missing, so the crate did not compile (E0063) |
| `src/main.rs` | `eval` errors on embedding dimension mismatch; removed unused `EmbeddingCache` import | Mismatched dims were silently truncated by `zip` |
| `src/pool.rs` | `with_intra_threads(config.intra_threads)` (no `as u16`) | `ort 2.0.0-rc.9` takes `usize` |
| `src/pool.rs`, `src/embedder.rs` | Removed `optimization_level` from `PoolConfig`; Level3 set where sessions are built | `GraphOptimizationLevel` is not `Clone`/`Copy` in rc.9, so `PoolConfig: Clone` did not compile |
| `src/cache.rs` | Capacity 0 clamps to 1 instead of panicking | Public constructor panicked on 0 |
| `src/embedder.rs` | `cosine_similarity` uses `assert_eq!` on length (was `debug_assert`) | Release builds silently truncated mismatched vectors |
| `src/embedder.rs` | `embed_batch_parallel` takes `&self`, runs in waves of `num_sessions` threads | Was one OS thread per chunk with no bound; unused `Arc` receiver |
| `src/batcher.rs` | Skip requests whose client already timed out or disconnected (`response_tx.is_closed()`) | Timed-out requests still consumed inference CPU |
| `Cargo.toml` | `ort` uses `default-features = false, features = ["load-dynamic","ndarray"]` | Default features enable `download-binaries`, which fails on `windows-gnu` |
| `Cargo.toml` | Added `ort-sys = "=2.0.0-rc.9"`; removed unused `rayon` | `ort-sys` floated to rc.10, breaking the "pinned" intent |
| `Cargo.lock` | Regenerated for the above | |

## 2. Issues faced and fixes

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `An Application Control policy has blocked this file (os error 4551)` on cargo build scripts | Windows **Smart App Control** blocks freshly built unsigned `.exe` files | Turned Smart App Control off (Windows Security > App & browser control). Irreversible without reinstalling Windows. Alternative: build in WSL2 or CI |
| 2 | `error calling dlltool 'dlltool.exe': program not found` | Active toolchain is `stable-x86_64-pc-windows-gnu`, which needs MinGW; none installed | `winget install BrechtSanders.WinLibs.POSIX.UCRT` (MinGW gcc/dlltool). Open a new terminal afterwards so PATH updates |
| 3 | `ort-sys` build panics: `downloaded binaries not available for target x86_64-pc-windows-gnu` | `ort` default features enable `download-binaries` | `default-features = false` (see above). Also pin `ort-sys` |
| 4 | Two compile errors in `pool.rs` (`GraphOptimizationLevel: Clone`, move out of shared ref) | rc.9 type is not `Clone`/`Copy` | Removed it from the config struct |
| 5 | `cargo update` stalled: `transfer too slow` | Flaky crates.io index connection | `CARGO_HTTP_TIMEOUT=300` and `CARGO_HTTP_MULTIPLEXING=false`, then retry |
| 6 | `winget` failed with `0x80070020` (file in use) | A second winget install of the same package was already running | Wait for the first to finish; do not start two |
| 7 | No `onnxruntime.dll` on the machine (needed because of `load-dynamic`) | Library is not bundled | `pip install onnxruntime==1.20.1` into a throwaway venv and point `ORT_DYLIB_PATH` at its `capi\onnxruntime.dll` |
| 8 | `ConvertFrom-Json` failed on `embed` output | The service writes **tracing logs to stdout**, so they precede the JSON | Set `RUST_LOG=error` when piping or parsing output (open item: log to stderr instead) |
| 9 | Hindi text corrupted when piped to the exe from PowerShell | PS 5.1 pipes native input as ASCII | Write a UTF-8 file and use `Start-Process -RedirectStandardInput` |
| 10 | Exit code 255 on `embed` when output was cut with `Select-Object -First N` | Pipe closed early; service panics on broken-pipe `println!` | Not a service bug; do not truncate the pipe |
| 11 | Release build very slow (minutes) | `lto = "fat"`, `codegen-units = 1`, all deps rebuilt | Expected on first build; use `cargo build` (debug) for quick checks |

## 3. Checklist before running again (Windows)

1. Smart App Control is off (`(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy').VerifiedAndReputablePolicyState` returns `0`).
2. MinGW is on PATH: `gcc --version` and `dlltool --version` work in a **new** terminal.
3. ONNX Runtime DLL exists and is set (version 1.20.x, matching `ort` rc.9):
   ```powershell
   $env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime.dll"
   ```
   The copy used in this session lived in a temporary scratchpad venv and may be gone. Recreate it or copy the DLL somewhere permanent.
4. Quiet the logs when parsing output: `$env:RUST_LOG = "error"`.
5. Build and test:
   ```powershell
   cargo check --all-targets
   cargo test --lib
   cargo build --release
   ```
6. Run:
   ```powershell
   .\target\release\embed-service.exe embed --text "I want to order pizza"
   .\target\release\embed-service.exe serve --port 8080
   ```

## 4. Verified results (2026-11-15, Intel i5-9300H, 4 cores / 8 threads)

- `cargo check --all-targets`: passes. `cargo test --lib`: 3 passed.
- `serve`: `/health`, `/embed`, `/embed/single` work; 300 texts is rejected with HTTP 400; duplicates are embedded once and returned in order.
- `eval` (quantized vs FP32 reference, 202 utterances, same Rust pipeline): mean cosine **0.9967**, min 0.9904, max 0.9983. Falls in the tool's warning band (>= 0.99, < 0.999), expected for int8.

`bench`, 4 sessions x 2 threads, 30 iterations, quantized model, MinGW (`windows-gnu`) build:

| Batch | 1 client p50 / p95 (ms) | 1 client texts/s | 4 clients p50 / p95 (ms) | 4 clients texts/s |
|---:|---:|---:|---:|---:|
| 1 | 6.96 / 10.57 | 136.1 | 10.06 / 29.97 | 283.7 |
| 8 | 48.67 / 53.66 | 163.3 | 110.21 / 270.33 | 227.8 |
| 16 | 90.43 / 100.93 | 176.5 | 244.74 / 556.54 | 221.6 |
| 64 | 322.56 / 364.29 | 197.5 | 849.41 / 1224.70 | 287.1 |

Notes on these numbers:
- These were **slower than the README headline** (5.15 ms p50 at batch 1). A later re-run with 100 iterations and 10 warmups gave 5.05 to 5.29 ms at batch 1 on the same MinGW build (section 7), matching the README. So the MinGW build is not slower. The earlier slow run (30 iterations) was most likely noise or background load, which was not proven. Use 100+ iterations on an idle machine before quoting numbers.
- With 4 clients, batch-1 throughput roughly doubles (136 to 284 texts/s) but tail latency grows sharply (p95 about 30 ms). 4 clients x 2 intra-threads oversubscribe a 4-core CPU. For latency-sensitive voice use, consider fewer threads per session.

## 5. Not verified / open items

- The README's Recall@1 and Recall@5 figures come from the Python scripts and were not reproduced here.
- The FP32 reference was produced by this same Rust pipeline, so `eval` measures quantization loss only, not tokenizer or pooling correctness against the Python reference.
- Logs go to stdout (issue 8); switching tracing to stderr is a one-line change not yet made.
- `scripts/*.py` and `benches/throughput.rs` were not run (benches only compile-checked).

## 6. OpenVINO backend and EmbeddingGemma (added 2026-11-15)

### What was added
- `--backend cpu|openvino`, `--ov-device CPU|GPU|NPU`, `--ov-cache-dir` (all subcommands). The OpenVINO execution provider registers through `ort`'s existing `load-dynamic` feature, so no new Cargo feature is needed. If the loaded ONNX Runtime has no OpenVINO support, startup fails with a clear error instead of silently running on plain CPU.
- The embedder now adapts to the model: it sends `token_type_ids` only if the model declares it, uses a pooled `sentence_embedding` output when present (else masked mean pooling), and supports `--prefix-style e5|gemma|none` and `--truncate-dim N` (Matryoshka).
- Gemma example:
  ```powershell
  embed-service --model models/gemma/model_quantized.onnx --tokenizer models/gemma/tokenizer.json `
    --prefix-style gemma --sessions 1 --threads 4 --backend openvino embed
  ```

### Runtime pairings tested (the ONNX Runtime and OpenVINO versions must match)
Install each in its own venv; point `ORT_DYLIB_PATH` at that venv's `onnxruntime\capi\onnxruntime.dll` and put its `openvino\libs` and `onnxruntime\capi` folders on `PATH`.

| ONNX Runtime | OpenVINO | Result |
|---|---|---|
| `onnxruntime` 1.20.1 (CPU) | none | Works. Baseline |
| `onnxruntime-openvino` 1.20.0 | `openvino==2024.4.0` | Works, **no speedup** (provider leaves the fused Gemma ops on CPU) |
| `onnxruntime-openvino` 1.24.1 | `openvino==2026.4.1` | **Fails**: `Error 127 procedure not found` loading the provider DLL (ABI mismatch) |
| `onnxruntime-openvino` 1.24.1 | `openvino==2025.4.1` | Works, about 2.2x faster at batch 1. `ort` rc.9 warns it expects 1.20.x but the legacy provider call still works |

### Gemma results (quantized INT8, 1 session x 4 threads, 30 iterations, i5-9300H)

| Path | batch 1 p50 | batch 8 p50 | batch 16 p50 |
|---|---:|---:|---:|
| ORT 1.20.1, CPU provider | 168.96 ms | 345.60 ms | 531.97 ms |
| ORT-OpenVINO 1.20.0 + OV 2024.4 | 163.20 ms | 360.19 ms | 579.58 ms |
| ORT-OpenVINO 1.24.1 + OV 2025.4.1 | **75.58 ms** | 300.54 ms | 573.44 ms |
| Native `openvino.Core` 2026.4.1 (Python probe, fixed seq 24, no tokenizer) | **39.4 ms** | 295.0 ms | 590.0 ms |

- Embeddings were identical across CPU and OpenVINO runs on the sample (English pizza vs Hinglish pizza cosine 0.9192, vs weather 0.2440).
- The report's 37.99 ms is reproduced only by **native OpenVINO 2026**, not by the ONNX Runtime provider. OpenVINO 2024.4's ONNX frontend cannot even read this file: it lacks `SimplifiedLayerNormalization`, `com.microsoft.RotaryEmbedding` and `com.microsoft.MultiHeadAttention`.
- The advantage is a **batch-1 latency** win only. At batch 8 and 16 every path is within noise of plain CPU (compute-bound), so OpenVINO does not help throughput here.
- Reaching about 39 ms inside the Rust service would need the native OpenVINO API (for example the `openvino` crate or the C API), a bigger change than the provider backend. Not done.
- The native probe used fixed token ids, so it measures forward-pass time only. Tokenization and pooling are excluded.

### SIMD in this project
- The repo contains **no explicit SIMD** (no `std::arch`, `portable_simd`, `#[target_feature]`, or SIMD crates). Inference SIMD (AVX2/FMA) lives inside ONNX Runtime and OpenVINO's C++ kernels, which pick instruction sets at runtime. Our Rust only does scalar `f32` loops for pooling, normalization and cosine similarity (a few hundred to 768 floats per text).
- `RUSTFLAGS="-C target-cpu=native"` was tested (compiler saw `avx avx2 bmi2 fma sse4.2`, no `avx512f`). e5-small, 4 sessions x 2 threads, 100 iterations, baseline vs native alternated twice:

| Batch | Baseline p50 (round 1 / 2) | Native p50 (round 1 / 2) |
|---:|---|---|
| 1 | 5.29 / 5.05 ms | 5.29 / 5.08 ms |
| 16 | 61.76 / 65.44 ms | 61.12 / 65.31 ms |

  No measurable gain, which is expected: inference runs in the C++ libraries, and our own loops are microseconds. A native build is also not portable to other CPUs, so it is **not** enabled by default. To try it anyway: `$env:RUSTFLAGS="-C target-cpu=native"; $env:CARGO_TARGET_DIR="target-native"; cargo build --release` (the `target-native/` folder is git-ignored).

### New issues faced
| # | Symptom | Cause | Fix |
|---|---|---|---|
| 12 | `unexpected argument 'want'` | `Start-Process -ArgumentList` splits strings with spaces | Feed texts through stdin from a UTF-8 file |
| 13 | OpenVINO provider loads but gives no speedup | Provider cannot offload ORT-specific fused ops | Use a newer matching pairing (table above) or native OpenVINO |
| 14 | `onnxruntime_providers_openvino.dll` Error 127 | ONNX Runtime and OpenVINO versions do not match | Pair 1.20.0 with 2024.4, 1.24.1 with 2025.4.1 |
| 15 | `ort` rc.9 warns about ONNX Runtime 1.24.1 | `ort` rc.9 targets 1.20 | Works for the OpenVINO provider call used here, but untested more broadly |
