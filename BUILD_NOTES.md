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
- They are **slower than the README headline** (5.15 ms p50 at batch 1, about 182 texts/s). The cause was not isolated. Candidates: this MinGW build vs the original build, other processes running during the test, laptop thermals, and only 30 iterations. Re-run on an idle machine before quoting them.
- With 4 clients, batch-1 throughput roughly doubles (136 to 284 texts/s) but tail latency grows sharply (p95 about 30 ms). 4 clients x 2 intra-threads oversubscribe a 4-core CPU. For latency-sensitive voice use, consider fewer threads per session.

## 5. Not verified / open items

- The README's Recall@1 and Recall@5 figures come from the Python scripts and were not reproduced here.
- The FP32 reference was produced by this same Rust pipeline, so `eval` measures quantization loss only, not tokenizer or pooling correctness against the Python reference.
- Logs go to stdout (issue 8); switching tracing to stderr is a one-line change not yet made.
- `scripts/*.py` and `benches/throughput.rs` were not run (benches only compile-checked).
