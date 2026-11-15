# Learning notes: how this embedding service gets (and loses) speed

A question-and-answer walk-through, ordered so each part builds on the one before. Numbers are from this project (Intel i5-9300H: 4 cores / 8 threads, AVX2 + FMA, no AVX-512, no VNNI). Detailed measurements and reproduction commands are in `BUILD_NOTES.md`; this file explains the ideas.

---

## Part 1. What are we building?

**Q1. What does the service do?**
It turns a sentence into a list of numbers (an embedding, 384 or 768 values) so that sentences with similar meaning end up close together. Callers send text over HTTP and get the vector back.

**Q2. What is the model?**
Two models were used:
- `multilingual-e5-small`: 384 dimensions, about 5 ms per request.
- `EmbeddingGemma-300M`: 768 dimensions, higher quality (Recall@1 about 96.5% vs 91%), slower. Most of the speed work in this repo is on Gemma.

**Q3. Where does the time go?**
Per request: tokenizing is about 0.07 ms, the model about 17 to 34 ms. More than 99.8% of the time is the model's matrix math. Everything else (the web server, the cache, the batching code) is negligible. That is why most of the work was about the model and the runtime, not the Rust code.

---

## Part 2. The software stack, top to bottom

**Q4. What are the layers between my code and the chip?**
```
Rust service (axum server, tokenizer, cache, session pool)
   | calls
Runtime: ONNX Runtime (ORT)  or  OpenVINO
   | calls
Kernel libraries: MLAS (inside ORT), oneDNN (inside OpenVINO), MKL, OpenBLAS
   | written with
CPU instructions: SSE, AVX, AVX2+FMA, AVX-512, VNNI, AMX
   | run on
CPU cores
```

**Q5. What is ONNX, and what is ONNX Runtime?**
ONNX is a file format: a graph of operations (MatMul, LayerNorm, ...) plus the weights. ONNX Runtime (ORT) is Microsoft's engine that loads that file and runs it. It has pluggable "execution providers" (the default CPU one, an OpenVINO one, others).

**Q6. What is OpenVINO?**
Intel's engine for the same job, with its own graph optimizer and kernels (oneDNN, formerly MKL-DNN). It can be used directly (native) or as an ORT execution provider.

**Q7. What is a kernel?**
A hand-tuned inner loop, such as "multiply these two matrices". These do about 99% of the work and are where CPU instruction sets matter.

---

## Part 3. How the CPU does math fast

**Q8. What is SIMD, and what is a lane?**
A normal instruction adds one pair of numbers. A SIMD instruction does a whole row at once. A **lane** is one slot in that row holding one number. Lanes = register width / number size: 128-bit = 4 fp32 lanes, 256-bit = 8, 512-bit = 16. The same 256-bit register holds 32 INT8 values, which is why smaller number types can fit more per instruction.

**Q9. How do SSE, AVX, AVX2, FMA, AVX-512, VNNI and AMX differ?**

| Name | Width | First appeared | What it adds |
|---|---|---|---|
| SSE / SSE2 | 128-bit (4 fp32) | 1999 to 2001 | Baseline; every x86-64 CPU has it |
| AVX | 256-bit (8 fp32) | Intel Sandy Bridge (2011), AMD Bulldozer | Wider float vectors, no fused multiply-add |
| AVX2 + FMA | 256-bit | Intel Haswell (2013), AMD Zen (2017) | 256-bit integers, fused multiply-add. **Your CPU** |
| AVX-512 | 512-bit (16 fp32) | Intel Xeon (2017), some client chips, AMD Zen 4 | Double the width, mask registers |
| VNNI | 256 or 512-bit | Cascade Lake, Ice Lake, Zen 4, Alder Lake (256-bit) | One-step INT8 dot-product; replaces about 3 AVX2 instructions |
| AMX | tile registers | Sapphire Rapids (2023) | Matrix engine for INT8/BF16 |

Market quirks: Intel 12th to 14th gen client chips have AVX2 and AVX-VNNI but AVX-512 disabled. AMD Zen 4 runs AVX-512 on 256-bit hardware; Zen 5 is full width.

**Q10. What is FMA, an FMA unit, and a port?**
FMA (fused multiply-add) computes `a x b + c` in one instruction. An FMA unit is the hardware inside a core that runs it. A port is the doorway the scheduler uses to send an instruction to a kind of hardware; only some ports lead to FMA units. On Intel Haswell/Skylake-class cores, two ports have FMA units, so a core does 2 FMAs per cycle. Loads and stores use other ports and can starve the FMA units if the kernel is not careful.

**Q11. What is the peak speed of my CPU, and what did we measure?**
Peak per core per cycle on AVX2 = 8 lanes x 2 operations x 2 FMA units = 32. With 4 cores and a clock speed that gives a spec-sheet figure of about 450 GFLOPS. The measured fp32 matrix-multiply ceiling (numpy/OpenBLAS, 4 threads) was about **200 GFLOPS** (113 at 24 tokens), roughly half. The gap comes from real clocks, memory traffic and small matrix shapes.

---

## Part 4. Compilers, languages and what did not help

**Q12. What do compilers have to do with this?**
A compiler turns source code into machine code for a specific CPU. `rustc` builds the Rust service. `icpc` was Intel's old C++ compiler; it is retired and replaced by `icx`/`icpx` (LLVM-based). **SYCL** is a way to write C++ once and run it on CPUs, GPUs and other accelerators; it is for writing your own kernels, not a speed switch.

**Q13. How is SIMD defined in this repo?**
It is not. There is no hand-written SIMD in the Rust code. The speed-critical SIMD lives inside ONNX Runtime (MLAS) and OpenVINO (oneDNN), which detect the CPU at runtime and pick the best kernel.

**Q14. Does `RUSTFLAGS="-C target-cpu=native"` help?**
No gain measured, because the heavy work is inside the prebuilt libraries, not in our Rust code.

**Q15. Is Rust slower or faster than Python here? Does the Rust/C++ boundary (ABI) cost anything?**
Same speed. On an identical setup the two tie, with inference above 99.8% of the time, so the language and the call into C++ are not the bottleneck. Rewriting the service in C++ would not help for the same reason.

**Q16. Would MKL-DNN (oneDNN) kernels help?**
They are already inside OpenVINO's CPU plugin. Native OpenVINO per token cost (about 1.30 ms) equals ONNX Runtime's (about 1.30 ms) once the fixed cost below is removed, so there is no kernel advantage left to gain.

**Q17. Would SYCL on the CPU help?**
It is technically possible (Intel DPC++ via the OpenCL CPU runtime, or AdaptiveCpp), but you would have to write your own matmul, attention and INT8 kernels and beat MLAS and oneDNN on the same AVX2 cores and the same ~200 GFLOPS ceiling. It is **not implemented or tried**; no SYCL compiler, MSVC or CMake is installed here. A small SYCL GEMM versus the 200 GFLOPS figure would be the cheap test.

---

## Part 5. The real bottleneck: hidden dequantization

**Q18. What are quantization and dequantization?**
Quantization stores weights as small integers (INT8: 4x smaller than fp32). Dequantization converts them back to fp32 before using them.

**Q19. What was wrong with the Gemma INT8 file?**
Its weights were INT8 only on disk. The graph had `DequantizeLinear` nodes that converted all weights to fp32 **on every call** (about 145 to 175 ms, independent of batch size), then ran fp32 matmuls. The latency fit was: ORT = 144.8 ms fixed + 1.298 ms per token; native OpenVINO = about 0 fixed + 1.302 ms per token. So the "6.4x OpenVINO speed-up" in the old report was mostly this avoidable fixed cost.

**Q20. How was it fixed?**
Fold the dequantization into the weights once, offline: `python scripts/fold_dequantize.py models/gemma/model_quantized.onnx models/gemma_fp32w`. In the Rust service on plain ORT CPU, Gemma batch 1 went from **147 to 171 ms down to 34 ms**, with identical embeddings (cosine 1.000000). Trade-off: the model file grows from 295 MB to 1.18 GB.

---

## Part 6. OpenVINO in practice

**Q21. Why is OpenVINO tricky to set up?**
Versions must match exactly. Tested pairings: onnxruntime-openvino 1.20.0 + OpenVINO 2024.4.0 works but gives no speed-up; 1.24.1 + 2026.4.1 fails with Windows error 127; 1.24.1 + **2025.4.1** works (75.6 ms at batch 1 through the provider). Native OpenVINO 2026.4.1 reads the Gemma model; 2024.4 cannot (missing operators).

---

## Part 7. Speed-up experiments after the fix

All numbers are batch-1 p50 on the 202-utterance eval set, plain ORT CPU, 4 threads (Python ORT 1.20.1 unless noted).

**Q22. Does a shorter prompt prefix help?**
Yes, modestly, with no measurable quality loss. Each token costs about 1.3 ms. Replacing the model-card prefix with `query: ` cut latency by about 10% and Recall@1 stayed at about 98% (differences of 1 to 3 utterances out of 202 are within noise). It departs from the model card, so recheck on real queries.

**Q23. Does INT4 weight-only quantization help?**
No. 55 ms vs 41 ms for fp32 on AVX2, due to unpacking overhead.

**Q24. Does dynamic INT8 help?**
Yes, the biggest gain after the fix. 41 ms to 34 ms with the model-card prefix, and **19 ms** with `query: `, Recall within noise. In the Rust service: batch 1 34.3 to 31.6 ms, but batch 16 got slower (about 355 to 401 ms), so use it for single-request latency, not large batches. Newer ORT (1.24.1) runs it faster still (22.8 ms at the model-card prefix).

**Q25. Is OpenVINO better with dynamic INT8?**
No. Through the OpenVINO provider it ran at 190 ms, about 8x slower than plain ORT on the same model and slower than OpenVINO fp32 (95 ms). Without VNNI, OpenVINO's INT8 has no hardware advantage.

---

## Part 8. The server

**Q26. What does axum do?**
Axum is the Rust web framework (`src/server.rs`). A router maps URLs to handlers: `GET /health`, `POST /embed` (batch) and `POST /embed/single`. Handlers parse JSON, check limits, call the embedder and return JSON. Shared state (embedder, cache, limits) is held in an `Arc`.

**Q27. How does the service avoid blocking?**
Axum runs on tokio, an async runtime, where a few threads juggle many waiting requests. Inference is heavy CPU work, so it runs under `tokio::task::spawn_blocking` on a separate thread pool, fed by a batcher (`src/batcher.rs`) that collects requests through channels and returns results through one-shot channels. Batching helps throughput: batch 16 is about 22 ms per text vs 34 ms single.

---

## Part 9. Summary of what we learned

1. The big win was not a faster language, instruction set or engine. The Gemma file was doing wasteful work on every call; fixing that gave about 4x.
2. After that, the cost is real math at about 1.3 ms per token on 4 cores, so the remaining levers are fewer tokens (shorter prefix) and cheaper math (dynamic INT8).
3. Best result so far: plain ORT CPU, dynamic INT8, `query: ` prefix, about **17 to 19 ms** per request, versus 147 to 171 ms at the start.
4. Next big steps are hardware: AVX-512/VNNI/AMX CPUs, or a GPU/NPU.

## Part 10. Open items
- Not measured: cosine fidelity of the dynamic-INT8 model vs fp32; non-synthetic query data; Rust batch 1 with the shorter prefix; a newer `ort` crate and DLL (1.24.x) in the Rust service.
- Not tried: SYCL; the OpenVINO `GPU` device on the integrated graphics; padding experiment; e5-small checked for the same dequantization pattern.

## Quick commands
```powershell
# Fold the Gemma weights (once)
python scripts/fold_dequantize.py models/gemma/model_quantized.onnx models/gemma_fp32w
# Run the service
embed-service --model models/gemma_fp32w/model_fp32w.onnx --tokenizer models/gemma/tokenizer.json --prefix-style gemma --sessions 1 --threads 4 serve
# Prefix / quantization experiments
python scripts/experiments/exp7_quant.py <fp32w.onnx> <out_dir>
python scripts/experiments/exp6_prefix.py <model.onnx> models/gemma/tokenizer.json
```

## Hands-on and deeper dives
- [learning/matmul/](learning/matmul/README.md): the same matrix multiply in six versions (naive to AVX2 + threads), runnable C++ that shows the constant-factor ideas.
- [learning/KERNEL_LIBRARIES.md](learning/KERNEL_LIBRARIES.md): MLAS, oneDNN and MKL explained for newcomers.
