# Kernel libraries explained: MLAS, oneDNN and MKL

Read `../LEARNING_NOTES.md` first for the big picture, and run `matmul/` to see the ideas in code. This file zooms into the layer that does about 99% of the work in this project: the **kernel libraries**.

---

## 1. What is a kernel library, in DSA terms?

You never write your own `std::sort` in a contest; you call the library, because experts already tuned it. Matrix multiply is the same: nobody writes the hot loop by hand for a neural network. They call a **kernel library**, a collection of heavily tuned functions like "multiply these two matrices" (GEMM), "convolution", "layer norm", "softmax".

```
ONNX Runtime ──calls──▶ MLAS        (Microsoft's own kernels)
OpenVINO     ──calls──▶ oneDNN      (Intel's neural-network kernels)
numpy/PyTorch/SciPy ─▶ MKL / OpenBLAS (general math kernels)
```

Same job, three different libraries, written by different teams.

## 2. Two words you need first

- **GEMM** = GEneral Matrix Multiply, `C = A x B`. The transformer's attention and feed-forward layers are mostly GEMMs. Speed of GEMM is roughly speed of the model.
- **BLAS** = Basic Linear Algebra Subprograms. A 1979-era standard *interface* ("a function called `sgemm` that multiplies matrices"). Many libraries implement it: MKL, OpenBLAS, BLIS, Apple Accelerate. Same function name, different speed.

## 3. The three libraries

### MLAS (Microsoft Linear Algebra Subprograms)
- **What:** the CPU kernel library built into **ONNX Runtime**. Open source (inside the onnxruntime repo), compiled into `onnxruntime.dll`.
- **Contains:** fp32 GEMM, quantized (INT8) GEMM, convolutions, and activation functions, written with intrinsics and assembly.
- **How it adapts:** at startup it checks which instructions your CPU has (AVX2, AVX-512, VNNI, AMX, or NEON on ARM) and picks the matching kernel. That is "runtime dispatch".
- **Where we used it:** every "plain ORT CPU" number in this project. The dynamic-INT8 speed-up (`MatMulInteger`) ran through MLAS's INT8 GEMM.
- **Think of it as:** the standard library that comes with the engine. Good, general, and portable across Intel, AMD and ARM.

### oneDNN (oneAPI Deep Neural Network Library, formerly MKL-DNN)
- **What:** Intel's open-source library of **neural-network** building blocks: matmul, convolution, normalization, softmax, and so on. It is **inside OpenVINO's CPU plugin**, and also used by PyTorch and TensorFlow.
- **Key tricks beyond a plain GEMM:**
  - **JIT code generation:** it writes machine code *at run time* for your exact CPU and exact matrix shape, instead of shipping one fixed kernel.
  - **Fusion ("post-ops"):** it can do matmul + bias + activation in one pass over memory instead of three, which saves memory traffic.
  - **Layout choice:** it re-arranges data into blocked formats that suit the CPU, so loads are contiguous.
- **Where we used it:** every OpenVINO run (native and through the ORT provider).
- **Think of it as:** a specialist: tuned for deep learning rather than general math, and strongest on newer Intel features (AVX-512, VNNI, AMX) that your i5-9300H doesn't have.

### MKL (oneAPI Math Kernel Library, "oneMKL")
- **What:** Intel's long-standing **general math** library: BLAS, LAPACK (linear solvers), FFT, sparse math, random numbers. Closed-source binaries, free to use.
- **Where it shows up:** numpy/SciPy and PyTorch CPU builds often use it (my 200 GFLOPS ceiling test used numpy with **OpenBLAS**, MKL's open-source rival).
- **How it differs from oneDNN:** MKL knows math (`sgemm`, FFT), not neural networks, so it has no fusion of "matmul + GELU" and no model awareness.
- **Think of it as:** the premium, battle-tested `sgemm`. Not used directly by our Rust service.

## 4. Side by side

| | MLAS | oneDNN | MKL |
|---|---|---|---|
| Made by | Microsoft | Intel (open source) | Intel (closed source) |
| Lives inside | ONNX Runtime | OpenVINO, PyTorch | numpy, PyTorch, SciPy |
| Aimed at | ONNX models on CPU | Deep-learning layers | General numeric computing |
| Special strength | Portable; INT8 and fp32 GEMM | JIT, fusion, newest Intel features | Mature BLAS/LAPACK/FFT |
| Used in this project | Yes (ORT CPU) | Yes (OpenVINO) | No (not directly) |

## 5. What they all do that your naive loop does not

These are the same tricks as the `matmul/` demo, just pushed much further:

1. **SIMD with FMA**: 8 floats per instruction on AVX2.
2. **Register blocking**: hold a block of C in registers across the whole k loop.
3. **Cache blocking and packing**: copy tiles of A and B into a tidy contiguous buffer sized for L1/L2 cache.
4. **Multithreading**: split the work across cores.
5. **Runtime dispatch / JIT**: choose or generate the right kernel for this CPU and shape.
6. **Fusion** (oneDNN): skip intermediate memory round trips.
7. **Weight pre-packing**: re-arrange constant weights once at load, not on every call.

## 6. How close does the demo get?

On this laptop (4 cores, AVX2 + FMA):

| | GFLOPS |
|---|---|
| Naive triple loop | about 1 |
| Demo: AVX2 + register blocking, 1 thread | about 42 |
| Demo: + 4 threads | about 97 |
| Library (numpy/OpenBLAS, 4 threads, measured) | about 200 |
| Spec-sheet peak | about 450 |

So the toy reaches roughly half of what the real libraries do. The rest comes from details like packing, prefetching, better block sizes, and careful instruction scheduling. That is the effort a library saves you.

## 7. Why ORT and OpenVINO ended up equal here

Once the wasted dequantization was removed, both cost about **1.30 ms per token** (ORT with MLAS: 1.298; OpenVINO with oneDNN: 1.302). Both libraries sit near the same AVX2 ceiling on this CPU, so switching libraries gains nothing. That changes on newer hardware: oneDNN and MLAS both have faster paths for VNNI and AMX, and those machines would show larger differences.

## 8. Quick questions to test yourself

1. Is `sgemm` a library or an interface? (Interface: a BLAS function; MKL and OpenBLAS are libraries that implement it.)
2. Which library is inside `onnxruntime.dll`? (MLAS.)
3. Which one writes machine code at run time? (oneDNN.)
4. Why can a library beat your tiled AVX2 loop by 2x? (Packing, prefetching, tuned block sizes, scheduling.)
5. Which would help a chip with AMX most? (Libraries with AMX kernels: oneDNN and recent MLAS.)

## 9. Not measured here
- I did not profile which MLAS or oneDNN kernel names ran for each op.
- The ~200 GFLOPS figure is from numpy with OpenBLAS, not from MLAS or oneDNN directly.
- Differences on AMD or ARM, and on AVX-512/VNNI/AMX hardware, are from general knowledge, not from tests here.
