# Matrix multiply, six ways

One C++ file, [matmul.cpp](matmul.cpp), multiplies two N x N matrices in six versions. The arithmetic is identical every time (O(N^3), 2 x N^3 floating-point operations), and only the **constant factor** changes. This is the same game the ONNX Runtime and OpenVINO kernels play, in a form you can read in one sitting.

## The six versions

| # | Version | The idea | DSA analogy |
|---|---|---|---|
| 1 | naive (i-j-k) | Textbook triple loop. The inner loop reads `B[k][j]`, jumping a whole row each step | Iterating a 2D array column-wise |
| 2 | reorder (i-k-j) | Swap two loops so the inner loop walks along a row | Loop reordering for cache locality |
| 3 | blocked tiles | Process 64 x 64 tiles so data stays in cache | Cache blocking |
| 4 | AVX2 + FMA | 8 floats per instruction with fused multiply-add | Processing 8 elements per word, like bitsets |
| 5 | AVX2 register block | A 4 x 16 block of C lives in 8 vector registers for a whole run of k | Keeping hot variables in registers |
| 6 | + threads | Each of 4 threads takes a band of rows | Divide and conquer across cores |

## Build and run
Needs a C++ compiler with AVX2 (g++ from MinGW/WinLibs, or any GCC/Clang). From this folder:

```bash
g++ -O2 -mavx2 -mfma -fno-tree-vectorize -pthread -static matmul.cpp -o matmul
./matmul 1024 3        # N=1024, best of 3 repetitions
```

- `-mavx2 -mfma` lets the code use AVX2 and FMA instructions (the same as `#pragma GCC target("avx2,fma")` in contest code).
- `-fno-tree-vectorize` stops the compiler from vectorizing versions 1 to 3, so you see the *hand-written* SIMD effect. Rebuild with `-O3` (and without that flag) to see what the compiler does by itself.
- `-static` bundles the runtime so the program runs without MinGW on PATH.
- N must be a multiple of 16.

## Results (i5-9300H, 4 cores, AVX2 + FMA, 2026-11-16)

**N = 1024** (2.15 GFLOP per multiply, best of 3), `-O2`, auto-vectorization off:

| Version | Time | GFLOPS | Speed-up vs naive |
|---|---|---|---|
| 1 naive | 2122 ms | 1.0 | 1.0x |
| 2 reorder | 610 ms | 3.5 | 3.5x |
| 3 blocked tiles | 593 ms | 3.6 | 3.6x |
| 4 AVX2 + FMA | 131 ms | 16.4 | 16x |
| 5 AVX2 register block | 51 ms | 41.8 | 41x |
| 6 + 4 threads | **22 ms** | **96.8** | **96x** |

**N = 2048** (17.2 GFLOP, 1 rep): 74,478 ms naive (0.23 GFLOPS) down to **225 ms** threaded (76 GFLOPS), a 331x speed-up.

**`-O3` with auto-vectorization on (N = 1024):** version 2 jumps from 610 ms to **131 ms**, essentially the same as my hand-written AVX2 (126 ms). Versions 5 and 6 are unchanged (56 ms and 22 ms), since they were already hand-vectorized.

All versions agree with version 1 to a relative error of about 3.5e-07. Float addition order differs, so tiny differences are normal.

## What this teaches

1. **Memory access order alone gave 3.5x** (version 1 to 2) with zero new instructions. Big-O never changed.
2. **The compiler can do the SIMD for you** when the loop is simple. `-O3` turned the plain i-k-j loop into AVX2 code as fast as my intrinsics. Hand-written SIMD only pays off when the structure is harder, as in version 5.
3. **Register blocking was the next big step** (3x over plain AVX2). The simple AVX2 version reloads and stores C on every step of k; the register-blocked one does that once per block. The limit is memory traffic, not arithmetic.
4. **Threads gave 2.3x on 4 cores**, not 4x. Cores share memory bandwidth and cache, so scaling is below linear.
5. **Where the real libraries go next:** about 200 GFLOPS (measured with numpy/OpenBLAS), roughly twice this demo. See `../KERNEL_LIBRARIES.md`.

## Honest notes
- **Blocking did not help as written.** Version 3 matched version 2 at N = 1024 and was a bit slower at N = 2048 (6019 vs 5445 ms), and with `-O3` it is slower than version 2. I did not investigate why. Plausible reasons: row access in version 2 was already contiguous, and the tile loop bounds block the compiler's vectorization. Real libraries also *pack* tiles into a contiguous buffer, which this version does not.
- Timings come from one machine and few repetitions (N = 2048 used one). Treat the ratios as rough.
- Version 5 uses a fixed 4 x 16 block and strided loads of B. A real kernel would pack B first.
