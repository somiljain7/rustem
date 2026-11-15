// Matrix multiply C = A * B (all N x N, row-major float) in six versions.
// Same O(N^3) arithmetic every time. Only the constant factor changes.
//
//   1 naive     i-j-k loops                         (cache-hostile B access)
//   2 reorder   i-k-j loops                         (contiguous access)
//   3 blocked   tiles so the working set fits cache (cache blocking)
//   4 avx2      reorder + 8 floats per instruction  (SIMD + FMA)
//   5 regblock  4x16 block of C kept in registers   (what real kernels do)
//   6 threads   version 5 split across CPU cores    (multithreading)
//
// Build (MinGW/GCC):  g++ -O2 -mavx2 -mfma -fno-tree-vectorize -pthread matmul.cpp -o matmul
// Run:                matmul [N=1024] [reps=3]
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using Mat = std::vector<float>;
static int N;

// 1. naive: B[k*N + j] jumps a whole row (N floats) every step of k -> cache misses.
static void mm_naive(const float* A, const float* B, float* C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

// 2. reorder: the inner loop now walks B and C along a row (contiguous memory).
static void mm_reorder(const float* A, const float* B, float* C) {
    std::fill(C, C + (size_t)N * N, 0.f);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
}

// 3. blocked: work on BS x BS tiles so A, B, C tiles stay in L1/L2 while reused.
static void mm_blocked(const float* A, const float* B, float* C) {
    const int BS = 64;
    std::fill(C, C + (size_t)N * N, 0.f);
    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS)
                for (int i = ii; i < std::min(ii + BS, N); i++)
                    for (int k = kk; k < std::min(kk + BS, N); k++) {
                        float a = A[i * N + k];
                        for (int j = jj; j < std::min(jj + BS, N); j++) C[i * N + j] += a * B[k * N + j];
                    }
}

// 4. avx2: same loop order as 2, but 8 columns per instruction using fused multiply-add.
static void mm_avx2(const float* A, const float* B, float* C) {
    std::fill(C, C + (size_t)N * N, 0.f);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            __m256 a = _mm256_set1_ps(A[i * N + k]);  // broadcast one number to 8 lanes
            for (int j = 0; j < N; j += 8) {
                __m256 c = _mm256_loadu_ps(&C[i * N + j]);
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                _mm256_storeu_ps(&C[i * N + j], _mm256_fmadd_ps(a, b, c));  // c += a*b, 8 lanes
            }
        }
}

// 5. register blocking: a 4 x 16 piece of C lives in 8 AVX registers for a whole run of k,
//    so C is read and written once per piece instead of once per k. Needs N % 16 == 0.
static void rb_rows(const float* A, const float* B, float* C, int i0, int i1) {
    const int KC = 256;  // keep the slice of B being reused small enough for cache
    for (int kb = 0; kb < N; kb += KC) {
        int ke = std::min(kb + KC, N);
        for (int i = i0; i < i1; i += 4)
            for (int j = 0; j < N; j += 16) {
                __m256 c00 = _mm256_loadu_ps(&C[(i + 0) * N + j]), c01 = _mm256_loadu_ps(&C[(i + 0) * N + j + 8]);
                __m256 c10 = _mm256_loadu_ps(&C[(i + 1) * N + j]), c11 = _mm256_loadu_ps(&C[(i + 1) * N + j + 8]);
                __m256 c20 = _mm256_loadu_ps(&C[(i + 2) * N + j]), c21 = _mm256_loadu_ps(&C[(i + 2) * N + j + 8]);
                __m256 c30 = _mm256_loadu_ps(&C[(i + 3) * N + j]), c31 = _mm256_loadu_ps(&C[(i + 3) * N + j + 8]);
                for (int k = kb; k < ke; k++) {
                    __m256 b0 = _mm256_loadu_ps(&B[k * N + j]);
                    __m256 b1 = _mm256_loadu_ps(&B[k * N + j + 8]);
                    __m256 a;
                    a = _mm256_set1_ps(A[(i + 0) * N + k]); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
                    a = _mm256_set1_ps(A[(i + 1) * N + k]); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
                    a = _mm256_set1_ps(A[(i + 2) * N + k]); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
                    a = _mm256_set1_ps(A[(i + 3) * N + k]); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
                }
                _mm256_storeu_ps(&C[(i + 0) * N + j], c00); _mm256_storeu_ps(&C[(i + 0) * N + j + 8], c01);
                _mm256_storeu_ps(&C[(i + 1) * N + j], c10); _mm256_storeu_ps(&C[(i + 1) * N + j + 8], c11);
                _mm256_storeu_ps(&C[(i + 2) * N + j], c20); _mm256_storeu_ps(&C[(i + 2) * N + j + 8], c21);
                _mm256_storeu_ps(&C[(i + 3) * N + j], c30); _mm256_storeu_ps(&C[(i + 3) * N + j + 8], c31);
            }
    }
}
static void mm_regblock(const float* A, const float* B, float* C) {
    std::fill(C, C + (size_t)N * N, 0.f);
    rb_rows(A, B, C, 0, N);
}

// 6. threads: each thread takes its own band of rows of C (no sharing, no locks).
static int g_threads = 4;
static void mm_threads(const float* A, const float* B, float* C) {
    std::fill(C, C + (size_t)N * N, 0.f);
    std::vector<std::thread> ts;
    int rows = N / g_threads / 4 * 4;  // multiple of 4
    for (int t = 0; t < g_threads; t++) {
        int i0 = t * rows, i1 = (t == g_threads - 1) ? N : i0 + rows;
        ts.emplace_back(rb_rows, A, B, C, i0, i1);
    }
    for (auto& th : ts) th.join();
}

struct Version { const char* name; void (*fn)(const float*, const float*, float*); int max_n; };

int main(int argc, char** argv) {
    N = argc > 1 ? std::atoi(argv[1]) : 1024;
    int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    if (N % 16 != 0) { std::printf("N must be a multiple of 16\n"); return 1; }
    g_threads = (int)std::max(1u, std::min(4u, std::thread::hardware_concurrency()));  // 4 physical cores here

    Mat A((size_t)N * N), B((size_t)N * N), C((size_t)N * N), ref((size_t)N * N);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(0.f, 1.f);
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);

    std::vector<Version> vs = {
        {"1 naive (i-j-k)", mm_naive, 1 << 30}, {"2 reorder (i-k-j)", mm_reorder, 1 << 30},
        {"3 blocked tiles", mm_blocked, 1 << 30}, {"4 AVX2 + FMA", mm_avx2, 1 << 30},
        {"5 AVX2 register block", mm_regblock, 1 << 30}, {"6 + threads", mm_threads, 1 << 30}};

    double flops = 2.0 * N * N * N;
    std::printf("N=%d  (%.2f GFLOP per multiply), threads in v6: %d, best of %d reps\n\n", N, flops / 1e9, g_threads, reps);
    std::printf("%-24s %10s %10s %10s %8s\n", "version", "time ms", "GFLOPS", "vs naive", "max err");
    double base = 0;
    for (size_t v = 0; v < vs.size(); v++) {
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            auto t0 = std::chrono::steady_clock::now();
            vs[v].fn(A.data(), B.data(), C.data());
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            best = std::min(best, ms);
        }
        if (v == 0) { ref = C; base = best; }
        double err = 0;
        for (size_t i = 0; i < C.size(); i++) err = std::max(err, (double)std::fabs(C[i] - ref[i]) / std::max(1.0, (double)std::fabs(ref[i])));
        std::printf("%-24s %10.1f %10.2f %9.1fx %8.1e\n", vs[v].name, best, flops / best / 1e6, base / best, err);
    }
    std::printf("\n'max err' is relative to version 1. Float addition order differs between versions, so tiny non-zero values are expected.\n");
    return 0;
}
