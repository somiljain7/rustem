"""Experiment 5: achievable fp32 GEMM throughput on this CPU (numpy/OpenBLAS), 4 threads,
at Gemma-like shapes (tokens x hidden) @ (hidden x out)."""
import time, numpy as np

np.random.seed(0)


def gflops(m, k, n, reps=20):
    a = np.random.rand(m, k).astype(np.float32)
    b = np.random.rand(k, n).astype(np.float32)
    for _ in range(3):
        a @ b
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        a @ b
        t.append(time.perf_counter() - s)
    t.sort()
    return 2 * m * k * n / t[len(t) // 2] / 1e9


for name, (m, k, n) in {
    "large square 2048^3": (2048, 2048, 2048),
    "384 tokens x 768 -> 1152 (qkv-ish)": (384, 768, 1152),
    "384 tokens x 768 -> 1152*... 3072 (mlp up)": (384, 768, 3072),
    "384 tokens x 3072 -> 768 (mlp down)": (384, 3072, 768),
    "24 tokens x 768 -> 3072 (batch 1 mlp up)": (24, 768, 3072),
}.items():
    print(f"{name:46s} {gflops(m, k, n):7.1f} GFLOPS")
