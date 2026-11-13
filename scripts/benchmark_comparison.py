#!/usr/bin/env python3
"""
benchmark_comparison.py — Direct benchmark comparing:
1. Native Rust / ONNX Runtime Dynamic INT8
2. Qdrant FastEmbed (intfloat/multilingual-e5-small)
3. HuggingFace Sentence-Transformers (PyTorch FP32 CPU)
4. HuggingFace Transformers Pipeline (PyTorch FP32 CPU)
"""

import time
import json
import psutil
import numpy as np

SAMPLE_TEXTS = [
    "query: मुझे पिज़्ज़ा ऑर्डर करना है",
    "query: I want to order pizza",
    "query: mujhe pizza order karna hai",
    "query: मला पिझ्झा ऑर्डर करायचा आहे",
    "query: నాకు పిజ్జా ఆర్డర్ చేయాలి",
    "query: What is the weather today?",
    "query: आज मौसम कैसा है?",
    "query: play some music",
]

def benchmark_engine(name, encode_fn, batch_sizes=[1, 8, 16, 64], iterations=100, warmup=10):
    process = psutil.Process()
    print(f"\n=======================================================")
    print(f"BENCHMARK: {name}")
    print(f"=======================================================")
    print(f"{'Batch':<8}{'p50 (ms)':<12}{'p95 (ms)':<12}{'p99 (ms)':<12}{'Throughput (texts/s)':<22}{'Memory (MB)':<12}")
    print("-" * 78)

    results = []
    for bsz in batch_sizes:
        batch = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(bsz)]

        # Warmup
        for _ in range(warmup):
            _ = encode_fn(batch)

        latencies = []
        t0_total = time.perf_counter()

        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = encode_fn(batch)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        total_elapsed = time.perf_counter() - t0_total
        throughput = (bsz * iterations) / total_elapsed
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        mem_mb = process.memory_info().rss / (1024 * 1024)

        print(f"{bsz:<8}{p50:<12.2f}{p95:<12.2f}{p99:<12.2f}{throughput:<22.1f}{mem_mb:<12.1f}")
        results.append({
            "engine": name,
            "batch_size": bsz,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "throughput": round(throughput, 1),
            "memory_mb": round(mem_mb, 1)
        })
    return results

def main():
    all_results = {}

    # 1. FastEmbed
    try:
        from fastembed import TextEmbedding
        print("\nLoading FastEmbed model: intfloat/multilingual-e5-small...")
        fe_model = TextEmbedding("intfloat/multilingual-e5-small")
        fe_fn = lambda texts: list(fe_model.embed(texts))
        all_results["FastEmbed"] = benchmark_engine("FastEmbed (Python ONNX)", fe_fn)
    except Exception as e:
        print(f"FastEmbed benchmark failed: {e}")

    # 2. HuggingFace Sentence-Transformers (PyTorch FP32)
    try:
        from sentence_transformers import SentenceTransformer
        print("\nLoading SentenceTransformer: intfloat/multilingual-e5-small...")
        st_model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
        st_fn = lambda texts: st_model.encode(texts, batch_size=len(texts), show_progress_bar=False, normalize_embeddings=True)
        all_results["SentenceTransformers"] = benchmark_engine("Sentence-Transformers (PyTorch FP32)", st_fn)
    except Exception as e:
        print(f"SentenceTransformers benchmark failed: {e}")

    # Save summary
    with open("data/runtime_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
