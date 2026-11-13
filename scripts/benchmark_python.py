#!/usr/bin/env python3
"""
benchmark_python.py — Python baseline benchmark for text embeddings.

Measures:
- Latency (p50, p95, p99 in ms) for batch sizes 1, 8, 16, 64
- Throughput (texts/sec)
- CPU utilization & memory consumption (RSS)
- Compares ONNX Runtime (CPU), FastEmbed (if available), and Sentence-Transformers (if available)
"""

import os
import sys
import time
import argparse
import random
import psutil
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Python baseline benchmark")
    parser.add_argument("--model-path", type=str, default="models/model_quantized.onnx", help="Path to int8 ONNX model")
    parser.add_argument("--tokenizer-path", type=str, default="models/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--iterations", type=int, default=1000, help="Benchmark iterations (default: 1000)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations (default: 50)")
    return parser.parse_args()

SAMPLE_TEXTS = [
    "query: मुझे पिज़्ज़ा ऑर्डर करना है",
    "query: I want to order pizza",
    "query: mujhe pizza order karna hai",
    "query: मला पिझ्झा ऑर्डर करायचा आहे",
    "query: నాకు పిజ్జా ఆర్డర్ చేయాలి",
    "query: What is the weather today?",
    "query: आज मौसम कैसा है?",
    "query: aaj mausam kaisa hai?",
    "query: play some relaxing music",
    "query: set an alarm for 7 AM",
    "query: book a cab to the railway station",
    "query: check my bank account balance",
    "query: transfer five hundred rupees to Rahul",
    "query: turn off the bedroom lights",
    "query: tell me the latest news headline",
    "query: how do I get to Connaught Place?",
]

def benchmark_onnx(model_path, tokenizer_path, batch_sizes, iterations, warmup):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")

    process = psutil.Process()
    results = []

    print(f"\nBenchmarking ONNX Runtime (int8, 2 threads):")
    print(f"{'Batch Size':<12}{'p50 (ms)':<12}{'p95 (ms)':<12}{'p99 (ms)':<12}{'Throughput (texts/s)':<22}{'Memory (MB)':<12}{'CPU %':<10}")
    print("-" * 88)

    for bsz in batch_sizes:
        # Construct batch
        batch = [random.choice(SAMPLE_TEXTS) for _ in range(bsz)]
        
        # Tokenize ahead to measure raw inference or include tokenization
        encodings = tokenizer.encode_batch(batch)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        model_inputs = [i.name for i in session.get_inputs()]
        if "token_type_ids" in model_inputs:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        # Warmup
        for _ in range(warmup):
            _ = session.run(None, inputs)

        latencies = []
        cpu_samples = []
        start_time = time.perf_counter()

        # Run benchmark
        for i in range(iterations):
            t0 = time.perf_counter()
            _ = session.run(None, inputs)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if i % 100 == 0:
                cpu_samples.append(process.cpu_percent())

        total_elapsed = time.perf_counter() - start_time
        total_texts = bsz * iterations
        throughput = total_texts / total_elapsed

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        mem_mb = process.memory_info().rss / (1024 * 1024)
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0.0

        print(f"{bsz:<12}{p50:<12.2f}{p95:<12.2f}{p99:<12.2f}{throughput:<22.1f}{mem_mb:<12.1f}{avg_cpu:<10.1f}")
        results.append({
            "batch_size": bsz,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "throughput": throughput,
            "mem_mb": mem_mb,
            "cpu_percent": avg_cpu,
        })

    return results

def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)

    batch_sizes = [1, 8, 16, 64]
    benchmark_onnx(args.model_path, args.tokenizer_path, batch_sizes, args.iterations, args.warmup)

if __name__ == "__main__":
    main()
