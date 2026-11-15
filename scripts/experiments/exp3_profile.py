import os, tempfile
"""Experiment 3: per-operator time in ONNX Runtime CPU for Gemma INT8 at batch 1 and 16 (seq 24)."""
import json, collections, os
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "models", "gemma", "model_quantized.onnx")
OUT = tempfile.gettempdir()
rng = np.random.default_rng(0)


def profile(b, s, runs):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(OUT, f"prof_b{b}")
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    f = {"input_ids": rng.integers(5, 5000, size=(b, s), dtype=np.int64), "attention_mask": np.ones((b, s), dtype=np.int64)}
    for _ in range(3):
        sess.run(None, f)  # warmup (also profiled; discard by using only last `runs` via event order is hard, so average)
    for _ in range(runs):
        sess.run(None, f)
    path = sess.end_profiling()
    ev = json.load(open(path))
    per_op = collections.defaultdict(float)
    per_op_n = collections.Counter()
    session_us = []
    for e in ev:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            op = e["args"].get("op_name", "?")
            per_op[op] += e["dur"]
            per_op_n[op] += 1
        elif e.get("cat") == "Session" and e["name"] == "model_run":
            session_us.append(e["dur"])
    calls = len(session_us)
    total_kernel = sum(per_op.values()) / calls / 1000.0
    run_ms = sorted(session_us)[len(session_us) // 2] / 1000.0
    print(f"\n=== batch {b}, seq {s}: {calls} profiled runs, median model_run {run_ms:.1f} ms, sum of kernel times/run {total_kernel:.1f} ms ===")
    rows = sorted(per_op.items(), key=lambda kv: -kv[1])[:10]
    for op, us in rows:
        print(f"  {op:34s} {us / calls / 1000.0:8.2f} ms/run  {100 * us / sum(per_op.values()):5.1f}%  ({per_op_n[op] // calls} nodes)")


profile(1, 24, 10)
profile(16, 24, 5)
