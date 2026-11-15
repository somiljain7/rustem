import os, tempfile
"""Experiment 1 (Python side): same model, tokenizer, prefix, padding, ORT settings as the Rust
`bench` command (1 session, 4 intra threads, full graph optimisation, CPU provider).
Reports p50 of: tokenize, session.run, post-processing, total."""
import sys, time, statistics
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREFIX = "task: search result | query: "
TEXTS = [
    "I want to order pizza", "मुझे पिज़्ज़ा ऑर्डर करना है", "mujhe pizza order karna hai",
    "What is the weather today?", "आज मौसम कैसा है?", "play some music",
    "set an alarm for 7am", "book a cab to airport",
]
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100
warm = int(sys.argv[2]) if len(sys.argv) > 2 else 10

tok = Tokenizer.from_file(rf"{ROOT}\models\gemma\tokenizer.json")
tok.enable_truncation(max_length=128)
tok.no_padding()
pad_id = tok.token_to_id("<pad>") or 0

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1
sess = ort.InferenceSession(rf"{ROOT}\models\gemma\model_quantized.onnx", so, providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]
pooled = names.index("sentence_embedding")


def embed(batch):
    t0 = time.perf_counter()
    enc = tok.encode_batch([PREFIX + t for t in batch], add_special_tokens=True)
    n = max(len(e.ids) for e in enc)
    ids = np.full((len(enc), n), pad_id, dtype=np.int64)
    mask = np.zeros((len(enc), n), dtype=np.int64)
    for i, e in enumerate(enc):
        ids[i, : len(e.ids)] = e.ids
        mask[i, : len(e.attention_mask)] = e.attention_mask
    t1 = time.perf_counter()
    outs = sess.run(None, {"input_ids": ids, "attention_mask": mask})
    t2 = time.perf_counter()
    e = outs[pooled]
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    _ = e.tolist()  # Rust returns Vec<Vec<f32>>; make the copy comparable
    t3 = time.perf_counter()
    return (t3 - t0) * 1e3, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3


for bs in (1, 16):
    batch = [TEXTS[i % len(TEXTS)] for i in range(bs)]
    for _ in range(warm):
        embed(batch)
    rows = [embed(batch) for _ in range(iters)]
    med = lambda k: statistics.median(r[k] for r in rows)
    print(f"python batch {bs:>2}: p50 total {med(0):.2f} ms = tokenize {med(1):.2f} + inference {med(2):.2f} + other {med(3):.2f}")
