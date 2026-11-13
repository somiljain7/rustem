#!/usr/bin/env python3
"""
compare_gemma.py - Compare EmbeddingGemma-300M against multilingual-e5-small (FP32 / INT8)
on the same intent-retrieval dataset, plus CPU latency.

Metrics (identical protocol to eval.py): Recall@1, Recall@5, MRR on
data/test_utterances.json (leave-one-out nearest neighbour by intent), and
per-batch latency / throughput on CPU.

Prerequisites (NOT downloaded by this script automatically):
  * google/embeddinggemma-300m is a gated model: accept the Gemma terms on
    Hugging Face and `huggingface-cli login`, then export or download an ONNX
    build, e.g. onnx-community/embeddinggemma-300m-ONNX (onnx/model_quantized.onnx
    + tokenizer.json) into models/gemma/.
  * pip install onnxruntime tokenizers numpy

Usage:
  python scripts/compare_gemma.py \
      --gemma-model models/gemma/model_quantized.onnx \
      --gemma-tokenizer models/gemma/tokenizer.json \
      --out data/gemma_comparison.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# EmbeddingGemma prompt formats (from the model card).
GEMMA_QUERY = "task: search result | query: "
GEMMA_DOC = "title: none | text: "


class OnnxEmbedder:
    def __init__(self, name, model_path, tokenizer_path, query_prefix, max_len=128,
                 threads=2, truncate_dim=None):
        self.name = name
        self.query_prefix = query_prefix
        self.truncate_dim = truncate_dim
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.enable_truncation(max_length=max_len)
        pad_id = self.tok.token_to_id("<pad>")
        self.tok.enable_padding(pad_id=pad_id if pad_id is not None else 0, pad_token="<pad>")

    def encode(self, texts):
        enc = self.tok.encode_batch([self.query_prefix + t for t in texts])
        feeds = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}
        outs = self.session.run(None, feeds)
        by_name = dict(zip(self.output_names, outs))

        if "sentence_embedding" in by_name:  # already pooled (EmbeddingGemma ONNX export)
            emb = by_name["sentence_embedding"]
        else:  # last_hidden_state -> masked mean pool
            hs = outs[0]
            m = feeds["attention_mask"][:, :, None].astype(np.float32)
            emb = (hs * m).sum(1) / np.clip(m.sum(1), 1e-9, None)

        if self.truncate_dim:  # Matryoshka truncation
            emb = emb[:, : self.truncate_dim]
        return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)


def retrieval_metrics(emb, intents):
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    n = len(intents)
    r1 = r5 = 0
    rr = []
    for i in range(n):
        order = np.argsort(sim[i])[::-1]
        hits = [intents[j] == intents[i] for j in order]
        first = next((r for r, h in enumerate(hits, 1) if h), None)
        if first is None:
            continue
        r1 += hits[0]
        r5 += any(hits[:5])
        rr.append(1.0 / first)
    return {"Recall@1": r1 / n, "Recall@5": r5 / n, "MRR": float(np.mean(rr))}


def latency(embedder, texts, batch_sizes=(1, 8, 16, 64), iters=20, warmup=3):
    out = {}
    for bs in batch_sizes:
        batch = (texts * (bs // len(texts) + 1))[:bs]
        for _ in range(warmup):
            embedder.encode(batch)
        times = []
        for _ in range(iters):
            t = time.perf_counter()
            embedder.encode(batch)
            times.append((time.perf_counter() - t) * 1000)
        times = np.array(times)
        out[bs] = {
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "texts_per_s": float(bs / (times.mean() / 1000)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5-fp32", default="models/model.onnx")
    ap.add_argument("--e5-int8", default="models/model_quantized.onnx")
    ap.add_argument("--e5-tokenizer", default="models/tokenizer.json")
    ap.add_argument("--gemma-model", default="models/gemma/model_quantized.onnx")
    ap.add_argument("--gemma-tokenizer", default="models/gemma/tokenizer.json")
    ap.add_argument("--gemma-dim", type=int, default=None, help="Matryoshka truncation, e.g. 256")
    ap.add_argument("--data", default="data/test_utterances.json")
    ap.add_argument("--out", default="data/gemma_comparison.json")
    args = ap.parse_args()

    utts = json.load(open(args.data, encoding="utf-8"))
    texts = [u["text"] for u in utts]
    intents = [u["intent"] for u in utts]

    models = [
        OnnxEmbedder("e5-small FP32", args.e5_fp32, args.e5_tokenizer, "query: "),
        OnnxEmbedder("e5-small INT8", args.e5_int8, args.e5_tokenizer, "query: "),
        OnnxEmbedder("EmbeddingGemma-300M", args.gemma_model, args.gemma_tokenizer,
                     GEMMA_QUERY, truncate_dim=args.gemma_dim),
    ]

    results = {}
    for m in models:
        emb = m.encode(texts)
        results[m.name] = {
            "dim": int(emb.shape[1]),
            "model_mb": round(Path(m.session._model_path).stat().st_size / 1e6, 1),
            "quality": retrieval_metrics(emb, intents),
            "latency": latency(m, texts[:8]),
        }

    print(f"{'Model':24s} {'dim':>5s} {'MB':>7s} {'R@1':>7s} {'R@5':>7s} {'MRR':>7s} "
          f"{'bs1 p50':>9s} {'bs16 txt/s':>11s}")
    for name, r in results.items():
        q, l = r["quality"], r["latency"]
        print(f"{name:24s} {r['dim']:5d} {r['model_mb']:7.1f} {q['Recall@1']*100:6.2f}% "
              f"{q['Recall@5']*100:6.2f}% {q['MRR']:7.4f} {l[1]['p50_ms']:8.1f}ms "
              f"{l[16]['texts_per_s']:11.1f}")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
