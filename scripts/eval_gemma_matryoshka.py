import json
import time
from pathlib import Path
import numpy as np
from compare_gemma import OnnxEmbedder, retrieval_metrics, latency, GEMMA_QUERY

def main():
    utts = json.load(open("data/test_utterances.json", encoding="utf-8"))
    texts = [u["text"] for u in utts]
    intents = [u["intent"] for u in utts]

    dims = [768, 512, 256, 128]
    print(f"=== Testing EmbeddingGemma Matryoshka Dimensions: {dims} ===")

    m = OnnxEmbedder("EmbeddingGemma-300M", "models/gemma/model_quantized.onnx", "models/gemma/tokenizer.json", GEMMA_QUERY)

    # Encode raw unnormalized
    enc = m.tok.encode_batch([m.query_prefix + t for t in texts])
    feeds = {
        "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
    }
    feeds = {k: v for k, v in feeds.items() if k in m.input_names}
    outs = m.session.run(None, feeds)
    by_name = dict(zip(m.output_names, outs))
    raw_emb = by_name.get("sentence_embedding", outs[0])

    print(f"{'Dimension':<12} {'Recall@1':<12} {'Recall@5':<12} {'MRR':<12} {'Sim Time (1000q)':<18}")
    print("-" * 66)

    results = {}
    for d in dims:
        sub_emb = raw_emb[:, :d]
        norm_emb = sub_emb / np.clip(np.linalg.norm(sub_emb, axis=1, keepdims=True), 1e-12, None)
        metrics = retrieval_metrics(norm_emb, intents)

        # Vector similarity search benchmark (cosine matching over 202 utterances)
        q = norm_emb[0:1]
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = q @ norm_emb.T
        sim_time_ms = (time.perf_counter() - t0) * 1000

        r1 = metrics["Recall@1"] * 100
        r5 = metrics["Recall@5"] * 100
        mrr = metrics["MRR"]
        print(f"{d:<12} {r1:<12.2f}% {r5:<12.2f}% {mrr:<12.4f} {sim_time_ms:<18.2f} ms")
        results[d] = {
            "dim": d,
            "Recall@1": r1,
            "Recall@5": r5,
            "MRR": mrr,
            "sim_search_time_ms": round(sim_time_ms, 2)
        }

    with open("data/gemma_matryoshka_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
