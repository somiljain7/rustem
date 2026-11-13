#!/usr/bin/env python3
"""
eval.py — Quality evaluation suite for multilingual-e5-small embeddings.

Checks:
1. Fidelity: Cosine similarity between fp32 reference and int8 ONNX (Target: mean >= 0.99, min >= 0.98)
2. Heatmap: Cross-lingual similarity heatmap (Devanagari Hindi, Marathi, Telugu, Tamil, Hinglish, English)
3. Nearest Neighbours: Semantic sanity check for multilingual paraphrase / translation queries
4. Retrieval Metrics: Recall@1, Recall@5, and MRR on 200+ multi-intent voice utterances
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Ensure matplotlib runs headlessly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate embedding quality and cross-lingual intent retrieval")
    parser.add_argument("--model-fp32", type=str, default="models/model.onnx", help="Path to fp32 ONNX model")
    parser.add_argument("--model-int8", type=str, default="models/model_quantized.onnx", help="Path to int8 ONNX model")
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--test-data", type=str, default="data/test_utterances.json", help="Path to intent utterances dataset")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for heatmap & report")
    parser.add_argument("--rust-embeddings", type=str, default=None, help="Optional JSON file of Rust-generated embeddings")
    return parser.parse_args()


class ONNXEmbedder:
    """Lightweight ONNX Runtime inference engine with masked mean pooling and L2 normalization."""
    def __init__(self, model_path: str, tokenizer_path: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=0, pad_token="<pad>")

    def encode(self, texts, is_query=True):
        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [f"{prefix}{t}" for t in texts]

        encodings = self.tokenizer.encode_batch(prefixed_texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # Filter only expected inputs in case token_type_ids is not present in model
        model_inputs = [i.name for i in self.session.get_inputs()]
        inputs = {k: v for k, v in inputs.items() if k in model_inputs}

        outputs = self.session.run(None, inputs)
        last_hidden_state = outputs[0]  # [B, S, D]

        # Masked mean pooling
        expanded_mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)  # [B, S, 1]
        sum_embeddings = np.sum(last_hidden_state * expanded_mask, axis=1)  # [B, D]
        sum_mask = np.clip(np.sum(expanded_mask, axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        # L2 Normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        normalized = pooled / norms
        return normalized


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise or matrix cosine similarity."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return np.dot(a, b.T)


def run_fidelity_check(embedder_fp32, embedder_int8, texts):
    """Compare fp32 vs int8 embeddings cosine fidelity."""
    print("\n" + "=" * 60)
    print("1. FIDELITY CHECK: FP32 Reference vs INT8 Quantized")
    print("=" * 60)

    emb_fp32 = embedder_fp32.encode(texts, is_query=True)
    emb_int8 = embedder_int8.encode(texts, is_query=True)

    similarities = np.sum(emb_fp32 * emb_int8, axis=1)
    mean_sim = float(np.mean(similarities))
    min_sim = float(np.min(similarities))
    max_sim = float(np.max(similarities))
    p50_sim = float(np.percentile(similarities, 50))
    p05_sim = float(np.percentile(similarities, 5))

    print(f"Sample count:   {len(texts)}")
    print(f"Mean Cosine:    {mean_sim:.5f} (Target: >= 0.990)")
    print(f"Min Cosine:     {min_sim:.5f} (Target: >= 0.980)")
    print(f"P05 Cosine:     {p05_sim:.5f}")
    print(f"P50 Cosine:     {p50_sim:.5f}")
    print(f"Max Cosine:     {max_sim:.5f}")

    passed = (mean_sim >= 0.99) and (min_sim >= 0.98)
    status = "PASSED" if passed else "FAILED / INVESTIGATE"
    print(f"Status:         [{status}]")
    return {"mean": mean_sim, "min": min_sim, "passed": passed}


def generate_multilingual_heatmap(embedder, output_path: Path):
    """Generate and save cross-lingual similarity heatmap."""
    print("\n" + "=" * 60)
    print("2. MULTILINGUAL SIMILARITY HEATMAP")
    print("=" * 60)

    # Multilingual clusters covering 4 intent groups across 6 language forms
    clusters = [
        # Cluster A: Food order
        ("Hindi: Pizza", "मुझे पिज़्ज़ा ऑर्डर करना है"),
        ("English: Pizza", "I want to order pizza"),
        ("Hinglish: Pizza", "mujhe pizza order karna hai"),
        ("Marathi: Pizza", "मला पिझ्झा ऑर्डर करायचा आहे"),
        ("Telugu: Pizza", "నాకు పిజ్జా ఆర్డర్ చేయాలి"),
        ("Tamil: Pizza", "எனக்கு பீட்சா ஆர்டர் செய்ய வேண்டும்"),
        # Cluster B: Weather query
        ("English: Weather", "What is the weather today?"),
        ("Hindi: Weather", "आज मौसम कैसा है?"),
        ("Hinglish: Weather", "aaj mausam kaisa hai?"),
        ("Marathi: Weather", "आज हवामान कसे आहे?"),
        # Cluster C: Joke request
        ("English: Joke", "Tell me a joke"),
        ("Hindi: Joke", "मुझे एक चुटकुला सुनाओ"),
        ("Hinglish: Joke", "mujhe ek joke sunao"),
        # Cluster D: Ride booking
        ("English: Cab", "Book a cab to the airport"),
        ("Hindi: Cab", "एयरपोर्ट के लिए कैब बुक करो"),
        ("Hinglish: Cab", "airport ke liye cab book karo"),
    ]

    labels = [c[0] for c in clusters]
    texts = [c[1] for c in clusters]

    embeddings = embedder.encode(texts, is_query=True)
    sim_matrix = cosine_sim(embeddings, embeddings)

    plt.figure(figsize=(11, 9))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        vmin=0.70,
        vmax=1.00,
        cbar_kws={"label": "Cosine Similarity"}
    )
    plt.title("Cross-Lingual Intent Semantic Similarity Heatmap (multilingual-e5-small INT8)", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Heatmap successfully saved to: {output_path}")


def nearest_neighbor_sanity(embedder, sample_queries):
    """Sanity check top nearest neighbors for test utterances."""
    print("\n" + "=" * 60)
    print("3. NEAREST-NEIGHBOR MULTILINGUAL SANITY CHECK")
    print("=" * 60)

    embeddings = embedder.encode(sample_queries, is_query=True)
    sim_matrix = cosine_sim(embeddings, embeddings)

    for i, query in enumerate(sample_queries[:6]):
        scores = sim_matrix[i].copy()
        scores[i] = -1.0  # exclude self
        top_k_indices = np.argsort(scores)[::-1][:3]
        print(f"\nQuery: '{query}'")
        for rank, idx in enumerate(top_k_indices, 1):
            print(f"  Top {rank} [{scores[idx]:.4f}]: '{sample_queries[idx]}'")


def evaluate_retrieval(embedder_fp32, embedder_int8, utterances):
    """Compute Recall@1, Recall@5, and MRR for intent matching."""
    print("\n" + "=" * 60)
    print("4. RETRIEVAL METRICS (Recall@1, Recall@5, MRR)")
    print("=" * 60)

    texts = [u["text"] for u in utterances]
    intents = [u["intent"] for u in utterances]
    n = len(texts)

    results = {}
    for name, embedder in [("FP32 Reference", embedder_fp32), ("INT8 Quantized", embedder_int8)]:
        embeddings = embedder.encode(texts, is_query=True)
        sim_matrix = cosine_sim(embeddings, embeddings)
        np.fill_diagonal(sim_matrix, -np.inf)  # exclude self-match

        recall_1 = 0
        recall_5 = 0
        reciprocal_ranks = []

        for i in range(n):
            target_intent = intents[i]
            sorted_indices = np.argsort(sim_matrix[i])[::-1]

            # Find matching intents
            matches = [intents[idx] == target_intent for idx in sorted_indices]

            if matches[0]:
                recall_1 += 1
            if any(matches[:5]):
                recall_5 += 1

            # First rank index (1-based)
            first_hit = next(rank for rank, hit in enumerate(matches, 1) if hit)
            reciprocal_ranks.append(1.0 / first_hit)

        r1 = recall_1 / n
        r5 = recall_5 / n
        mrr = float(np.mean(reciprocal_ranks))

        results[name] = {"Recall@1": r1, "Recall@5": r5, "MRR": mrr}
        print(f"\n--- {name} ---")
        print(f"Total utterances: {n}")
        print(f"Recall@1:         {r1 * 100:.2f}%")
        print(f"Recall@5:         {r5 * 100:.2f}%")
        print(f"MRR:              {mrr:.4f}")

    # Compute drop
    r1_gap = (results["FP32 Reference"]["Recall@1"] - results["INT8 Quantized"]["Recall@1"]) * 100
    mrr_gap = (results["FP32 Reference"]["MRR"] - results["INT8 Quantized"]["MRR"])
    print("\n--- INT8 vs FP32 Delta ---")
    print(f"Recall@1 Gap:     {r1_gap:+.2f}% (Requirement: < 1.0% gap)")
    print(f"MRR Gap:          {mrr_gap:+.4f}")
    if abs(r1_gap) <= 1.0:
        print("Status:           [ACCEPTABLE - Degradation is within <1% threshold]")
    else:
        print("Status:           [ATTENTION - Degradation exceeds 1%]")

    return results


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing embedders for quality evaluation...")
    if not os.path.exists(args.model_fp32) or not os.path.exists(args.model_int8):
        print(f"Models not found at {args.model_fp32} or {args.model_int8}.")
        print("Please ensure ONNX files are downloaded or exported into the models directory.")
        sys.exit(1)

    embedder_fp32 = ONNXEmbedder(args.model_fp32, args.tokenizer)
    embedder_int8 = ONNXEmbedder(args.model_int8, args.tokenizer)

    with open(args.test_data, "r", encoding="utf-8") as f:
        utterances = json.load(f)

    test_texts = [u["text"] for u in utterances]

    # 1. Fidelity
    fidelity_res = run_fidelity_check(embedder_fp32, embedder_int8, test_texts)

    # 2. Multilingual Heatmap
    heatmap_path = out_dir / "similarity_heatmap.png"
    generate_multilingual_heatmap(embedder_int8, heatmap_path)

    # 3. Nearest Neighbour Sanity
    nearest_neighbor_sanity(embedder_int8, test_texts)

    # 4. Retrieval metrics
    evaluate_retrieval(embedder_fp32, embedder_int8, utterances)

    print("\nEvaluation successfully completed.")


if __name__ == "__main__":
    main()
