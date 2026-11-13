#!/usr/bin/env python3
"""
plot_tsne_comparison.py — Generate high-resolution 2D t-SNE projection plots
comparing the semantic embedding spaces of:
1. Multilingual-E5-Small (FP32)
2. Multilingual-E5-Small (INT8 Quantized)
3. EmbeddingGemma-300M (INT8 Quantized, Full 768-dim)

Evaluates on the 202-utterance multilingual intent dataset across Indian languages
(Hindi, Marathi, Telugu, Tamil, Hinglish, English) covering distinct intent clusters.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from compare_gemma import OnnxEmbedder, GEMMA_QUERY

def main():
    print("[1/4] Loading test dataset...")
    utts = json.load(open("data/test_utterances.json", encoding="utf-8"))
    texts = [u["text"] for u in utts]
    intents = [u["intent"] for u in utts]
    unique_intents = sorted(list(set(intents)))
    intent_to_id = {intent: idx for idx, intent in enumerate(unique_intents)}
    intent_ids = np.array([intent_to_id[i] for i in intents])

    # Pick top 8 most frequent intents for clean visual representation
    from collections import Counter
    top_intents = [item[0] for item in Counter(intents).most_common(8)]
    mask = [i in top_intents for i in intents]
    filtered_texts = [t for t, m in zip(texts, mask) if m]
    filtered_intents = [i for i, m in zip(intents, mask) if m]

    print(f"Total utterances: {len(texts)} (Plotting top {len(top_intents)} intent clusters: {len(filtered_texts)} utterances)")

    # 1. Embed with e5-small FP32
    print("[2/4] Generating embeddings from e5-small FP32...")
    m_e5_fp32 = OnnxEmbedder("e5-small FP32", "models/model.onnx", "models/tokenizer.json", "query: ")
    emb_e5_fp32 = m_e5_fp32.encode(filtered_texts)

    # 2. Embed with e5-small INT8
    print("[2/4] Generating embeddings from e5-small INT8...")
    m_e5_int8 = OnnxEmbedder("e5-small INT8", "models/model_quantized.onnx", "models/tokenizer.json", "query: ")
    emb_e5_int8 = m_e5_int8.encode(filtered_texts)

    # 3. Embed with EmbeddingGemma-300M (Full 768-dim)
    print("[2/4] Generating embeddings from EmbeddingGemma-300M (768-dim)...")
    m_gemma = OnnxEmbedder("EmbeddingGemma-300M", "models/gemma/model_quantized.onnx", "models/gemma/tokenizer.json", GEMMA_QUERY)
    emb_gemma = m_gemma.encode(filtered_texts)

    # Compute t-SNE projections
    print("[3/4] Computing t-SNE 2D projections (perplexity=25, random_state=42)...")
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, max_iter=1500)
    tsne_e5_fp32 = tsne.fit_transform(emb_e5_fp32)
    tsne_e5_int8 = tsne.fit_transform(emb_e5_int8)
    tsne_gemma = tsne.fit_transform(emb_gemma)

    # Plot 3-panel figure
    print("[4/4] Rendering t-SNE comparison figure...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)

    # Color palette
    palette = sns.color_palette("tab10", len(top_intents))
    color_map = {intent: palette[i] for i, intent in enumerate(top_intents)}

    models_data = [
        ("Multilingual-E5-Small (FP32, 384d)", tsne_e5_fp32, axes[0]),
        ("Multilingual-E5-Small (INT8, 384d)", tsne_e5_int8, axes[1]),
        ("EmbeddingGemma-300M (INT8, 768d)", tsne_gemma, axes[2]),
    ]

    for title, coords, ax in models_data:
        for intent in top_intents:
            idx = [i for i, val in enumerate(filtered_intents) if val == intent]
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                label=intent.replace("_", " "),
                color=color_map[intent],
                s=55,
                alpha=0.85,
                edgecolors="none"
            )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Add shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=10,
        frameon=True
    )

    plt.suptitle("Cross-Lingual Intent Semantic Space t-SNE Comparison\n(Hindi, Marathi, Telugu, Tamil, Hinglish, English)", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = r"C:\Users\somil\.gemini\antigravity-cli\brain\4c30425c-dd40-4d84-bf9c-eaeed4dd2a60\tsne_models_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"t-SNE comparison plot successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
