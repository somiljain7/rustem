#!/usr/bin/env python3
"""
export_onnx.py — Export intfloat/multilingual-e5-small to ONNX (fp32 + int8).

Downloads the model from HuggingFace, exports to ONNX with dynamic axes,
applies dynamic int8 quantization, and saves both models + tokenizer.

Output layout (relative to this script):
    ../models/model_fp32.onnx
    ../models/model_int8.onnx
    ../models/tokenizer.json
    ../models/config.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export multilingual-e5-small to ONNX")
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"HuggingFace model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save ONNX models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max sequence length for dummy input (default: 512)",
    )
    return parser.parse_args()


def download_model(model_name: str):
    """Download and return model + tokenizer from HuggingFace."""
    print(f"[1/5] Downloading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    print(f"       Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def export_onnx_fp32(
    model: torch.nn.Module,
    tokenizer,
    output_path: Path,
    opset: int,
    max_seq_len: int,
) -> None:
    """Export PyTorch model to ONNX fp32 with dynamic batch and sequence axes."""
    print(f"[2/5] Exporting ONNX fp32 → {output_path}")

    # Create dummy input
    dummy_text = "query: hello world"
    encoded = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get(
        "token_type_ids", torch.zeros_like(input_ids)
    )

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
    }

    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        str(output_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"       FP32 model size: {size_mb:.1f} MB ✓")


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """Apply dynamic INT8 quantization to the ONNX model."""
    print(f"[3/5] Quantizing to INT8 → {int8_path}")

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,  # Better for non-VNNI CPUs like i5-9300H
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

    size_mb = int8_path.stat().st_size / (1024 * 1024)
    print(f"       INT8 model size: {size_mb:.1f} MB ✓")


def save_tokenizer(tokenizer, output_dir: Path) -> None:
    """Save tokenizer.json and config to the output directory."""
    print(f"[4/5] Saving tokenizer → {output_dir}")

    # Save using HuggingFace's save method (creates tokenizer.json etc.)
    tokenizer.save_pretrained(str(output_dir))

    # Verify tokenizer.json exists
    tokenizer_json = output_dir / "tokenizer.json"
    if tokenizer_json.exists():
        size_kb = tokenizer_json.stat().st_size / 1024
        print(f"       tokenizer.json: {size_kb:.0f} KB ✓")
    else:
        print("       ⚠ tokenizer.json not found, saving manually")
        tokenizer.backend_tokenizer.save(str(tokenizer_json))


def verify_onnx(fp32_path: Path, int8_path: Path, tokenizer) -> None:
    """Run both models and compare outputs to ensure correctness."""
    print("[5/5] Verifying ONNX models...")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4

    test_texts = [
        "query: I want to order pizza",
        "query: मुझे पिज़्ज़ा ऑर्डर करना है",
        "query: mujhe pizza order karna hai",
    ]

    encoded = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="np",
    )

    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
        "token_type_ids": encoded.get(
            "token_type_ids",
            np.zeros_like(encoded["input_ids"]),
        ).astype(np.int64),
    }

    # Run fp32
    fp32_sess = ort.InferenceSession(str(fp32_path), sess_opts)
    fp32_out = fp32_sess.run(None, inputs)[0]

    # Mean pooling (with attention mask)
    mask = inputs["attention_mask"]
    fp32_emb = (fp32_out * mask[:, :, np.newaxis]).sum(axis=1) / mask.sum(
        axis=1, keepdims=True
    )

    # Run int8
    int8_sess = ort.InferenceSession(str(int8_path), sess_opts)
    int8_out = int8_sess.run(None, inputs)[0]
    int8_emb = (int8_out * mask[:, :, np.newaxis]).sum(axis=1) / mask.sum(
        axis=1, keepdims=True
    )

    # Cosine similarity between fp32 and int8
    for i, text in enumerate(test_texts):
        cos_sim = np.dot(fp32_emb[i], int8_emb[i]) / (
            np.linalg.norm(fp32_emb[i]) * np.linalg.norm(int8_emb[i])
        )
        label = text[:50]
        print(f"       {label:50s} fp32↔int8 cos_sim = {cos_sim:.6f}")

    # Cross-lingual: pizza texts should be similar
    cos_en_hi = np.dot(fp32_emb[0], fp32_emb[1]) / (
        np.linalg.norm(fp32_emb[0]) * np.linalg.norm(fp32_emb[1])
    )
    print(f"       EN↔HI pizza similarity (fp32): {cos_en_hi:.4f}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = output_dir / "model_fp32.onnx"
    int8_path = output_dir / "model_int8.onnx"

    # Step 1: Download
    model, tokenizer = download_model(args.model_name)

    # Step 2: Export fp32
    export_onnx_fp32(model, tokenizer, fp32_path, args.opset, args.max_seq_len)

    # Step 3: Quantize to int8
    quantize_int8(fp32_path, int8_path)

    # Step 4: Save tokenizer
    save_tokenizer(tokenizer, output_dir)

    # Step 5: Verify
    verify_onnx(fp32_path, int8_path, tokenizer)

    print("\n✅ Export complete!")
    print(f"   FP32: {fp32_path}")
    print(f"   INT8: {int8_path}")
    print(f"   Tokenizer: {output_dir / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
