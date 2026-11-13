import json
import numpy as np
import openvino as ov
from tokenizers import Tokenizer
from compare_gemma import GEMMA_QUERY, retrieval_metrics

def main():
    core = ov.Core()
    ov_model = core.read_model("models/gemma/model_quantized.onnx")
    compiled = core.compile_model(ov_model, "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "4"})
    output_layer = compiled.output("sentence_embedding") if "sentence_embedding" in [o.get_any_name() for o in compiled.outputs] else compiled.output(0)

    utts = json.load(open("data/test_utterances.json", encoding="utf-8"))
    texts = [u["text"] for u in utts]
    intents = [u["intent"] for u in utts]

    tok = Tokenizer.from_file("models/gemma/tokenizer.json")
    tok.enable_truncation(max_length=64)
    pad_id = tok.token_to_id("<pad>") or 0
    tok.enable_padding(pad_id=pad_id, pad_token="<pad>")

    enc = tok.encode_batch([GEMMA_QUERY + t for t in texts])
    inp = {
        "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)
    }

    res = compiled(inp)[output_layer]
    if len(res.shape) == 3:
        m = inp["attention_mask"][:, :, None].astype(np.float32)
        res = (res * m).sum(1) / np.clip(m.sum(1), 1e-9, None)

    norm_emb = res / np.clip(np.linalg.norm(res, axis=1, keepdims=True), 1e-12, None)
    metrics = retrieval_metrics(norm_emb, intents)
    print("OpenVINO Retrieval Metrics (Full 768-dim):")
    print(f"Recall@1: {metrics['Recall@1']*100:.2f}%")
    print(f"Recall@5: {metrics['Recall@5']*100:.2f}%")
    print(f"MRR:      {metrics['MRR']:.4f}")

if __name__ == "__main__":
    main()
