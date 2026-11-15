"""Gemma prompt-prefix sweep: retrieval quality (Recall@1/5, MRR) vs token count and batch-1 latency.
usage: python exp6_prefix.py <model.onnx> [tokenizer.json]
Run it on the pre-dequantized model (scripts/fold_dequantize.py) so latency reflects real compute."""
import json, sys, time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, __file__.rsplit("experiments", 1)[0])
from compare_gemma import retrieval_metrics

MODEL = sys.argv[1]
TOK = sys.argv[2] if len(sys.argv) > 2 else "models/gemma/tokenizer.json"
PREFIXES = {
    "full (model card)": "task: search result | query: ",
    "task: search | ": "task: search | ",
    "query: ": "query: ",
    "none": "",
}

utts = json.load(open("data/test_utterances.json", encoding="utf-8"))
texts = [u["text"] for u in utts]
intents = [u["intent"] for u in utts]

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
import os
if os.environ.get("EP") == "openvino":
    providers = [("OpenVINOExecutionProvider", {"device_type": "CPU", "num_of_threads": 4}), "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]
sess = ort.InferenceSession(MODEL, opts, providers=providers)
print("providers in use:", sess.get_providers(), flush=True)
names = [o.name for o in sess.get_outputs()]
tok = Tokenizer.from_file(TOK)
tok.enable_truncation(max_length=128)
tok.enable_padding(pad_id=tok.token_to_id("<pad>") or 0, pad_token="<pad>")


def embed(batch):
    enc = tok.encode_batch(batch)
    feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
             "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
    outs = dict(zip(names, sess.run(None, feeds)))
    e = outs["sentence_embedding"]
    return e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None), enc


print(f"{'prefix':20s} {'avg tok':>8s} {'R@1':>7s} {'R@5':>7s} {'MRR':>7s} {'bs1 p50':>9s}")
for label, pre in PREFIXES.items():
    embs = []
    toks = []
    for i in range(0, len(texts), 16):
        e, enc = embed([pre + t for t in texts[i:i + 16]])
        embs.append(e)
    emb = np.vstack(embs)
    for t in texts:
        toks.append(sum(tok.encode(pre + t).attention_mask))
    q = retrieval_metrics(emb, intents)
    sample = [pre + t for t in texts[:24]]
    for s in sample[:3]:
        embed([s])
    lat = []
    for s in sample * 3:
        t0 = time.perf_counter()
        embed([s])
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"{label:20s} {np.mean(toks):8.1f} {q['Recall@1']*100:6.2f}% {q['Recall@5']*100:6.2f}% "
          f"{q['MRR']:7.4f} {np.percentile(lat, 50):8.1f}ms")
