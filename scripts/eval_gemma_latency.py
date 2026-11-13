import json
import time
from compare_gemma import OnnxEmbedder, latency, GEMMA_QUERY

def main():
    utts = json.load(open("data/test_utterances.json", encoding="utf-8"))
    texts = [u["text"] for u in utts]

    for d in [768, 256, 128]:
        m = OnnxEmbedder(f"Gemma-{d}", "models/gemma/model_quantized.onnx", "models/gemma/tokenizer.json", GEMMA_QUERY, truncate_dim=d)
        lat = latency(m, texts[:8], batch_sizes=(1, 16), iters=8, warmup=2)
        p50 = lat[1]["p50_ms"]
        tp16 = lat[16]["texts_per_s"]
        print(f"Gemma Dim={d}: batch=1 p50 = {p50:.1f} ms | batch=16 throughput = {tp16:.1f} texts/s")

if __name__ == "__main__":
    main()
