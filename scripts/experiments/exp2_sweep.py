import os, tempfile
"""Experiments 2 and 4: latency vs (batch, seq) sweep and config sweep on the Gemma INT8 ONNX.
Synthetic token ids (forward pass only, no tokenizer).
usage: python exp2_sweep.py <ort|ov> <sweep|cfg>"""
import sys, time, statistics
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "models", "gemma", "model_quantized.onnx")
engine, mode = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)


def make_runner(**cfg):
    if engine == "ort":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = cfg.get("threads", 4)
        so.inter_op_num_threads = 1
        if cfg.get("nospin"):
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        s = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
        return lambda f: s.run(None, f)
    import openvino as ov
    core = ov.Core()
    props = {"PERFORMANCE_HINT": cfg.get("hint", "LATENCY"), "INFERENCE_NUM_THREADS": str(cfg.get("threads", 4))}
    if "streams" in cfg:
        props["NUM_STREAMS"] = str(cfg["streams"])
    c = core.compile_model(core.read_model(MODEL), "CPU", props)
    return lambda f: c(f)


def feeds(b, s):
    return {"input_ids": rng.integers(5, 5000, size=(b, s), dtype=np.int64),
            "attention_mask": np.ones((b, s), dtype=np.int64)}


def measure(run, b, s, iters, warm=3):
    f = feeds(b, s)
    for _ in range(warm):
        run(f)
    lat = []
    for _ in range(iters):
        t = time.perf_counter()
        run(f)
        lat.append((time.perf_counter() - t) * 1e3)
    lat.sort()
    return statistics.median(lat), lat[max(0, int(len(lat) * 0.95) - 1)]


if mode == "sweep":
    run = make_runner()
    pts = []
    for b in (1, 2, 4, 8, 16, 32):
        for s in (8, 24, 64, 128):
            tokens = b * s
            if tokens > 1024:
                continue
            p50, p95 = measure(run, b, s, 20 if tokens <= 256 else 10)
            pts.append((b, s, tokens, p50))
            print(f"{engine} sweep batch={b:2d} seq={s:3d} tokens={tokens:4d}  p50={p50:8.1f} ms  p95={p95:8.1f} ms", flush=True)
    x = np.array([p[2] for p in pts], float)
    y = np.array([p[3] for p in pts], float)
    slope, icpt = np.polyfit(x, y, 1)
    resid = y - (icpt + slope * x)
    print(f"{engine} FIT latency = {icpt:.1f} ms + {slope*1000:.2f} us/token   (rmse {np.sqrt((resid**2).mean()):.1f} ms, n={len(pts)})")
else:
    cfgs = ([dict(hint="LATENCY", threads=4), dict(hint="LATENCY", threads=8),
             dict(hint="THROUGHPUT", threads=4), dict(hint="THROUGHPUT", threads=8)]
            if engine == "ov" else
            [dict(threads=4), dict(threads=8), dict(threads=4, nospin=True), dict(threads=2)])
    for cfg in cfgs:
        run = make_runner(**cfg)
        out = []
        for b, s in ((1, 24), (16, 24)):
            p50, p95 = measure(run, b, s, 25)
            out.append(f"batch {b:2d}: p50={p50:7.1f} p95={p95:7.1f}")
        print(f"{engine} cfg {cfg}: " + " | ".join(out), flush=True)
