"""Build weight-only / dynamic quantized variants of the pre-dequantized Gemma model (ORT 1.20 API).
usage: python exp7_quant.py <model_fp32w.onnx> <out_dir>
Then evaluate each with exp6_prefix.py <variant.onnx>."""
import os, sys, time
import onnx

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

which = sys.argv[3] if len(sys.argv) > 3 else "both"

if which in ("both", "int4"):
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
    t0 = time.time()
    q = MatMul4BitsQuantizer(onnx.load(src), block_size=32, is_symmetric=True)
    q.process()
    p = os.path.join(out, "gemma_int4.onnx")
    q.model.save_model_to_file(p, use_external_data_format=True)
    print(f"int4 -> {p} in {time.time()-t0:.0f}s", flush=True)

if which in ("both", "dyn8"):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    t0 = time.time()
    p = os.path.join(out, "gemma_dyn8.onnx")
    quantize_dynamic(src, p, weight_type=QuantType.QInt8, use_external_data_format=True)
    print(f"dyn8 -> {p} in {time.time()-t0:.0f}s", flush=True)
