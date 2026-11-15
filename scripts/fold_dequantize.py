"""usage: python fold_dequantize.py [src.onnx] [dst_dir]`nFold DequantizeLinear nodes whose input is a constant initializer into fp32 initializers (offline),
so ONNX Runtime does not dequantize ~all weights on every call. Writes a new model + external data."""
import os, sys, time, collections
import numpy as np
import onnx
from onnx import numpy_helper

SRC = sys.argv[1] if len(sys.argv) > 1 else "models/gemma/model_quantized.onnx"
DST_DIR = sys.argv[2] if len(sys.argv) > 2 else "models/gemma_fp32w"
os.makedirs(DST_DIR, exist_ok=True)
DST = os.path.join(DST_DIR, "model_fp32w.onnx")

t0 = time.time()
m = onnx.load(SRC, load_external_data=True)
print(f"loaded in {time.time()-t0:.1f}s; opset {[o.version for o in m.opset_import]}, nodes {len(m.graph.node)}")
inits = {i.name: i for i in m.graph.initializer}
ops = collections.Counter(n.op_type for n in m.graph.node)
print("top ops:", ops.most_common(8))

folded, kept, bytes_before, bytes_after = 0, 0, 0, 0
new_nodes, new_inits, remove_inits = [], [], set()
for n in m.graph.node:
    if n.op_type == "DequantizeLinear" and n.input[0] in inits and all((i == "" or i in inits) for i in n.input[1:]):
        x = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
        scale = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
        zp = numpy_helper.to_array(inits[n.input[2]]).astype(np.float32) if len(n.input) > 2 and n.input[2] else np.zeros_like(scale)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        axis = attrs.get("axis", 1)
        block = attrs.get("block_size", 0)
        if scale.ndim == 0 or scale.size == 1:
            y = (x - zp.reshape(-1)[0]) * scale.reshape(-1)[0]
        else:
            ax = axis if axis >= 0 else axis + x.ndim
            if block:
                scale = np.repeat(scale, block, axis=ax)[tuple(slice(0, s) for s in x.shape)]
                zp = np.repeat(zp, block, axis=ax)[tuple(slice(0, s) for s in x.shape)]
                y = (x - zp) * scale
            else:
                shp = [1] * x.ndim
                shp[ax] = -1
                y = (x - zp.reshape(shp)) * scale.reshape(shp)
        bytes_before += inits[n.input[0]].raw_data.__len__() if inits[n.input[0]].raw_data else x.size
        bytes_after += y.nbytes
        new_inits.append(numpy_helper.from_array(y.astype(np.float32), name=n.output[0]))
        for i in n.input:
            if i:
                remove_inits.add(i)
        folded += 1
    else:
        new_nodes.append(n)
        if n.op_type == "DequantizeLinear":
            kept += 1

used = {i for n in new_nodes for i in n.input}
del m.graph.node[:]
m.graph.node.extend(new_nodes)
keep_inits = [i for i in m.graph.initializer if i.name not in remove_inits or i.name in used]
del m.graph.initializer[:]
m.graph.initializer.extend(keep_inits)
m.graph.initializer.extend(new_inits)
print(f"folded {folded} DequantizeLinear nodes, kept {kept}; ~{bytes_after/1e6:.0f} MB of fp32 weights created")
t0 = time.time()
onnx.save(m, DST, save_as_external_data=True, all_tensors_to_one_file=True, location="model_fp32w.onnx_data", size_threshold=1024)
print(f"saved {DST} in {time.time()-t0:.1f}s")
