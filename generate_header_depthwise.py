# === Paste this cell into Kaggle and run ===
import torch
import numpy as np
import json
from pathlib import Path

# ---------- USER CONFIG ----------
PTH_PATH = "tinydet_grayscale_depthwise.pth"   # path to your .pth
OUT_DIR = Path("C:/Users/Deepu/BTP")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HEADER_FP32 = OUT_DIR / "tinydet_weights.h"
HEADER_INT8 = OUT_DIR / "tinydet_weights_int8.h"   # optional quantized header
META_JSON = OUT_DIR / "tinydet_weights_meta.json"
EXPORT_INT8 = True   # set False to skip int8 export
INT8_SYMMETRIC = True  # symmetric quantization (scale only); keep False if you want asymmetric
# ----------------------------------

def load_state_dict(pth):
    d = torch.load(pth, map_location="cpu")
    # checkpoint could be {'model_state': state_dict} or {'state_dict': ...} or raw state_dict
    if isinstance(d, dict) and any(k in d for k in ("model_state", "state_dict")):
        key = "model_state" if "model_state" in d else "state_dict"
        sd = d[key]
    else:
        sd = d
    if not isinstance(sd, dict):
        raise RuntimeError("Loaded object is not a state_dict")
    return sd

def fold_conv_bn(conv_w, conv_b, bn_weight, bn_bias, bn_rm, bn_rv, eps):
    """
    conv_w: numpy array shape (out_c, in_c, k, k)
    conv_b: numpy array or None shape (out_c,)
    bn_*: numpy arrays shape (out_c,)
    returns: (w_folded, b_folded)
    Formula:
      scale = gamma / sqrt(running_var + eps)
      W' = W * scale.reshape(-1,1,1,1)
      b' = beta - gamma * running_mean / sqrt(running_var + eps)  (+ scale * conv_b if conv had bias)
    """
    gamma = bn_weight
    beta = bn_bias
    rm = bn_rm
    rv = bn_rv
    scale = gamma / np.sqrt(rv + eps)
    w_fold = conv_w * scale.reshape((-1,1,1,1))
    if conv_b is None:
        b_conv = np.zeros_like(scale)
    else:
        b_conv = conv_b
    b_fold = beta - gamma * rm / np.sqrt(rv + eps) + scale * b_conv
    return w_fold.astype(np.float32), b_fold.astype(np.float32)

def array_to_c_braces(arr, float_fmt="{:.7f}"):
    """Convert numpy array to nested C braces string."""
    if arr.ndim == 0:
        return float_fmt.format(float(arr))
    elif arr.ndim == 1:
        return "{" + ", ".join(float_fmt.format(float(x)) for x in arr) + "}"
    else:
        inner = ",\n".join(array_to_c_braces(a, float_fmt) for a in arr)
        return "{" + inner + "}"

# ----------------- MAIN -----------------
sd = load_state_dict(PTH_PATH)
print("Loaded state_dict keys:", list(sd.keys())[:30], " ... total", len(sd.keys()))

# We'll build a dict of arrays to export: name -> (np_array, shape)
export = {}
meta = {}  # metadata: shapes, dtype, scale if quantized

eps_default = 1e-5

# 1) handle stem: expecting keys like 'stem.0.weight', 'stem.1.weight' (bn)
def safe_get(k):
    return sd[k].cpu().numpy() if k in sd else None

# Fold stem conv + bn if present
stem_conv_w = safe_get("stem.0.weight")
stem_conv_b = safe_get("stem.0.bias")  # likely None because bias=False
stem_bn_w   = safe_get("stem.1.weight")
stem_bn_b   = safe_get("stem.1.bias")
stem_bn_rm  = safe_get("stem.1.running_mean")
stem_bn_rv  = safe_get("stem.1.running_var")

if stem_conv_w is not None and stem_bn_w is not None:
    w_fold, b_fold = fold_conv_bn(stem_conv_w, stem_conv_b, stem_bn_w, stem_bn_b, stem_bn_rm, stem_bn_rv, eps_default)
    export["stem_conv_weight"] = w_fold
    export["stem_conv_bias"] = b_fold
    meta["stem_conv_weight"] = {"shape": list(w_fold.shape), "dtype": "float32"}
    meta["stem_conv_bias"]   = {"shape": list(b_fold.shape), "dtype": "float32"}
    print("Folded stem Conv+BN -> stem_conv_weight/bias")
else:
    # fallback: export raw conv (and bias if present)
    if stem_conv_w is not None:
        export["stem_conv_weight"] = stem_conv_w.astype(np.float32)
        meta["stem_conv_weight"] = {"shape": list(stem_conv_w.shape), "dtype": "float32"}
        print("Exported raw stem conv weight")
    if stem_conv_b is not None:
        export["stem_conv_bias"] = stem_conv_b.astype(np.float32)
        meta["stem_conv_bias"] = {"shape": list(stem_conv_b.shape), "dtype": "float32"}

# 2) handle stageX.Y blocks (depthwise separable)
# keys pattern example: 'stage2.0.depthwise.weight', 'stage2.0.pointwise.weight', 'stage2.0.bn.weight'
for stage in ("stage2","stage3","stage4","stage5"):
    # find how many blocks in this stage by scanning keys
    indices = set()
    prefix_len = len(stage) + 1
    for k in sd.keys():
        if k.startswith(stage + "."):
            # key like stage2.0.depthwise.weight
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                indices.add(int(parts[1]))
    if not indices:
        continue
    for idx in sorted(indices):
        base = f"{stage}.{idx}"
        dw_k = base + ".depthwise.weight"
        pw_k = base + ".pointwise.weight"
        bn_prefix = base + ".bn"
        # load arrays
        dw_w = safe_get(dw_k)   # shape: [C_in,1,k,k]
        pw_w = safe_get(pw_k)   # shape: [C_out,C_in,1,1]
        pw_b = safe_get(base + ".pointwise.bias")  # usually None (bias=False)
        bn_w = safe_get(bn_prefix + ".weight")
        bn_b = safe_get(bn_prefix + ".bias")
        bn_rm = safe_get(bn_prefix + ".running_mean")
        bn_rv = safe_get(bn_prefix + ".running_var")
        if pw_w is None:
            raise RuntimeError(f"Expected pointwise weight at {pw_k} not found in state_dict keys.")
        # fold BN into pointwise if BN exists
        if bn_w is not None:
            # conv to fold is the pointwise conv: pw_w shape (C_out, C_in,1,1)
            # conv bias may be None; fold BN into PW conv
            w_fold, b_fold = fold_conv_bn(pw_w, pw_b, bn_w, bn_b, bn_rm, bn_rv, eps_default)
            export[f"{base}_depthwise_weight"] = dw_w.astype(np.float32)
            export[f"{base}_pointwise_weight"] = w_fold
            export[f"{base}_pointwise_bias"] = b_fold
            meta[f"{base}_depthwise_weight"] = {"shape": list(dw_w.shape), "dtype": "float32"}
            meta[f"{base}_pointwise_weight"] = {"shape": list(w_fold.shape), "dtype": "float32"}
            meta[f"{base}_pointwise_bias"] = {"shape": list(b_fold.shape), "dtype": "float32"}
            print(f"Folded {base} pointwise BN -> pointwise_weight/bias and exported depthwise.")
        else:
            # no BN: export raw dw and pw and possible bias
            export[f"{base}_depthwise_weight"] = (dw_w.astype(np.float32) if dw_w is not None else None)
            export[f"{base}_pointwise_weight"] = pw_w.astype(np.float32)
            if pw_b is not None:
                export[f"{base}_pointwise_bias"] = pw_b.astype(np.float32)
            meta[f"{base}_depthwise_weight"] = {"shape": list(dw_w.shape) if dw_w is not None else None, "dtype": "float32"}
            meta[f"{base}_pointwise_weight"] = {"shape": list(pw_w.shape), "dtype": "float32"}
            if pw_b is not None:
                meta[f"{base}_pointwise_bias"] = {"shape": list(pw_b.shape), "dtype": "float32"}
            print(f"Exported raw {base} depthwise + pointwise (no BN folding found).")

# 3) head conv: exports weight and bias as-is (head has bias by default)
head_w = safe_get("head.weight")
head_b = safe_get("head.bias")
if head_w is not None:
    export["head_weight"] = head_w.astype(np.float32)
    meta["head_weight"] = {"shape": list(head_w.shape), "dtype": "float32"}
if head_b is not None:
    export["head_bias"] = head_b.astype(np.float32)
    meta["head_bias"] = {"shape": list(head_b.shape), "dtype": "float32"}
    print("Exported head weight & bias")

# Remove any None entries from export
export = {k:v for k,v in export.items() if v is not None}

# 4) Write float32 header
def write_c_header_fp32(export_dict, out_path):
    with open(out_path, "w") as f:
        f.write("// Auto-generated TinyDet weights (float32) - BN folded where possible\n")
        f.write("#ifndef TINYDET_WEIGHTS_H\n#define TINYDET_WEIGHTS_H\n\n")
        for name, arr in export_dict.items():
            shape = arr.shape
            # convert to C-friendly name
            cname = name.replace(".", "_").replace("-", "_")
            dim_str = "".join(f"[{d}]" for d in shape)
            f.write(f"// {name}  shape={shape}\n")
            f.write(f"static const float {cname}{dim_str} = \n")
            f.write(array_to_c_braces(arr))
            f.write(";\n\n")
        f.write("#endif // TINYDET_WEIGHTS_H\n")
    print("Written FP32 header to:", out_path)

write_c_header_fp32(export, HEADER_FP32)

# 5) Optional: export INT8 arrays with per-tensor scale (symmetric quant)
def quantize_to_int8(export_dict, int8_path, meta_out, int8_symmetric=True):
    int8_export = {}
    scales = {}
    for name, arr in export_dict.items():
        # For per-tensor symmetric quantization
        max_val = float(np.max(np.abs(arr)))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 127.0  # map to int8 range [-127,127]
        q = np.round(arr / scale).astype(np.int8)
        int8_export[name] = q
        scales[name] = float(scale)
    # Write header
    with open(int8_path, "w") as f:
        f.write("// Auto-generated TinyDet weights (int8) with per-tensor scales\n")
        f.write("#ifndef TINYDET_WEIGHTS_INT8_H\n#define TINYDET_WEIGHTS_INT8_H\n\n")
        for name, qarr in int8_export.items():
            cname = name.replace(".", "_").replace("-", "_")
            shape = qarr.shape
            dim_str = "".join(f"[{d}]" for d in shape)
            f.write(f"// {name}  shape={shape}  scale={scales[name]:.9f}\n")
            f.write(f"static const signed char {cname}{dim_str} = \n")
            # format as integers
            if qarr.ndim == 0:
                f.write(str(int(qarr)) )
            else:
                def int_braces(x):
                    if x.ndim == 0:
                        return str(int(x))
                    elif x.ndim == 1:
                        return "{" + ", ".join(str(int(v)) for v in x) + "}"
                    else:
                        return "{" + ",\n ".join(int_braces(a) for a in x) + "}"
                f.write(int_braces(qarr))
            f.write(";\n\n")
        # also write scales array
        f.write("// Per-tensor scales (float)\n")
        f.write("static const float tinydet_param_scales[] = {\n")
        f.write(",\n".join(f"/*{k}*/ {scales[k]:.9f}" for k in scales))
        f.write("\n};\n\n")
        f.write("#endif // TINYDET_WEIGHTS_INT8_H\n")
    # add scales to meta
    meta_out["int8_scales"] = scales
    print("Written INT8 header to:", int8_path)
    return meta_out

if EXPORT_INT8:
    meta = quantize_to_int8(export, HEADER_INT8, meta, int8_symmetric=INT8_SYMMETRIC)

# 6) write meta json
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)
print("Wrote metadata JSON:", META_JSON)

print("Done. Files:")
print(" - float32 header:", HEADER_FP32)
if EXPORT_INT8:
    print(" - int8 header:", HEADER_INT8)
print(" - meta:", META_JSON)