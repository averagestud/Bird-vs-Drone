import torch
import numpy as np
from yolo_nano_model import YoloNano

# ======== Load Trained Model ========
model = YoloNano(num_classes=2, input_ch=1, base_filters=24)
model.load_state_dict(torch.load("yolonanobest_global.pth", map_location="cpu"))
model.eval()

# ======== Mapping Long Keys to Short Names ========
name_mapping = {
    # Stem
    "stem.0.weight": "stem_conv_weight",
    "stem.1.weight": "stem_bn_weight",
    "stem.1.bias": "stem_bn_bias",
    "stem.1.running_mean": "stem_bn_mean",
    "stem.1.running_var": "stem_bn_var",

    # Stage 1 (DWConv)
    "stage1.depthwise.weight": "stage1_dw_weight",
    "stage1.pointwise.weight": "stage1_pw_weight",
    "stage1.bn.weight": "stage1_bn_weight",
    "stage1.bn.bias": "stage1_bn_bias",
    "stage1.bn.running_mean": "stage1_bn_mean",
    "stage1.bn.running_var": "stage1_bn_var",

    # Stage 2
    "stage2.depthwise.weight": "stage2_dw_weight",
    "stage2.pointwise.weight": "stage2_pw_weight",
    "stage2.bn.weight": "stage2_bn_weight",
    "stage2.bn.bias": "stage2_bn_bias",
    "stage2.bn.running_mean": "stage2_bn_mean",
    "stage2.bn.running_var": "stage2_bn_var",

    # Stage 3
    "stage3.depthwise.weight": "stage3_dw_weight",
    "stage3.pointwise.weight": "stage3_pw_weight",
    "stage3.bn.weight": "stage3_bn_weight",
    "stage3.bn.bias": "stage3_bn_bias",
    "stage3.bn.running_mean": "stage3_bn_mean",
    "stage3.bn.running_var": "stage3_bn_var",

    # Stage 4
    "stage4.depthwise.weight": "stage4_dw_weight",
    "stage4.pointwise.weight": "stage4_pw_weight",
    "stage4.bn.weight": "stage4_bn_weight",
    "stage4.bn.bias": "stage4_bn_bias",
    "stage4.bn.running_mean": "stage4_bn_mean",
    "stage4.bn.running_var": "stage4_bn_var",

    # Stage 5
    "stage5.depthwise.weight": "stage5_dw_weight",
    "stage5.pointwise.weight": "stage5_pw_weight",
    "stage5.bn.weight": "stage5_bn_weight",
    "stage5.bn.bias": "stage5_bn_bias",
    "stage5.bn.running_mean": "stage5_bn_mean",
    "stage5.bn.running_var": "stage5_bn_var",

    # Detection Head
    "head.weight": "head_weight",
    "head.bias": "head_bias",
}


# ======== Helper: numpy array to C-style braces ========
def array_to_c_braces(arr):
    if arr.ndim == 0:
        return f"{arr.item():.6f}"
    elif arr.ndim == 1:
        return "{" + ", ".join(f"{x:.6f}" for x in arr) + "}"
    else:
        return "{" + ",\n ".join(array_to_c_braces(a) for a in arr) + "}"

# ======== Export to .h File ========
with open("yolonano_detection.h", "w") as f:
    f.write("// Auto-generated weights for TinyDet\n\n")
    for key, tensor in model.state_dict().items():
        if key not in name_mapping:
            continue
        var_name = name_mapping[key]
        np_arr = tensor.cpu().detach().numpy()
        dim_str = "".join(f"[{d}]" for d in np_arr.shape)
        f.write(f"float {var_name}{dim_str} = \n")
        f.write(array_to_c_braces(np_arr))
        f.write(";\n\n")

print("✅ Conversion complete — yolonano_detection.h generated.")
