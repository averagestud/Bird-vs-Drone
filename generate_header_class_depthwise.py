import torch
import numpy as np
from depthwise_cnn import DepthwiseCNN  # import your model definition file

# ======== Load Trained Model ========
model = DepthwiseCNN(num_classes=2)
model.load_state_dict(torch.load("best_model_depthwise_classification.pth", map_location="cpu"))
model.eval()

# ======== Mapping Long Keys to Short Names ========
# (You can modify these names to match your C inference code variable names)
name_mapping = {
    "features.0.depthwise.weight": "conv1_dw_weight",
    "features.0.pointwise.weight": "conv1_pw_weight",
    "features.0.bn.weight": "bn1_weight",
    "features.0.bn.bias": "bn1_bias",
    "features.0.bn.running_mean": "bn1_running_mean",
    "features.0.bn.running_var": "bn1_running_var",

    "features.2.depthwise.weight": "conv2_dw_weight",
    "features.2.pointwise.weight": "conv2_pw_weight",
    "features.2.bn.weight": "bn2_weight",
    "features.2.bn.bias": "bn2_bias",
    "features.2.bn.running_mean": "bn2_running_mean",
    "features.2.bn.running_var": "bn2_running_var",

    "features.4.depthwise.weight": "conv3_dw_weight",
    "features.4.pointwise.weight": "conv3_pw_weight",
    "features.4.bn.weight": "bn3_weight",
    "features.4.bn.bias": "bn3_bias",
    "features.4.bn.running_mean": "bn3_running_mean",
    "features.4.bn.running_var": "bn3_running_var",

    "classifier.1.weight": "fc_weight",
    "classifier.1.bias": "fc_bias"
}

# ======== Helper to Convert NumPy Arrays to C-Style Braces ========
def array_to_c_braces(arr):
    """Recursively convert a numpy array into a C brace-enclosed string."""
    if arr.ndim == 0:
        return f"{arr.item():.6f}"
    elif arr.ndim == 1:
        return "{" + ", ".join(f"{x:.6f}" for x in arr) + "}"
    else:
        return "{" + ",\n ".join(array_to_c_braces(a) for a in arr) + "}"

# ======== Write to .h File ========
with open("model_weights_depthwise_classification.h", "w") as f:
    f.write("// Auto-generated weights for DepthwiseCNN\n\n")
    for key, tensor in model.state_dict().items():
        if key not in name_mapping:
            continue  # skip buffers not mapped
        var_name = name_mapping[key]
        np_arr = tensor.cpu().detach().numpy()
        dim_str = "".join(f"[{d}]" for d in np_arr.shape)
        f.write(f"float {var_name}{dim_str} = \n")
        f.write(array_to_c_braces(np_arr))
        f.write(";\n\n")

print("✅ Conversion complete — model_weights_depthwise_classification.h generated.")
