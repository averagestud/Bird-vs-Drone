import torch
import numpy as np
from simple_cnn import SimpleCNN_Grayscale  # Your model definition; for color images change Conv2d(1,32,...) to Conv2d(3,32,...)

# Load the trained model
model = SimpleCNN_Grayscale(num_classes=2)
model.load_state_dict(torch.load("best_model_grayscale.pth", map_location="cpu"))
model.eval()

# Hardcoded dimensions for each parameter (must match your model architecture)
dims = {
    "features.0.weight": (32, 1, 3, 3),            # conv1: 3 input channels, 32 output
    "features.0.bias": (32,),
    "features.1.weight": (32,),
    "features.1.bias": (32,),
    "features.1.running_mean": (32,),
    "features.1.running_var": (32,),
    "features.4.weight": (64, 32, 3, 3),           # conv2: 32->64
    "features.4.bias": (64,),
    "features.5.weight": (64,),
    "features.5.bias": (64,),
    "features.5.running_mean": (64,),
    "features.5.running_var": (64,),
    "features.8.weight": (128, 64, 3, 3),          # conv3: 64->128
    "features.8.bias": (128,),
    "features.9.weight": (128,),
    "features.9.bias": (128,),
    "features.9.running_mean": (128,),
    "features.9.running_var": (128,),
    "classifier.1.weight": (2, 128),               # FC: 128->2
    "classifier.1.bias": (2,)
}

# Mapping of long parameter names to shorter names for use in C code
name_mapping = {
    "features.0.weight": "conv1_weight",
    "features.0.bias": "conv1_bias",
    "features.1.weight": "bn1_weight",
    "features.1.bias": "bn1_bias",
    "features.1.running_mean": "bn1_running_mean",
    "features.1.running_var": "bn1_running_var",
    "features.4.weight": "conv2_weight",
    "features.4.bias": "conv2_bias",
    "features.5.weight": "bn2_weight",
    "features.5.bias": "bn2_bias",
    "features.5.running_mean": "bn2_running_mean",
    "features.5.running_var": "bn2_running_var",
    "features.8.weight": "conv3_weight",
    "features.8.bias": "conv3_bias",
    "features.9.weight": "bn3_weight",
    "features.9.bias": "bn3_bias",
    "features.9.running_mean": "bn3_running_mean",
    "features.9.running_var": "bn3_running_var",
    "classifier.1.weight": "fc_weight",
    "classifier.1.bias": "fc_bias"
}

# A recursive function that converts a NumPy array into nested C braces.
def array_to_c_braces(arr):
    if arr.ndim == 0:
        return f"{arr.item():.6f}"
    elif arr.ndim == 1:
        return "{" + ", ".join(f"{x:.6f}" for x in arr) + "}"
    else:
        return "{" + ",\n ".join(array_to_c_braces(a) for a in arr) + "}"

with open("model_weights_grayscale.h", "w") as f:
    f.write("// Auto-generated model weights\n\n")
    for key, tensor in model.state_dict().items():
        if key in dims and key in name_mapping:
            shape = dims[key]
            var_name = name_mapping[key]
            np_arr = tensor.cpu().detach().numpy().reshape(shape)
            # Create a dimension string like [32][3][3][3]
            dim_str = "".join(f"[{d}]" for d in shape)
            f.write(f"const float {var_name}{dim_str} = \n")
            f.write(array_to_c_braces(np_arr))
            f.write(";\n\n")
print("Conversion complete. 'model_weights_grayscale.h' generated.")
