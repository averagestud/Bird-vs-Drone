import os
import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# 1. RE-DEFINE THE MODEL (Must be identical to your training script)
# =============================================================================
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=stride, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # This forward is not used for export, 
        # but is here for completeness
        x = self.depthwise(x)
        x = self.dw_bn(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.pw_bn(x)
        x = self.act(x)
        return x

class PatchClassifier6(nn.Module):
    def __init__(self, in_ch=1, base_filters=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_filters, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.layer1 = DWConvBlock(base_filters, base_filters*2, stride=2)
        self.layer2 = DWConvBlock(base_filters*2, base_filters*4, stride=2)
        self.layer3 = DWConvBlock(base_filters*4, base_filters*8, stride=2)
        self.layer4 = DWConvBlock(base_filters*8, base_filters*16, stride=2)
        self.layer5 = DWConvBlock(base_filters*16, base_filters*16, stride=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_patch = nn.Linear(base_filters*16, 32)
        self.fc_final = nn.Linear(32, 1)

    def _init_weights(self):
        # Not needed for loading, but good practice
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Forward pass not used for export
        pass

# =============================================================================
# 2. HELPER FUNCTIONS FOR FUSION AND C-CODE GENERATION
# =============================================================================

def fuse_conv_bn(conv, bn):
    """
    Fuses a Conv2d/DWConv2d layer with a subsequent BatchNorm2d layer.
    
    Args:
        conv (nn.Module): The Conv2d or Depthwise Conv2d layer (bias=False).
        bn (nn.Module): The BatchNorm2d layer.
        
    Returns:
        (torch.Tensor, torch.Tensor): Fused (weight, bias) tensors.
    """
    # Get BN parameters
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    
    # Get Conv weights
    w_conv = conv.weight.clone()
    
    # Conv bias is assumed to be False, so we start with zeros
    b_conv = torch.zeros(conv.out_channels)
    
    # Perform the fusion calculation
    # W_fused = W_conv * (gamma / sqrt(var + eps))
    # b_fused = (b_conv - mean) * (gamma / sqrt(var + eps)) + beta
    
    scale = gamma / torch.sqrt(running_var + eps)
    
    # Reshape scale for broadcasting with conv weights
    # For Conv2d (N, C, H, W) -> scale shape (N, 1, 1, 1)
    # For DWConv2d (N, 1, H, W) -> scale shape (N, 1, 1, 1)
    scale_rs = scale.reshape(-1, 1, 1, 1)
    
    w_fused = w_conv * scale_rs
    b_fused = (b_conv - running_mean) * scale + beta
    
    return w_fused.detach(), b_fused.detach()

def format_tensor_recursive(f, tensor, dims):
    """
    Recursively formats a tensor into a C-style nested initializer list.
    
    Example:
    { // dim 0
      { // dim 1
        {1.0f, 2.0f}, {3.0f, 4.0f}
      },
      { // dim 1
        {5.0f, 6.0f}, {7.0f, 8.0f}
      }
    }
    """
    if len(dims) == 1:
        # Base case: 1D array
        f.write("{")
        f.write(", ".join([f"{x:.8e}f" for x in tensor]))
        f.write("}")
        return

    f.write("{\n")
    for i in range(dims[0]):
        format_tensor_recursive(f, tensor[i], dims[1:])
        if i < dims[0] - 1:
            f.write(",\n")
    f.write("\n}")

def write_c_array(f_c, f_h, tensor, name):
    """
    Writes a tensor to the .c and .h files.
    
    Args:
        f_c (file): File handle for weights.c
        f_h (file): File handle for weights.h
        tensor (torch.Tensor): The tensor to write.
        name (str): The variable name in C.
    """
    dims = list(tensor.shape)
    
    # --- Write to .h file (extern declaration) ---
    dims_str = "".join([f"[{d}]" for d in dims])
    f_h.write(f"extern const float {name}{dims_str};\n")
    
    # --- Write to .c file (definition) ---
    f_c.write(f"const float {name}{dims_str} = \n")
    format_tensor_recursive(f_c, tensor.data, dims)
    f_c.write(";\n\n")

# =============================================================================
# 3. MAIN EXPORT SCRIPT
# =============================================================================
def main():
    # --- Configuration ---
    BASE_FILTERS = 16
    IN_CH = 1
    # *** IMPORTANT: Update this path to your trained model ***
    MODEL_PATH = "BTP/yolo_btp_class.pth"
    
    OUTPUT_H_FILE = "BTP/btp_class_weights.h"
    OUTPUT_C_FILE = "BTP/btp_class_weights.c"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please update the MODEL_PATH variable in this script.")
        return

    # --- Load Model ---
    print("Loading trained model...")
    model = PatchClassifier6(in_ch=IN_CH, base_filters=BASE_FILTERS)
    
    # Load state dict. If on GPU, map to CPU for export.
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        
    # *** CRITICAL: Set model to eval() mode ***
    # This switches BatchNorm layers to use their running averages
    model.eval()
    print("Model loaded successfully and set to eval() mode.")

    # --- Open output files ---
    with open(OUTPUT_H_FILE, 'w') as f_h, open(OUTPUT_C_FILE, 'w') as f_c:
        
        # --- Write File Headers ---
        f_h.write("/*\n * weights.h\n")
        f_h.write(" * \n * Exported weights for PatchClassifier6.\n")
        f_h.write(" * Generated by export_weights.py\n")
        f_h.write(" */\n\n")
        f_h.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        
        f_c.write("/*\n * weights.c\n")
        f_c.write(" * \n * Exported weights for PatchClassifier6.\n")
        f_c.write(" * Generated by export_weights.py\n")
        f_c.write(" */\n\n")
        f_c.write(f'#include "{OUTPUT_H_FILE}"\n\n')

        # --- Write Layer Dimensions to .h file ---
        f_h.write("// === Layer Dimensions ===\n")
        f_h.write("#define PATCH_H_IN 162\n")
        f_h.write("#define PATCH_W_IN 162\n")
        f_h.write("#define NUM_PATCHES 16\n\n")
        
        f_h.write("// Stem (1->16)\n")
        f_h.write("#define STEM_C_IN 1\n")
        f_h.write(f"#define STEM_C_OUT {BASE_FILTERS}\n\n")
        
        f_h.write("// Layer 1 (16->32)\n")
        f_h.write(f"#define L1_C_IN {BASE_FILTERS}\n")
        f_h.write(f"#define L1_C_MID {BASE_FILTERS}\n")
        f_h.write(f"#define L1_C_OUT {BASE_FILTERS*2}\n\n")
        
        f_h.write("// Layer 2 (32->64)\n")
        f_h.write(f"#define L2_C_IN {BASE_FILTERS*2}\n")
        f_h.write(f"#define L2_C_MID {BASE_FILTERS*2}\n")
        f_h.write(f"#define L2_C_OUT {BASE_FILTERS*4}\n\n")
        
        f_h.write("// Layer 3 (64->128)\n")
        f_h.write(f"#define L3_C_IN {BASE_FILTERS*4}\n")
        f_h.write(f"#define L3_C_MID {BASE_FILTERS*4}\n")
        f_h.write(f"#define L3_C_OUT {BASE_FILTERS*8}\n\n")
        
        f_h.write("// Layer 4 (128->256)\n")
        f_h.write(f"#define L4_C_IN {BASE_FILTERS*8}\n")
        f_h.write(f"#define L4_C_MID {BASE_FILTERS*8}\n")
        f_h.write(f"#define L4_C_OUT {BASE_FILTERS*16}\n\n")
        
        f_h.write("// Layer 5 (256->256)\n")
        f_h.write(f"#define L5_C_IN {BASE_FILTERS*16}\n")
        f_h.write(f"#define L5_C_MID {BASE_FILTERS*16}\n")
        f_h.write(f"#define L5_C_OUT {BASE_FILTERS*16}\n\n")
        
        f_h.write("// FC Layers\n")
        f_h.write(f"#define FC_PATCH_IN {BASE_FILTERS*16}\n")
        f_h.write("#define FC_PATCH_OUT 32\n")
        f_h.write("#define FC_FINAL_IN 32\n")
        f_h.write("#define FC_FINAL_OUT 1\n\n")

        # --- Process and Write Weights ---
        print("Fusing and exporting Stem...")
        stem_conv = model.stem[0]
        stem_bn = model.stem[1]
        w_fused, b_fused = fuse_conv_bn(stem_conv, stem_bn)
        write_c_array(f_c, f_h, w_fused, "stem_conv_w")
        write_c_array(f_c, f_h, b_fused, "stem_conv_b")
        
        # Process Layers 1-5
        for i in range(1, 6):
            print(f"Fusing and exporting Layer {i}...")
            layer = getattr(model, f'layer{i}')
            
            # Depthwise part
            dw_conv = layer.depthwise
            dw_bn = layer.dw_bn
            w_fused, b_fused = fuse_conv_bn(dw_conv, dw_bn)
            write_c_array(f_c, f_h, w_fused, f"l{i}_dw_conv_w")
            write_c_array(f_c, f_h, b_fused, f"l{i}_dw_conv_b")
            
            # Pointwise part
            pw_conv = layer.pointwise
            pw_bn = layer.pw_bn
            w_fused, b_fused = fuse_conv_bn(pw_conv, pw_bn)
            write_c_array(f_c, f_h, w_fused, f"l{i}_pw_conv_w")
            write_c_array(f_c, f_h, b_fused, f"l{i}_pw_conv_b")

        # Process FC Layers (no fusion needed)
        print("Exporting FC layers...")
        
        # fc_patch
        w_fc_patch = model.fc_patch.weight.detach()
        b_fc_patch = model.fc_patch.bias.detach()
        write_c_array(f_c, f_h, w_fc_patch, "fc_patch_w")
        write_c_array(f_c, f_h, b_fc_patch, "fc_patch_b")
        
        # fc_final
        w_fc_final = model.fc_final.weight.detach()
        b_fc_final = model.fc_final.bias.detach()
        write_c_array(f_c, f_h, w_fc_final, "fc_final_w")
        write_c_array(f_c, f_h, b_fc_final, "fc_final_b")
        
        # --- Write File Footers ---
        f_h.write("\n#endif // WEIGHTS_H\n")
        
    print("\n=======================================================")
    print(f"Successfully exported weights to:")
    print(f"  Header:   {os.path.abspath(OUTPUT_H_FILE)}")
    print(f"  Source:   {os.path.abspath(OUTPUT_C_FILE)}")
    print("=======================================================")

if __name__ == "__main__":
    main()