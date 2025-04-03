import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# ---------------------------------------------------------
# Define the model architecture for grayscale images
# ---------------------------------------------------------
class SimpleCNN_Grayscale(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN_Grayscale, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1-channel input
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained grayscale model weights (trained on grayscale images)
model = SimpleCNN_Grayscale(num_classes=2).to(device)
model.load_state_dict(torch.load("best_model_grayscale.pth", map_location=device))
model.eval()

# ---------------------------------------------------------
# Helper function to load the .bin file (grayscale)
# ---------------------------------------------------------
def load_bin_image(bin_file, H=2048, W=2048, channels=1):
    total_elements = H * W * channels
    arr = np.fromfile(bin_file, dtype=np.float32)
    if arr.size != total_elements:
        raise ValueError(f"Expected {total_elements} elements, got {arr.size}")
    arr = arr.reshape((channels, H, W))
    return arr

# ---------------------------------------------------------
# Preprocess function:
# - Reads the .bin file (assumed to have pixel values in [0,255] as float32)
# - Converts the array to uint8 and then to a PIL image
# - Resizes to the model's expected input size (640x640)
# - Converts to tensor and normalizes (using mean=0.5, std=0.5 for grayscale)
# ---------------------------------------------------------
def preprocess_bin_image(bin_file):
    # Load raw data from .bin file
    H, W = 2048, 2048
    arr = load_bin_image(bin_file, H=H, W=W, channels=1)
    
    # Clip and convert to uint8 (assuming values in 0-255)
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    
    # Remove the channel dimension for PIL (shape becomes (H, W))
    img = Image.fromarray(arr_uint8.squeeze(0), mode="L")
    
    # Resize the image to the model's expected input size (e.g., 640x640)
    img = img.resize((2048, 2048))
    
    # Define transforms: convert to tensor and normalize.
    # Here we assume that during training you normalized with mean=0.5 and std=0.5.
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to tensor with values in [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    tensor_image = transform(img)  # shape: (1, 640, 640)
    tensor_image = tensor_image.unsqueeze(0)  # add batch dimension -> (1, 1, 640, 640)
    return tensor_image

# ---------------------------------------------------------
# Inference function
# ---------------------------------------------------------
def predict(bin_file):
    input_tensor = preprocess_bin_image(bin_file).to(device)
    print("Input tensor shape:", input_tensor.shape)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    inference_time_ms = (time.time() - start_time) * 1000  # milliseconds
    
    logits = output.cpu().numpy().squeeze()
    probs = F.softmax(output, dim=1).cpu().numpy().squeeze()
    print("Logits:", logits)
    print("Softmax probabilities:", probs)
    
    _, predicted = torch.max(output, 1)
    label = "Bird" if predicted.item() == 0 else "Drone"
    return label, inference_time_ms

# ---------------------------------------------------------
# Main block
# ---------------------------------------------------------
if __name__ == "__main__":
    bin_file_path = "BT_Test_gray.bin"  # Path to your .bin file
    label, inf_time = predict(bin_file_path)
    print(f"Predicted label: {label}")
    print(f"Inference time: {inf_time:.2f} ms")
