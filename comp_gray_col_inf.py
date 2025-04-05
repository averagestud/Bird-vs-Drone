import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# ----------------------------
# 1. Model Definitions (unchanged)
# ----------------------------
class SimpleCNN_Color(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1),
            nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)

class SimpleCNN_Gray(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1),
            nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)

# ----------------------------
# 2. Load Models
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
color_model = SimpleCNN_Color().to(device)
color_model.load_state_dict(torch.load("best_model.pth", map_location=device))
color_model.eval()

gray_model = SimpleCNN_Gray().to(device)
gray_model.load_state_dict(torch.load("best_model_grayscale.pth", map_location=device))
gray_model.eval()

# ----------------------------
# 3. Preprocessing Functions
# ----------------------------
def load_color_bin(path, H=1024, W=1024):
    arr = np.fromfile(path, dtype=np.float32)
    arr = arr.reshape(3, H, W).clip(0,255).astype(np.uint8)
    return Image.fromarray(arr.transpose(1,2,0), mode="RGB")

def load_gray_bin(path, H=1024, W=1024):
    arr = np.fromfile(path, dtype=np.float32)
    arr = arr.reshape(1, H, W).clip(0,255).astype(np.uint8)
    return Image.fromarray(arr.squeeze(0), mode="L")

# Transforms
color_transform = transforms.Compose([
    # transforms.Resize((1024,1024)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# **Downscaled** grayscale transform to get ~3× speedup
gray_transform = transforms.Compose([
    # transforms.Resize((1024,1024)),    # 368^2 / 640^2 ≈ 1/3
    transforms.ToTensor(),
    # transforms.Normalize([0.5],[0.5])
])

# ----------------------------
# 4. Inference + Timing
# ----------------------------
def measure(model, img, transform, device, warmup=5, iters=20):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(tensor)
    times = []
    with torch.no_grad():
        for _ in range(iters):
            if device.type=="cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(tensor)
            if device.type=="cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1-t0)*1000)
    return sum(times)/len(times), times

if __name__=="__main__":
    color_img = load_color_bin("DT_Test.bin")
    gray_img  = load_gray_bin("DT_Test_gray.bin")

    avg_color, _ = measure(color_model, color_img, color_transform, device)
    avg_gray,  _ = measure(gray_model,  gray_img,  gray_transform,  device)

    print(f"Color inference average: {avg_color:.2f} ms")
    print(f"Gray  inference average: {avg_gray:.2f} ms")
