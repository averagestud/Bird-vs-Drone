#!/usr/bin/env python3
import os
import glob
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm

# =======================
# 1. Imports & Setup
# =======================
IMG_SIZE = 640
NUM_CLASSES = 2  # not used for detector, kept for compatibility
GRID_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 15
DATASET_MAX_FRAMES = 400000
MAX_FRAMES_PER_VIDEO = 5000

# Paths (adjust to your environment)
RAW_VIDEO_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/wosdetc_train_videos/"
ANNOTATION_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/challenge-master/annotations/"

# =======================
# 2. Parse WOSDETC line (unchanged)
# =======================
def parse_wosdetc_line(line):
    """Parses a single line from a WOSDETC annotation file."""
    tokens = line.strip().split()
    if len(tokens) < 2:
        return None, []
    try:
        frame_num = int(tokens[0])
        num_objs = int(tokens[1])
    except ValueError:
        return None, []
        
    objs = []
    offset = 2
    for _ in range(num_objs):
        if offset + 5 > len(tokens): break
        try:
            x = float(tokens[offset+0])
            y = float(tokens[offset+1])
            w = float(tokens[offset+2])
            h = float(tokens[offset+3])
            cls_token = tokens[offset+4]
            cls = 1 if cls_token.lower() == "drone" else 0
            objs.append([cls, x, y, w, h])
            offset += 5
        except ValueError:
            offset += 5
            continue
    return frame_num, objs

# =======================
# 3. Video & Annotation Pairing
# =======================
video_info_list = []
print("Pairing videos with annotations and collecting frame metadata...")

for ann_file in tqdm(sorted(os.listdir(ANNOTATION_DIR))):
    if not ann_file.endswith('.txt'):
        continue
    ann_path = os.path.join(ANNOTATION_DIR, ann_file)
    basename = os.path.splitext(ann_file)[0]

    videos_found = glob.glob(os.path.join(RAW_VIDEO_DIR, f"{basename}*"))
    if len(videos_found) == 0:
        continue

    video_path = videos_found[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        continue

    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    nframes = min(nframes, MAX_FRAMES_PER_VIDEO)

    frame_to_objs = {}
    try:
        with open(ann_path, 'r') as f:
            for line in f:
                frame_num, objs = parse_wosdetc_line(line)
                if frame_num is not None:
                    frame_to_objs[frame_num] = objs
    except FileNotFoundError:
        continue

    video_info_list.append({
        'video_path': video_path,
        'nframes': nframes,
        'w_orig': w_orig,
        'h_orig': h_orig,
        'frame_to_objs': frame_to_objs
    })

print(f"Total videos successfully paired: {len(video_info_list)}")

# =======================
# 4. Classification Dataset (NEW)
# =======================
class WosdetcVideoDatasetCls(Dataset):
    """
    Loads frames directly from video files on demand and provides a single binary label per frame:
    - 1 if the frame contains a drone, 0 otherwise.
    The label is derived from the first character of each annotation line for that frame:
    - '0' => bird (label 0)
    - '1' => drone (label 1)
    If a frame has no annotation line, it's treated as 0 (no drone).
    """
    def __init__(self, video_info_list, img_size=IMG_SIZE, transform=None, max_frames=DATASET_MAX_FRAMES):
        self.img_size = img_size
        self.transform = transform
        self.samples = []  # List of (video_index, frame_index)
        print(f"Aggregating all frame indices (up to {max_frames} total)...")
        current_frame_count = 0
        for vid_idx, info in enumerate(video_info_list):
            for frame_idx in range(info['nframes']):
                self.samples.append((vid_idx, frame_idx))
                current_frame_count += 1
                if current_frame_count >= max_frames:
                    break
            if current_frame_count >= max_frames:
                break
        self.video_info_list = video_info_list
        print(f"Dataset size (sampled frames): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_idx, frame_idx = self.samples[idx]
        info = self.video_info_list[vid_idx]
        video_path = info['video_path']
        w_orig, h_orig = info['w_orig'], info['h_orig']

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            img = transforms.ToTensor()(img)
            label = 0
            return img, torch.tensor(label, dtype=torch.float32)

        # grayscale and resize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Determine label from annotation line first character
        objs = info['frame_to_objs'].get(frame_idx, [])
        label = 0  # default
        if len(objs) > 0:
            # Use the first object's first token presence: bbox-classes already parsed as 0/1
            # We only need presence of drone (1) anywhere; otherwise bird (0)
            first_cls = objs[0][0]
            label = int(first_cls)  # 1 for drone, 0 for bird
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img, label_tensor

def collate_fn_cls(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(labels)

# Data transforms
tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Build dataset and dataloader for classification
cls_ds = WosdetcVideoDatasetCls(
    video_info_list,
    img_size=IMG_SIZE,
    transform=tf,
    max_frames=DATASET_MAX_FRAMES
)

cls_dl = DataLoader(cls_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn_cls)

# =======================
# 5. 6-Layer Depthwise-Separable Classifier (NEW)
# =======================
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=stride, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.dw_bn(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.pw_bn(x)
        x = self.act(x)
        return x

class Classifier6(nn.Module):
    """
    6-layer depthwise-separable CNN classifier
    - 1 initial conv (stride 2)
    - 5 DWConvBlock layers
    - GlobalAvgPool -> 1x1 linear to single logit
    """
    def __init__(self, in_ch=1, base_filters=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.layer1 = DWConvBlock(base_filters, base_filters*2, stride=2)
        self.layer2 = DWConvBlock(base_filters*2, base_filters*4, stride=2)
        self.layer3 = DWConvBlock(base_filters*4, base_filters*8, stride=2)
        self.layer4 = DWConvBlock(base_filters*8, base_filters*16, stride=2)
        self.layer5 = DWConvBlock(base_filters*16, base_filters*16, stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(base_filters*16, 1)  # single logit for drone vs non-drone

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out  # [N, 1]

# Instantiate model
model = Classifier6(in_ch=1, base_filters=16).to(DEVICE)
print(f"Classifier model on device: {DEVICE}")

# =======================
# 6. Training Loop (Classification)
# =======================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
print(f"Starting classification training on {len(cls_ds)} samples for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for imgs, labels in tqdm(cls_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)  # [B,1]

        optimizer.zero_grad()
        logits = model(imgs)  # [B,1]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

    # Optional: quick per-epoch accuracy (on training stream)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for imgs, labels in cls_dl:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} training accuracy: {acc:.4f}")

# =======================
# 7. Validation Utility (Optional)
# =======================
def evaluate_once(model, dataloader, threshold=0.5):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > threshold).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    return acc

# Optional quick eval on training set (for demonstration)
train_acc = evaluate_once(model, cls_dl, threshold=0.5)
print(f"Validation-like (training data) accuracy: {train_acc:.4f}")

# =======================
# 8. Saving the Model
# =======================
SAVE_DIR = "/scratch/deepaprakash.k.ece22@itbhu/"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "bird_drone_classifier6_dwconv.pth")
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Optional small inference demo on a single batch
def demo_inference(model, dataset, which=0):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    imgs, lbls = next(iter(loader))
    imgs = imgs.to(DEVICE)
    with torch.no_grad():
        logit = model(imgs)
        prob = torch.sigmoid(logit).cpu().item()
    print(f"Demo inference - drone probability: {prob:.4f} (label {lbls[0].item()})")

# Uncomment to run a quick demo after training
# demo_inference(model, cls_ds, which=0)

