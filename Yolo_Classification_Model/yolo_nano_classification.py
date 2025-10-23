import os
import glob
import math
import torch
import torch.nn as nn
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
PATCH_GRID = 4  # 4x4 = 16 patches
PATCH_SIZE = IMG_SIZE // PATCH_GRID
PAD = 1
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 15
DATASET_MAX_FRAMES = 400000
MAX_FRAMES_PER_VIDEO = 5000

RAW_VIDEO_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/wosdetc_train_videos/"
ANNOTATION_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/challenge-master/annotations/"

# =======================
# 2. Parse WOSDETC line (unchanged)
# =======================
def parse_wosdetc_line(line):
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
# 4. Classification Dataset (PATCH-BASED)
# =======================
class WosdetcVideoPatchDataset(Dataset):
    """
    For each frame, splits the image into 16 grids (4x4);
    each patch is padded by 1 pixel, converted to tensor.
    Label: 1 if frame contains a drone (from annotation), else 0.
    Returns: list of 16 [1, PATCH_SIZE + 2, PATCH_SIZE + 2] tensors, label
    """
    def __init__(self, video_info_list, img_size=IMG_SIZE, patch_grid=PATCH_GRID, pad=PAD, transform=None, max_frames=DATASET_MAX_FRAMES):
        self.img_size = img_size
        self.patch_grid = patch_grid
        self.patch_size = self.img_size // self.patch_grid
        self.pad = pad
        self.transform = transform
        self.samples = []
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
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_size, self.img_size))
        if self.transform:
            fullimg = self.transform(img)
        else:
            fullimg = transforms.ToTensor()(img)
        # Make PATCHES, each patch is padded by 1 pixel
        patches = []
        for row in range(self.patch_grid):
            for col in range(self.patch_grid):
                x0 = col * self.patch_size
                y0 = row * self.patch_size
                patch = img[y0:y0+self.patch_size, x0:x0+self.patch_size]
                patch = cv2.copyMakeBorder(patch, self.pad, self.pad, self.pad, self.pad, borderType=cv2.BORDER_REFLECT)
                patch_tensor = transforms.ToTensor()(patch)
                patches.append(patch_tensor)
        # Label: 1 if any drone present, else 0
        objs = info['frame_to_objs'].get(frame_idx, [])
        label = 0
        if len(objs) > 0:
            label = 1 if any(obj[0] == 1 for obj in objs) else 0
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return torch.stack(patches), label_tensor  # [16, 1, PATCH+2, PATCH+2], label

def collate_fn_patch(batch):
    patch_batches, labels = zip(*batch)
    return torch.stack(patch_batches), torch.stack(labels)

tf = None  # patches are already tensor
cls_ds = WosdetcVideoPatchDataset(
    video_info_list,
    img_size=IMG_SIZE,
    patch_grid=PATCH_GRID,
    pad=PAD,
    transform=tf,
    max_frames=DATASET_MAX_FRAMES
)
cls_dl = DataLoader(cls_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn_patch)

# =======================
# 5. Patch-wise Classifier (per patch ? pooled)
# =======================
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=stride, groups=in_ch, bias=False)
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
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, patches):  # [B, 16, 1, PATCH+2, PATCH+2]
        B, NP, C, H, W = patches.shape
        feats = []
        for p in range(NP):  # For each patch in batch
            x = patches[:,p]
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.global_pool(x)
            x = x.view(B, -1)
            x = self.fc_patch(x)
            feats.append(x)  # [B, 32]
        feats = torch.stack(feats, dim=1)  # [B, 16, 32]
        # Pool across all patches per image (mean, max, etc)
        pooled = feats.mean(dim=1)  # [B, 32]
        out = self.fc_final(pooled)  # [B, 1]
        return out

model = PatchClassifier6(in_ch=1, base_filters=16).to(DEVICE)
print(f"Patch-based classifier model on device: {DEVICE}")

# =======================
# 6. Training Loop (PATCH-BASED)
# =======================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Starting PATCH-based training on {len(cls_ds)} samples for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for patchbatch, labels in tqdm(cls_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        patchbatch = patchbatch.to(DEVICE)  # [B, 16, 1, PATCH+2, PATCH+2]
        labels = labels.to(DEVICE).unsqueeze(1)  # [B,1]
        optimizer.zero_grad()
        logits = model(patchbatch)  # [B,1]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
    # Optional: quick accuracy
    model.eval()
    with torch.no_grad():
        correct = 0; total = 0
        for patchbatch, labels in cls_dl:
            patchbatch = patchbatch.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            logits = model(patchbatch)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} training accuracy: {acc:.4f}")

# =======================
# 7. Saving the Model
# =======================
SAVE_DIR = "/home/deepaprakash.k.ece22@itbhu/"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "bird_drone_patchclassifier6_dwconv.pth")
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Optional: demo on a batch
def demo_inference(model, dataset, which=0):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    patchbatch, lbl = next(iter(loader))
    patchbatch = patchbatch.to(DEVICE)
    with torch.no_grad():
        logit = model(patchbatch)
        prob = torch.sigmoid(logit).cpu().item()
    print(f"Demo inference - drone probability: {prob:.4f} (label {lbl[0].item()})")
# Uncomment below to run after training
demo_inference(model, cls_ds, which=0)

