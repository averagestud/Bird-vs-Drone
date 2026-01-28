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
from PIL import Image  # <-- Added for transforms
from sklearn.model_selection import train_test_split  # <-- Added for val split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score # <-- Added for metrics

# =======================
# 1. Imports & Setup
# =======================
IMG_SIZE = 640
PATCH_GRID = 4  # 4x4 = 16 patches
PATCH_SIZE = IMG_SIZE // PATCH_GRID
PAD = 1
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Robust Training Hyperparameters ---
BATCH_SIZE = 32
NUM_EPOCHS = 20       # <-- Increased from 10
LEARNING_RATE = 1e-3  # <-- Slightly lower for more stable training
WEIGHT_DECAY = 1e-4   # <-- For AdamW optimizer
LABEL_SMOOTHING = 0.05 # <-- Regularization to prevent overconfidence
# ---

DATASET_MAX_FRAMES = 400000
MAX_FRAMES_PER_VIDEO = 5000

RAW_VIDEO_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/wosdetc_train_videos/"
ANNOTATION_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/challenge-master/annotations/"
FRAME_SAVE_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/wosdetc_train_frames/"
BEST_MODEL_SAVE_PATH = "/scratch/deepaprakash.ece22.itbhu/yolo_nano_kria.pth" # <-- Will save best model here


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
# 3. Video & Annotation Pairing (unchanged)
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


# ==========================================================
# 4. *** NEW *** Calculate Class Weights & Split Data
# ==========================================================
print("Calculating class weights for imbalance...")
num_pos = 0 # (drone)
num_neg = 0 # (no drone)
total_frames_to_check = 0
for info in tqdm(video_info_list):
    for frame_idx in range(info['nframes']):
        objs = info['frame_to_objs'].get(frame_idx, [])
        label = 1 if any(obj[0] == 1 for obj in objs) else 0
        if label == 1:
            num_pos += 1
        else:
            num_neg += 1
        total_frames_to_check += 1
        if total_frames_to_check >= DATASET_MAX_FRAMES:
            break
    if total_frames_to_check >= DATASET_MAX_FRAMES:
        break

# Calculate pos_weight for BCEWithLogitsLoss
if num_pos > 0 and num_neg > 0:
    pos_weight = num_neg / num_pos
    print(f"Dataset Stats: {num_pos} DRONE frames, {num_neg} NO-DRONE frames.")
    print(f"Calculated pos_weight for loss: {pos_weight:.2f}")
else:
    print("Warning: Could not calculate pos_weight. Using 1.0.")
    pos_weight = 1.0

# Split into Train and Validation sets (90% train, 10% val)
train_info, val_info = train_test_split(video_info_list, test_size=0.1, random_state=42)
print(f"Data split: {len(train_info)} train videos, {len(val_info)} validation videos.")


# ==========================================================
# 5. *** NEW *** Define Data Augmentations
# ==========================================================
# Statistics for grayscale normalization (using 0.5 is a common default)
IMG_MEAN = [0.5]
IMG_STD = [0.5]

# Robust train transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
])

# Validation transforms (no augmentation, just normalization)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
])


# ==========================================================
# 6. Classification Dataset (PATCH-BASED) - *** UPDATED ***
#    Now applies augmentations to the full image *before*
#    patching the resulting tensor.
# ==========================================================
class WosdetcVideoPatchDataset(Dataset):
    def __init__(self, video_info_list, frame_base_dir, img_size=IMG_SIZE, patch_grid=PATCH_GRID, pad=PAD, transform=None, max_frames=DATASET_MAX_FRAMES):
        self.img_size = img_size
        self.patch_grid = patch_grid
        self.patch_size = self.img_size // self.patch_grid
        self.pad = pad
        self.transform = transform # <-- This will now be used
        self.samples = []
        
        # This is a tensor-based padding function
        self.pad_fn = torch.nn.ReflectionPad2d(self.pad) 
        
        print(f"Aggregating all frame indices (up to {max_frames} total)...")
        current_frame_count = 0
        for vid_idx, info in enumerate(video_info_list):
            video_path = info['video_path']
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            
            for frame_idx in range(info['nframes']):
                frame_path = os.path.join(frame_base_dir, video_basename, f"frame_{frame_idx:08d}.jpg")
                
                # Get the label
                objs = info['frame_to_objs'].get(frame_idx, [])
                label = 0
                if len(objs) > 0:
                    label = 1 if any(obj[0] == 1 for obj in objs) else 0
                
                self.samples.append((frame_path, label))
                current_frame_count += 1
                if current_frame_count >= max_frames:
                    break
            if current_frame_count >= max_frames:
                break
                
        print(f"Dataset size (sampled frames): {len(self.samples)}")
        if len(self.samples) == 0:
            print("WARNING: Dataset is empty. Did you run the preprocessing script?")
            print(f"Expected frames in: {frame_base_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, label = self.samples[idx]
        
        # Read the pre-processed image from disk
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: could not read {frame_path}. Returning zero image.")
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # --- *** NEW: Apply transforms *** ---
        # 1. Convert to PIL Image for torchvision transforms
        img_pil = Image.fromarray(img)
        
        # 2. Apply transforms (ToTensor, Augmentations, Normalization)
        # This gives a tensor of shape [1, 640, 640]
        img_tensor = self.transform(img_pil) 
        # ---
        
        # 3. Make PATCHES from the *augmented tensor*
        patches = []
        for row in range(self.patch_grid):
            for col in range(self.patch_grid):
                x0 = col * self.patch_size
                y0 = row * self.patch_size
                
                # Slice the tensor
                patch_tensor = img_tensor[
                    :, 
                    y0 : y0 + self.patch_size, 
                    x0 : x0 + self.patch_size
                ]
                
                # Pad the tensor patch (add (1,1,1,1) padding)
                # Input shape [1, P, P] -> Output shape [1, P+2, P+2]
                padded_patch = self.pad_fn(patch_tensor)
                patches.append(padded_patch)
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        # Stack all 16 patches
        return torch.stack(patches), label_tensor  # [16, 1, PATCH+2, PATCH+2], label


def collate_fn_patch(batch):
    patch_batches, labels = zip(*batch)
    return torch.stack(patch_batches), torch.stack(labels)

# --- Create Train and Val Datasets ---
print("\nCreating Training Dataset...")
train_ds = WosdetcVideoPatchDataset(
    train_info,
    frame_base_dir=FRAME_SAVE_DIR,
    transform=train_transform,  # <-- Use augmentations
    max_frames=DATASET_MAX_FRAMES
)
print("\nCreating Validation Dataset...")
val_ds = WosdetcVideoPatchDataset(
    val_info,
    frame_base_dir=FRAME_SAVE_DIR,
    transform=val_transform,    # <-- Use NO augmentations
    max_frames=int(DATASET_MAX_FRAMES * 0.15) # Use a subset for faster validation
)

# --- Create Train and Val DataLoaders ---
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn_patch, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn_patch, pin_memory=True)


# ==========================================================
# 7. Patch-wise Classifier (UNCHANGED ARCHITECTURE)
# ==========================================================
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
    def __init__(self, in_ch=1, base_filters=16, patch_feature_dim=32):
        """
        Very small per-patch backbone. For FPGA choose base_filters=8.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        # Two lightweight DW blocks; keep strides so spatial dims shrink quickly
        self.block1 = DWConvBlock(base_filters, base_filters*2, stride=2)   # /4 overall from input patch
        self.block2 = DWConvBlock(base_filters*2, base_filters*4, stride=2) # /8

        # Global pooling -> small projection: patch_feature_dim
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_patch = nn.Linear(base_filters*4, patch_feature_dim)  # small
        self.fc_final = nn.Linear(patch_feature_dim, 1)               # binary output

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward_backbone(self, x):
        # x: [B*NP, C, H, W]
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)           # [B*NP, channels, 1,1]
        x = x.view(x.size(0), -1)         # [B*NP, channels]
        feat = self.fc_patch(x)           # [B*NP, patch_feature_dim]
        return feat

    def forward(self, patches):
        """
        patches: [B, NP, C, H, W]
        returns: logits [B,1]
        """
        B, NP, C, H, W = patches.shape
        x = patches.view(B*NP, C, H, W)              # batch all patches
        feats = self.forward_backbone(x)             # [B*NP, D]
        feats = feats.view(B, NP, -1)                # [B, NP, D]
        pooled = feats.max(dim=1)[0]                   # [B, D]
        logits = self.fc_final(pooled)               # [B,1]
        return logits

model = PatchClassifier6(in_ch=1, base_filters=16).to(DEVICE)
print(f"Patch-based classifier model on device: {DEVICE}")


# ==========================================================
# 8. *** NEW *** Robust Training & Validation Loop
# ==========================================================
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight, device=DEVICE) # <-- Use calculated weight
)
optimizer = optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY # <-- Use AdamW
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=NUM_EPOCHS, 
    eta_min=LEARNING_RATE / 100 # <-- Anneal to 1% of LR
)

print(f"Starting ROBUST training on {len(train_ds)} samples for {NUM_EPOCHS} epochs...")
print(f"Validating on {len(val_ds)} samples.")
print(f"Best model will be saved to: {BEST_MODEL_SAVE_PATH}")

best_val_f1 = -1.0  # Track best F1-score

for epoch in range(NUM_EPOCHS):
    
    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0
    
    pbar_train = tqdm(train_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for patchbatch, labels in pbar_train:
        patchbatch = patchbatch.to(DEVICE)  # [B, 16, 1, PATCH+2, PATCH+2]
        labels = labels.to(DEVICE).unsqueeze(1)  # [B,1]
        
        optimizer.zero_grad()
        
        logits = model(patchbatch)  # [B,1]
        
        # --- Apply Label Smoothing ---
        labels_smoothed = labels * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        
        loss = criterion(logits, labels_smoothed) # <-- Use smoothed labels
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        pbar_train.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_train_loss = total_train_loss / len(train_dl)
    
    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar_val = tqdm(val_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
    with torch.no_grad():
        for patchbatch, labels in pbar_val:
            patchbatch = patchbatch.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1) # [B,1]
            
            logits = model(patchbatch)
            loss = criterion(logits, labels) # <-- Use original labels for val loss
            
            total_val_loss += loss.item()
            
            # Store predictions and labels for metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # --- Calculate and Print Metrics ---
    avg_val_loss = total_val_loss / len(val_dl)
    
    all_preds_np = torch.cat(all_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    
    # Handle case where there are no positive predictions or labels in batch
    if len(all_labels_np) > 0:
        val_acc = accuracy_score(all_labels_np, all_preds_np)
        val_p = precision_score(all_labels_np, all_preds_np, average='binary', zero_division=0)
        val_r = recall_score(all_labels_np, all_preds_np, average='binary', zero_division=0)
        val_f1 = f1_score(all_labels_np, all_preds_np, average='binary', zero_division=0)
    else:
        val_acc, val_p, val_r, val_f1 = 0.0, 0.0, 0.0, 0.0

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    print(f"  Val Precision: {val_p:.4f} | Val Recall: {val_r:.4f}")
    
    # --- Step the scheduler ---
    scheduler.step()

    # --- Save the best model ---
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
        print(f"  ---> New best model saved with F1-score: {best_val_f1:.4f}\n")
    else:
        print(f"  (Best F1 remains: {best_val_f1:.4f})\n")

print("Training finished.")
print(f"Best model (F1: {best_val_f1:.4f}) saved to {BEST_MODEL_SAVE_PATH}")


# ===============================================
# 9. Final Inference Demo
# ===============================================
# Load the *best* model we saved for the demo
print("\nLoading best model for demo...")
try:
    model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH))
    model.to(DEVICE)
    print("Best model loaded successfully.")

    def demo_inference(model, dataset):
        model.eval()
        if len(dataset) == 0:
            print("Demo inference skipped: dataset is empty.")
            return
            
        # Use the validation dataset for the demo
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        try:
            patchbatch, lbl = next(iter(loader))
        except StopIteration:
            print("Demo inference skipped: DataLoader is empty.")
            return
            
        patchbatch = patchbatch.to(DEVICE)
        with torch.no_grad():
            logit = model(patchbatch)
            prob = torch.sigmoid(logit).cpu().item()
            
        print(f"Demo inference on one sample from validation set:")
        print(f"  - True Label: {lbl[0].item()}")
        print(f"  - Drone Probability: {prob:.4f}")

    # Run demo on the validation set (which uses val_transform)
    demo_inference(model, val_ds)

except FileNotFoundError:
    print(f"Could not load best model from {BEST_MODEL_SAVE_PATH}. Skipping demo.")
except Exception as e:
    print(f"An error occurred during demo: {e}")