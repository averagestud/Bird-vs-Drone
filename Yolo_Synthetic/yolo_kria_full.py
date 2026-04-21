import os
import glob
import math
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image  # <-- Added for transforms
from sklearn.model_selection import train_test_split  # <-- Added for val split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score # <-- Added for metrics

# =======================
# 1. Imports & Setup
# =======================
IMG_W = 640
IMG_H = 480
PATCH_GRID = 4  # 4x4 = 16 patches
PATCH_W = IMG_W // PATCH_GRID  # 160
PATCH_H = IMG_H // PATCH_GRID  # 120
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


DATASETS = [
    {
        "name": "REAL_WOSDETC",
        "video_dir": "/scratch/deepaprakash.ece22.itbhu/birdvdrone/wosdetc_train_videos/",
        "ann_dir": "/scratch/deepaprakash.ece22.itbhu/birdvdrone/challenge-master/annotations/",
        "frame_dir": "/scratch/deepaprakash.ece22.itbhu/birdvdrone/wosdetc_train_frames/"
    },
    {
        "name": "SYNTHETIC_BIRDS",
        "video_dir": "/scratch/deepaprakash.ece22.itbhu/birdvdrone/synthetic_videos/",
        "ann_dir": "/scratch/deepaprakash.ece22.itbhu/birdvdrone/synthetic_annotations/",
        "frame_dir": "/scratch/deepaprakash.ece22.itbhu/birdvdrone/synthetic_train_frames/"
    }
]

BEST_MODEL_SAVE_PATH = "/scratch/deepaprakash.ece22.itbhu/yolo_kria_qat_synth.pth"

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
# 3. Video & Annotation Pairing (Logical Merge)
# =======================
video_info_list = []
print("Pairing videos with annotations from ALL datasets...")

for dataset in DATASETS:
    print(f"Processing dataset: {dataset['name']}...")
    ann_dir = dataset["ann_dir"]
    vid_dir = dataset["video_dir"]
    frame_dir = dataset["frame_dir"]
    
    if not os.path.exists(ann_dir):
        print(f"  -> Skipping {dataset['name']}: Annotation directory not found.")
        continue

    for ann_file in tqdm(sorted(os.listdir(ann_dir))):
        if not ann_file.endswith('.txt'): continue
        
        ann_path = os.path.join(ann_dir, ann_file)
        basename = os.path.splitext(ann_file)[0]
        videos_found = glob.glob(os.path.join(vid_dir, f"{basename}*"))
        
        if len(videos_found) == 0: continue
        
        video_path = videos_found[0]
        
        # --- THE FIX: Bypass OpenCV for dummy synthetic videos ---
        if dataset["name"] == "SYNTHETIC_BIRDS":
            nframes = 4050 # We know this exact number from our generation script
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): continue
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        # ---------------------------------------------------------
        
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
            'w_orig': 640 if dataset["name"] == "SYNTHETIC_BIRDS" else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) if 'cap' in locals() and cap.isOpened() else 640),
            'h_orig': 640 if dataset["name"] == "SYNTHETIC_BIRDS" else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if 'cap' in locals() and cap.isOpened() else 640),
            'frame_to_objs': frame_to_objs,
            'frame_dir': frame_dir  # Store the specific folder
        })

print(f"Total combined videos/sequences ready: {len(video_info_list)}")


# ==========================================================
# 4. *** UPDATED *** Stratified Split & Class Weights
# ==========================================================
print("Extracting video-level labels for stratified splitting...")
# 1. Determine a single label for each video sequence
video_labels = []
for info in video_info_list:
    has_drone = False
    # Check all frames in this video to see if a drone ever appears
    for frame_idx, objs in info['frame_to_objs'].items():
        if any(obj[0] == 1 for obj in objs):  # Class 1 = Drone
            has_drone = True
            break
    video_labels.append(1 if has_drone else 0)

# 2. Perform a STRATIFIED split
# By passing stratify=video_labels, the 90/10 split ratio is guaranteed
# to have the exact same proportion of Bird/Drone videos in both sets.
train_info, val_info = train_test_split(
    video_info_list, 
    test_size=0.15, 
    random_state=42, 
    stratify=video_labels  # <-- THE CRITICAL ADDITION
)

print(f"Data split: {len(train_info)} train videos, {len(val_info)} validation videos (Stratified).")

# 3. Calculate exact frame-level class weights based ONLY on the training set
print("Calculating exact class weights from the Training split...")
num_pos = 0 # (drone)
num_neg = 0 # (no drone)

for info in train_info:
    for frame_idx in range(info['nframes']):
        objs = info['frame_to_objs'].get(frame_idx, [])
        label = 1 if any(obj[0] == 1 for obj in objs) else 0
        if label == 1:
            num_pos += 1
        else:
            num_neg += 1

if num_pos > 0 and num_neg > 0:
    pos_weight = num_neg / num_pos
    print(f"Train Set Stats: {num_pos} DRONE frames, {num_neg} NO-DRONE frames.")
    print(f"Calculated pos_weight for loss: {pos_weight:.2f}")
else:
    print("Warning: Could not calculate pos_weight. Using 1.0.")
    pos_weight = 1.0


# ==========================================================
# 5. *** NEW *** Define Data Augmentations
# ==========================================================
# Statistics for grayscale normalization (using 0.5 is a common default)
# ==========================================================
# 5. Define Data Augmentations
# ==========================================================
IMG_MEAN = [0.5]
IMG_STD = [0.5]

# Robust train transforms - NO CROPPING
train_transform = transforms.Compose([
    # Force strict resize to (Height, Width)
    transforms.Resize((IMG_H, IMG_W)), 
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
])

# Validation transforms 
val_transform = transforms.Compose([
    # Validation also needs the strict resize
    transforms.Resize((IMG_H, IMG_W)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
])


# ==========================================================
# 6. Classification Dataset (PATCH-BASED) 
# ==========================================================
class WosdetcVideoPatchDataset(Dataset):
    def __init__(self, video_info_list, img_w=IMG_W, img_h=IMG_H, patch_grid=PATCH_GRID, pad=PAD, transform=None, max_frames=DATASET_MAX_FRAMES):
        self.img_w = img_w
        self.img_h = img_h
        self.patch_grid = patch_grid
        
        # Calculate distinct width and height for the patches
        self.patch_w = self.img_w // self.patch_grid
        self.patch_h = self.img_h // self.patch_grid
        
        self.pad = pad
        self.transform = transform 
        self.samples = []
        
        self.pad_fn = torch.nn.ReflectionPad2d(self.pad) 
        
        print(f"Aggregating all frame indices (up to {max_frames} total)...")
        current_frame_count = 0
        
        for info in video_info_list:
            video_path = info['video_path']
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            
            specific_frame_dir = info['frame_dir'] 
            
            for frame_idx in range(info['nframes']):
                frame_path = os.path.join(specific_frame_dir, video_basename, f"frame_{frame_idx:08d}.jpg")
                
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, label = self.samples[idx]
        
        # Read the pre-processed image from disk
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: could not read {frame_path}. Returning zero image.")
            img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        # --- *** NEW: Apply transforms *** ---
        # 1. Convert to PIL Image for torchvision transforms
        img_pil = Image.fromarray(img)
        
        # 2. Apply transforms (ToTensor, Augmentations, Normalization)
        # This gives a tensor of shape [1, 640, 480]
        img_tensor = self.transform(img_pil) 
        # ---
        
        # 3. Make PATCHES from the *augmented tensor*
        patches = []
        for row in range(self.patch_grid):
            for col in range(self.patch_grid):
                x0 = col * PATCH_W
                y0 = row * PATCH_H
                
                # Slice the tensor
                patch_tensor = img_tensor[
                    :, 
                    y0 : y0 + PATCH_H, 
                    x0 : x0 + PATCH_W
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

# --- When instantiating the datasets, remove the frame_base_dir argument ---
print("\nCreating Training Dataset...")
train_ds = WosdetcVideoPatchDataset(
    train_info,
    transform=train_transform,  
    max_frames=DATASET_MAX_FRAMES
)
print("\nCreating Validation Dataset...")
val_ds = WosdetcVideoPatchDataset(
    val_info,
    transform=val_transform,    
    max_frames=int(DATASET_MAX_FRAMES * 0.15) 
)

# --- Create Train and Val DataLoaders ---
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn_patch, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn_patch, pin_memory=True)


# ==========================================================
# 7. Patch-wise Classifier with PoT Quantization (QAT)
# ==========================================================

class PoTQuantizerSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for Power-of-Two Quantization.
    Forward: Quantizes weights to nearest power of 2.
    Backward: Passes gradients through unchanged (STE).
    """
    @staticmethod
    def forward(ctx, x, min_exp=-8, max_exp=7):
        # 1. Store the sign and get absolute values
        sign = torch.sign(x)
        x_abs = torch.abs(x)
        
        # 2. Prevent log2(0) by clamping to a minimum threshold
        eps = 2.0 ** (min_exp - 1)
        x_abs = torch.clamp(x_abs, min=eps)
        
        # 3. Find the nearest power of 2
        log2_x = torch.round(torch.log2(x_abs))
        
        # 4. Clamp the exponents to simulate hardware bit-width limits
        log2_x = torch.clamp(log2_x, min=min_exp, max=max_exp)
        
        # 5. Reconstruct the PoT tensor
        x_pot = sign * (2.0 ** log2_x)
        
        # 6. Hard zeroing: If the original weight was extremely small, snap to 0
        x_pot[torch.abs(x) < (2.0 ** min_exp)] = 0.0
        
        return x_pot

    @staticmethod
    def backward(ctx, grad_output):
        # Pass the gradient straight through to the FP32 weights
        return grad_output, None, None

# --- PoT Layer Wrappers ---
class PoTConv2d(nn.Conv2d):
    def forward(self, input):
        # Apply PoT quantization to weights before convolution
        quantized_weight = PoTQuantizerSTE.apply(self.weight)
        return self._conv_forward(input, quantized_weight, self.bias)

class PoTLinear(nn.Linear):
    def forward(self, input):
        # Apply PoT quantization to weights before linear transformation
        quantized_weight = PoTQuantizerSTE.apply(self.weight)
        return nn.functional.linear(input, quantized_weight, self.bias)

# --- Updated Network Architecture ---
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Replaced standard Conv2d with PoTConv2d
        self.depthwise = PoTConv2d(in_ch, in_ch, 3, padding=1, stride=stride, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pointwise = PoTConv2d(in_ch, out_ch, 1, bias=False)
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
        super().__init__()
        # Replaced standard Conv2d with PoTConv2d
        self.stem = nn.Sequential(
            PoTConv2d(in_ch, base_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.block1 = DWConvBlock(base_filters, base_filters*2, stride=2)  
        self.block2 = DWConvBlock(base_filters*2, base_filters*4, stride=2) 

        self.global_pool = nn.AdaptiveMaxPool2d((1,1))
        # Replaced standard Linear with PoTLinear
        self.fc_patch = PoTLinear(base_filters*4, patch_feature_dim)  
        self.fc_final = PoTLinear(patch_feature_dim, 1)               

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (PoTConv2d, PoTLinear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward_backbone(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)           
        x = x.view(x.size(0), -1)         
        feat = self.fc_patch(x)           
        return feat

    def forward(self, patches):
        B, NP, C, H, W = patches.shape
        x = patches.view(B*NP, C, H, W)              
        feats = self.forward_backbone(x)             
        feats = feats.view(B, NP, -1)                
        pooled = feats.max(dim=1)[0]                   
        logits = self.fc_final(pooled)               
        return logits

model = PatchClassifier6(in_ch=1, base_filters=16).to(DEVICE)
print(f"PoT Quantized Patch-based classifier model initialized on: {DEVICE}")


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

# ... [Keep Sections 1-7 Unchanged] ...

# ==========================================================
# 7.5. *** NEW *** Binary Focal Loss Definition
# ==========================================================
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for the positive class (0 to 1).
                           If None, no alpha weighting is applied.
            gamma (float): Focusing parameter. Higher gamma reduces loss for easy examples.
                           (Standard is 2.0)
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, 1] (raw output from model, no sigmoid yet)
        # targets: [B, 1] (0 or 1)
        
        # 1. Calculate Standard Binary Cross Entropy (BCE)
        #    reduction='none' allows us to weigh individual samples
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. Calculate probabilities (pt)
        #    p_t = p if y=1 else (1-p)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # 3. Calculate Focal Component: (1 - p_t)^gamma
        focal_factor = (1 - p_t) ** self.gamma
        
        # 4. Apply Alpha Balancing
        #    alpha_t = alpha if y=1 else (1-alpha)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_factor * bce_loss
        else:
            loss = focal_factor * bce_loss

        # 5. Apply Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ==========================================================
# 7.8 *** NEW *** Hard PoT Export Helper
# ==========================================================
def get_hard_pot_state_dict(current_model):
    """Converts the model's FP32 weights into hard PoT values for deployment."""
    pot_state_dict = {}
    with torch.no_grad():
        for name, param in current_model.state_dict().items():
            # We only quantize multi-dimensional weights (Conv2d and Linear).
            # We leave 1D parameters (BatchNorm weights, biases) as they are.
            if 'weight' in name and param.dim() >= 2:
                # Apply the PoT math from PoTQuantizerSTE
                sign = torch.sign(param)
                x_abs = torch.clamp(torch.abs(param), min=2.0**-9)
                log2_x = torch.clamp(torch.round(torch.log2(x_abs)), min=-8.0, max=7.0)
                pot_weight = sign * (2.0 ** log2_x)
                
                # Hard zeroing for extremely small weights
                pot_weight[torch.abs(param) < (2.0 ** -8)] = 0.0
                
                pot_state_dict[name] = pot_weight
            else:
                pot_state_dict[name] = param
    return pot_state_dict

# ==========================================================
# 8. *** UPDATED *** Robust Training with Focal Loss
# ==========================================================

# --- Configure Focal Loss Parameters ---
# ==========================================================
# 8. *** UPDATED *** Training with Focal Loss & Confusion Matrix
# ==========================================================

# --- Configure Focal Loss Parameters ---
# Convert your calculated 'pos_weight' into 'alpha' for Focal Loss
# Formula: alpha = pos_weight / (1 + pos_weight)   <-- Fixed character here
# Example: If pos_weight is 9 (1 positive for every 9 negatives), alpha becomes 0.9

if 'pos_weight' in locals() and pos_weight is not None:
    calculated_alpha = pos_weight / (1.0 + pos_weight)
    calculated_alpha = max(0.1, min(0.9, calculated_alpha))
else:
    calculated_alpha = 0.75

print(f"Initializing Focal Loss with Alpha={calculated_alpha:.2f}, Gamma=2.0")

criterion = BinaryFocalLoss(alpha=0.1, gamma=2.0, reduction='mean')

optimizer = optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=1e-3 
)

# Slightly aggresisive scheduler to fix FP and FN
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_dl),
    epochs=NUM_EPOCHS,
    pct_start=0.3
)

print(f"Starting TRAINING on {len(train_ds)} samples for {NUM_EPOCHS} epochs...")
print(f"Validating on {len(val_ds)} samples.")

best_val_f1 = -1.0 
best_threshold = 0.5 # We will track the best threshold

for epoch in range(NUM_EPOCHS):
    
    # ==========================
    # 1. Training Phase
    # ==========================
    model.train()
    total_train_loss = 0.0
    
    # Lists for training confusion matrix (fixed 0.5 threshold for monitoring)
    train_preds_list = []
    train_labels_list = []
    
    pbar_train = tqdm(train_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for patchbatch, labels in pbar_train:
        patchbatch = patchbatch.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        
        logits = model(patchbatch)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step() # <--- Important: Step per batch for OneCycleLR
        
        total_train_loss += loss.item()
        pbar_train.set_postfix(loss=f"{loss.item():.4f}")
        
        # Monitor training with default 0.5 threshold
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_preds_list.append(preds.cpu())
            train_labels_list.append(labels.cpu())
        
    avg_train_loss = total_train_loss / len(train_dl)
    
    # Print Train CM (Just for sanity check)
    train_preds_np = torch.cat(train_preds_list).numpy()
    train_labels_np = torch.cat(train_labels_list).numpy()
    tn, fp, fn, tp = confusion_matrix(train_labels_np, train_preds_np, labels=[0, 1]).ravel()
    print(f"\n  [Train] CM (Thresh=0.5): TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    
    # ==========================
    # 2. Validation Phase (Strict Hard PoT)
    # ==========================
    model.eval()
    
    # --- *** NEW: Swap to Hard PoT weights for Validation *** ---
    # 1. Backup the current FP32 weights so we don't ruin training
    fp32_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # 2. Generate the strictly quantized PoT weights using our helper
    hard_pot_dict = get_hard_pot_state_dict(model)
    
    # 3. Load the PoT weights into the model
    model.load_state_dict(hard_pot_dict)
    # -------------------------------------------------------------
    
    total_val_loss = 0.0
    val_probs_list = []  
    val_labels_list = []
    
    pbar_val = tqdm(val_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val (PoT)]")
    with torch.no_grad():
        for patchbatch, labels in pbar_val:
            patchbatch = patchbatch.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1) 
            
            # The model is now running on strict PoT weights!
            logits = model(patchbatch)
            loss = criterion(logits, labels) 
            
            total_val_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            val_probs_list.append(probs.cpu())
            val_labels_list.append(labels.cpu())

    avg_val_loss = total_val_loss / len(val_dl)
    
    # Concatenate all batches
    val_probs_np = torch.cat(val_probs_list).numpy()
    val_labels_np = torch.cat(val_labels_list).numpy()
    
    # --- Metric Calculation ---
    current_best_thresh = 0.5
    current_best_preds = (val_probs_np > current_best_thresh).astype(int)
    current_best_f1 = f1_score(val_labels_np, current_best_preds, average='binary', zero_division=0)
    
    val_acc = accuracy_score(val_labels_np, current_best_preds)
    val_p = precision_score(val_labels_np, current_best_preds, average='binary', zero_division=0)
    val_r = recall_score(val_labels_np, current_best_preds, average='binary', zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(val_labels_np, current_best_preds, labels=[0, 1]).ravel()

    print(f"  [Val PoT] Best Thresh: {current_best_thresh:.2f}")
    print(f"  [Val PoT] CM: TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print(f"  Stats:  Loss: {avg_val_loss:.4f} | F1: {current_best_f1:.4f} | Prec: {val_p:.4f} | Rec: {val_r:.4f}")

    # --- Save Best Model ---
    if current_best_f1 > best_val_f1:
        best_val_f1 = current_best_f1
        best_threshold = current_best_thresh
        
        # Because we already loaded the hard_pot_dict into the model above,
        # model.state_dict() now inherently contains the strict PoT weights!
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
        print(f"  ---> New best PoT model saved! (F1: {best_val_f1:.4f})\n")
    else:
        print(f"  (Best F1 remains: {best_val_f1:.4f})\n")

    # --- *** NEW: Restore FP32 weights for the next training epoch *** ---
    model.load_state_dict(fp32_state_dict)
    # ---------------------------------------------------------------------

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