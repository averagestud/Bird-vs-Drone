import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# =======================
# 1. Paths & Setup
# =======================
IMG_SIZE = 640
NUM_SYNTHETIC_VIDEOS = 20      
FRAMES_PER_SYNTH_VIDEO = 4050  # Total = 81,000 frames

# Your existing DRDO paths
RAW_VIDEO_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/wosdetc_train_videos/"
ANNOTATION_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/challenge-master/annotations/"
FRAME_SAVE_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/wosdetc_train_frames/"

# --- NEW: CUB-200-2011 Paths ---
# Point these to the specific folders you just extracted
CUB_IMAGES_DIR = "/scratch/deepaprakash.ece22.itbhu/bird_cutouts/CUB_200_2011/images/"
CUB_MASKS_DIR = "/scratch/deepaprakash.ece22.itbhu/bird_cutouts/segmentations/"

# Synthetic output paths
SYNTH_FRAME_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/synthetic_train_frames/"
SYNTH_ANN_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/synthetic_annotations/"
SYNTH_VIDEO_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/synthetic_videos/" 

for d in [SYNTH_FRAME_DIR, SYNTH_ANN_DIR, SYNTH_VIDEO_DIR]:
    os.makedirs(d, exist_ok=True)

# =======================
# 2. Mine Empty Backgrounds
# =======================
def parse_wosdetc_line(line):
    tokens = line.strip().split()
    if len(tokens) < 2: return None, []
    try:
        frame_num, num_objs = int(tokens[0]), int(tokens[1])
    except ValueError: return None, []
    
    objs, offset = [], 2
    for _ in range(num_objs):
        if offset + 5 > len(tokens): break
        try:
            x, y, w, h = float(tokens[offset]), float(tokens[offset+1]), float(tokens[offset+2]), float(tokens[offset+3])
            cls = 1 if tokens[offset+4].lower() == "drone" else 0
            objs.append([cls, x, y, w, h])
            offset += 5
        except ValueError:
            offset += 5
            continue
    return frame_num, objs

print("Mining existing dataset for empty background frames...")
empty_background_paths = []

for ann_file in tqdm(sorted(os.listdir(ANNOTATION_DIR))):
    if not ann_file.endswith('.txt'): continue
    ann_path = os.path.join(ANNOTATION_DIR, ann_file)
    basename = os.path.splitext(ann_file)[0]
    vid_frame_dir = os.path.join(FRAME_SAVE_DIR, basename)
    
    if not os.path.exists(vid_frame_dir): continue

    try:
        with open(ann_path, 'r') as f:
            for line in f:
                frame_num, objs = parse_wosdetc_line(line)
                if frame_num is not None and len(objs) == 0:
                    frame_file = os.path.join(vid_frame_dir, f"frame_{frame_num:08d}.jpg")
                    if os.path.exists(frame_file):
                        empty_background_paths.append(frame_file)
    except FileNotFoundError:
        continue

print(f"Found {len(empty_background_paths)} clean canvas frames.")

# =======================
# 3. Process CUB-200 Pairs
# =======================
def add_sensor_noise(image, noise_level):
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy_img = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def process_cub_bird(image_path, mask_path, target_w, target_h):
    """Combines the raw JPG and the ground-truth PNG mask."""
    # Read the bird image (convert to grayscale)
    bird_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    # Read the segmentation mask (ensure grayscale)
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if bird_img is None or mask_img is None:
        return None, None
        
    # Resize both to the tiny target dimensions
    bird_img = cv2.resize(bird_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    mask_img = cv2.resize(mask_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if random.choice([True, False]): 
        bird_img = cv2.flip(bird_img, 1)
        mask_img = cv2.flip(mask_img, 1)

    # Convert the 0-255 mask into a 0.0-1.0 alpha multiplier
    alpha_mask = mask_img / 255.0

    return bird_img, alpha_mask

# =======================
# 4. Match Images to Masks
# =======================
print("Pairing CUB images with their segmentation masks...")
bird_pairs = []

# Walk through the images directory
for img_path in Path(CUB_IMAGES_DIR).rglob("*.jpg"):
    # The CUB dataset matches folder and file names exactly between images and segmentations
    # Example: images/001.Albatross/bird.jpg -> segmentations/001.Albatross/bird.png
    relative_path = img_path.relative_to(CUB_IMAGES_DIR)
    mask_path = Path(CUB_MASKS_DIR) / relative_path.with_suffix(".png")
    
    if mask_path.exists():
        bird_pairs.append((str(img_path), str(mask_path)))

if not bird_pairs:
    raise FileNotFoundError("Could not match any JPGs to their PNG segmentations. Check your folder paths!")

print(f"Successfully paired {len(bird_pairs)} birds. Generating 81,000 synthetic frames...")

# =======================
# 5. Generation Pipeline
# =======================
for vid_idx in range(NUM_SYNTHETIC_VIDEOS):
    synth_vid_name = f"synthetic_bird_sequence_{vid_idx:03d}"
    
    with open(os.path.join(SYNTH_VIDEO_DIR, f"{synth_vid_name}.mp4"), 'w') as f: 
        f.write("dummy")
    
    vid_out_dir = os.path.join(SYNTH_FRAME_DIR, synth_vid_name)
    os.makedirs(vid_out_dir, exist_ok=True)
    
    ann_out_path = os.path.join(SYNTH_ANN_DIR, f"{synth_vid_name}.txt")
    
    with open(ann_out_path, "w") as ann_file:
        for f_idx in tqdm(range(FRAMES_PER_SYNTH_VIDEO), desc=f"Video {vid_idx+1}/{NUM_SYNTHETIC_VIDEOS}"):
            
            bg_path = random.choice(empty_background_paths)
            bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
            bg = cv2.resize(bg, (IMG_SIZE, IMG_SIZE))
            
            # Pick a random paired bird & mask
            img_path, mask_path = random.choice(bird_pairs)
            bw, bh = random.randint(15, 45), random.randint(15, 45)
            
            bird_gray, alpha_mask = process_cub_bird(img_path, mask_path, bw, bh)
            if bird_gray is None: continue
                
            x = random.randint(0, IMG_SIZE - bw - 1)
            y = random.randint(0, IMG_SIZE - bh - 1)
            
            # Blend
            roi = bg[y:y+bh, x:x+bw]
            blended = (bird_gray * alpha_mask) + (roi * (1.0 - alpha_mask))
            bg[y:y+bh, x:x+bw] = blended.astype(np.uint8)
            
            # Kria Edge Device Noise Simulation
            bg = cv2.GaussianBlur(bg, (3, 3), 0)
            bg = add_sensor_noise(bg, noise_level=random.randint(5, 18))
            
            cv2.imwrite(os.path.join(vid_out_dir, f"frame_{f_idx:08d}.jpg"), bg)
            ann_file.write(f"{f_idx} 1 {x} {y} {bw} {bh} bird\n")

print("\nDataset perfectly balanced! You now have an additional 81,000 bird frames.")