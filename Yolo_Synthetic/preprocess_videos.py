import os
import cv2
from tqdm import tqdm
import glob
import numpy as np

# --- Configuration ---
# Input directories (same as your script)
RAW_VIDEO_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/wosdetc_train_videos/"
ANNOTATION_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/challenge-master/annotations/"

# *** New output directory for fast-loading frames ***
FRAME_SAVE_DIR = "/scratch/deepaprakash.ece22.itbhu/birdvdrone/wosdetc_train_frames/" 

# Settings (must match your training script)
MAX_FRAMES_PER_VIDEO = 5000
IMG_SIZE = 640

# --- Start Pre-processing ---
# os.makedirs(FRAME_SAVE_DIR, exist_ok=True)
print(f"Saving extracted frames to: {FRAME_SAVE_DIR}")

# --- Find videos that have annotations ---
video_paths_to_process = []
for ann_file in sorted(os.listdir(ANNOTATION_DIR)):
    if not ann_file.endswith('.txt'):
        continue
    basename = os.path.splitext(ann_file)[0]
    videos_found = glob.glob(os.path.join(RAW_VIDEO_DIR, f"{basename}*"))
    if len(videos_found) > 0:
        video_paths_to_process.append(videos_found[0])

print(f"Found {len(video_paths_to_process)} videos with annotations to process.")

# --- Loop over all videos and extract frames ---
for video_path in tqdm(video_paths_to_process, desc="Extracting Videos"):
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create a specific directory for this video's frames
    video_frame_dir = os.path.join(FRAME_SAVE_DIR, video_basename)
    os.makedirs(video_frame_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        continue
        
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nframes = min(nframes, MAX_FRAMES_PER_VIDEO)
    
    for frame_idx in range(nframes):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale and resize (same as your original dataset)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Save the frame as a JPEG image
        # Use 8-digit padding for correct file ordering
        save_path = os.path.join(video_frame_dir, f"frame_{frame_idx:08d}.jpg")
        cv2.imwrite(save_path, img)
        
    cap.release()

print("--- Frame extraction complete! ---")
print(f"Your frames are now in: {FRAME_SAVE_DIR}")