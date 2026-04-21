import zipfile
import os

# Make sure the destination folder exists
os.makedirs("/scratch/deepaprakash.ece22.itbhu/birdvdrone/", exist_ok=True)

# List your zip files
zip_files = ["/scratch/deepaprakash.ece22.itbhu/challenge-master.zip", "/scratch/deepaprakash.ece22.itbhu/wosdetc_train_videos.zip"]

for zf in zip_files:
    with zipfile.ZipFile(zf, 'r') as zip_ref:
        zip_ref.extractall("/scratch/deepaprakash.ece22.itbhu/birdvdrone")

print("All files extracted into birdvdrone/")
