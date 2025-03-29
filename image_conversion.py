from PIL import Image
import numpy as np

# Open and resize the image to the required size (e.g., 64x64)
img = Image.open("DT (12).jpg").resize((64, 64))
img = img.convert("RGB")  # ensure it has 3 channels
img_array = np.array(img).astype(np.float32)  # convert to float32 if needed
img_array = img_array.transpose(2, 0, 1)  # change from HxWxC to CxHxW

# Optionally normalize pixel values, e.g., to [0,1]
img_array /= 255.0

# Save the raw pixel values to a binary file
img_array.tofile("DT (12).bin")
