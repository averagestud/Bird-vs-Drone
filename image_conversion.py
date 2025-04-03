from PIL import Image
import numpy as np

# Open the image, resize it to the desired dimensions (e.g., 2048x2048)
img = Image.open("DT.jpeg").resize((1024, 1024))

# Convert the image to grayscale (L mode means 1 channel)
img = img.convert("L")

# Convert the image to a NumPy array of type float32
img_array = np.array(img).astype(np.float32)

# Optionally, normalize pixel values to [0,1] if desired (uncomment the next line)
# img_array /= 255.0

# Expand dimensions to have shape (1, H, W) if needed by your model
img_array = np.expand_dims(img_array, axis=0)  # Now shape is (1, 2048, 2048)

# Save the array as a binary file
img_array.tofile("DT_Test_gray.bin")
print("Saved grayscale .bin file with shape:", img_array.shape)
