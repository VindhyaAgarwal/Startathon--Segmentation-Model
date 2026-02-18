%%writefile visualize.py
import cv2
import numpy as np
import os
from pathlib import Path

# Use the prediction folder we just created
input_folder = "/content/predictions" 
output_folder = "/content/visualizations"
os.makedirs(output_folder, exist_ok=True)

# Official Challenge Color Palette
color_map = {
    0: [0, 0, 0],        # Background
    1: [34, 139, 34],    # Trees
    2: [0, 255, 0],      # Lush Bushes
    3: [210, 180, 140],  # Dry Grass
    4: [139, 90, 43],    # Dry Bushes
    5: [128, 128, 0],    # Ground Clutter
    6: [139, 69, 19],    # Logs
    7: [128, 128, 128],  # Rocks
    8: [160, 82, 45],    # Landscape
    9: [135, 206, 235],  # Sky
}

image_files = list(Path(input_folder).glob("*.png"))
print(f"Colorizing {len(image_files)} predictions...")

for image_file in image_files:
    im = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
    if im is None: continue
    
    # Create colored image
    color_img = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    for val, color in color_map.items():
        color_img[im == val] = color[::-1] # Convert RGB to BGR for OpenCV
    
    cv2.imwrite(os.path.join(output_folder, image_file.name), color_img)

print(f"Visualizations ready in: {output_folder}")