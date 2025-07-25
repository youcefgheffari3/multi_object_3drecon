import os
import cv2
import json
from tqdm import tqdm
import numpy as np

# Paths
IMG_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tum_freiburg1_desk/rgb"
TRACKS_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tracks/"
OUT_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/background/"
os.makedirs(OUT_DIR, exist_ok=True)

# Get sorted frame list
frames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])

for fname in tqdm(frames, desc="Generating background frames"):
    img_path = os.path.join(IMG_DIR, fname)
    track_path = os.path.join(TRACKS_DIR, fname.replace('.png', '.json'))

    image = cv2.imread(img_path)
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255  # white mask

    # If we have tracked objects, mark their regions
    if os.path.exists(track_path):
        with open(track_path, 'r') as f:
            tracks = json.load(f)

        for obj in tracks:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=0, thickness=-1)  # black = remove

    # Apply mask
    background = cv2.bitwise_and(image, image, mask=mask)

    # Optional: inpaint to fill in removed regions (OpenCV inpainting)
    # background = cv2.inpaint(image, 255 - mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Save masked image
    out_path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_path, background)

print("✅ Background frames saved to:", OUT_DIR)
