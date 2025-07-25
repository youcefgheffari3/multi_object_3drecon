import os
import cv2
import json
from tqdm import tqdm

# Directories
IMG_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tum_freiburg1_desk/rgb"
TRACKS_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tracks/"
OUT_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/objects/"
os.makedirs(OUT_DIR, exist_ok=True)

# Load and sort all frame names
frames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])

for idx, fname in tqdm(enumerate(frames), total=len(frames), desc="Extracting"):
    img_path = os.path.join(IMG_DIR, fname)
    track_path = os.path.join(TRACKS_DIR, fname.replace('.png', '.json'))

    if not os.path.exists(track_path):
        continue

    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    with open(track_path, 'r') as f:
        track_data = json.load(f)

    for obj in track_data:
        track_id = obj["id"]
        cls = obj["class"]
        x1, y1, x2, y2 = map(int, obj["bbox"])

        # Clamp to image bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Save path: objects/class_id/frame_id.png
        obj_dir = os.path.join(OUT_DIR, f"{cls}_{track_id}")
        os.makedirs(obj_dir, exist_ok=True)

        out_name = os.path.join(obj_dir, f"{idx:06d}.png")
        cv2.imwrite(out_name, crop)

print("✅ Object sequences saved to:", OUT_DIR)
