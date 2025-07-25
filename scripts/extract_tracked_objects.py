import os
import json
import cv2
from tqdm import tqdm
from collections import defaultdict

# === Paths ===
IMG_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tum_freiburg1_desk/rgb_indexed"
TRACK_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tracks"
OUT_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/objects"
os.makedirs(OUT_DIR, exist_ok=True)

# === Collect all tracked frames ===
track_files = sorted([f for f in os.listdir(TRACK_DIR) if f.endswith('.json')])

# === Group detections by object ID ===
object_data = defaultdict(list)

for fname in tqdm(track_files, desc="Processing tracks"):
    frame_id = fname.replace(".json", "")
    image_path = os.path.join(IMG_DIR, f"{frame_id}.png")
    if not os.path.exists(image_path):
        continue
    image = cv2.imread(image_path)

    with open(os.path.join(TRACK_DIR, fname), 'r') as f:
        tracks = json.load(f)

    for obj in tracks:
        obj_id = obj["id"]
        cls = obj["class"]
        bbox = list(map(int, obj["bbox"]))  # [x1, y1, x2, y2]

        crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if crop.size == 0:
            continue

        obj_folder = os.path.join(OUT_DIR, f"{cls}_{obj_id}")
        os.makedirs(obj_folder, exist_ok=True)

        save_path = os.path.join(obj_folder, f"{frame_id}.png")
        cv2.imwrite(save_path, crop)

print("✅ Object image folders created at:", OUT_DIR)
