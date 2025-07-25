import os
import json
from tqdm import tqdm
from ultralytics import YOLO
import cv2

# ✅ Paths
IMG_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tum_freiburg1_desk/rgb_indexed"
OUT_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/detections/"
os.makedirs(OUT_DIR, exist_ok=True)

# ✅ Load YOLOv8n (COCO weights)
model = YOLO("yolov8n.pt")

# ✅ Keep only driving-relevant classes
TARGET_CLASSES = ["car", "truck", "bus", "person"]

# ✅ Process selected range of frames only (e.g. 000050–000080)
frame_range = range(50, 81)  # inclusive of 80

for i in tqdm(frame_range, desc="Running detection"):
    fname = f"{i:06d}.png"
    img_path = os.path.join(IMG_DIR, fname)

    if not os.path.exists(img_path):
        print(f"⚠️ Skipping missing frame: {fname}")
        continue

    image = cv2.imread(img_path)

    # Run YOLOv8 inference
    results = model(image)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        if cls_name not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append({
            "class": cls_name,
            "bbox": [x1, y1, x2, y2],
            "conf": conf
        })

    # Save detection results
    out_path = os.path.join(OUT_DIR, fname.replace(".png", ".json"))
    with open(out_path, 'w') as f:
        json.dump(detections, f, indent=2)

print("✅ Detection complete. Results saved to:", OUT_DIR)
