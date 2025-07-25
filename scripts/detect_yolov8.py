import os
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO

# Change as needed
dataset_dir = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tum_freiburg1_desk/rgb"
output_dir = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/detections/"
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv8 model (small version)
model = YOLO("yolov8n.pt")

# Supported classes (COCO)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Process all frames
image_files = sorted(os.listdir(dataset_dir))

for fname in tqdm(image_files, desc="Detecting"):
    if not fname.endswith('.png'):
        continue

    frame_path = os.path.join(dataset_dir, fname)
    results = model(frame_path, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = COCO_CLASSES[cls_id]
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        detections.append({
            "class": cls_name,
            "conf": conf,
            "bbox": [x1, y1, x2, y2]
        })

    # Save detection JSON per frame
    save_path = os.path.join(output_dir, fname.replace(".png", ".json"))
    with open(save_path, 'w') as f:
        json.dump(detections, f, indent=2)

    # Optional: Save visualized output
    # annotated = results.plot()
    # cv2.imwrite(os.path.join(output_dir, "vis_" + fname), annotated)

print("✅ Detection complete. Saved to:", output_dir)
