import os
import cv2
import json
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort

# Directories
IMG_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tum_freiburg1_desk/rgb_indexed"
DET_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/detections/"
OUT_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tracks/"
os.makedirs(OUT_DIR, exist_ok=True)

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Get sorted frame list
frames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])

for fname in tqdm(frames, desc="Tracking"):
    img_path = os.path.join(IMG_DIR, fname)
    det_path = os.path.join(DET_DIR, fname.replace('.png', '.json'))

    # Read image and detections
    image = cv2.imread(img_path)
    if not os.path.exists(det_path):
        continue
    with open(det_path, 'r') as f:
        detections = json.load(f)

    # Prepare input for Deep SORT: [[bbox], confidence, class_name]
    inputs = []
    for det in detections:
        if det["class"] not in ["car", "person", "truck", "bus"]:  # Filter classes
            continue
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        conf = det["conf"]
        cls_name = det["class"]
        inputs.append([bbox, conf, cls_name])  # ✅ Correct format

    # Update tracker
    tracks = tracker.update_tracks(inputs, frame=image)

    # Store per-frame tracking results
    frame_tracks = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # [left, top, right, bottom]
        cls = track.det_class   # ✅ Fixed: use .det_class instead of .get_class()
        frame_tracks.append({
            "id": int(track_id),
            "class": cls,
            "bbox": list(map(float, ltrb))
        })

    # Save tracking output
    out_path = os.path.join(OUT_DIR, fname.replace('.png', '.json'))
    with open(out_path, 'w') as f:
        json.dump(frame_tracks, f, indent=2)

print("✅ Tracking complete. Results saved to:", OUT_DIR)
