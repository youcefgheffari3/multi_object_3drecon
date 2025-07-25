import os
import json

# === CONFIGURATION ===
OBJ_DIR = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/objects/person_5/"
RGB_TIMESTAMP_FILE = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tum_freiburg1_desk/rgb.txt"
POSE_FILE = "C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/tum_freiburg1_desk/groundtruth.txt"

# === CAMERA INTRINSICS (TUM Freiburg1) ===
intrinsics = [
    525.0, 0.0, 319.5,
    0.0, 525.0, 239.5,
    0.0, 0.0, 1.0
]

# === Load RGB timestamps ===
timestamps = []
with open(RGB_TIMESTAMP_FILE, 'r') as f:
    for line in f:
        if line.startswith("#") or len(line.strip()) == 0:
            continue
        parts = line.strip().split()
        if len(parts) >= 1:
            timestamps.append(parts[0])

# === Load poses from groundtruth.txt ===
pose_map = {}
with open(POSE_FILE, 'r') as f:
    for line in f:
        if line.startswith("#") or len(line.strip()) == 0:
            continue
        parts = line.strip().split()
        if len(parts) != 8:
            continue
        ts = parts[0]
        tx, ty, tz = map(float, parts[1:4])
        qx, qy, qz, qw = map(float, parts[4:])
        pose_map[ts] = {
            "translation": [tx, ty, tz],
            "quaternion": [qx, qy, qz, qw]
        }

# === Generate poses.json for each image in object folder ===
pose_json = {}
cropped_imgs = sorted([f for f in os.listdir(OBJ_DIR) if f.endswith(".png")])

used = 0
for img_name in cropped_imgs:
    try:
        frame_idx = int(os.path.splitext(img_name)[0])
    except ValueError:
        print(f"⚠️ Skipping non-indexed image name: {img_name}")
        continue

    if frame_idx >= len(timestamps):
        print(f"⚠️ Frame {img_name} → index {frame_idx} out of range.")
        continue

    ts = timestamps[frame_idx]
    if ts not in pose_map:
        print(f"⚠️ No pose found for timestamp {ts}")
        continue

    pose_json[img_name] = {
        "intrinsics": intrinsics,
        "extrinsics": {
            "translation": pose_map[ts]["translation"],
            "quaternion": pose_map[ts]["quaternion"]
        }
    }
    used += 1

# === Save output JSON ===
out_path = os.path.join(OBJ_DIR, "poses.json")
with open(out_path, "w") as f:
    json.dump(pose_json, f, indent=2)

print(f"✅ poses.json rebuilt with {used} valid frames → {out_path}")
