import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# == Configuration ==
OBJ_NAME = "person_5"
OBJ_DIR = f"C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/data/objects/{OBJ_NAME}/"
SAVE_DIR = f"C:/Users/Gheffari Youcef/Videos/multi_object_3drecon/outputs/{OBJ_NAME}/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load camera poses
pose_file = os.path.join(OBJ_DIR, "poses.json")
pose_data = {}

if os.path.exists(pose_file):
    with open(pose_file, "r") as f:
        pose_data = json.load(f)
else:
    print(f"⚠️ poses.json not found. Proceeding with dummy poses.")

# Load all available images + their poses
images = []
poses = []
missing = 0

print("🔍 Loading images and poses...")
image_files = sorted([f for f in os.listdir(OBJ_DIR) if f.endswith(".png")])

for img_name in image_files:
    img_path = os.path.join(OBJ_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {img_name}")
        missing += 1
        continue

    image = Image.open(img_path).convert("RGB")
    images.append(torch.tensor(np.array(image) / 255.0).permute(2, 0, 1))  # (3, H, W)

    # Use pose from file if available, else dummy
    if img_name in pose_data:
        t = pose_data[img_name]["extrinsics"]["translation"]
        q = pose_data[img_name]["extrinsics"]["quaternion"]
    else:
        t = [0.0, 0.0, 0.0]
        q = [0.0, 0.0, 0.0, 1.0]
    poses.append((t, q))

print(f"✅ Loaded {len(images)} images | ❌ Skipped {missing} missing")

if len(images) == 0:
    raise RuntimeError("No valid images loaded. Check object folder content.")

# Simulate a dummy 3D volume (like TensoRF tensor)
H, W = images[0].shape[1], images[0].shape[2]
vol = torch.nn.Parameter(torch.rand(32, 32, 32), requires_grad=True)

# Optimizer
optimizer = torch.optim.Adam([vol], lr=1e-2)

# Dummy training loop (image matching loss)
print("🚀 Training dummy TensoRF tensor...")
for epoch in range(100):
    loss_total = 0.0
    for i in range(len(images)):
        img = images[i]
        proj = torch.mean(vol) * torch.ones_like(img)  # fake render output
        loss = torch.nn.functional.mse_loss(proj, img)
        loss_total += loss

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"📦 Epoch {epoch:03d} | Loss: {loss_total.item():.4f}")

# Save results
tensor_path = os.path.join(SAVE_DIR, "tensor.pt")
torch.save(vol.detach().cpu(), tensor_path)
print(f"✅ Saved tensor: {tensor_path}")

# Save slice for visualization
slice_path = os.path.join(SAVE_DIR, "tensor_slice.png")
plt.imshow(vol.detach().cpu().numpy()[16], cmap='viridis')
plt.title("Mid Slice of 3D Volume")
plt.savefig(slice_path)
print(f"🖼️ Saved slice visualization: {slice_path}")
