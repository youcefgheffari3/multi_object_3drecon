import os
import numpy as np
import cv2
from glob import glob


def load_tum_rgb_depth_poses(folder):
    rgb_files = sorted(glob(os.path.join(folder, 'rgb', '*.png')))
    depth_files = sorted(glob(os.path.join(folder, 'depth', '*.png')))

    # Load ground truth poses
    poses = {}
    with open(os.path.join(folder, 'groundtruth.txt')) as f:
        for line in f:
            if line.startswith('#'): continue
            vals = list(map(float, line.strip().split()))
            timestamp = vals[0]
            t = np.array(vals[1:4])
            q = np.array(vals[4:])  # Quaternion
            poses[timestamp] = (t, q)

    return rgb_files, depth_files, poses


# Usage
if __name__ == '__main__':
    rgb_files, depth_files, poses = load_tum_rgb_depth_poses('data/tum_freiburg1_desk/')
    print(f"{len(rgb_files)} RGB frames loaded.")
    print(f"{len(poses)} poses available.")
