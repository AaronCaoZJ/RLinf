#!/usr/bin/env python3
"""
Visualize camera observations from a generated MimicGen HDF5.

Reads obs/external_cam_image from demo_0, samples every --step timesteps,
arranges in a grid (max --cols per row), and saves next to the HDF5 as
{prefix}_camera_view.png  (prefix = part before "_gen" in the HDF5 filename).

Usage:
    python real_franka/real2sim_env/check_camera_view.py \
        --hdf5 real_franka/real2sim_env/mg_dataset/blockpap_gen.hdf5
"""
import argparse
import os

import h5py
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5",  type=str, required=True,
                        help="Path to the generated HDF5 file")
    parser.add_argument("--demo",  type=str, default="demo_0",
                        help="Demo key to read (default: demo_0)")
    parser.add_argument("--step",  type=int, default=20,
                        help="Sample every N timesteps (default: 20)")
    parser.add_argument("--cols",  type=int, default=5,
                        help="Max images per row (default: 5)")
    parser.add_argument("--cam",   type=str, default="external_cam",
                        help="Camera name (default: external_cam); "
                             "reads obs/{cam}_image from the HDF5")
    args = parser.parse_args()

    obs_key = args.cam if args.cam.endswith("_image") else f"{args.cam}_image"
    with h5py.File(args.hdf5, "r") as f:
        key = f"data/{args.demo}/obs/{obs_key}"
        if key not in f:
            available = list(f[f"data/{args.demo}/obs"].keys())
            raise KeyError(f"Key '{key}' not found. Available obs keys: {available}")
        imgs = f[key][:]   # (T, H, W, 3)

    T = imgs.shape[0]
    indices = list(range(0, T, args.step))
    frames = [imgs[i] for i in indices]
    n = len(frames)

    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols

    H, W = frames[0].shape[:2]
    canvas = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)

    for idx, img in enumerate(frames):
        r, c = divmod(idx, cols)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = img

    # Derive prefix from "{prefix}_gen.hdf5" → "{prefix}_camera_view.png"
    basename = os.path.splitext(os.path.basename(args.hdf5))[0]  # e.g. "blockpap_gen"
    if "_gen" in basename:
        prefix = basename[:basename.rindex("_gen")]               # e.g. "blockpap"
    else:
        prefix = basename
    out_dir = os.path.dirname(os.path.abspath(args.hdf5))
    out_path = os.path.join(out_dir, f"{prefix}_camera_view.png")
    Image.fromarray(canvas).save(out_path)
    print(f"Saved {n} frames (t=0,{args.step},...) → {out_path}")
    print(f"Grid: {rows} rows × {cols} cols, each frame {W}×{H}")


if __name__ == "__main__":
    main()
