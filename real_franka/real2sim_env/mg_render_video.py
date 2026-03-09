#!/usr/bin/env python3
"""
Standalone video rendering script for BlockPAP-v1 generated HDF5 datasets.

Replays saved states from an HDF5 file and writes a video — no physics
simulation, just state restore + render (step_sim=False).

Demos are randomly sampled (with optional --seed) rather than taken
in order, so the video is a representative cross-section of the dataset.

Usage (standalone):
    python mg_render_video.py \
        --hdf5   .../blockpap_gen.hdf5 \
        --video  .../output.mp4

    # randomly sample 10 demos, skip every other frame
    python mg_render_video.py \
        --hdf5       .../blockpap_gen.hdf5 \
        --video      .../output.mp4 \
        --num-render 10 \
        --video-skip 2 \
        --cam-t      og \
        --seed       42

API (called from another script):
    from mg_render_video import render_hdf5_to_video
    render_hdf5_to_video(
        hdf5_path  = ".../blockpap_gen.hdf5",
        video_path = ".../output.mp4",
        cam_t      = "og",
        video_skip = 1,
        num_renders = None,
        seed        = None,
    )
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pick_and_place  # noqa: F401 – registers BlockPAP-v1


def render_hdf5_to_video(
    hdf5_path,
    video_path,
    cam_t="og",
    video_skip=1,
    num_renders=None,
    seed=None,
):
    """
    Replay saved states from an HDF5 file and write to video.

    Args:
        hdf5_path  : path to HDF5 file containing data/demo_*/states
        video_path : output .mp4 path
        cam_t      : camera calibration preset ('og', '0302', '0303')
        video_skip : render every Nth frame (1 = all frames)
        num_renders: max number of demos to render (None = all); demos are
                     randomly sampled (sorted by key after sampling)
        seed       : random seed for demo sampling (None = non-deterministic)
    """
    import imageio
    from mg_blockpap_wrapper import EnvManiskillBlockPAP

    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)

    env = EnvManiskillBlockPAP(
        env_name="BlockPAP-v1",
        render=False,
        render_offscreen=True,
        use_image_obs=False,
        cam_t=cam_t,
    )
    # gymnasium OrderEnforcing requires reset() before render()
    env.reset()

    writer = imageio.get_writer(video_path, fps=20)
    try:
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys())
            if num_renders is not None and num_renders < len(demo_keys):
                rng = np.random.default_rng(seed)
                demo_keys = rng.choice(demo_keys, size=num_renders, replace=False).tolist()
                demo_keys = sorted(demo_keys)
            print(f"  Replaying {len(demo_keys)} demos → {video_path}")
            for dk in demo_keys:
                states = f[f"data/{dk}/states"][:]
                for t, state_vec in enumerate(states):
                    if t % video_skip != 0:
                        continue
                    # step_sim=False: skip physics, restore pose exactly as saved.
                    # We do not want gravity to perturb the block between frames.
                    env.reset_to({"states": state_vec}, step_sim=False)
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        frame = frame.copy()
                        cv2.putText(frame, dk, (8, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3,
                                    cv2.LINE_AA)
                        cv2.putText(frame, dk, (8, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        writer.append_data(frame)
    finally:
        writer.close()
        env.env.close()

    print(f"  Video saved: {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render BlockPAP HDF5 demos to video by replaying saved states"
    )
    parser.add_argument("--hdf5",        type=str, required=True,
                        help="Input HDF5 file (data/demo_*/states)")
    parser.add_argument("--video",       type=str, required=True,
                        help="Output .mp4 path")
    parser.add_argument("--cam-t",       type=str, default="og",
                        help="Camera calibration preset (default: og)")
    parser.add_argument("--video-skip",  type=int, default=1,
                        help="Render every Nth frame (default: 1 = all)")
    parser.add_argument("--num-render",  type=int, default=None,
                        help="Max demos to render, randomly sampled (default: all)")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed for demo sampling (default: non-deterministic)")
    args = parser.parse_args()

    render_hdf5_to_video(
        hdf5_path   = args.hdf5,
        video_path  = args.video,
        cam_t       = args.cam_t,
        video_skip  = args.video_skip,
        num_renders = args.num_render,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
