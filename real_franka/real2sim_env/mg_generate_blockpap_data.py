#!/usr/bin/env python3
"""
MimicGen data generation script for BlockPAP-v1 — direct LeRobot v2.1 output.

Generates successful demonstrations and writes them immediately as a LeRobot v2.1
dataset.  Failed demos are silently discarded (no storage, no rendering).

Output layout:
  <lerobot-dir>/
    meta/
      info.json
      tasks.jsonl
      episodes.jsonl
    data/
      chunk-000/episode_000000.parquet
      chunk-000/episode_000001.parquet
      ...
    videos/
      chunk-000/observation.images.image/episode_000000.mp4
      chunk-000/observation.images.wrist_image/episode_000000.mp4  (black — no wrist cam)
      chunk-000/observation.images.back_image/episode_000000.mp4
      ...

Multi-GPU (6 workers, GPUs 0-5):
    python mg_generate_blockpap_data.py \\
        --src    .../blockpap_src.hdf5 \\
        --lerobot-dir .../blockpap_lerobot \\
        --num-demos 900 --num-workers 6 --gpu-ids 0,1,2,3,4,5

Single-GPU:
    python mg_generate_blockpap_data.py \\
        --src    .../blockpap_src.hdf5 \\
        --lerobot-dir .../blockpap_lerobot \\
        --num-demos 200
"""
import argparse
import json
import multiprocessing
import os
import sys
import traceback

import h5py
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mg_blockpap_interface  # noqa: F401 – registers MG_BlockPAP
import pick_and_place          # noqa: F401 – registers BlockPAP-v1

from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator
from mimicgen.env_interfaces.base import make_interface
import mimicgen.utils.file_utils as MG_FileUtils


# ── Task spec ────────────────────────────────────────────────────────────────

def create_task_spec():
    """
    2-subtask task spec — mirrors MimicGen's official Stack task convention:

      1. Grasp (ref: block, term: grasped)
         offset=(10,20) naturally covers the lift motion after the grasp
         signal fires, so the arm is already elevated at the transition.

      2. Place (ref: target_coaster, term: None)
         5-step interpolation bridges from the lifted position to the
         start of the coaster-relative place trajectory.
    """
    spec = MG_TaskSpec()
    spec.add_subtask(
        object_ref="block",
        subtask_term_signal="grasped",
        subtask_term_offset_range=(10, 20),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs=dict(nn_k=3),
        action_noise=1e-6,
        num_interpolation_steps=0,
        num_fixed_steps=0,
    )
    spec.add_subtask(
        object_ref="target_coaster",
        subtask_term_signal=None,
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs=dict(nn_k=3),
        action_noise=1e-6,
        num_interpolation_steps=5,
        num_fixed_steps=0,
    )
    return spec


# ── LeRobot v2.1 format helpers ───────────────────────────────────────────────

_VIDEO_KEYS = ["observation.images.image", "observation.images.wrist_image", "observation.images.back_image"]
_GRIPPER_OPEN_THRESHOLD_M = 0.04


def _quat_sapien_to_scipy(q):
    """SAPIEN [w,x,y,z] → scipy [x,y,z,w]."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def _wrap_angle(a):
    """Wrap angle array to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _obs_list_to_state_action(obs_list):
    """
    Convert a list of T obs dicts to LeRobot state/action arrays.

    State  (T, 7): [x, y, z, roll, pitch, yaw, gripper_width]
        Action (T, 7): [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_abs01]
      (last action repeats second-to-last)

        For pi0.5 training, keep arm motion as delta and encode gripper as
        absolute binary command in {0, 1}. The gripper target for step t is
        derived from the next state's width (t + 1), matching standard action
        alignment in sequence datasets.
    """
    from scipy.spatial.transform import Rotation as _R

    T = len(obs_list)
    ee_pos   = np.array([o["ee_pos"]   for o in obs_list], dtype=np.float32)   # (T,3)
    ee_quat  = np.array([o["ee_quat"]  for o in obs_list], dtype=np.float32)   # (T,4)
    jpos     = np.array([o["joint_pos"] for o in obs_list], dtype=np.float32)  # (T,9)

    euler = np.array([
        _R.from_quat(_quat_sapien_to_scipy(ee_quat[t])).as_euler("xyz")
        for t in range(T)
    ], dtype=np.float32)

    gripper = jpos[:, 7] + jpos[:, 8]   # total width in meters (0 ~ 0.08)
    gripper_abs01 = (gripper >= _GRIPPER_OPEN_THRESHOLD_M).astype(np.float32)

    states = np.zeros((T, 7), dtype=np.float32)
    states[:, :3] = ee_pos
    states[:, 3:6] = euler
    states[:, 6]   = gripper

    # Match SFT training convention: z+10cm, euler_z-45°
    states[:, 2] += 0.1
    states[:, 5] += -np.pi / 4

    actions = np.zeros((T, 7), dtype=np.float32)
    for t in range(T - 1):
        actions[t, :3] = ee_pos[t + 1] - ee_pos[t]
        actions[t, 3:6] = _wrap_angle(euler[t + 1] - euler[t])
        actions[t, 6]   = gripper_abs01[t + 1]
    if T > 1:
        actions[-1] = actions[-2]

    return states, actions


def _trim_episode(obs_list, state_vecs,
                  head_min_skip=2,
                  jump_thresh=0.05,
                  tail_motion_thresh=0.001,
                  tail_min_frames=3):
    """
    Trim head/tail of a MimicGen episode:

    Head:
      - Always skip the first head_min_skip frames (interpolation artifacts).
      - Additionally skip any frame where the EE position jump from the
        previous frame exceeds jump_thresh (teleportation artifact).

    Tail:
      - Walk backwards and drop trailing frames where the EE barely moves
        (robot idling after task completion). Stop trimming once we find
        tail_min_frames consecutive frames with meaningful motion.
    """
    T = len(obs_list)
    ee_pos = np.array([o["ee_pos"] for o in obs_list], dtype=np.float32)

    # ── head trim ────────────────────────────────────────────────────────────
    start = head_min_skip
    for i in range(head_min_skip + 1, min(head_min_skip + 10, T)):
        delta = float(np.linalg.norm(ee_pos[i] - ee_pos[i - 1]))
        if delta > jump_thresh:
            start = i  # extend head skip past the jump
        else:
            break  # first non-jump frame found

    # ── tail trim ────────────────────────────────────────────────────────────
    end = T
    static_run = 0
    for t in range(T - 1, start, -1):
        delta = float(np.linalg.norm(ee_pos[t] - ee_pos[t - 1]))
        if delta < tail_motion_thresh:
            static_run += 1
        else:
            if static_run >= tail_min_frames:
                end = t + 1   # trim the trailing static segment
            break

    end = max(start + 1, end)   # always keep at least 1 frame
    return obs_list[start:end], state_vecs[start:end]


def _collect_replay_frames(env, state_vecs, frame_skip=1):
    """Replay env states (no physics) and collect RGB frames from front + side cameras.
    Both cameras use the sensor pipeline (no sky shader) → black background."""
    front_frames, side_frames = [], []
    front_sensor = env.base_env._sensors.get("external_cam")
    side_sensor  = env.base_env._sensors.get("back_cam")

    def _sensor_to_rgb(sensor):
        sensor.capture()
        obs = sensor.get_obs(rgb=True, depth=False, position=False, segmentation=False)
        rgb = obs["rgb"]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return rgb

    for t, sv in enumerate(state_vecs):
        if t % frame_skip != 0:
            continue
        env.reset_to({"states": sv}, step_sim=False)
        # front: use external_cam sensor for consistent black background
        if front_sensor is not None:
            front_frames.append(_sensor_to_rgb(front_sensor))
        else:
            frame = env.render(mode="rgb_array")
            if frame is not None:
                front_frames.append(frame.copy())
        # side
        if side_sensor is not None:
            side_frames.append(_sensor_to_rgb(side_sensor))
    return front_frames, side_frames


def _encode_video(frames, path, fps, codec="libx264", pix_fmt="yuv420p", img_size=None):
    """Encode a list of (H,W,3) uint8 frames to MP4 via ffmpeg subprocess."""
    import subprocess
    if not frames:
        return
    h, w = frames[0].shape[:2]
    if img_size is not None and (h != img_size or w != img_size):
        import cv2
        frames = [cv2.resize(f, (img_size, img_size)) for f in frames]
        h = w = img_size
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", codec, "-pix_fmt", pix_fmt,
        "-loglevel", "error",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg returned {rc} encoding {path}")


def _write_episode_parquet(path, states, actions, frames, side_frames, ep_idx, frame_offset, fps, img_size):
    """Write one episode to parquet with inline images + state/action/timestamps/indices."""
    from datasets import Dataset, Features, Sequence, Value, Image as ImageFeature
    from PIL import Image as PIL_Image
    import cv2 as _cv2

    T = len(states)

    # Convert frames to PIL Images (resize if requested)
    if frames:
        if img_size is not None:
            front_pil = [
                PIL_Image.fromarray(_cv2.resize(f, (img_size, img_size)))
                for f in frames
            ]
            bk = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            front_pil = [PIL_Image.fromarray(f) for f in frames]
            bk = np.zeros_like(frames[0])
    else:
        sz = img_size or 480
        bk = np.zeros((sz, sz, 3), dtype=np.uint8)
        front_pil = [PIL_Image.fromarray(bk)] * T

    black_pil = [PIL_Image.fromarray(bk)] * T

    # side camera frames
    if side_frames and len(side_frames) == T:
        if img_size is not None:
            side_pil = [
                PIL_Image.fromarray(_cv2.resize(f, (img_size, img_size)))
                for f in side_frames
            ]
        else:
            side_pil = [PIL_Image.fromarray(f) for f in side_frames]
    else:
        side_pil = black_pil

    features = Features({
        "image":         ImageFeature(),
        "wrist_image":   ImageFeature(),
        "back_image":    ImageFeature(),
        "state":         Sequence(Value("float32"), length=7),
        "actions":       Sequence(Value("float32"), length=7),
        "timestamp":     Value("float32"),
        "frame_index":   Value("int64"),
        "episode_index": Value("int64"),
        "index":         Value("int64"),
        "task_index":    Value("int64"),
    })
    data = {
        "image":         front_pil,
        "wrist_image":   black_pil,
        "back_image":    side_pil,
        "state":         states.tolist(),
        "actions":       actions.tolist(),
        "timestamp":     (np.arange(T, dtype=np.float32) / fps).tolist(),
        "frame_index":   np.arange(T, dtype=np.int64).tolist(),
        "episode_index": np.full(T, ep_idx, dtype=np.int64).tolist(),
        "index":         (np.arange(T, dtype=np.int64) + frame_offset).tolist(),
        "task_index":    np.zeros(T, dtype=np.int64).tolist(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Dataset.from_dict(data, features=features).to_parquet(path)


def _episode_parquet_path(base_dir, ep_idx, chunks_size):
    chunk = ep_idx // chunks_size
    return os.path.join(base_dir, "data",
                        f"chunk-{chunk:03d}",
                        f"episode_{ep_idx:06d}.parquet")


def _episode_video_path(base_dir, video_key, ep_idx, chunks_size):
    chunk = ep_idx // chunks_size
    return os.path.join(base_dir, "videos",
                        f"chunk-{chunk:03d}", video_key,
                        f"episode_{ep_idx:06d}.mp4")


def _write_lerobot_episode(
    base_dir, ep_idx, frame_offset, states, actions,
    frames, side_frames, chunks_size, fps, codec, pix_fmt, img_size, task_text
):
    """Write parquet + 3 videos for one episode (front cam + black wrist cam + side cam).
    Returns the episodes_stats record for this episode."""
    T = len(states)

    # Parquet (with inline PIL images)
    parquet_path = _episode_parquet_path(base_dir, ep_idx, chunks_size)
    _write_episode_parquet(parquet_path, states, actions, frames, side_frames, ep_idx, frame_offset, fps, img_size)

    # Front camera video
    front_path = _episode_video_path(base_dir, "observation.images.image", ep_idx, chunks_size)
    _encode_video(frames, front_path, fps, codec, pix_fmt, img_size)

    # Wrist camera video (black frames — no wrist cam in sim)
    h = w = (img_size if img_size is not None else
             (frames[0].shape[0] if frames else 480))
    black = np.zeros((h, w, 3), dtype=np.uint8)
    black_frames = [black] * len(frames)
    wrist_path = _episode_video_path(base_dir, "observation.images.wrist_image", ep_idx, chunks_size)
    _encode_video(black_frames, wrist_path, fps, codec, pix_fmt, img_size)

    # Side camera video
    side_path = _episode_video_path(base_dir, "observation.images.back_image", ep_idx, chunks_size)
    _encode_video(side_frames if side_frames else black_frames, side_path, fps, codec, pix_fmt, img_size)

    # episodes.jsonl entry
    _append_episode_meta(base_dir, ep_idx, T, task_text)

    # Compute and return stats (caller decides where to write)
    return _compute_episode_stats(states, actions, frames, side_frames, ep_idx, frame_offset, fps)


def _append_episode_meta(base_dir, ep_idx, length, task_text):
    path = os.path.join(base_dir, "meta", "episodes.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({
            "episode_index": ep_idx,
            "tasks": [task_text],
            "length": length,
        }) + "\n")


def _image_channel_stats_np(frames):
    """Per-channel min/max/mean/std over a list of HxWx3 uint8 numpy frames, normalized [0,1]."""
    n = len(frames)
    if n == 0:
        return {"min": [[[0.]],[[0.]],[[0.]]], "max": [[[0.]],[[0.]],[[0.]]],
                "mean": [[[0.]],[[0.]],[[0.]]], "std": [[[0.]],[[0.]],[[0.]]], "count": [0]}
    C = 3
    ch_min = np.full(C, np.inf, dtype=np.float64)
    ch_max = np.full(C, -np.inf, dtype=np.float64)
    ch_sum = np.zeros(C, dtype=np.float64)
    ch_sum_sq = np.zeros(C, dtype=np.float64)
    for frame in frames:
        arr = frame.astype(np.float32) / 255.0
        for c in range(C):
            ch = arr[:, :, c]
            ch_min[c] = min(ch_min[c], float(ch.min()))
            ch_max[c] = max(ch_max[c], float(ch.max()))
            ch_sum[c] += float(ch.mean())
            ch_sum_sq[c] += float((ch ** 2).mean())
    ch_mean = ch_sum / n
    ch_std = np.sqrt(np.maximum(ch_sum_sq / n - ch_mean ** 2, 0.0))
    def nested(a): return [[[float(v)]] for v in a]
    return {"min": nested(ch_min), "max": nested(ch_max),
            "mean": nested(ch_mean), "std": nested(ch_std), "count": [n]}


def _compute_episode_stats(states, actions, frames, side_frames, ep_idx, frame_offset, fps):
    """Compute episodes_stats record for one episode from raw numpy arrays."""
    T = len(states)

    def vec_stats(arr):
        a = np.asarray(arr, dtype=np.float64)
        return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
                "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [T]}

    def scalar_stats(arr):
        a = np.asarray(arr, dtype=np.float64).ravel()
        return {"min": [float(a.min())], "max": [float(a.max())],
                "mean": [float(a.mean())], "std": [float(a.std())], "count": [T]}

    timestamps    = np.arange(T, dtype=np.float32) / fps
    frame_indices = np.arange(T, dtype=np.int64)
    indices       = frame_indices + frame_offset
    img_stats     = _image_channel_stats_np(frames)

    # wrist camera is always black in this sim dataset
    h, w = (frames[0].shape[:2] if frames else (480, 640))
    wrist_stats = _image_channel_stats_np([np.zeros((h, w, 3), dtype=np.uint8)] * T)
    side_stats  = _image_channel_stats_np(side_frames) if side_frames else wrist_stats

    return {
        "episode_index": ep_idx,
        "stats": {
            "observation.images.image":       img_stats,
            "observation.images.wrist_image": wrist_stats,
            "observation.images.back_image":  side_stats,
            "image":         img_stats,
            "wrist_image":   wrist_stats,
            "back_image":    side_stats,
            "state":         vec_stats(states),
            "actions":       vec_stats(actions),
            "timestamp":     scalar_stats(timestamps),
            "frame_index":   scalar_stats(frame_indices),
            "episode_index": scalar_stats(np.full(T, ep_idx,       dtype=np.int64)),
            "index":         scalar_stats(indices),
            "task_index":    scalar_stats(np.zeros(T,              dtype=np.int64)),
        },
    }


def _append_episode_stats(base_dir, stats_record):
    path = os.path.join(base_dir, "meta", "episodes_stats.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(stats_record) + "\n")


def _write_lerobot_meta(base_dir, total_episodes, total_frames,
                        fps, chunks_size, task_text, img_h, img_w,
                        codec, pix_fmt="yuv420p"):
    meta_dir = os.path.join(base_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    video_info = {
        "video.height":     img_h,
        "video.width":      img_w,
        "video.codec":      codec,
        "video.pix_fmt":    pix_fmt,
        "video.is_depth_map": False,
        "video.fps":        fps,
        "video.channels":   3,
        "has_audio":        False,
    }
    total_chunks = max(1, (total_episodes + chunks_size - 1) // chunks_size)
    info = {
        "codebase_version": "v2.1",
        "robot_type":       "franka_panda",
        "total_episodes":   total_episodes,
        "total_frames":     total_frames,
        "total_tasks":      1,
        "total_videos":     total_episodes * len(_VIDEO_KEYS),
        "total_chunks":     total_chunks,
        "chunks_size":      chunks_size,
        "fps":              fps,
        "splits":           {"train": f"0:{total_episodes}"},
        "data_path":        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path":       "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.image": {
                "names": ["channel", "height", "width"],
                "dtype": "video",
                "shape": [3, img_h, img_w],
                "info":  video_info,
            },
            "observation.images.wrist_image": {
                "names": ["channel", "height", "width"],
                "dtype": "video",
                "shape": [3, img_h, img_w],
                "info":  video_info,
            },
            "observation.images.back_image": {
                "names": ["channel", "height", "width"],
                "dtype": "video",
                "shape": [3, img_h, img_w],
                "info":  video_info,
            },
            "image": {
                "dtype": "image",
                "shape": [img_h, img_w, 3],
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": [img_h, img_w, 3],
                "names": ["height", "width", "channel"],
            },
            "back_image": {
                "dtype": "image",
                "shape": [img_h, img_w, 3],
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32", "shape": [7],
                "names": ["ee_pose_and_gripper_width"],
            },
            "actions": {
                "dtype": "float32", "shape": [7],
                "names": ["delta_ee_pose_and_gripper_action"],
            },
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        },
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_text}) + "\n")


def _merge_lerobot_dirs(output_dir, temp_dirs, chunks_size, fps, task_text,
                        img_h, img_w, codec, pix_fmt):
    """
    Merge per-worker temp LeRobot dirs into the final output_dir.

    Each temp dir has LOCAL episode numbering 0, 1, 2...
    This function renumbers them globally and rewrites parquet metadata columns.
    Videos are moved (not decoded) so this is fast even at scale.
    """
    import shutil
    import pyarrow.parquet as pq
    import pyarrow as pa

    global_ep_idx    = 0
    global_frame_off = 0
    total_frames     = 0

    for tmp_dir in temp_dirs:
        if tmp_dir is None or not os.path.exists(tmp_dir):
            continue

        # Collect parquet files in episode order across all chunks
        data_dir = os.path.join(tmp_dir, "data")
        if not os.path.exists(data_dir):
            continue
        parquet_files = []
        for chunk_name in sorted(os.listdir(data_dir)):
            cdir = os.path.join(data_dir, chunk_name)
            if not os.path.isdir(cdir):
                continue
            for fname in sorted(os.listdir(cdir)):
                if fname.endswith(".parquet"):
                    parquet_files.append(os.path.join(cdir, fname))

        for local_parquet in parquet_files:
            # Parse local episode index from filename
            local_ep_idx = int(
                os.path.basename(local_parquet)
                .replace("episode_", "").replace(".parquet", "")
            )

            # Read, update episode_index + index columns, write to final path
            table = pq.read_table(local_parquet)
            T = len(table)
            table = table.set_column(
                table.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array(np.full(T, global_ep_idx, dtype=np.int64)),
            )
            table = table.set_column(
                table.schema.get_field_index("index"),
                "index",
                pa.array(np.arange(T, dtype=np.int64) + global_frame_off),
            )
            final_parquet = _episode_parquet_path(output_dir, global_ep_idx, chunks_size)
            os.makedirs(os.path.dirname(final_parquet), exist_ok=True)
            pq.write_table(table, final_parquet)
            _append_episode_meta(output_dir, global_ep_idx, T, task_text)

            # Move videos (no decoding needed)
            local_chunk = local_ep_idx // chunks_size
            for vk in _VIDEO_KEYS:
                src = os.path.join(
                    tmp_dir, "videos",
                    f"chunk-{local_chunk:03d}", vk,
                    f"episode_{local_ep_idx:06d}.mp4",
                )
                dst = _episode_video_path(output_dir, vk, global_ep_idx, chunks_size)
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)

            # Load temp stats, fix global episode_index + index, append to final
            stats_path = local_parquet.replace(".parquet", ".stats.json")
            if os.path.exists(stats_path):
                with open(stats_path) as _sf:
                    stats_record = json.load(_sf)
                stats_record["episode_index"] = global_ep_idx
                stats_record["stats"]["episode_index"] = {
                    "min": [float(global_ep_idx)], "max": [float(global_ep_idx)],
                    "mean": [float(global_ep_idx)], "std": [0.0], "count": [T],
                }
                stats_record["stats"]["index"] = {
                    "min": [float(global_frame_off)],
                    "max": [float(global_frame_off + T - 1)],
                    "mean": [float(global_frame_off + (T - 1) / 2)],
                    "std": [float(np.arange(T, dtype=np.float64).std())],
                    "count": [T],
                }
                _append_episode_stats(output_dir, stats_record)

            global_ep_idx    += 1
            global_frame_off += T
            total_frames     += T

    # Clean up temp dirs
    for tmp_dir in temp_dirs:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return global_ep_idx, total_frames


# ── Worker ───────────────────────────────────────────────────────────────────

def _worker_generate(
    rank,
    gpu_id,
    target_demos,
    max_attempts,
    src_path,
    cam_t,
    lerobot_dir,       # output dir for this worker (temp or final)
    frame_skip,
    fps,
    img_size,
    codec,
    pix_fmt,
    chunks_size,
    task_text,
    traj_id_to_src_idx,
    demo_keys,
):
    """
    Worker function for multi-GPU parallel generation.

    Writes LeRobot v2.1 data (parquet + videos) with LOCAL episode numbering.
    Failed demos are silently discarded.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import mg_blockpap_interface  # noqa: F401
    import pick_and_place as _pap  # noqa: F401

    from mimicgen.configs.task_spec import MG_TaskSpec
    from mimicgen.datagen.data_generator import DataGenerator
    from mimicgen.env_interfaces.base import make_interface
    from mg_blockpap_wrapper import EnvManiskillBlockPAP
    import numpy as np, traceback

    print(f"[Worker {rank}] GPU={gpu_id}, target={target_demos} demos")

    _pap.TRAJ_ID = "random"
    task_spec = create_task_spec()

    env = EnvManiskillBlockPAP(
        env_name="BlockPAP-v1",
        render=False,
        render_offscreen=True,   # always on: needed for video replay
        use_image_obs=False,     # images come from render, not obs
        cam_t=cam_t,
    )

    env_interface = make_interface(
        name="MG_BlockPAP",
        interface_type="maniskill",
        env=env.base_env,
    )

    data_gen = DataGenerator(
        task_spec=task_spec,
        dataset_path=src_path,
        demo_keys=demo_keys,
    )

    _orig_select = data_gen.select_source_demo

    def _forced_select(eef_pose, object_pose, subtask_ind, src_subtask_inds,
                       subtask_object_name, selection_strategy_name,
                       selection_strategy_kwargs=None):
        forced_tid = getattr(env.base_env, "_nearest_traj_id", None)
        if forced_tid is not None and forced_tid in traj_id_to_src_idx:
            return traj_id_to_src_idx[forced_tid]
        return _orig_select(
            eef_pose=eef_pose, object_pose=object_pose,
            subtask_ind=subtask_ind, src_subtask_inds=src_subtask_inds,
            subtask_object_name=subtask_object_name,
            selection_strategy_name=selection_strategy_name,
            selection_strategy_kwargs=selection_strategy_kwargs,
        )

    data_gen.select_source_demo = _forced_select

    n_success    = 0
    num_attempts = 0
    local_frame_off = 0

    while n_success < target_demos:
        num_attempts += 1
        try:
            result = data_gen.generate(
                env=env,
                env_interface=env_interface,
                select_src_per_subtask=False,
                transform_first_robot_pose=True,
                interpolate_from_last_target_pose=True,
                video_writer=None,
                video_skip=1,
                camera_names=None,
            )

            if not result["success"]:
                if num_attempts % 10 == 0:
                    print(f"[Worker {rank}] Attempt {num_attempts}: failed "
                          f"({n_success}/{target_demos} so far)")
                continue

            obs_list = result.get("observations", [])
            if not obs_list:
                print(f"[Worker {rank}] Warning: no observations in result, skipping")
                continue

            nearest_tid = getattr(env.base_env, "_nearest_traj_id", "?")
            src_idx     = traj_id_to_src_idx.get(nearest_tid, "?")
            print(f"[Worker {rank}] [{n_success + 1}/{target_demos}] Success "
                  f"(attempt {num_attempts}, T={len(obs_list)}, "
                  f"nearest_traj={nearest_tid}, src_demo={src_idx})")

            # Subsample by frame_skip so frames/states/actions align
            state_vecs = result["states"]
            if frame_skip > 1:
                obs_list   = obs_list[::frame_skip]
                state_vecs = state_vecs[::frame_skip]

            # Trim head (teleport/interpolation artifacts) and tail (post-task idle)
            obs_list, state_vecs = _trim_episode(obs_list, state_vecs)

            # Compute state / action from (subsampled) obs
            states, actions = _obs_list_to_state_action(obs_list)
            T = len(states)

            # Render frames by replaying (already-subsampled) env states
            frames, side_frames = _collect_replay_frames(env, state_vecs, frame_skip=1)

            # Write parquet + videos for this episode
            stats_record = _write_lerobot_episode(
                base_dir     = lerobot_dir,
                ep_idx       = n_success,
                frame_offset = local_frame_off,
                states       = states,
                actions      = actions,
                frames       = frames,
                side_frames  = side_frames,
                chunks_size  = chunks_size,
                fps          = fps,
                codec        = codec,
                pix_fmt      = pix_fmt,
                img_size     = img_size,
                task_text    = task_text,
            )

            # Save temp stats file for merge step to pick up
            stats_path = _episode_parquet_path(lerobot_dir, n_success, chunks_size).replace(".parquet", ".stats.json")
            with open(stats_path, "w") as _sf:
                json.dump(stats_record, _sf)

            n_success     += 1
            local_frame_off += T

        except Exception as e:
            print(f"[Worker {rank}] Attempt {num_attempts}: error - {e}")
            traceback.print_exc()

        if num_attempts >= max_attempts:
            print(f"[Worker {rank}] Reached max attempts ({num_attempts}). "
                  f"Generated {n_success}/{target_demos} demos.")
            break

    env.env.close()
    print(f"[Worker {rank}] Done: {n_success} success in {num_attempts} attempts.")
    return n_success


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate BlockPAP demos → LeRobot v2.1 dataset directly"
    )
    parser.add_argument("--src",          type=str, required=True,
                        help="Source MimicGen HDF5 path")
    parser.add_argument("--lerobot-dir",  type=str, required=True,
                        help="Output root directory for LeRobot v2.1 dataset")
    parser.add_argument("--num-demos",    type=int, default=900,
                        help="Total number of successful demos to generate")
    parser.add_argument("--max-attempts", type=int, default=99999,
                        help="Max generation attempts per worker before giving up")
    parser.add_argument("--cam-t",        type=str, default="og",
                        help="Camera calibration preset (og / 0302 / 0303)")
    parser.add_argument("--fps",          type=float, default=20.0,
                        help="Dataset FPS for videos and timestamps (default: 20). "
                             "frame_skip is auto-computed as round(sim_freq / fps).")
    parser.add_argument("--sim-freq",     type=float, default=20.0,
                        help="Simulation control frequency in Hz (default: 20). "
                             "Used to compute frame_skip = round(sim_freq / fps).")
    parser.add_argument("--image-size",   type=int, default=None,
                        help="Resize frames to (image_size × image_size). "
                             "Default: keep original render resolution.")
    parser.add_argument("--video-codec",  type=str, default="libx264",
                        help="ffmpeg video codec (default: libx264). "
                             "Use 'libaom-av1' for AV1 (slower).")
    parser.add_argument("--chunks-size",  type=int, default=500,
                        help="Episodes per chunk directory (default: 500)")
    parser.add_argument("--task",         type=str,
                        default="pick up the block and place it on the coaster",
                        help="Task description string written to tasks.jsonl")
    parser.add_argument("--num-workers",  type=int, default=1,
                        help="Number of parallel generation workers (default: 1)")
    parser.add_argument("--gpu-ids",      type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. '0,1,2,3'. "
                             "Defaults to 0,1,...,num_workers-1")
    args = parser.parse_args()

    # Resolve GPU IDs
    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.num_workers))
    if len(gpu_ids) < args.num_workers:
        gpu_ids = [gpu_ids[i % len(gpu_ids)] for i in range(args.num_workers)]

    # Derive frame_skip from sim frequency and target dataset fps
    frame_skip = max(1, round(args.sim_freq / args.fps))
    print(f"sim_freq={args.sim_freq} Hz  fps={args.fps}  →  frame_skip={frame_skip} "
          f"(store 1 frame every {frame_skip} control steps)")

    pix_fmt = "yuv420p"

    task_spec = create_task_spec()
    print(f"Task spec: {len(task_spec.spec)} subtasks")
    for i, s in enumerate(task_spec.spec):
        print(f"  Subtask {i}: object_ref={s['object_ref']}, "
              f"term_signal={s['subtask_term_signal']}")

    print(f"\nLoading source dataset: {args.src}")
    demo_keys = MG_FileUtils.get_all_demos_from_dataset(args.src)
    print(f"Source demos: {len(demo_keys)}")

    MG_FileUtils.parse_source_dataset(
        dataset_path=args.src,
        demo_keys=demo_keys,
        task_spec=task_spec,
    )

    # Build traj_id → source demo index mapping
    traj_id_to_src_idx = {}
    with h5py.File(args.src, "r") as f_src:
        for i, dk in enumerate(sorted(f_src["data"].keys())):
            real_tid = f_src[f"data/{dk}"].attrs.get("real_traj_id", None)
            if real_tid is not None:
                traj_id_to_src_idx[str(real_tid)] = i
    print(f"Traj ID → source demo index: {traj_id_to_src_idx}")

    # ── Multi-GPU parallel generation ─────────────────────────────────────────
    if args.num_workers > 1:
        print(f"\nLaunching {args.num_workers} workers on GPUs {gpu_ids[:args.num_workers]}")

        base_demos = args.num_demos // args.num_workers
        remainder  = args.num_demos % args.num_workers
        targets    = [base_demos + (1 if i < remainder else 0)
                      for i in range(args.num_workers)]

        # Each worker writes to a temp subdir
        temp_dirs = [
            os.path.join(args.lerobot_dir, f"_worker{i}")
            for i in range(args.num_workers)
        ]

        common = dict(
            src_path           = args.src,
            cam_t              = args.cam_t,
            frame_skip         = frame_skip,
            fps                = args.fps,
            img_size           = args.image_size,
            codec              = args.video_codec,
            pix_fmt            = pix_fmt,
            chunks_size        = args.chunks_size,
            task_text          = args.task,
            traj_id_to_src_idx = traj_id_to_src_idx,
            demo_keys          = demo_keys,
        )

        ctx   = multiprocessing.get_context("spawn")
        procs = []
        for rank in range(args.num_workers):
            p = ctx.Process(
                target=_worker_generate,
                args=(
                    rank,
                    gpu_ids[rank],
                    targets[rank],
                    args.max_attempts,
                    common["src_path"],
                    common["cam_t"],
                    temp_dirs[rank],
                    common["frame_skip"],
                    common["fps"],
                    common["img_size"],
                    common["codec"],
                    common["pix_fmt"],
                    common["chunks_size"],
                    common["task_text"],
                    common["traj_id_to_src_idx"],
                    common["demo_keys"],
                ),
                daemon=False,
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        print(f"\nMerging worker outputs → {args.lerobot_dir}")
        total_episodes, total_frames = _merge_lerobot_dirs(
            output_dir  = args.lerobot_dir,
            temp_dirs   = temp_dirs,
            chunks_size = args.chunks_size,
            fps         = args.fps,
            task_text   = args.task,
            img_h       = args.image_size or 480,
            img_w       = args.image_size or 640,
            codec       = args.video_codec,
            pix_fmt     = pix_fmt,
        )

    # ── Single-GPU generation ─────────────────────────────────────────────────
    else:
        pick_and_place.TRAJ_ID = "random"

        from mg_blockpap_wrapper import EnvManiskillBlockPAP

        env = EnvManiskillBlockPAP(
            env_name="BlockPAP-v1",
            render=False,
            render_offscreen=True,
            use_image_obs=False,
            cam_t=args.cam_t,
        )
        print(f"Created environment: {env.name}")

        env_interface = make_interface(
            name="MG_BlockPAP",
            interface_type="maniskill",
            env=env.base_env,
        )

        data_gen = DataGenerator(
            task_spec=task_spec,
            dataset_path=args.src,
            demo_keys=demo_keys,
        )

        _orig_select = data_gen.select_source_demo

        def _forced_select(eef_pose, object_pose, subtask_ind, src_subtask_inds,
                           subtask_object_name, selection_strategy_name,
                           selection_strategy_kwargs=None):
            forced_tid = getattr(env.base_env, "_nearest_traj_id", None)
            if forced_tid is not None and forced_tid in traj_id_to_src_idx:
                return traj_id_to_src_idx[forced_tid]
            return _orig_select(
                eef_pose=eef_pose, object_pose=object_pose,
                subtask_ind=subtask_ind, src_subtask_inds=src_subtask_inds,
                subtask_object_name=subtask_object_name,
                selection_strategy_name=selection_strategy_name,
                selection_strategy_kwargs=selection_strategy_kwargs,
            )

        data_gen.select_source_demo = _forced_select

        n_success      = 0
        num_attempts   = 0
        frame_offset   = 0
        total_frames   = 0

        print(f"\nGenerating {args.num_demos} demonstrations...")
        while n_success < args.num_demos:
            num_attempts += 1
            try:
                result = data_gen.generate(
                    env=env,
                    env_interface=env_interface,
                    select_src_per_subtask=False,
                    transform_first_robot_pose=True,
                    interpolate_from_last_target_pose=True,
                    video_writer=None,
                    video_skip=1,
                    camera_names=None,
                )

                if not result["success"]:
                    if num_attempts % 10 == 0:
                        print(f"  Attempt {num_attempts}: failed "
                              f"({n_success}/{args.num_demos} so far)")
                    continue

                obs_list = result.get("observations", [])
                if not obs_list:
                    print(f"  Warning: no observations in result, skipping")
                    continue

                nearest_tid = getattr(env.base_env, "_nearest_traj_id", "?")
                src_idx     = traj_id_to_src_idx.get(nearest_tid, "?")
                print(f"  [{n_success + 1}/{args.num_demos}] Success "
                      f"(attempt {num_attempts}, T={len(obs_list)}, "
                      f"nearest_traj={nearest_tid}, src_demo={src_idx})")

                # Subsample by frame_skip so frames/states/actions align
                state_vecs = result["states"]
                if frame_skip > 1:
                    obs_list   = obs_list[::frame_skip]
                    state_vecs = state_vecs[::frame_skip]

                # Trim head (teleport/interpolation artifacts) and tail (post-task idle)
                obs_list, state_vecs = _trim_episode(obs_list, state_vecs)

                states, actions = _obs_list_to_state_action(obs_list)
                T = len(states)

                frames = _collect_replay_frames(env, state_vecs, frame_skip=1)

                stats_record = _write_lerobot_episode(
                    base_dir     = args.lerobot_dir,
                    ep_idx       = n_success,
                    frame_offset = frame_offset,
                    states       = states,
                    actions      = actions,
                    frames       = frames,
                    chunks_size  = args.chunks_size,
                    fps          = args.fps,
                    codec        = args.video_codec,
                    pix_fmt      = pix_fmt,
                    img_size     = args.image_size,
                    task_text    = args.task,
                )
                _append_episode_stats(args.lerobot_dir, stats_record)

                n_success    += 1
                frame_offset += T
                total_frames += T

            except Exception as e:
                print(f"  Attempt {num_attempts}: error - {e}")
                traceback.print_exc()

            if num_attempts >= args.max_attempts:
                print(f"\nReached max attempts ({num_attempts}). "
                      f"Generated {n_success}/{args.num_demos} demos.")
                break

        env.env.close()
        total_episodes = n_success
        print(f"\nDone! {n_success} demos in {num_attempts} attempts. "
              f"Success rate: {n_success / max(num_attempts, 1) * 100:.1f}%")

    # ── Write final info.json + tasks.jsonl ───────────────────────────────────
    # Determine actual rendered image size
    if args.image_size is not None:
        img_h = img_w = args.image_size
    else:
        # Try to infer from one video; default to 480×640
        sample_vid = _episode_video_path(
            args.lerobot_dir, "observation.images.image", 0, args.chunks_size
        )
        img_h, img_w = 480, 640
        if os.path.exists(sample_vid):
            try:
                import cv2
                cap = cv2.VideoCapture(sample_vid)
                img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            except Exception:
                pass

    _write_lerobot_meta(
        base_dir       = args.lerobot_dir,
        total_episodes = total_episodes,
        total_frames   = total_frames,
        fps            = args.fps,
        chunks_size    = args.chunks_size,
        task_text      = args.task,
        img_h          = img_h,
        img_w          = img_w,
        codec          = args.video_codec,
        pix_fmt        = pix_fmt,
    )

    print(f"\nLeRobot v2.1 dataset written to: {args.lerobot_dir}")
    print(f"  total_episodes={total_episodes}  total_frames={total_frames}")
    print(f"  fps={args.fps}  frame_skip={frame_skip}  codec={args.video_codec}")


if __name__ == "__main__":
    main()
