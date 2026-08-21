#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVTOKEN (LlamaFactory rollout_lite.json) -> LeRobot v2.1

沿用 real_data_convert.py 的方案：state/actions 的构造、parquet schema、
episodes_stats / episodes / tasks / info.json 的写法全部直接复用其函数，
只把数据源从 HDF5 换成 MVTOKEN 的 (rollout_lite.json + per-rollout actions.jsonl + PNG)。

与 real_data_convert.py 的差异（MVTOKEN 数据本身决定的）：
  1. 只有 2 路相机 (agentview / wrist)。back_image 按 RLinf load_hdf5 的约定填黑帧，
     franka_policy.FrankaEEInputs 会检测 `back_image.any() == False` 并把它 mask 掉。
  2. 不做 SECTION 1 的静态段清洗：MVTOKEN 已经是原子步离散化的数据
     (每帧一个 2cm/0.15rad 的动作)，再清洗会破坏与 rollout_lite.json 样本的 1:1 对应。
  3. rollout_lite.json 每集末尾多一个 DONE 样本（复用最后一张图，机械臂没动）。
     默认保留，pose 复制上一帧 -> 该帧 delta 为 0，与 RLinf `actions[-1] = actions[-2]` 同源。
  4. actions.jsonl 里的 wall-clock 间隔很不均匀 (median 0.245s, mean 0.60s)，
     所以和 RLinf 清洗后一样重新生成均匀 timestamp = frame_index / fps。

用法:
    python mvtoken_to_lerobot.py \
        --rollout-json /path/to/v3/rollout_lite.json \
        --output-root  /path/to/lerobot
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 单一事实来源：state/actions 语义与落盘格式全部来自 RLinf 的真机转换脚本
from real_data_convert import (  # noqa: E402
    _encode_video_from_pil,
    append_episode_meta,
    append_episode_stats,
    compute_actions_from_ee_pose,
    compute_episode_stats,
    save_episode_with_datasets,
    write_info_json,
    write_tasks_jsonl,
)

TASK_RE = re.compile(r"^Task:\s*(.+?)\s*$", re.MULTILINE)
ROLLOUT_RE = re.compile(r"^(.*/rollout_\d+)/")


# ============================================================================
# SECTION 1: 解析 rollout_lite.json
# ============================================================================


def parse_rollout_json(rollout_json: str) -> "OrderedDict[str, List[dict]]":
    """按 rollout 目录把样本分组，保持在 json 里首次出现的顺序。"""
    with open(rollout_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    episodes: "OrderedDict[str, List[dict]]" = OrderedDict()
    for idx, sample in enumerate(samples):
        images = sample.get("images") or []
        if len(images) != 2:
            raise ValueError(f"sample {idx}: expected 2 images (agentview, wrist), got {len(images)}")
        m = ROLLOUT_RE.match(images[0])
        if m is None:
            raise ValueError(f"sample {idx}: cannot locate rollout dir in {images[0]}")
        episodes.setdefault(m.group(1), []).append(sample)
    return episodes


def extract_task(samples: List[dict], rollout_dir: str) -> str:
    tasks = set()
    for s in samples:
        m = TASK_RE.search(s["instruction"])
        if m is None:
            raise ValueError(f"{rollout_dir}: no 'Task:' line in instruction")
        tasks.add(m.group(1))
    if len(tasks) != 1:
        raise ValueError(f"{rollout_dir}: inconsistent task text {sorted(tasks)}")
    return tasks.pop()


def load_action_rows(rollout_dir: str) -> List[dict]:
    path = os.path.join(rollout_dir, "actions.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def align_episode(
    rollout_dir: str,
    samples: List[dict],
    keep_done_frame: bool,
) -> Tuple[np.ndarray, List[Tuple[str, str]], List[str], bool]:
    """
    把 rollout_lite.json 的样本和 actions.jsonl 的位姿对齐。

    Returns:
        ee_pose  (T, 8) = [x, y, z, qx, qy, qz, qw, gripper_width]
        img_paths  长度 T 的 (agentview_path, wrist_path)
        tokens     长度 T 的离散动作 token
        done_appended  是否补了末尾 DONE 帧
    """
    rows = load_action_rows(rollout_dir)
    n_rows, n_samples = len(rows), len(samples)

    if n_samples not in (n_rows, n_rows + 1):
        raise ValueError(f"{rollout_dir}: {n_samples} samples vs {n_rows} action rows")
    done_appended = n_samples == n_rows + 1
    if done_appended and samples[-1]["output"] != "DONE":
        raise ValueError(f"{rollout_dir}: trailing extra sample is {samples[-1]['output']!r}, expected DONE")

    # 逐帧交叉校验：图像文件名 + token 必须一致，否则说明两份数据不同源
    for i, row in enumerate(rows):
        sample = samples[i]
        for key, path in zip(("agentview", "wrist"), sample["images"]):
            if os.path.basename(row[key]) != os.path.basename(path):
                raise ValueError(f"{rollout_dir} frame {i}: image mismatch {row[key]} vs {path}")
        if row["token"] != sample["output"]:
            raise ValueError(f"{rollout_dir} frame {i}: token mismatch {row['token']} vs {sample['output']}")

    ee_pose = np.asarray(
        [list(map(float, r["ee_pose"])) + [float(r["gripper_width"])] for r in rows],
        dtype=np.float64,
    )
    if ee_pose.shape[1] != 8:
        raise ValueError(f"{rollout_dir}: expected ee_pose 7D + gripper, got shape {ee_pose.shape}")

    img_paths = [(s["images"][0], s["images"][1]) for s in samples]
    tokens = [s["output"] for s in samples]

    if done_appended:
        if keep_done_frame:
            # DONE 帧机械臂未动：复制最后一帧位姿 -> delta 为 0
            ee_pose = np.vstack([ee_pose, ee_pose[-1]])
        else:
            img_paths = img_paths[:-1]
            tokens = tokens[:-1]
            done_appended = False

    if len(ee_pose) != len(img_paths):
        raise ValueError(f"{rollout_dir}: pose/image length mismatch {len(ee_pose)} vs {len(img_paths)}")
    return ee_pose, img_paths, tokens, done_appended


# ============================================================================
# SECTION 2: 构造 episode（对齐 real_data_convert.build_episode_data）
# ============================================================================


def circular_mean(angles: np.ndarray) -> np.ndarray:
    """按列求圆均值 —— 角度不能直接取算术平均（+pi 和 -pi 会平均成 0）。"""
    return np.arctan2(np.sin(angles).mean(axis=0), np.cos(angles).mean(axis=0))


def unwrap_to_reference(angles: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """把角度解环绕到以 ref 为中心的 2pi 窗口 [ref-pi, ref+pi)。

    动机：本数据集夹爪常年朝下 = 绕水平轴约 180deg，roll 的真实值就压在
    欧拉角 +-pi 的奇异点上抖动，导致同一个物理姿态被记成 +3.14 / -3.14 两簇
    （实测 74.7% / 25.3%，84/101 集在单集内部就跨界）。这样的 state 送进
    openpi 的 quantile 归一化后会变成满量程、10.8% 帧翻转的伪信号。

    ref 必须在全数据集层面统一计算，逐集各算各的会造成集间不一致。
    只影响 state 的绝对姿态角；action 的角增量走 wrap_angle_delta，本来就是对的。
    """
    return ref + (angles - ref + np.pi) % (2 * np.pi) - np.pi


def _load_images(paths: List[str], height: int, width: int, workers: int) -> List[Image.Image]:
    import cv2

    def load_one(p: str) -> Image.Image:
        img = Image.open(p).convert("RGB")
        if img.size != (width, height):
            arr = cv2.resize(np.asarray(img), (width, height), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(arr)
        return img

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(load_one, paths))


def build_episode_data(
    ee_pose: np.ndarray,
    img_paths: List[Tuple[str, str]],
    episode_index: int,
    fps: float,
    image_height: int,
    image_width: int,
    task_index: int,
    frame_offset: int,
    workers: int,
    done_appended: bool = False,
    repeat_last_real_action: bool = True,
    euler_ref: Optional[np.ndarray] = None,
) -> dict:
    T = len(ee_pose)
    states, actions = compute_actions_from_ee_pose(ee_pose)

    # actions[:, 3:6] 在 compute_actions_from_ee_pose 内部由局部 euler_angles 算出，
    # 这里改 states 不会波及 actions。
    if euler_ref is not None:
        states[:, 3:6] = unwrap_to_reference(states[:, 3:6].astype(np.float64), euler_ref).astype(np.float32)

    if done_appended and T >= 3:
        # 最后一个真实帧的动作已执行但没有后续观测（DONE 帧是复制上一帧位姿），
        # 直接算会得到 0，等于给一个确实在运动的帧打了"不动"的标签。
        # 这里沿用 RLinf 对不可观测末动作的处理 (actions[-1] = actions[-2])，
        # 再把 DONE 帧本身归零 —— 终止帧确实不该有位移。
        if repeat_last_real_action:
            actions[T - 2] = actions[T - 3]
        else:
            actions[T - 2] = 0.0
            actions[T - 2, 6] = actions[T - 3, 6]
        actions[T - 1, :6] = 0.0
        actions[T - 1, 6] = actions[T - 2, 6]

    front = _load_images([p[0] for p in img_paths], image_height, image_width, workers)
    wrist = _load_images([p[1] for p in img_paths], image_height, image_width, workers)
    # MVTOKEN 只有两路相机；back 按 RLinf 的缺相机约定填黑帧（下游会 mask 掉）
    black = Image.new("RGB", (image_width, image_height), (0, 0, 0))
    back = [black] * T

    timestamps = (np.arange(T, dtype=np.float32) / float(fps)).astype(np.float32)

    return {
        "image": front,
        "wrist_image": wrist,
        "back_image": back,
        "state": states.tolist(),
        "actions": actions.tolist(),
        "timestamp": timestamps.tolist(),
        "frame_index": np.arange(T, dtype=np.int64).tolist(),
        "episode_index": np.full(T, episode_index, dtype=np.int64).tolist(),
        "index": np.arange(frame_offset, frame_offset + T, dtype=np.int64).tolist(),
        "task_index": np.full(T, task_index, dtype=np.int64).tolist(),
    }


# ============================================================================
# SECTION 3: 主流程
# ============================================================================


def clear_output(output_root: str) -> None:
    for sub in ("data", "videos", "meta"):
        path = os.path.join(output_root, sub)
        if os.path.exists(path):
            print(f"[INFO] Removing stale {path}")
            shutil.rmtree(path)
    report = os.path.join(output_root, "conversion_report.json")
    if os.path.exists(report):
        os.remove(report)


def convert(
    rollout_json: str,
    output_root: str,
    fps: float,
    image_height: int,
    image_width: int,
    chunk_size: int,
    keep_done_frame: bool,
    repeat_last_real_action: bool,
    unwrap_euler: bool,
    write_videos: bool,
    workers: int,
    limit_episodes: Optional[int],
) -> None:
    episodes = parse_rollout_json(rollout_json)
    ep_dirs = list(episodes.keys())
    if limit_episodes is not None:
        ep_dirs = ep_dirs[:limit_episodes]

    print("=" * 80)
    print("MVTOKEN -> LeRobot v2.1 (RLinf real_franka scheme)")
    print("=" * 80)
    print(f"Source     : {rollout_json}")
    print(f"Output     : {output_root}")
    print(f"Episodes   : {len(ep_dirs)}")
    print(f"Resolution : {image_width}x{image_height} @ {fps} fps (uniform timestamps)")
    print(f"DONE frame : {'kept' if keep_done_frame else 'dropped'}")
    print(f"Euler      : {'unwrapped to global circular mean' if unwrap_euler else 'raw (may straddle +-pi)'}")
    print(f"Videos     : {'on' if write_videos else 'off'}")
    print("=" * 80)

    # 先全量对齐校验，任何一集不一致就直接失败，避免写出半份数据集
    aligned: List[dict] = []
    all_tasks: List[str] = []
    for ep_dir in tqdm(ep_dirs, desc="Validating"):
        samples = episodes[ep_dir]
        task = extract_task(samples, ep_dir)
        ee_pose, img_paths, tokens, done_appended = align_episode(ep_dir, samples, keep_done_frame)
        missing = [p for pair in img_paths for p in pair if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{ep_dir}: {len(missing)} missing images, e.g. {missing[0]}")
        if task not in all_tasks:
            all_tasks.append(task)
        aligned.append(
            {
                "source_dir": ep_dir,
                "task": task,
                "ee_pose": ee_pose,
                "img_paths": img_paths,
                "tokens": tokens,
                "done_frame_appended": done_appended,
            }
        )

    print(f"\nUnique tasks: {len(all_tasks)}")
    for i, task in enumerate(all_tasks):
        print(f"  Task {i}: {task}")

    # 全数据集统一的欧拉角参考（圆均值），用于解掉 roll 压在 +-pi 奇异点上的环绕
    euler_ref = None
    if unwrap_euler:
        from scipy.spatial.transform import Rotation as _R

        all_euler = np.vstack(
            [_R.from_quat(ep["ee_pose"][:, 3:7]).as_euler("xyz") for ep in aligned]
        )
        euler_ref = circular_mean(all_euler)
        wrapped = unwrap_to_reference(all_euler, euler_ref)
        moved = int((np.abs(wrapped - all_euler) > 1e-9).any(axis=1).sum())
        print("\nEuler unwrap (state only, actions untouched):")
        print(f"  global circular-mean ref (rad): {np.round(euler_ref, 4)}")
        print(f"  frames re-wrapped: {moved} / {len(all_euler)}")
        for j, n in enumerate(("roll", "pitch", "yaw")):
            print(
                f"    {n:<6} before [{all_euler[:, j].min():+.4f}, {all_euler[:, j].max():+.4f}] "
                f"std={all_euler[:, j].std():.4f}"
                f"  ->  after [{wrapped[:, j].min():+.4f}, {wrapped[:, j].max():+.4f}] "
                f"std={wrapped[:, j].std():.4f}"
            )
    print()

    clear_output(output_root)
    data_root = os.path.join(output_root, "data")
    meta_root = os.path.join(output_root, "meta")
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(meta_root, exist_ok=True)

    write_tasks_jsonl(meta_root, all_tasks)
    sidecar_path = os.path.join(meta_root, "mvtoken_episodes.jsonl")

    total_frames = 0
    for ep_idx, ep in enumerate(tqdm(aligned, desc="Converting")):
        chunk_id = ep_idx // chunk_size
        task_index = all_tasks.index(ep["task"])

        data = build_episode_data(
            ee_pose=ep["ee_pose"],
            img_paths=ep["img_paths"],
            episode_index=ep_idx,
            fps=fps,
            image_height=image_height,
            image_width=image_width,
            task_index=task_index,
            frame_offset=total_frames,
            workers=workers,
            done_appended=ep["done_frame_appended"],
            repeat_last_real_action=repeat_last_real_action,
            euler_ref=euler_ref,
        )
        episode_length = len(data["timestamp"])

        chunk_dir = os.path.join(data_root, f"chunk-{chunk_id:03d}")
        os.makedirs(chunk_dir, exist_ok=True)
        save_episode_with_datasets(data, os.path.join(chunk_dir, f"episode_{ep_idx:06d}.parquet"))

        append_episode_stats(meta_root, compute_episode_stats(data, ep_idx))
        append_episode_meta(meta_root, ep_idx, length=episode_length, task_text=ep["task"])

        if write_videos:
            video_chunk_dir = os.path.join(output_root, "videos", f"chunk-{chunk_id:03d}")
            for key in ("image", "wrist_image", "back_image"):
                _encode_video_from_pil(
                    data[key],
                    os.path.join(video_chunk_dir, f"observation.images.{key}/episode_{ep_idx:06d}.mp4"),
                    fps=fps,
                )

        # 旁挂：保留 MVTOKEN 的离散 token 与来源，parquet 本身保持 RLinf schema 不变
        with open(sidecar_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "episode_index": ep_idx,
                        "source_dir": ep["source_dir"],
                        "task": ep["task"],
                        "task_index": task_index,
                        "length": episode_length,
                        "done_frame_appended": ep["done_frame_appended"],
                        "tokens": ep["tokens"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        total_frames += episode_length

    write_info_json(
        meta_root,
        total_episodes=len(aligned),
        total_frames=total_frames,
        total_tasks=len(all_tasks),
        fps=fps,
        chunk_size=chunk_size,
        image_height=image_height,
        image_width=image_width,
    )
    if not write_videos:
        # 没写视频就别在 info.json 里谎报
        info_path = os.path.join(meta_root, "info.json")
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        info["total_videos"] = 0
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    report = {
        "source_rollout_json": os.path.abspath(rollout_json),
        "output_root": os.path.abspath(output_root),
        "scheme": "RLinf/real_franka/real_data_convert.py (LeRobot v2.1)",
        "total_episodes": len(aligned),
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "fps": fps,
        "image_height": image_height,
        "image_width": image_width,
        "keep_done_frame": keep_done_frame,
        "last_real_action": "repeat_previous" if repeat_last_real_action else "zero",
        "euler_unwrapped": bool(unwrap_euler),
        "euler_reference_rad": (euler_ref.tolist() if euler_ref is not None else None),
        "videos_written": write_videos,
        "cameras": {"image": "agentview", "wrist_image": "wrist", "back_image": "black (absent)"},
        "state_layout": "[x, y, z, roll, pitch, yaw, gripper_width]",
        "action_layout": "[dx, dy, dz, droll, dpitch, dyaw, gripper_binary(open=1)]",
        "source_dirs": [ep["source_dir"] for ep in aligned],
    }
    with open(os.path.join(output_root, "conversion_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Episodes: {len(aligned)}   Frames: {total_frames}   Tasks: {len(all_tasks)}")
    print(f"Output  : {output_root}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MVTOKEN rollout_lite.json -> LeRobot v2.1 (RLinf scheme)")
    parser.add_argument(
        "--rollout-json",
        default="/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/mix_22_27_04/v3/rollout_lite.json",
    )
    parser.add_argument(
        "--output-root",
        default="/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/mix_22_27_04/lerobot",
    )
    parser.add_argument("--fps", type=float, default=4.0, help="nominal fps for uniform timestamps")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--drop-done-frame", action="store_true", help="drop the trailing DONE frame of each episode")
    parser.add_argument(
        "--zero-last-real-action",
        action="store_true",
        help="label the last observed frame with a zero action instead of repeating the previous one",
    )
    parser.add_argument(
        "--no-unwrap-euler",
        action="store_true",
        help="keep raw euler angles (roll will straddle +-pi and poison norm stats)",
    )
    parser.add_argument("--skip-videos", action="store_true", help="only write parquet + meta")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-episodes", type=int, default=None, help="smoke test on the first N episodes")
    args = parser.parse_args()

    convert(
        rollout_json=args.rollout_json,
        output_root=args.output_root,
        fps=args.fps,
        image_height=args.image_height,
        image_width=args.image_width,
        chunk_size=args.chunk_size,
        keep_done_frame=not args.drop_done_frame,
        repeat_last_real_action=not args.zero_last_real_action,
        unwrap_euler=not args.no_unwrap_euler,
        write_videos=not args.skip_videos,
        workers=args.workers,
        limit_episodes=args.limit_episodes,
    )


if __name__ == "__main__":
    main()
