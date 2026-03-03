#!/usr/bin/env python3
"""
Dataset Inspector: HDF5 结构 + 视频画幅（只扫一个 episode 文件夹的三个视频）
用法: python inspect_data.py [数据集目录]
输出: 同时打印到终端，并 append 到 data_log.txt
"""

import sys, json, subprocess
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

DATASET_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/storage/zhijun/real_franka/pick_and_place")
VIDEO_DIR   = DATASET_DIR / "videos"
LOG_FILE    = Path(__file__).parent / "data_log.txt"


# ── Tee: 同时写终端和日志文件 ─────────────────────────────────────────────────

class Tee:
    def __init__(self, file):
        self._file   = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


# ── HDF5 ─────────────────────────────────────────────────────────────────────

def inspect_hdf5(hdf5_path: Path):
    print(f"\n{'='*60}")
    print(f"  HDF5 详细结构: {hdf5_path.name}")
    print(f"{'='*60}")

    with h5py.File(hdf5_path, "r") as f:
        def visitor(name, obj):
            depth  = name.count("/")
            prefix = "  " * depth
            short  = name.split("/")[-1]
            if isinstance(obj, h5py.Dataset):
                size_mb = obj.nbytes / 1024**2
                extra = ""
                if obj.size <= 20 and obj.dtype.kind in ("f", "i", "u"):
                    extra = f"  val={np.array(obj).tolist()}"
                print(f"  {prefix}├─ {short}  {obj.shape}  {obj.dtype}  ({size_mb:.3f} MB){extra}")
            else:
                print(f"  {prefix}└─ [{short}/]")
        f.visititems(visitor)

        if f.attrs:
            print("\n  [Root Attributes]")
            for k, v in f.attrs.items():
                print(f"    {k}: {v}")

        print("\n" + "-" * 60)
        print("  关键数据内容摘要")
        print("-" * 60)
        inspect_key_datasets(f)


def summarize_numeric_dataset(name: str, arr: np.ndarray, head_rows: int = 999):
    print(f"\n  [{name}]")
    print(f"    shape={arr.shape}, dtype={arr.dtype}")

    if arr.size == 0:
        print("    empty dataset")
        return

    flat = arr.reshape(-1)
    print(
        f"    min={float(np.min(flat)):.6f}, max={float(np.max(flat)):.6f}, "
        f"mean={float(np.mean(flat)):.6f}, std={float(np.std(flat)):.6f}"
    )

    if arr.ndim == 1:
        n = min(head_rows, arr.shape[0])
        print(f"    head({n})={arr[:n].tolist()}")
        if arr.shape[0] > n:
            print(f"    tail({n})={arr[-n:].tolist()}")
        return

    n = min(head_rows, arr.shape[0])
    print(f"    head({n})=")
    for i in range(n):
        print(f"      t={i}: {arr[i].tolist()}")

    if arr.shape[0] > n:
        print(f"    tail({n})=")
        for i in range(arr.shape[0] - n, arr.shape[0]):
            print(f"      t={i}: {arr[i].tolist()}")


def inspect_key_datasets(f: h5py.File):
    def get_dataset(path: str):
        if "/" in path:
            g, d = path.split("/", 1)
            if g in f and d in f[g]:
                return f[g][d][:]
            return None
        return f[path][:] if path in f else None

    numeric_targets = [
        "timestamp",
        "stage",
        "joint_action",
        "observations/joint_pos",
        "observations/full_joint_pos",
        "observations/ee_pose",
    ]

    for key in numeric_targets:
        arr = get_dataset(key)
        if arr is None:
            print(f"\n  [{key}] not found")
            continue
        if np.issubdtype(arr.dtype, np.number):
            summarize_numeric_dataset(key, arr)
        else:
            print(f"\n  [{key}] shape={arr.shape}, dtype={arr.dtype} (non-numeric)")

    joint_pos = get_dataset("observations/joint_pos")
    if joint_pos is not None and joint_pos.ndim == 2 and joint_pos.shape[1] >= 9:
        gripper = joint_pos[:, -2:]
        gripper_delta = np.diff(gripper, axis=0) if gripper.shape[0] > 1 else np.zeros_like(gripper)
        print("\n  [gripper from observations/joint_pos last 2 dims]")
        print(f"    range=({float(np.min(gripper)):.6f}, {float(np.max(gripper)):.6f})")
        print(f"    mean_abs_delta={float(np.mean(np.abs(gripper_delta))):.6f}")
        if gripper.shape[0] > 0:
            print(f"    first={gripper[0].tolist()}, last={gripper[-1].tolist()}")

    if "observations" in f and "images" in f["observations"]:
        print("\n  [observations/images]")
        img_group = f["observations"]["images"]
        for img_key in img_group.keys():
            ds = img_group[img_key]
            print(f"    {img_key}: shape={ds.shape}, dtype={ds.dtype}")


# ── 视频 ──────────────────────────────────────────────────────────────────────

def get_video_info(video_path: Path) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", str(video_path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        for s in json.loads(out).get("streams", []):
            if s.get("codec_type") == "video":
                num, den = map(int, s["r_frame_rate"].split("/"))
                return {
                    "width":    s["width"],
                    "height":   s["height"],
                    "fps":      round(num / den, 2) if den else 0,
                    "codec":    s["codec_name"],
                    "pix_fmt":  s.get("pix_fmt"),
                    "duration": round(float(s.get("duration", 0)), 2),
                }
    except Exception as e:
        return {"error": str(e)}
    return {}


def inspect_videos():
    # 取第一个 episode 文件夹
    ep_dirs = sorted(
        [d for d in VIDEO_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    )
    if not ep_dirs:
        print(f"\n[视频目录不存在或为空] {VIDEO_DIR}")
        return

    sample_ep = ep_dirs[0]
    videos    = sorted(sample_ep.glob("*.mp4"))

    print(f"\n{'='*60}")
    print(f"  视频信息（采样自 episode {sample_ep.name}，共 {len(ep_dirs)} 个 episode）")
    print(f"{'='*60}")

    for vid in videos:
        info = get_video_info(vid)
        cam  = vid.stem
        if "error" in info:
            print(f"  Camera {cam}: ERROR - {info['error']}")
        else:
            print(f"  Camera {cam}: {info['width']}x{info['height']} @ {info['fps']} fps"
                  f"  codec={info['codec']}  pix_fmt={info['pix_fmt']}"
                  f"  duration={info['duration']}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'#'*60}")
    print(f"  Dataset Inspector  |  {DATASET_DIR}")
    print(f"{'#'*60}")

    hdf5_files = sorted(DATASET_DIR.glob("episode_*.hdf5"))
    if hdf5_files:
        inspect_hdf5(hdf5_files[0])
    else:
        print("[未找到 HDF5 文件]")

    inspect_videos()
    print()


if __name__ == "__main__":
    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_f.write(f"\n{'#'*60}\n")
        log_f.write(f"# LOG TIME : {ts}\n")
        log_f.write(f"# DATASET  : {DATASET_DIR.resolve()}\n")
        log_f.write(f"{'#'*60}\n")

        sys.stdout = Tee(log_f)
        try:
            main()
        finally:
            sys.stdout = sys.stdout._stdout
