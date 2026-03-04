#!/usr/bin/env python3
"""
批量叠加真实机器人视频（底层）与仿真回放视频（上层）。

修改下方的全局变量后直接运行：
  python overlay_videos.py

路径约定：
  底层：/storage/zhijun/real_franka/pick_and_place/videos/{traj}/1.mp4
  上层：real_franka/real2sim_env/render/traj{traj}/{CAM_T}/replay.mp4
  输出：real_franka/real2sim_env/render/traj{traj}/{CAM_T}/overlay.mp4
"""

import json
import os
import subprocess
import sys

# ── 配置 ─────────────────────────────────────────────────────────────────────
TRAJ_LIST = [0, 15, 25, 40, 45]  # 要叠加的轨迹 ID 列表
CAM_T     = "og"                 # 相机预设，与 replay_traj.py 保持一致
ALPHA     = 0.5                  # 上层视频透明度（0~1）
# ─────────────────────────────────────────────────────────────────────────────

_HDF5_VIDEO_BASE = "/storage/zhijun/real_franka/pick_and_place/videos"
_RENDER_BASE     = "RLinf/real_franka/real2sim_env/render"


def get_video_info(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    for s in data["streams"]:
        if s["codec_type"] == "video":
            num, den = map(int, s["r_frame_rate"].split("/"))
            return s["width"], s["height"], num / den
    raise ValueError(f"No video stream in {path}")


def overlay_videos(video_bottom, video_top, output, alpha=0.5):
    for path in [video_bottom, video_top]:
        if not os.path.exists(path):
            print(f"  [ERROR] 文件不存在: {path}")
            sys.exit(1)

    w,  h,  fps  = get_video_info(video_bottom)
    w2, h2, fps2 = get_video_info(video_top)
    print(f"  底层: {w}x{h} @ {fps:.2f}fps")
    print(f"  上层: {w2}x{h2} @ {fps2:.2f}fps")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    scale_filter = f"scale={w}:{h}" if (w2 != w or h2 != h) else "null"
    filter_complex = (
        f"[1:v]{scale_filter},format=rgba,"
        f"colorchannelmixer=aa={alpha}[top];"
        f"[0:v][top]overlay=0:0[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_bottom,
        "-i", video_top,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-shortest",
        output,
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"  [OK] → {output}")
    else:
        print(f"  [ERROR] ffmpeg 失败，返回码: {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    for traj in TRAJ_LIST:
        print(f"\n=== traj {traj} ===")
        bottom = f"{_HDF5_VIDEO_BASE}/{traj}/1.mp4"
        top    = f"{_RENDER_BASE}/traj{traj}/{CAM_T}/replay.mp4"
        output = f"{_RENDER_BASE}/traj{traj}/{CAM_T}/overlay.mp4"
        overlay_videos(bottom, top, output, alpha=ALPHA)
