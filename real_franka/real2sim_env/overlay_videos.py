#!/usr/bin/env python3
"""
将两个视频叠加：底层视频 + 上层视频（透明度50%）
底层：/storage/zhijun/real_franka/pick_and_place/video/0/1.mp4
上层：BlockPAP-v1_replay_0.mp4（透明度50%）
"""

import subprocess
import sys
import os

# 视频路径
VIDEO_BOTTOM = "/storage/zhijun/real_franka/pick_and_place/videos/0/1.mp4"
VIDEO_TOP = "/workspace1/zhijun/RLinf/real_franka/real2sim_env/render/BlockPAP-v1_traj_0.mp4"
OUTPUT = "/workspace1/zhijun/RLinf/real_franka/real2sim_env/render/overlay_BlockPAP-v1_traj_0.mp4"

# 可通过命令行参数覆盖
if len(sys.argv) >= 3:
    VIDEO_BOTTOM = sys.argv[1]
    VIDEO_TOP = sys.argv[2]
if len(sys.argv) >= 4:
    OUTPUT = sys.argv[3]


def get_video_info(path):
    """获取视频的宽高和帧率"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)
    for s in data["streams"]:
        if s["codec_type"] == "video":
            w, h = s["width"], s["height"]
            fps_str = s["r_frame_rate"]
            num, den = map(int, fps_str.split("/"))
            fps = num / den
            return w, h, fps
    raise ValueError(f"No video stream found in {path}")


def overlay_videos(video_bottom, video_top, output, alpha=0.5):
    """
    叠加两个视频，video_top 在上层，透明度为 alpha（0~1）。
    如果两个视频尺寸不同，video_top 会缩放到 video_bottom 的尺寸。
    取两个视频中较短的时长。
    """
    for path in [video_bottom, video_top]:
        if not os.path.exists(path):
            print(f"[ERROR] 文件不存在: {path}")
            sys.exit(1)

    w, h, fps = get_video_info(video_bottom)
    print(f"底层视频: {w}x{h} @ {fps:.2f}fps")

    w2, h2, fps2 = get_video_info(video_top)
    print(f"上层视频: {w2}x{h2} @ {fps2:.2f}fps")

    os.makedirs(os.path.dirname(output), exist_ok=True)

    # ffmpeg filter_complex:
    # 1. 将上层视频缩放到底层视频尺寸（如果不同）
    # 2. 将上层视频设为 RGBA 格式，应用 alpha 透明度
    # 3. 叠加到底层视频上
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
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output
    ]

    print(f"\n执行命令:\n{' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n[OK] 输出视频已保存到: {output}")
    else:
        print(f"\n[ERROR] ffmpeg 失败，返回码: {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    print(f"底层视频: {VIDEO_BOTTOM}")
    print(f"上层视频: {VIDEO_TOP}")
    print(f"输出路径: {OUTPUT}")
    overlay_videos(VIDEO_BOTTOM, VIDEO_TOP, OUTPUT, alpha=0.3)
