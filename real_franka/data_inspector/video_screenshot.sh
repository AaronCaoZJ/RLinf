#!/bin/bash
# 截取/storage/zhijun/real_franka/pick_and_place/videos/
# 0 & 15 & 25 & 40 & 45
# 共5个起始点，初始化时刻和接近抓取时刻分别命名为:
# BlockPAP_traj0_t0.png, BlockPAP_traj15_t0.png, BlockPAP_traj25_t0.png, BlockPAP_traj40_t0.png, BlockPAP_traj45_t0.png
# BlockPAP_traj0_t12.png, BlockPAP_traj15_t12.png, BlockPAP_traj25_t12.png, BlockPAP_traj40_t12.png, BlockPAP_traj45_t12.png

VIDEO_BASE="/storage/zhijun/real_franka/pick_and_place/videos"
OUTPUT_DIR="/workspace1/zhijun/RLinf/real_franka/data_inspector/BlockPAP_ref_screenshot"

mkdir -p "$OUTPUT_DIR"

for traj in 0 15 25 40 45; do
    VIDEO="${VIDEO_BASE}/${traj}/1.mp4"

    # t=0: 初始化时刻
    ffmpeg -ss 00:00:00 \
        -i "$VIDEO" \
        -vframes 1 \
        "${OUTPUT_DIR}/BlockPAP_traj${traj}_t0.png"

    # t=12: 接近抓取时刻
    ffmpeg -ss 00:00:12 \
        -i "$VIDEO" \
        -vframes 1 \
        "${OUTPUT_DIR}/BlockPAP_traj${traj}_t12.png"
done
