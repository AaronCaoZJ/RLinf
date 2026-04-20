#!/usr/bin/env bash
# render_multi_assets.sh
#
# 阶段 1 — 预览（20 张，每维独立扫描，不加相机偏转）：
#   Round 1: 10 种桌面纹理（ambient=default, bg=none）
#   Round 2:  4 种环境光   （tex=006, bg=none）
#   Round 3:  3 种背景墙   （tex=006, ambient=default）
#
# 阶段 2 — 全量正式渲染（5×4×3=60 张，每张带随机相机偏转）：
#   TABLE_TEXTURES=(003 006 010 white black)
#   AMBIENT_PRESETS=(bright dim default warm)
#   BG_PRESETS=(gray dark white)
#   每种组合带 --cam-jitter（roll/pitch/yaw 各 ±2°）
#
# 用法（从 RLinf/ 根目录执行）：
#   bash real_franka/real2sim_env/render_multi_assets.sh [--skip-preview] [--preview-only]
#
# 可选：CUDA_VISIBLE_DEVICES=0 bash ...

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY_SCRIPT="$SCRIPT_DIR/pick_and_place.py"
OUT_BASE="$SCRIPT_DIR/render/MultiAssets"

SKIP_PREVIEW=false
PREVIEW_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--skip-preview" ]] && SKIP_PREVIEW=true
    [[ "$arg" == "--preview-only" ]] && PREVIEW_ONLY=true
done

cd "$REPO_ROOT"

# # ═════════════════════════════════════════════════════════════════════════════
# # 阶段 1：预览扫描（无相机偏转）
# # ═════════════════════════════════════════════════════════════════════════════
# if ! $SKIP_PREVIEW; then

# # ─────────────────────────────────────────────────────────────────────────────
# # Round 1：桌面纹理（含 white / black；ambient=default, bg=none 固定）
# # ─────────────────────────────────────────────────────────────────────────────
# PREVIEW_TEX=(001 002 003 004 005 006 007 008 009 010 white black)

# echo "========================================================="
# echo " Round 1: 桌面纹理预览（共 ${#PREVIEW_TEX[@]} 张）"
# echo "========================================================="
# count=0
# for tex in "${PREVIEW_TEX[@]}"; do
#     count=$(( count + 1 ))
#     out_dir="$OUT_BASE/preview_tex/tex_${tex}"
#     echo "  [$count/${#PREVIEW_TEX[@]}] tex=$tex → $out_dir"
#     python "$PY_SCRIPT" --table-tex "$tex" --ambient default --bg none --out-dir "$out_dir"
# done

# # ─────────────────────────────────────────────────────────────────────────────
# # Round 2：环境光（tex=006, bg=none 固定）
# # ─────────────────────────────────────────────────────────────────────────────
# PREVIEW_AMB=(bright dim default warm)

# echo ""
# echo "========================================================="
# echo " Round 2: 环境光预览（共 ${#PREVIEW_AMB[@]} 张）"
# echo "========================================================="
# count=0
# for amb in "${PREVIEW_AMB[@]}"; do
#     count=$(( count + 1 ))
#     out_dir="$OUT_BASE/preview_amb/amb_${amb}"
#     echo "  [$count/${#PREVIEW_AMB[@]}] ambient=$amb → $out_dir"
#     python "$PY_SCRIPT" --table-tex 006 --ambient "$amb" --bg none --out-dir "$out_dir"
# done

# # ─────────────────────────────────────────────────────────────────────────────
# # Round 3：背景墙（tex=006, ambient=default 固定）
# # ─────────────────────────────────────────────────────────────────────────────
# PREVIEW_BG=(gray dark white)

# echo ""
# echo "========================================================="
# echo " Round 3: 背景墙预览（共 ${#PREVIEW_BG[@]} 张）"
# echo "========================================================="
# count=0
# for bg in "${PREVIEW_BG[@]}"; do
#     count=$(( count + 1 ))
#     out_dir="$OUT_BASE/preview_bg/bg_${bg}"
#     echo "  [$count/${#PREVIEW_BG[@]}] bg=$bg → $out_dir"
#     python "$PY_SCRIPT" --table-tex 006 --ambient default --bg "$bg" --out-dir "$out_dir"
# done

# total_preview=$(( ${#PREVIEW_TEX[@]} + ${#PREVIEW_AMB[@]} + ${#PREVIEW_BG[@]} ))
# echo ""
# echo "✅ 预览完成！共 $total_preview 张"
# echo "   $OUT_BASE/preview_tex/   preview_amb/   preview_bg/"

# fi  # end !SKIP_PREVIEW

# ═════════════════════════════════════════════════════════════════════════════
# 阶段 2：正式全量渲染（5 tex × 4 amb × 3 bg = 60 张，每张带随机相机偏转）
# ═════════════════════════════════════════════════════════════════════════════
if ! $PREVIEW_ONLY; then

TABLE_TEXTURES=(003 006 010 white black)
AMBIENT_PRESETS=(bright dim default warm)
BG_PRESETS=(gray dark white)

total=$(( ${#TABLE_TEXTURES[@]} * ${#AMBIENT_PRESETS[@]} * ${#BG_PRESETS[@]} ))
count=0

echo ""
echo "========================================================="
echo " 阶段 2: 全量正式渲染（$total 张，含随机相机偏转 ±2°）"
echo "========================================================="

for tex in "${TABLE_TEXTURES[@]}"; do
    for amb in "${AMBIENT_PRESETS[@]}"; do
        for bg in "${BG_PRESETS[@]}"; do
            count=$(( count + 1 ))
            out_dir="$OUT_BASE/tex_${tex}__amb_${amb}__bg_${bg}"
            echo "  [$count/$total] tex=$tex  amb=$amb  bg=$bg → $out_dir"
            python "$PY_SCRIPT" \
                --table-tex "$tex" \
                --ambient   "$amb" \
                --bg        "$bg"  \
                --cam-jitter        \
                --out-dir   "$out_dir"
        done
    done
done

echo ""
echo "✅ 全量完成！$total 张截图已保存到 $OUT_BASE"

fi  # end !PREVIEW_ONLY
