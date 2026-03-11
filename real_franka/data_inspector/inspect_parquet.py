#!/usr/bin/env python3
"""
LeRobot parquet 检查脚本：
1) 打印 HuggingFace metadata 中的 feature _type（Sequence / List / Image / Value）
2) 打印 Arrow schema（物理存储类型）
3) 打印文件结构（列名、shape、dtype、大小）
4) 对比两个文件的 index 行为（global index 连续性、frame_index 重置）
5) 打印数值列的范围（min/max/mean/std）及前 N 步实际数据
6) 输出日志文件与 parquet 同目录

用法:
  # 只检查一个文件
  python inspect_parquet.py /path/to/episode_000000.parquet

  # 同时对比两个文件（会额外输出跨文件 index 分析）
  python inspect_parquet.py /path/to/episode_000000.parquet /path/to/episode_000001.parquet

  # 传目录 → 自动选前两个文件对比
  python inspect_parquet.py /path/to/lerobot_dir/

  # 控制打印步数
  python inspect_parquet.py /path/to/file.parquet --max-timesteps 5
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def find_parquets(path: Path) -> List[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    return sorted(path.rglob("*.parquet"))


def is_image_col(col_name: str, first_val) -> bool:
    if "image" in col_name.lower():
        return True
    return isinstance(first_val, dict) and "bytes" in first_val


def array_stats(arr: np.ndarray) -> Tuple[float, float, float, float]:
    flat = arr.astype(np.float64).ravel()
    return float(flat.min()), float(flat.max()), float(flat.mean()), float(flat.std())


def _hf_features(schema) -> dict:
    """从 parquet schema metadata 中提取 HuggingFace feature 字典。"""
    meta = schema.metadata or {}
    raw = meta.get(b"huggingface")
    if raw is None:
        return {}
    return json.loads(raw).get("info", {}).get("features", {})


def _type_tag(feat: dict) -> str:
    """返回 HF feature 的 _type 标签，以及是否有问题。"""
    t = feat.get("_type", "?")
    sub_dtype = feat.get("feature", {}).get("dtype") or feat.get("dtype", "")
    length = feat.get("length", "")
    parts = [t]
    if sub_dtype:
        parts.append(f"dtype={sub_dtype}")
    if length != "":
        parts.append(f"len={length}")
    tag = "  ".join(parts)
    warn = " ← [WARN: should be Sequence]" if t == "List" else ""
    return tag + warn


# ── 单文件检查 ─────────────────────────────────────────────────────────────────

def inspect_one_file(parquet_path: Path, max_timesteps: int = 3) -> List[str]:
    lines: List[str] = []

    lines.append("=" * 65)
    lines.append(f"  FILE : {parquet_path.name}")
    lines.append("=" * 65)

    table  = pq.read_table(parquet_path)
    schema = table.schema
    df     = table.to_pandas()
    T      = len(df)
    hf     = _hf_features(schema)

    lines.append(f"  总行数 (timesteps): {T}")
    lines.append(f"  总列数            : {len(schema)}")
    lines.append("")

    # ── HuggingFace feature types ────────────────────────────────────────────
    lines.append("── HuggingFace Feature Types (" + ("来自 parquet metadata" if hf else "无 metadata") + ") ──")
    if hf:
        any_list = False
        for col_name, feat in hf.items():
            tag = _type_tag(feat)
            ok  = "WARN" if "WARN" in tag else " OK "
            lines.append(f"  [{ok}] {col_name:<20} {tag}")
            if "WARN" in tag:
                any_list = True
        if any_list:
            lines.append("")
            lines.append("  !! 存在 _type='List'，请运行 fix_parquet_metadata.py 修复 !!")
    else:
        lines.append("  (未找到 huggingface metadata，可能不是 LeRobot v2 格式)")
    lines.append("")

    # ── Arrow schema (物理类型) ──────────────────────────────────────────────
    lines.append("── Arrow Schema (物理存储类型) ──────────────────────────────────")
    for field in schema:
        hf_type = _type_tag(hf.get(field.name, {})) if field.name in hf else "-"
        lines.append(f"  {field.name:<20} arrow={str(field.type):<30} hf={hf_type}")
    lines.append("")

    # ── 列结构 ───────────────────────────────────────────────────────────────
    lines.append("── 列结构（shape / size）──────────────────────────────────────")
    for field in schema:
        col   = df[field.name]
        first = col.iloc[0] if T > 0 else None

        if is_image_col(field.name, first):
            try:
                from PIL import Image
                img  = Image.open(io.BytesIO(first["bytes"]))
                desc = f"image  {img.size[0]}×{img.size[1]} {img.mode}"
            except Exception:
                desc = "image  (无法解码)"
            size_kb = sum(
                len(v["bytes"]) for v in col if isinstance(v, dict) and "bytes" in v
            ) / 1024.0
            lines.append(f"  {field.name:<20} {desc}  total={size_kb:.1f} KB")
        elif isinstance(first, (list, np.ndarray)):
            arr     = np.array(first)
            size_mb = col.memory_usage(deep=True) / 1024 / 1024
            lines.append(
                f"  {field.name:<20} shape=({T},{len(arr)})  "
                f"dtype={arr.dtype}  ({size_mb:.3f} MB)"
            )
        else:
            size_kb = col.memory_usage(deep=True) / 1024
            lines.append(
                f"  {field.name:<20} scalar  arrow_dtype={field.type}  ({size_kb:.1f} KB)"
            )
    lines.append("")

    # ── Index 字段详情 ────────────────────────────────────────────────────────
    lines.append("── Index 字段详情 ──────────────────────────────────────────────")
    for idx_col in ("index", "frame_index", "episode_index", "task_index"):
        if idx_col not in df.columns:
            continue
        vals = df[idx_col].tolist()
        unique_vals = sorted(set(vals))
        # 连续性检查
        if len(vals) > 1:
            gaps = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
            non_unit = [(i, g) for i, g in enumerate(gaps) if g != 1]
        else:
            non_unit = []

        if idx_col in ("index", "frame_index"):
            cont = "连续 ✓" if not non_unit else f"不连续！{len(non_unit)} 处跳变"
            lines.append(
                f"  {idx_col:<20} [{vals[0]} .. {vals[-1]}]  "
                f"count={len(vals)}  {cont}"
            )
            if non_unit:
                for pos, gap in non_unit[:5]:
                    lines.append(f"    row {pos}→{pos+1}: {vals[pos]}→{vals[pos+1]}  gap={gap}")
        else:
            lines.append(f"  {idx_col:<20} 唯一值={unique_vals}")
    lines.append("")

    # ── 数值列统计 + 前 N 帧 ─────────────────────────────────────────────────
    n_label = "所有帧" if max_timesteps <= 0 else f"前 {max_timesteps} 帧"
    lines.append(f"── 数值列摘要（全量统计 + {n_label}）────────────────────────────")

    for field in schema:
        col   = df[field.name]
        first = col.iloc[0] if T > 0 else None
        if is_image_col(field.name, first):
            continue

        lines.append("")
        lines.append(f"[{field.name}]")

        if isinstance(first, (list, np.ndarray)):
            full_arr = np.array(col.tolist(), dtype=np.float64)
            lines.append(f"  shape=({T},{full_arr.shape[1]})  dtype=float64")
            vmin, vmax, mean, std = array_stats(full_arr)
            lines.append(
                f"  全量: min={vmin:.6f}  max={vmax:.6f}  "
                f"mean={mean:.6f}  std={std:.6f}"
            )
            # per-dim range
            per_dim_min = full_arr.min(axis=0)
            per_dim_max = full_arr.max(axis=0)
            dim_str = "  ".join(
                f"d{i}=[{per_dim_min[i]:.4g},{per_dim_max[i]:.4g}]"
                for i in range(full_arr.shape[1])
            )
            lines.append(f"  per-dim: {dim_str}")
            n_print = T if max_timesteps <= 0 else min(max_timesteps, T)
            lines.append(f"  first {n_print} timesteps:")
            for t in range(n_print):
                lines.append(f"    t={t}: {full_arr[t].tolist()}")
        else:
            try:
                arr = col.to_numpy(dtype=float)
                vmin, vmax, mean, std = array_stats(arr)
                lines.append(
                    f"  全量: min={vmin:.6f}  max={vmax:.6f}  "
                    f"mean={mean:.6f}  std={std:.6f}"
                )
                n_print = T if max_timesteps <= 0 else min(max_timesteps, T)
                lines.append(f"  first {n_print}: {arr[:n_print].tolist()}")
            except Exception:
                lines.append("  (非数值列，跳过统计)")

    lines.append("")
    return lines


# ── 跨文件 index 对比 ─────────────────────────────────────────────────────────

def compare_index(path_a: Path, path_b: Path) -> List[str]:
    lines: List[str] = []
    lines.append("=" * 65)
    lines.append("  跨文件 Index 行为对比")
    lines.append("=" * 65)

    ta = pq.read_table(path_a).to_pandas()
    tb = pq.read_table(path_b).to_pandas()

    for col in ("index", "frame_index", "episode_index", "task_index"):
        has_a = col in ta.columns
        has_b = col in tb.columns
        if not (has_a and has_b):
            continue

        va = ta[col].tolist()
        vb = tb[col].tolist()

        lines.append(f"\n[{col}]")
        lines.append(f"  {path_a.name}: [{va[0]} .. {va[-1]}]  count={len(va)}")
        lines.append(f"  {path_b.name}: [{vb[0]} .. {vb[-1]}]  count={len(vb)}")

        if col == "index":
            # global index 应跨文件连续
            expected_start = va[-1] + 1
            if vb[0] == expected_start:
                lines.append(f"  ✓ global index 跨文件连续 ({va[-1]} → {vb[0]})")
            else:
                gap = vb[0] - va[-1] - 1
                lines.append(
                    f"  ✗ global index 跨文件不连续！期望 {expected_start}，"
                    f"实际 {vb[0]}  (gap={gap})"
                )

        elif col == "frame_index":
            # frame_index 应在每个文件内从 0 开始
            ok_a = va[0] == 0
            ok_b = vb[0] == 0
            lines.append(
                f"  {path_a.name}: 从 {va[0]} 开始 {'✓' if ok_a else '✗ (应从0开始)'}"
            )
            lines.append(
                f"  {path_b.name}: 从 {vb[0]} 开始 {'✓' if ok_b else '✗ (应从0开始)'}"
            )

        elif col == "episode_index":
            same = set(va) == set(vb)
            lines.append(
                f"  {path_a.name}: {sorted(set(va))}"
            )
            lines.append(
                f"  {path_b.name}: {sorted(set(vb))}"
            )
            if same:
                lines.append("  [WARN] 两文件 episode_index 相同，可能有数据重复问题")
            else:
                lines.append("  ✓ episode_index 各自独立")

    lines.append("")
    return lines


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeRobot parquet 检查脚本")
    parser.add_argument(
        "input_path", type=str,
        help="parquet 文件路径、两个文件路径，或目录路径"
    )
    parser.add_argument(
        "input_path2", type=str, nargs="?", default=None,
        help="第二个 parquet 文件路径（可选，用于跨文件 index 对比）"
    )
    parser.add_argument(
        "--max-timesteps", type=int, default=3,
        help="打印的最大时间步数（默认3，0表示全部）"
    )
    return parser.parse_args()


def main() -> int:
    args  = parse_args()
    path  = Path(args.input_path)
    files = find_parquets(path)

    if not files:
        print(f"[ERROR] 未找到 parquet 文件: {path}")
        return 1

    # 确定要检查的文件
    if args.input_path2:
        path2 = Path(args.input_path2)
        if not path2.exists():
            print(f"[ERROR] 第二个文件不存在: {path2}")
            return 1
        file_a = files[0]
        file_b = path2
    elif len(files) >= 2:
        file_a = files[0]
        file_b = files[1]
    else:
        file_a = files[0]
        file_b = None

    print(f"找到 {len(files)} 个 parquet 文件，检查前 {'2' if file_b else '1'} 个")
    print(f"  A: {file_a}")
    if file_b:
        print(f"  B: {file_b}")
    print()

    all_lines: List[str] = []
    all_lines.append(f"# LOG TIME : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_lines.append(f"# INPUT    : {path}")
    all_lines.append("")

    # 单文件检查
    all_lines += inspect_one_file(file_a, max_timesteps=args.max_timesteps)
    if file_b:
        all_lines += inspect_one_file(file_b, max_timesteps=args.max_timesteps)
        all_lines += compare_index(file_a, file_b)

    # 写日志
    log_dir = path if path.is_dir() else path.parent
    out_path = log_dir / "inspect_log.txt"
    out_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    # 同时打印到 stdout
    print("\n".join(all_lines))
    print(f"\n[OK] 日志已保存至: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
