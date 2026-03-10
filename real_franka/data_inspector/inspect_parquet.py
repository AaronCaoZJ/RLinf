#!/usr/bin/env python3
"""
LeRobot parquet 检查脚本：
1) 打印文件结构（列名、shape、dtype、大小）
2) 打印数值列的范围（min/max/mean/std）
3) 打印前 N 个时间步的实际数据
4) 输出日志文件与 parquet 同目录

用法:
  python inspect_parquet.py /path/to/episode_000000.parquet
  python inspect_parquet.py /path/to/lerobot_dir/          # 递归找第一个
  python inspect_parquet.py /path/to/file.parquet --max-timesteps 5
"""

from __future__ import annotations

import argparse
import io
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


def is_image_col(col_name: str, col_data) -> bool:
    """判断列是否为图像（HuggingFace 存储为 dict with 'bytes' key）。"""
    if "image" in col_name.lower():
        return True
    try:
        first = col_data.iloc[0]
        return isinstance(first, dict) and "bytes" in first
    except Exception:
        return False


def array_stats(arr: np.ndarray) -> Tuple[float, float, float, float]:
    flat = arr.astype(np.float64).ravel()
    return float(flat.min()), float(flat.max()), float(flat.mean()), float(flat.std())


# ── 核心检查 ──────────────────────────────────────────────────────────────────

def inspect_one_file(parquet_path: Path, max_timesteps: int = 3,
                     log_dir: Path = None) -> Path:
    lines: List[str] = []

    lines.append("#" * 60)
    lines.append(f"# LOG TIME : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# FILE     : {parquet_path}")
    lines.append("#" * 60)
    lines.append("")

    table  = pq.read_table(parquet_path)
    schema = table.schema
    df     = table.to_pandas()
    T      = len(df)

    # ── 结构树 ────────────────────────────────────────────────────────────────
    lines.append("=" * 60)
    lines.append(f"  Parquet 结构: {parquet_path.name}")
    lines.append("=" * 60)
    lines.append(f"  总行数 (timesteps): {T}")
    lines.append(f"  总列数            : {len(schema)}")
    lines.append("")

    for field in schema:
        col   = df[field.name]
        first = col.iloc[0] if T > 0 else None

        if is_image_col(field.name, col):
            try:
                from PIL import Image
                img  = Image.open(io.BytesIO(first["bytes"]))
                desc = f"image  {img.size[0]}×{img.size[1]} {img.mode}"
            except Exception:
                desc = "image  (无法解码)"
            size_kb = sum(
                len(v["bytes"]) for v in col if isinstance(v, dict) and "bytes" in v
            ) / 1024.0
            lines.append(f"  {field.name:<35} {desc}  total={size_kb:.1f} KB")
        elif isinstance(first, (list, np.ndarray)):
            arr     = np.array(first)
            size_mb = col.memory_usage(deep=True) / 1024 / 1024
            lines.append(
                f"  {field.name:<35} shape=({T},{len(arr)})  "
                f"dtype={arr.dtype}  ({size_mb:.3f} MB)"
            )
        else:
            size_kb = col.memory_usage(deep=True) / 1024
            lines.append(
                f"  {field.name:<35} scalar  dtype={field.type}  ({size_kb:.1f} KB)"
            )

    # ── 数值列统计 ────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 60)
    n_label = "所有帧" if max_timesteps <= 0 else f"前 {max_timesteps} 帧"
    lines.append(f"  数值列摘要（范围 + {n_label}）")
    lines.append("=" * 60)

    for field in schema:
        col   = df[field.name]
        first = col.iloc[0] if T > 0 else None

        if is_image_col(field.name, col):
            continue

        lines.append("")
        lines.append(f"[{field.name}]")

        if isinstance(first, (list, np.ndarray)):
            full_arr = np.array(col.tolist(), dtype=np.float64)
            lines.append(f"  shape=({T},{full_arr.shape[1]})  dtype=float")
            vmin, vmax, mean, std = array_stats(full_arr)
            lines.append(
                f"  range: min={vmin:.6f}, max={vmax:.6f}, "
                f"mean={mean:.6f}, std={std:.6f}"
            )
            n_print = T if max_timesteps <= 0 else min(max_timesteps, T)
            lines.append(f"  first {n_print} timesteps:")
            for t in range(n_print):
                lines.append(f"    t={t}: {full_arr[t].tolist()}")
        else:
            try:
                arr = col.to_numpy(dtype=float)
                vmin, vmax, mean, std = array_stats(arr)
                lines.append(
                    f"  range: min={vmin:.6f}, max={vmax:.6f}, "
                    f"mean={mean:.6f}, std={std:.6f}"
                )
                n_print = T if max_timesteps <= 0 else min(max_timesteps, T)
                lines.append(f"  first {n_print}: {arr[:n_print].tolist()}")
            except Exception:
                lines.append("  (非数值列，跳过统计)")

    save_dir = log_dir if log_dir is not None else parquet_path.parent
    out_path = save_dir / f"{parquet_path.stem}_log.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeRobot parquet 检查脚本")
    parser.add_argument("input_path", type=str, help="parquet 文件路径或目录路径")
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

    print(f"找到 {len(files)} 个 parquet 文件")
    for f in files:
        print(f"  {f}")

    # 日志存到用户传入的根目录（若传的是文件则存在其所在目录）
    log_dir = path if path.is_dir() else path.parent
    out = inspect_one_file(files[0], max_timesteps=args.max_timesteps, log_dir=log_dir)
    print(f"\n[OK] {files[0]}\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
