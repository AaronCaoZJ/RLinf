#!/usr/bin/env python3
"""
通用 HDF5 检查脚本：
1) 打印文件结构（group/dataset、shape、dtype、大小）
2) 打印关键字段的数值范围（min/max/mean/std）
3) 打印前 3 个时间步的实际数据

输出文件：与 hdf5 同目录，命名为 <hdf5_stem>_log.txt

用法:
  python inspect_hdf5.py /path/to/file.hdf5
  python inspect_hdf5.py /path/to/dir --recursive
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np


def sizeof_mb(nbytes: int) -> float:
    return float(nbytes) / (1024.0 * 1024.0)


def format_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return np.array2string(value, threshold=20)
    return str(value)


def collect_datasets(f: h5py.File) -> List[Tuple[str, h5py.Dataset]]:
    datasets: List[Tuple[str, h5py.Dataset]] = []

    def visitor(name: str, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))

    f.visititems(visitor)
    datasets.sort(key=lambda x: x[0])
    return datasets


DEMO_NAME_RE = re.compile(r"^demo_(\d+)$")


def parse_demo_index(name: str):
    match = DEMO_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def get_expanded_demo_name_from_keys(keys: List[str]):
    demo_names = set()
    for key in keys:
        parts = key.split("/")
        for part in parts:
            if parse_demo_index(part) is not None:
                demo_names.add(part)
                break

    if len(demo_names) < 2:
        return None
    if "demo_0" in demo_names:
        return "demo_0"
    return min(demo_names, key=lambda n: parse_demo_index(n))


def extract_demo_name_from_path(path: str):
    for part in path.split("/"):
        if parse_demo_index(part) is not None:
            return part
    return None


def render_group_tree(lines: List[str], group: h5py.Group, prefix: str = "") -> None:
    items = list(group.items())
    total = len(items)

    demo_group_names = []
    for name, obj in items:
        if isinstance(obj, h5py.Group) and parse_demo_index(name) is not None:
            demo_group_names.append(name)

    expanded_demo_name = None
    if len(demo_group_names) >= 2:
        # 优先展开 demo_0；若不存在则展开索引最小的 demo_x。
        if "demo_0" in demo_group_names:
            expanded_demo_name = "demo_0"
        else:
            expanded_demo_name = min(demo_group_names, key=lambda n: parse_demo_index(n))

    for idx, (name, obj) in enumerate(items):
        is_last = idx == total - 1
        connector = "└─ " if is_last else "├─ "
        child_prefix = prefix + ("   " if is_last else "│  ")

        if isinstance(obj, h5py.Group):
            if expanded_demo_name is not None and parse_demo_index(name) is not None:
                if name != expanded_demo_name:
                    lines.append(
                        f"{prefix}{connector}[{name}/]  (same as {expanded_demo_name}, omitted)"
                    )
                    continue
            lines.append(f"{prefix}{connector}[{name}/]")
            render_group_tree(lines, obj, child_prefix)
        elif isinstance(obj, h5py.Dataset):
            size_mb = sizeof_mb(obj.nbytes)
            lines.append(
                f"{prefix}{connector}{name}  {obj.shape}  {obj.dtype}  ({size_mb:.3f} MB)"
            )


def is_image_like(name: str, ds: h5py.Dataset) -> bool:
    lname = name.lower()
    if any(k in lname for k in ("image", "camera", "rgb", "depth", "mask")):
        return True
    if ds.ndim >= 3 and ds.dtype == np.uint8 and ds.shape[-1] in (1, 3, 4):
        return True
    return False


def is_key_field(name: str, ds: h5py.Dataset) -> bool:
    if not np.issubdtype(ds.dtype, np.number):
        return False
    if is_image_like(name, ds):
        return False

    lname = name.lower()
    keywords = (
        "time",
        "timestamp",
        "stage",
        "action",
        "joint",
        "pose",
        "state",
        "reward",
        "done",
        "success",
        "gripper",
        "ee",
    )
    if any(k in lname for k in keywords):
        return True

    # 兜底：常见时序小张量（避免把超大体素/图像类拉进来）
    if ds.ndim >= 1 and ds.size <= 2_000_000:
        return True

    return False


def stats_for_dataset(ds: h5py.Dataset) -> Tuple[float, float, float, float]:
    # 小数组直接一次读取
    if ds.size <= 1_000_000:
        arr = ds[()]
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        return float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std())

    # 大数组按第 0 维流式统计，避免峰值内存过高
    if ds.ndim == 0:
        v = float(ds[()])
        return v, v, v, 0.0

    chunk = min(max(1, 1024), ds.shape[0])
    total_count = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    vmin = float("inf")
    vmax = float("-inf")

    for start in range(0, ds.shape[0], chunk):
        end = min(start + chunk, ds.shape[0])
        part = np.asarray(ds[start:end], dtype=np.float64).reshape(-1)
        if part.size == 0:
            continue
        total_count += part.size
        total_sum += float(part.sum())
        total_sq_sum += float(np.dot(part, part))
        pvmin = float(part.min())
        pvmax = float(part.max())
        if pvmin < vmin:
            vmin = pvmin
        if pvmax > vmax:
            vmax = pvmax

    if total_count == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    mean = total_sum / total_count
    var = max(total_sq_sum / total_count - mean * mean, 0.0)
    std = var ** 0.5
    return vmin, vmax, mean, std


def head_timesteps(ds: h5py.Dataset, n: int = 3):
    if ds.ndim == 0:
        return ds[()]
    n = min(n, ds.shape[0])
    return ds[:n]


def write_header(lines: List[str], hdf5_path: Path) -> None:
    lines.append("#" * 60)
    lines.append(f"# LOG TIME : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# FILE     : {hdf5_path}")
    lines.append("#" * 60)
    lines.append("")


def inspect_one_file(hdf5_path: Path, max_timesteps: int = 3) -> Path:
    lines: List[str] = []
    write_header(lines, hdf5_path)

    with h5py.File(hdf5_path, "r") as f:
        lines.append("=" * 60)
        lines.append(f"  HDF5 结构: {hdf5_path.name}")
        lines.append("=" * 60)

        datasets = collect_datasets(f)

        # 打印 group / dataset 树形结构
        render_group_tree(lines, f)

        if f.attrs:
            lines.append("")
            lines.append("[Root Attributes]")
            for k, v in f.attrs.items():
                lines.append(f"  {k}: {format_value(v)}")

        lines.append("")
        lines.append("=" * 60)
        if max_timesteps <= 0:
            lines.append("  关键字段摘要（范围 + 所有帧）")
        else:
            lines.append(f"  关键字段摘要（范围 + 前{max_timesteps}帧）")
        lines.append("=" * 60)

        key_fields = [(name, ds) for name, ds in datasets if is_key_field(name, ds)]
        expanded_demo_name = get_expanded_demo_name_from_keys([name for name, _ in key_fields])
        omitted_demo_counter = {}

        if not key_fields:
            lines.append("  (未发现可用关键字段)")

        for name, ds in key_fields:
            demo_name = extract_demo_name_from_path(name)
            if (
                expanded_demo_name is not None
                and demo_name is not None
                and demo_name != expanded_demo_name
            ):
                omitted_demo_counter[demo_name] = omitted_demo_counter.get(demo_name, 0) + 1
                continue

            lines.append("")
            lines.append(f"[{name}]")
            lines.append(f"  shape={ds.shape}, dtype={ds.dtype}")
            vmin, vmax, mean, std = stats_for_dataset(ds)
            lines.append(
                f"  range: min={vmin:.6f}, max={vmax:.6f}, "
                f"mean={mean:.6f}, std={std:.6f}"
            )

            # 打印时间步
            n_print = ds.shape[0] if (ds.ndim > 0 and max_timesteps <= 0) else max_timesteps
            head = head_timesteps(ds, n_print)
            if ds.ndim == 0:
                lines.append(f"  value={format_value(head)}")
            elif ds.ndim == 1:
                lines.append(f"  first3={np.asarray(head).tolist()}")
            else:
                arr = np.asarray(head)
                lines.append(f"  first{arr.shape[0]} timesteps:")
                for t in range(arr.shape[0]):
                    lines.append(f"    t={t}: {arr[t].tolist()}")

        if omitted_demo_counter:
            lines.append("")
            lines.append("[Other demos]")
            lines.append(
                f"  expanded: {expanded_demo_name}; other demo key-field details are omitted."
            )
            for demo_name in sorted(
                omitted_demo_counter.keys(), key=lambda n: parse_demo_index(n)
            ):
                lines.append(
                    f"  - {demo_name}: omitted {omitted_demo_counter[demo_name]} key fields"
                )

    out_path = hdf5_path.with_name(f"{hdf5_path.stem}_log.txt")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def find_hdf5_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file() and path.suffix.lower() in (".h5", ".hdf5"):
        return [path]

    if not path.is_dir():
        return []

    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    files = sorted(path.glob(pattern))
    return [p for p in files if p.is_file()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通用 HDF5 检查脚本")
    parser.add_argument("input_path", type=str, help="hdf5 文件路径或目录路径")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="当输入是目录时，递归查找子目录中的 .hdf5",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=0,
        help="打印的最大时间步数（默认0，设为0或负数表示打印所有）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)

    hdf5_files = find_hdf5_files(input_path, args.recursive)
    if not hdf5_files:
        print(f"[ERROR] 未找到 hdf5 文件: {input_path}")
        return 1

    print(f"找到 {len(hdf5_files)} 个 hdf5 文件")
    for fp in hdf5_files:
        out = inspect_one_file(fp, max_timesteps=args.max_timesteps)
        print(f"[OK] {fp} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
