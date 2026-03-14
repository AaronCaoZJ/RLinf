#!/usr/bin/env python3
"""
LeRobot parquet 检查脚本：
默认输出精简报告，聚焦：
1) HuggingFace metadata 中关键列是 Sequence 还是 List
2) index 是否在文件内连续、以及跨 episode 是否全局连续
3) state[6] / actions[6] 的真实范围与关键校验

可选通过 --full-report 输出完整详细报告。

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
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

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


def _type_status(hf: dict, key: str, expect_sequence: bool = False) -> Tuple[str, str]:
    feat = hf.get(key, {})
    t = feat.get("_type", "?")
    if expect_sequence:
        ok = t == "Sequence"
        return ("PASS" if ok else "FAIL"), t
    return "INFO", t


def _parse_episode_id_from_name(name: str) -> int | None:
    m = re.search(r"episode_(\d+)\.parquet$", name)
    if not m:
        return None
    return int(m.group(1))


def _feat_dtype(feat: dict) -> str:
    return feat.get("feature", {}).get("dtype") or feat.get("dtype", "")


def _arrow_field_type(schema, col: str) -> str:
    idx = schema.get_field_index(col)
    if idx < 0:
        return "<missing>"
    return str(schema.field(col).type)


def _is_expected_image_arrow_type(t: str) -> bool:
    # Expected physical storage for HF Image column in parquet.
    return t == "struct<bytes: binary, path: string>"


def _is_expected_vec_arrow_type(t: str) -> bool:
    return t == "fixed_size_list<element: float>[7]"


def _arrow_to_np(arr) -> np.ndarray:
    """Convert an Arrow array (possibly FixedSizeList) to numpy float64."""
    if arr is None:
        return np.array([])
    return np.asarray(arr.to_pylist(), dtype=np.float32)


def _is_binary01(arr: np.ndarray, tol: float = 1e-6) -> bool:
    if arr.size == 0:
        return True
    return np.all((np.abs(arr - 0.0) <= tol) | (np.abs(arr - 1.0) <= tol))


def _validate_contiguous(vals: np.ndarray) -> Tuple[bool, str]:
    if vals.size <= 1:
        return True, "样本不足，无连续性问题"
    diffs = np.diff(vals)
    bad = np.where(diffs != 1)[0]
    if bad.size == 0:
        return True, "连续"
    i = int(bad[0])
    return False, f"不连续: row {i}->{i+1}, {vals[i]}->{vals[i+1]}, gap={diffs[i]}"


def _estimate_column_size_kb(pf: pq.ParquetFile, col_name: str) -> float:
    """Estimate on-disk compressed size (KB) from Parquet metadata."""
    if pf.metadata is None:
        return 0.0
    schema_names = list(pf.schema_arrow.names)
    if col_name not in schema_names:
        return 0.0
    col_idx = schema_names.index(col_name)
    total = 0
    for rg in range(pf.metadata.num_row_groups):
        total += pf.metadata.row_group(rg).column(col_idx).total_compressed_size
    size_kb = total / 1024.0
    if size_kb > 0:
        return size_kb

    # Fallback: some struct/image columns may report zero compressed size.
    try:
        t = pf.read(columns=[col_name])
        vals = t[col_name].to_pylist()
        raw = 0
        external_paths = []
        for v in vals:
            if isinstance(v, dict) and isinstance(v.get("bytes"), (bytes, bytearray)):
                raw += len(v["bytes"])
            elif isinstance(v, dict) and isinstance(v.get("path"), str) and v["path"]:
                external_paths.append(v["path"])

        if raw > 0:
            return raw / 1024.0

        ext = 0
        for p in external_paths:
            try:
                pp = Path(p)
                if pp.exists() and pp.is_file():
                    ext += pp.stat().st_size
            except Exception:
                continue
        return ext / 1024.0
    except Exception:
        return size_kb


def _image_storage_mode(pf: pq.ParquetFile, col_name: str) -> str:
    """Return 'bytes', 'path' or 'unknown' for HF Image storage style."""
    try:
        t = pf.read(columns=[col_name])
        vals = t[col_name].to_pylist()
        for v in vals:
            if not isinstance(v, dict):
                continue
            if isinstance(v.get("bytes"), (bytes, bytearray)) and len(v["bytes"]) > 0:
                return "bytes"
            if isinstance(v.get("path"), str) and v["path"]:
                return "path"
    except Exception:
        pass
    return "unknown"


def _decode_first_image_info(pf: pq.ParquetFile, col_name: str) -> str:
    """Decode first image bytes for (W,H,mode) summary."""
    try:
        t = pf.read(columns=[col_name])
        vals = t[col_name].to_pylist()
        first = vals[0] if vals else None
        if isinstance(first, dict) and "bytes" in first and first["bytes"] is not None:
            from PIL import Image

            img = Image.open(io.BytesIO(first["bytes"]))
            return f"image  {img.size[0]}×{img.size[1]} {img.mode}"
    except Exception:
        pass
    return "image  (无法解码)"


def _load_first_image_pil(pf: pq.ParquetFile, col_name: str):
    """Load first image from HF Image struct column as PIL.Image or return None."""
    try:
        from PIL import Image

        t = pf.read(columns=[col_name])
        vals = t[col_name].to_pylist()
        first = vals[0] if vals else None
        if not isinstance(first, dict):
            return None

        b = first.get("bytes")
        p = first.get("path")
        if isinstance(b, (bytes, bytearray)) and len(b) > 0:
            return Image.open(io.BytesIO(b)).convert("RGB")
        if isinstance(p, str) and p:
            pp = Path(p)
            if pp.exists() and pp.is_file():
                return Image.open(pp).convert("RGB")
    except Exception:
        return None
    return None


def _load_image_from_hf_cell(cell):
    """Decode one HF Image cell ({bytes,path}) to PIL.Image or return None."""
    try:
        from PIL import Image

        if not isinstance(cell, dict):
            return None

        b = cell.get("bytes")
        p = cell.get("path")

        if isinstance(b, (bytes, bytearray)) and len(b) > 0:
            return Image.open(io.BytesIO(b))

        if isinstance(p, str) and p:
            pp = Path(p)
            if pp.exists() and pp.is_file():
                return Image.open(pp)
    except Exception:
        return None
    return None


def _sample_positions(n: int, max_samples: int = 5) -> List[int]:
    if n <= 0:
        return []
    if n == 1:
        return [0]
    base = [0, n - 1, n // 2]
    # Add quartiles for better coverage on longer sequences.
    if n >= 4:
        base.extend([n // 4, (3 * n) // 4])
    out = sorted(set(base))
    return out[:max_samples]


def check_image_column_geometry(pf: pq.ParquetFile, col_name: str, max_samples: int = 5) -> Dict:
    """Check image decoding, HWC geometry, channel count and dtype from sampled rows."""
    result = {
        "exists": col_name in pf.schema_arrow.names,
        "sampled": 0,
        "decoded": 0,
        "shapes": set(),
        "modes": set(),
        "dtypes": set(),
        "decode_errors": 0,
    }
    if not result["exists"]:
        return result

    try:
        t = pf.read(columns=[col_name])
        vals = t[col_name].to_pylist()
    except Exception:
        result["decode_errors"] += 1
        return result

    for i in _sample_positions(len(vals), max_samples=max_samples):
        result["sampled"] += 1
        img = _load_image_from_hf_cell(vals[i])
        if img is None:
            result["decode_errors"] += 1
            continue

        try:
            arr = np.asarray(img)
            result["decoded"] += 1
            result["modes"].add(str(img.mode))
            result["dtypes"].add(str(arr.dtype))
            # Normalize grayscale to explicit channel for consistency in report.
            if arr.ndim == 2:
                h, w = arr.shape
                c = 1
            elif arr.ndim == 3:
                h, w, c = arr.shape
            else:
                h, w, c = -1, -1, -1
            result["shapes"].add((int(h), int(w), int(c)))
        except Exception:
            result["decode_errors"] += 1

    return result


def export_first_episode_images(parquet_path: Path, out_dir: Path) -> List[str]:
    """Export first frame of image and wrist_image from the first episode parquet."""
    lines: List[str] = []
    lines.append("── 首帧图像导出 ─────────────────────────────────────────────")

    pf = pq.ParquetFile(parquet_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for col_name in ("image", "wrist_image"):
        if col_name not in pf.schema_arrow.names:
            lines.append(f"  [WARN] {col_name}: 列不存在")
            continue

        img = _load_first_image_pil(pf, col_name)
        if img is None:
            lines.append(f"  [WARN] {col_name}: 无法解码首帧")
            continue

        save_path = out_dir / f"first_episode_first_frame_{col_name}.png"
        img.save(save_path)
        lines.append(
            f"  [PASS] {col_name}: 已保存 {save_path}  ({img.size[0]}x{img.size[1]})"
        )
        exported += 1

    if exported == 0:
        lines.append("  [WARN] 未导出任何图像")
    lines.append("")
    return lines


def validate_lerobot21(
    schema,
    hf: Dict,
    state_arr: np.ndarray,
    action_arr: np.ndarray,
    index_arr: np.ndarray,
    frame_index_arr: np.ndarray,
    episode_index_arr: np.ndarray,
    task_index_arr: np.ndarray,
    pf: pq.ParquetFile | None = None,
) -> List[str]:
    """Run LeRobot v2.1 focused checks and return report lines."""
    lines: List[str] = []
    lines.append("── LeRobot v2.1 专项校验 ─────────────────────────────────────")

    required_cols = {
        "image", "wrist_image", "state", "actions", "timestamp",
        "frame_index", "episode_index", "index", "task_index",
    }
    actual_cols = set(schema.names)
    missing = sorted(required_cols - actual_cols)
    extra = sorted(actual_cols - required_cols)

    lines.append(f"  [{'PASS' if not missing else 'FAIL'}] 必需列完整性: missing={missing if missing else '[]'}")
    if extra:
        lines.append(f"  [WARN] 额外列: {extra}")

    st_feat = hf.get("state", {})
    ac_feat = hf.get("actions", {})
    st_ok = (
        st_feat.get("_type") == "Sequence"
        and st_feat.get("length") == 7
        and _feat_dtype(st_feat) == "float32"
    )
    ac_ok = (
        ac_feat.get("_type") == "Sequence"
        and ac_feat.get("length") == 7
        and _feat_dtype(ac_feat) == "float32"
    )
    lines.append(f"  [{'PASS' if st_ok else 'FAIL'}] state metadata: _type={st_feat.get('_type')} len={st_feat.get('length')}")
    lines.append(f"  [{'PASS' if ac_ok else 'FAIL'}] actions metadata: _type={ac_feat.get('_type')} len={ac_feat.get('length')}")

    img_feat = hf.get("image", {})
    wimg_feat = hf.get("wrist_image", {})
    img_ok = img_feat.get("_type") == "Image"
    wimg_ok = wimg_feat.get("_type") == "Image"
    lines.append(f"  [{'PASS' if img_ok else 'FAIL'}] image metadata: _type={img_feat.get('_type', '?')}")
    lines.append(f"  [{'PASS' if wimg_ok else 'FAIL'}] wrist_image metadata: _type={wimg_feat.get('_type', '?')}")

    # Arrow physical types are critical for downstream decoding.
    image_arrow = _arrow_field_type(schema, "image")
    wrist_arrow = _arrow_field_type(schema, "wrist_image")
    state_arrow = _arrow_field_type(schema, "state")
    actions_arrow = _arrow_field_type(schema, "actions")
    lines.append(
        f"  [{'PASS' if _is_expected_image_arrow_type(image_arrow) else 'FAIL'}] image arrow type: {image_arrow}"
    )
    lines.append(
        f"  [{'PASS' if _is_expected_image_arrow_type(wrist_arrow) else 'FAIL'}] wrist_image arrow type: {wrist_arrow}"
    )
    lines.append(
        f"  [{'PASS' if _is_expected_vec_arrow_type(state_arrow) else 'FAIL'}] state arrow type: {state_arrow}"
    )
    lines.append(
        f"  [{'PASS' if _is_expected_vec_arrow_type(actions_arrow) else 'FAIL'}] actions arrow type: {actions_arrow}"
    )

    # Image geometry/type checks (HWC/channel/dtype) from sampled rows.
    if pf is not None:
        for col in ("image", "wrist_image"):
            img_ck = check_image_column_geometry(pf, col_name=col, max_samples=5)
            if not img_ck["exists"]:
                lines.append(f"  [FAIL] {col} 几何/类型校验: 列不存在")
                continue

            sampled = img_ck["sampled"]
            decoded = img_ck["decoded"]
            dec_ok = sampled > 0 and decoded == sampled
            lines.append(
                f"  [{'PASS' if dec_ok else 'FAIL'}] {col} 解码样本: decoded={decoded}/{sampled}"
            )

            shapes = sorted(img_ck["shapes"])
            dtypes = sorted(img_ck["dtypes"])
            modes = sorted(img_ck["modes"])

            if shapes:
                # Expect HWC with 3 channels and consistent shape across sampled rows.
                c_ok = all(s[2] == 3 for s in shapes)
                shape_consistent = len(shapes) == 1
                lines.append(
                    f"  [{'PASS' if c_ok else 'FAIL'}] {col} channel 校验: shapes={shapes} (期望 C=3)"
                )
                lines.append(
                    f"  [{'PASS' if shape_consistent else 'WARN'}] {col} 分辨率一致性: unique_shapes={shapes}"
                )

            if dtypes:
                dtype_ok = dtypes == ["uint8"]
                lines.append(
                    f"  [{'PASS' if dtype_ok else 'WARN'}] {col} dtype 校验: sampled_dtypes={dtypes} (期望 uint8)"
                )
            if modes:
                mode_ok = modes == ["RGB"]
                lines.append(
                    f"  [{'PASS' if mode_ok else 'WARN'}] {col} PIL mode: sampled_modes={modes} (期望 RGB)"
                )

    st_shape_ok = state_arr.ndim == 2 and state_arr.shape[1] == 7
    ac_shape_ok = action_arr.ndim == 2 and action_arr.shape[1] == 7
    lines.append(f"  [{'PASS' if st_shape_ok else 'FAIL'}] state shape: {state_arr.shape}")
    lines.append(f"  [{'PASS' if ac_shape_ok else 'FAIL'}] actions shape: {action_arr.shape}")

    if st_shape_ok:
        s6 = state_arr[:, 6]
        s6_min = float(np.min(s6))
        s6_max = float(np.max(s6))
        s6_ok = np.isfinite(s6).all() and s6_min >= -1e-6 and s6_max <= 0.12
        lines.append(
            f"  [INFO] state[6] 真实夹爪范围: min={s6_min:.6f}, max={s6_max:.6f}"
        )
        lines.append(
            f"  [{'PASS' if s6_ok else 'WARN'}] state[6] 夹爪宽度(米, 连续值/非二值): "
            f"min={s6_min:.6f}, max={s6_max:.6f} (参考约[0,0.08])"
        )

    if ac_shape_ok:
        g = action_arr[:, 6]
        g_min = float(np.min(g))
        g_max = float(np.max(g))
        g_range_ok = np.isfinite(g).all() and g_min >= -1e-6 and g_max <= 1.0 + 1e-6
        g_ok = _is_binary01(g)
        uniq = sorted(set(np.round(g, 6).tolist()))
        lines.append(
            f"  [INFO] actions[6] 真实夹爪范围: min={g_min:.6f}, max={g_max:.6f}"
        )
        lines.append(
            f"  [{'PASS' if g_range_ok else 'FAIL'}] actions[6] 夹爪范围校验: 期望[0,1]"
        )
        lines.append(
            f"  [{'PASS' if g_ok else 'FAIL'}] actions[6] 夹爪命令(应为二值0/1): unique={uniq[:8]}"
        )

        # pi0.5 sanity: translational deltas should be finite and small; rotational deltas in [-pi, pi].
        arm_delta = action_arr[:, :6].astype(np.float64)
        arm_finite_ok = np.isfinite(arm_delta).all()
        pos_max = float(np.max(np.abs(arm_delta[:, :3]))) if arm_delta.size else 0.0
        rot_max = float(np.max(np.abs(arm_delta[:, 3:6]))) if arm_delta.size else 0.0
        # keep loose thresholds to avoid false positives while catching clearly broken values
        pos_ok = pos_max <= 0.20
        rot_ok = rot_max <= (np.pi + 1e-3)
        lines.append(
            f"  [{'PASS' if arm_finite_ok else 'FAIL'}] actions[:6] 有限值校验: {'finite' if arm_finite_ok else 'contains NaN/Inf'}"
        )
        lines.append(
            f"  [{'PASS' if pos_ok else 'WARN'}] actions[:3] 位置增量绝对值上界: max_abs={pos_max:.6f} (参考<=0.20m)"
        )
        lines.append(
            f"  [{'PASS' if rot_ok else 'WARN'}] actions[3:6] 旋转增量绝对值上界: max_abs={rot_max:.6f} (参考<=pi)"
        )

    if frame_index_arr.size > 0:
        start_ok = int(frame_index_arr[0]) == 0
        cont_ok, cont_msg = _validate_contiguous(frame_index_arr)
        lines.append(f"  [{'PASS' if start_ok else 'FAIL'}] frame_index 从0开始: {int(frame_index_arr[0])}")
        lines.append(f"  [{'PASS' if cont_ok else 'FAIL'}] frame_index 连续性: {cont_msg}")

    if index_arr.size > 0:
        cont_ok, cont_msg = _validate_contiguous(index_arr)
        lines.append(f"  [{'PASS' if cont_ok else 'FAIL'}] index 连续性: {cont_msg}")

    if episode_index_arr.size > 0:
        ep_unique = sorted(set(episode_index_arr.tolist()))
        lines.append(f"  [{'PASS' if len(ep_unique) == 1 else 'WARN'}] episode_index 唯一值: {ep_unique}")

    if task_index_arr.size > 0:
        tk_unique = sorted(set(task_index_arr.tolist()))
        lines.append(f"  [{'PASS' if len(tk_unique) == 1 else 'WARN'}] task_index 唯一值: {tk_unique}")

    lines.append("")
    return lines


def validate_dataset_layout(dataset_path: Path, files: List[Path]) -> List[str]:
    """Validate LeRobot v2.1 directory organization and meta files."""
    lines: List[str] = []
    if not dataset_path.is_dir():
        return lines

    lines.append("=" * 65)
    lines.append("  数据集组织结构检查 (LeRobot v2.1)")
    lines.append("=" * 65)

    required_dirs = ["data", "meta", "videos"]
    for d in required_dirs:
        p = dataset_path / d
        lines.append(f"  [{'PASS' if p.exists() and p.is_dir() else 'FAIL'}] 目录: {d}")

    required_meta = ["info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"]
    for f in required_meta:
        p = dataset_path / "meta" / f
        lines.append(f"  [{'PASS' if p.exists() and p.is_file() else 'FAIL'}] meta/{f}")

    # info.json consistency checks
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            codebase = info.get("codebase_version")
            lines.append(f"  [{'PASS' if codebase == 'v2.1' else 'WARN'}] info.codebase_version={codebase}")

            total_episodes = info.get("total_episodes")
            if isinstance(total_episodes, int):
                lines.append(
                    f"  [{'PASS' if total_episodes == len(files) else 'WARN'}] info.total_episodes={total_episodes}, parquet_files={len(files)}"
                )

            feats = info.get("features", {})
            need = ["image", "wrist_image", "state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index"]
            miss = [k for k in need if k not in feats]
            lines.append(f"  [{'PASS' if not miss else 'FAIL'}] info.features 完整性: missing={miss if miss else '[]'}")
        except Exception as e:
            lines.append(f"  [FAIL] 解析 meta/info.json 失败: {e}")

    lines.append("")
    return lines


def scan_all_parquets_dataset_level(files: List[Path]) -> List[str]:
    """Lightweight full-dataset scan for schema/index/episode consistency."""
    lines: List[str] = []
    if not files:
        return lines

    lines.append("=" * 65)
    lines.append("  全量文件一致性扫描")
    lines.append("=" * 65)

    schema_ref = None
    schema_mismatch = 0
    hf_mismatch = 0
    non_contiguous_index = 0
    bad_episode_field = 0
    bad_frame_start = 0
    file_parse_errors = 0

    prev_last_index = None
    expected_episode = None

    for i, fp in enumerate(files):
        try:
            pf = pq.ParquetFile(fp)
            schema = pf.schema_arrow
            hf = _hf_features(schema)

            if schema_ref is None:
                schema_ref = schema
            elif str(schema) != str(schema_ref):
                schema_mismatch += 1

            # key hf metadata consistency
            for k, expected in (("image", "Image"), ("wrist_image", "Image"), ("state", "Sequence"), ("actions", "Sequence")):
                if hf.get(k, {}).get("_type") != expected:
                    hf_mismatch += 1
                    break

            cols = [c for c in ("index", "frame_index", "episode_index") if c in schema.names]
            t = pf.read(columns=cols)
            if "index" in cols:
                idx = np.asarray(t["index"].to_numpy(), dtype=np.int64)
                if idx.size > 1:
                    ok, _ = _validate_contiguous(idx)
                    if not ok:
                        non_contiguous_index += 1
                if idx.size > 0 and prev_last_index is not None and int(idx[0]) != prev_last_index + 1:
                    non_contiguous_index += 1
                if idx.size > 0:
                    prev_last_index = int(idx[-1])

            if "frame_index" in cols:
                fr = np.asarray(t["frame_index"].to_numpy(), dtype=np.int64)
                if fr.size > 0 and int(fr[0]) != 0:
                    bad_frame_start += 1

            if "episode_index" in cols:
                ep = np.asarray(t["episode_index"].to_numpy(), dtype=np.int64)
                if ep.size > 0:
                    uniq = sorted(set(ep.tolist()))
                    if len(uniq) != 1:
                        bad_episode_field += 1
                    ep_in_file = int(uniq[0]) if uniq else None
                    if expected_episode is None:
                        expected_episode = ep_in_file
                    elif ep_in_file is not None and ep_in_file != expected_episode:
                        bad_episode_field += 1
                    if expected_episode is not None:
                        expected_episode += 1
            else:
                # fallback to filename ordering check
                ep_name = _parse_episode_id_from_name(fp.name)
                if expected_episode is None:
                    expected_episode = ep_name
                elif ep_name is not None and expected_episode is not None and ep_name != expected_episode:
                    bad_episode_field += 1
                    expected_episode = ep_name
                if expected_episode is not None:
                    expected_episode += 1

        except Exception:
            file_parse_errors += 1

        if (i + 1) % 200 == 0 or i == len(files) - 1:
            lines.append(f"  [INFO] 扫描进度: {i+1}/{len(files)}")

    lines.append(f"  [{'PASS' if schema_mismatch == 0 else 'FAIL'}] schema 全量一致性: mismatched_files={schema_mismatch}")
    lines.append(f"  [{'PASS' if hf_mismatch == 0 else 'FAIL'}] HF 元数据关键类型一致性: mismatched_files={hf_mismatch}")
    lines.append(f"  [{'PASS' if non_contiguous_index == 0 else 'FAIL'}] index 全量连续性: violations={non_contiguous_index}")
    lines.append(f"  [{'PASS' if bad_frame_start == 0 else 'FAIL'}] frame_index 起点(应为0): violations={bad_frame_start}")
    lines.append(f"  [{'PASS' if bad_episode_field == 0 else 'FAIL'}] episode_index 组织一致性: violations={bad_episode_field}")
    lines.append(f"  [{'PASS' if file_parse_errors == 0 else 'FAIL'}] parquet 可读性: parse_errors={file_parse_errors}")
    lines.append("")
    return lines


# ── 单文件检查 ─────────────────────────────────────────────────────────────────

def inspect_one_file(parquet_path: Path, max_timesteps: int = 3) -> List[str]:
    lines: List[str] = []

    lines.append("=" * 65)
    lines.append(f"  FILE : {parquet_path.name}")
    lines.append("=" * 65)

    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    T = pf.metadata.num_rows if pf.metadata is not None else 0
    hf     = _hf_features(schema)

    non_image_cols = [
        c for c in schema.names if c not in ("image", "wrist_image")
    ]
    numeric_table = pf.read(columns=non_image_cols)

    arrays: Dict[str, np.ndarray] = {}
    for c in non_image_cols:
        arr = numeric_table[c]
        if c in ("state", "actions"):
            arrays[c] = _arrow_to_np(arr)
        else:
            arrays[c] = np.asarray(arr.to_numpy(), dtype=np.int64 if "index" in c else np.float64)

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

    lines += validate_lerobot21(
        schema=schema,
        hf=hf,
        state_arr=arrays.get("state", np.array([])),
        action_arr=arrays.get("actions", np.array([])),
        index_arr=arrays.get("index", np.array([], dtype=np.int64)),
        frame_index_arr=arrays.get("frame_index", np.array([], dtype=np.int64)),
        episode_index_arr=arrays.get("episode_index", np.array([], dtype=np.int64)),
        task_index_arr=arrays.get("task_index", np.array([], dtype=np.int64)),
        pf=pf,
    )

    # ── 列结构 ───────────────────────────────────────────────────────────────
    lines.append("── 列结构（shape / size）──────────────────────────────────────")
    for field in schema:
        name = field.name
        if name in ("image", "wrist_image"):
            desc = _decode_first_image_info(pf, name)
            size_kb = _estimate_column_size_kb(pf, name)
            mode = _image_storage_mode(pf, name)
            mode_tag = " (external path)" if mode == "path" else ""
            if size_kb > 0:
                lines.append(f"  {name:<20} {desc}  total~={size_kb:.1f} KB{mode_tag}")
            else:
                lines.append(f"  {name:<20} {desc}  total~=N/A{mode_tag}")
            continue

        arr = arrays.get(name)
        if arr is None:
            lines.append(f"  {name:<20} (未读取)")
            continue

        if arr.ndim == 2:
            size_mb = arr.nbytes / 1024 / 1024
            lines.append(
                f"  {name:<20} shape=({arr.shape[0]},{arr.shape[1]})  "
                f"dtype={arr.dtype}  ({size_mb:.3f} MB)"
            )
        else:
            size_kb = arr.nbytes / 1024
            lines.append(
                f"  {name:<20} scalar  arrow_dtype={field.type}  ({size_kb:.1f} KB)"
            )
    lines.append("")

    # ── Index 字段详情 ────────────────────────────────────────────────────────
    lines.append("── Index 字段详情 ──────────────────────────────────────────────")
    for idx_col in ("index", "frame_index", "episode_index", "task_index"):
        if idx_col not in arrays:
            continue
        vals = arrays[idx_col].astype(np.int64).tolist()
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
        if field.name in ("image", "wrist_image"):
            continue

        arr = arrays.get(field.name)
        if arr is None:
            lines.append("")
            lines.append(f"[{field.name}]")
            lines.append("  (未读取，跳过统计)")
            continue

        lines.append("")
        lines.append(f"[{field.name}]")

        if arr.ndim == 2:
            full_arr = arr.astype(np.float64)
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
            one_d = arr.astype(np.float64)
            vmin, vmax, mean, std = array_stats(one_d)
            lines.append(
                f"  全量: min={vmin:.6f}  max={vmax:.6f}  "
                f"mean={mean:.6f}  std={std:.6f}"
            )
            n_print = T if max_timesteps <= 0 else min(max_timesteps, T)
            lines.append(f"  first {n_print}: {one_d[:n_print].tolist()}")

    lines.append("")
    return lines


def inspect_one_file_concise(parquet_path: Path) -> List[str]:
    """Concise report focused on key checks requested by users."""
    lines: List[str] = []
    lines.append("=" * 65)
    lines.append(f"  FILE : {parquet_path.name}")
    lines.append("=" * 65)

    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    hf = _hf_features(schema)
    T = pf.metadata.num_rows if pf.metadata is not None else 0

    needed = [c for c in ["state", "actions", "index", "frame_index", "episode_index", "task_index"] if c in schema.names]
    tbl = pf.read(columns=needed)

    state = _arrow_to_np(tbl["state"]) if "state" in tbl.column_names else np.array([])
    actions = _arrow_to_np(tbl["actions"]) if "actions" in tbl.column_names else np.array([])
    index = np.asarray(tbl["index"].to_numpy(), dtype=np.int64) if "index" in tbl.column_names else np.array([], dtype=np.int64)
    frame_index = np.asarray(tbl["frame_index"].to_numpy(), dtype=np.int64) if "frame_index" in tbl.column_names else np.array([], dtype=np.int64)
    episode_index = np.asarray(tbl["episode_index"].to_numpy(), dtype=np.int64) if "episode_index" in tbl.column_names else np.array([], dtype=np.int64)
    task_index = np.asarray(tbl["task_index"].to_numpy(), dtype=np.int64) if "task_index" in tbl.column_names else np.array([], dtype=np.int64)

    lines.append(f"  timesteps={T}")
    lines.append("")

    lines.append("── 关键点1: HF 类型 (Sequence/List) ─────────────────────────")
    for key in ("state", "actions"):
        status, t = _type_status(hf, key, expect_sequence=True)
        lines.append(f"  [{status}] {key}: _type={t} (期望 Sequence)")
    lines.append("")

    lines.append("── 关键点2: index 连续性 ───────────────────────────────────")
    if frame_index.size > 0:
        start_ok = int(frame_index[0]) == 0
        cont_ok, cont_msg = _validate_contiguous(frame_index)
        lines.append(f"  [{'PASS' if start_ok else 'FAIL'}] frame_index 从0开始: {int(frame_index[0])}")
        lines.append(f"  [{'PASS' if cont_ok else 'FAIL'}] frame_index 连续: {cont_msg}")
    if index.size > 0:
        cont_ok, cont_msg = _validate_contiguous(index)
        lines.append(f"  [{'PASS' if cont_ok else 'FAIL'}] index 文件内连续: {cont_msg}")
        lines.append(f"  [INFO] index 范围: [{int(index[0])} .. {int(index[-1])}]  count={len(index)}")
    if episode_index.size > 0:
        lines.append(f"  [INFO] episode_index 唯一值: {sorted(set(episode_index.tolist()))}")
    if task_index.size > 0:
        lines.append(f"  [INFO] task_index 唯一值: {sorted(set(task_index.tolist()))}")
    lines.append("")

    lines.append("── 关键点3: state[6] / actions[6] (夹爪) ─────────────────────")
    if state.ndim == 2 and state.shape[1] >= 7:
        s6 = state[:, 6].astype(np.float64)
        lines.append(f"  [INFO] state[6] 真实范围: min={float(np.min(s6)):.6f}, max={float(np.max(s6)):.6f}")
        s_ok = np.isfinite(s6).all() and float(np.min(s6)) >= -1e-6 and float(np.max(s6)) <= 0.12
        lines.append(f"  [{'PASS' if s_ok else 'WARN'}] state[6] 含义: 夹爪宽度(米, 连续值)")
    else:
        lines.append("  [FAIL] state shape 无法用于检查第7维夹爪")

    if actions.ndim == 2 and actions.shape[1] >= 7:
        a6 = actions[:, 6].astype(np.float64)
        uniq = sorted(set(np.round(a6, 6).tolist()))
        a_range_ok = np.isfinite(a6).all() and float(np.min(a6)) >= -1e-6 and float(np.max(a6)) <= 1.0 + 1e-6
        a_binary_ok = _is_binary01(a6)
        lines.append(f"  [INFO] actions[6] 真实范围: min={float(np.min(a6)):.6f}, max={float(np.max(a6)):.6f}")
        lines.append(f"  [{'PASS' if a_range_ok else 'FAIL'}] actions[6] 范围校验: 期望[0,1]")
        lines.append(f"  [{'PASS' if a_binary_ok else 'FAIL'}] actions[6] 二值校验: unique={uniq[:8]}")
    else:
        lines.append("  [FAIL] actions shape 无法用于检查第7维夹爪")
    lines.append("")

    lines.append("── 关键点4: image / wrist_image (HWC/C/dtype) ───────────────")
    for col in ("image", "wrist_image"):
        ck = check_image_column_geometry(pf, col_name=col, max_samples=5)
        if not ck["exists"]:
            lines.append(f"  [FAIL] {col}: 列不存在")
            continue

        sampled = ck["sampled"]
        decoded = ck["decoded"]
        dec_ok = sampled > 0 and decoded == sampled
        lines.append(f"  [{'PASS' if dec_ok else 'FAIL'}] {col} 解码: {decoded}/{sampled}")

        shapes = sorted(ck["shapes"])
        dtypes = sorted(ck["dtypes"])
        modes = sorted(ck["modes"])
        if shapes:
            c_ok = all(s[2] == 3 for s in shapes)
            lines.append(f"  [{'PASS' if c_ok else 'FAIL'}] {col} shape(H,W,C): {shapes} (期望 C=3)")
        else:
            lines.append(f"  [FAIL] {col} shape(H,W,C): 无法解析")

        dtype_ok = dtypes == ["uint8"]
        lines.append(f"  [{'PASS' if dtype_ok else 'WARN'}] {col} dtype: {dtypes if dtypes else 'N/A'} (期望 uint8)")

        mode_ok = modes == ["RGB"]
        lines.append(f"  [{'PASS' if mode_ok else 'WARN'}] {col} mode: {modes if modes else 'N/A'} (期望 RGB)")
    lines.append("")

    return lines


# ── 跨文件 index 对比 ─────────────────────────────────────────────────────────

def compare_index(path_a: Path, path_b: Path) -> List[str]:
    lines: List[str] = []
    lines.append("=" * 65)
    lines.append("  跨文件 Index 行为对比")
    lines.append("=" * 65)

    idx_cols = ["index", "frame_index", "episode_index", "task_index"]
    ta_tbl = pq.read_table(path_a, columns=idx_cols)
    tb_tbl = pq.read_table(path_b, columns=idx_cols)

    for col in ("index", "frame_index", "episode_index", "task_index"):
        has_a = col in ta_tbl.column_names
        has_b = col in tb_tbl.column_names
        if not (has_a and has_b):
            continue

        va = np.asarray(ta_tbl[col].to_numpy(), dtype=np.int64).tolist()
        vb = np.asarray(tb_tbl[col].to_numpy(), dtype=np.int64).tolist()

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


def compare_index_concise(path_a: Path, path_b: Path) -> List[str]:
    """Concise cross-episode global index continuity report."""
    lines: List[str] = []
    lines.append("=" * 65)
    lines.append("  跨文件关键检查")
    lines.append("=" * 65)

    ta = pq.read_table(path_a, columns=["index", "frame_index", "episode_index"])
    tb = pq.read_table(path_b, columns=["index", "frame_index", "episode_index"])

    ia = np.asarray(ta["index"].to_numpy(), dtype=np.int64)
    ib = np.asarray(tb["index"].to_numpy(), dtype=np.int64)
    fa = np.asarray(ta["frame_index"].to_numpy(), dtype=np.int64)
    fb = np.asarray(tb["frame_index"].to_numpy(), dtype=np.int64)

    expected = int(ia[-1]) + 1 if ia.size > 0 else None
    global_ok = ib.size > 0 and expected is not None and int(ib[0]) == expected
    lines.append(
        f"  [{'PASS' if global_ok else 'FAIL'}] global index 跨episode连续: "
        f"A[{int(ia[0])}..{int(ia[-1])}] -> B[{int(ib[0])}..{int(ib[-1])}]"
    )

    fa_ok = fa.size > 0 and int(fa[0]) == 0
    fb_ok = fb.size > 0 and int(fb[0]) == 0
    lines.append(f"  [{'PASS' if fa_ok else 'FAIL'}] A frame_index 从0开始: {int(fa[0]) if fa.size else 'N/A'}")
    lines.append(f"  [{'PASS' if fb_ok else 'FAIL'}] B frame_index 从0开始: {int(fb[0]) if fb.size else 'N/A'}")
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
    parser.add_argument(
        "--full-report", action="store_true",
        help="输出完整详细报告（默认输出精简关键报告）"
    )
    parser.add_argument(
        "--scan-all", action="store_true",
        help="对目录下全部 parquet 执行一致性扫描（schema/index/episode）"
    )
    parser.add_argument(
        "--export-first-images", action="store_true",
        help="导出第一个 episode 的首帧 image / wrist_image 到 PNG"
    )
    parser.add_argument(
        "--export-dir", type=str, default=None,
        help="首帧图像导出目录（默认: 输入目录下 inspect_exports）"
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

    all_lines += validate_dataset_layout(path, files)

    # 单文件检查
    if args.full_report:
        all_lines += inspect_one_file(file_a, max_timesteps=args.max_timesteps)
        if file_b:
            all_lines += inspect_one_file(file_b, max_timesteps=args.max_timesteps)
            all_lines += compare_index(file_a, file_b)
    else:
        all_lines += inspect_one_file_concise(file_a)
        if file_b:
            all_lines += inspect_one_file_concise(file_b)
            all_lines += compare_index_concise(file_a, file_b)

    if args.scan_all and path.is_dir():
        all_lines += scan_all_parquets_dataset_level(files)

    if args.export_first_images:
        base_dir = path if path.is_dir() else path.parent
        export_dir = Path(args.export_dir) if args.export_dir else (base_dir / "inspect_exports")
        all_lines += export_first_episode_images(file_a, export_dir)

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
