"""Force gripper action to absolute binary 0/1 in LeRobot parquet datasets.

Supported source formats for actions[:, 6]:
1) Already binary {0,1}
2) Continuous absolute in [0,1]
3) Delta gripper width in meters (converted via state[:,6] width)

Examples:
  python real_franka/real2sim_env/convert_gripper_to_abs.py \
      --dataset-dir /workspace1/zhijun/mg_dataset/blockpap_cleaned_mimicgen

  python real_franka/real2sim_env/convert_gripper_to_abs.py \
      --dataset-dir /workspace1/zhijun/mg_dataset/BlockPAP-v1_Mix
"""

import argparse
import json
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def is_binary01(arr: np.ndarray, tol: float = 1e-6) -> bool:
    if arr.size == 0:
        return True
    return np.all((np.abs(arr - 0.0) <= tol) | (np.abs(arr - 1.0) <= tol))


def classify_gripper_action(g: np.ndarray, delta_threshold: float) -> str:
    g_min = float(np.min(g))
    g_max = float(np.max(g))
    if is_binary01(g):
        return "binary"
    if g_min >= -1e-6 and g_max <= 1.0 + 1e-6:
        return "abs01_cont"
    if g_min >= -delta_threshold and g_max <= delta_threshold:
        return "delta_m"
    return "unknown"


def convert_delta_to_abs01_binary(
    states: np.ndarray,
    delta_g: np.ndarray,
    gripper_max_m: float,
    binary_threshold: float,
) -> np.ndarray:
    t_len = len(delta_g)
    out = np.empty(t_len, dtype=np.float32)
    for t in range(max(0, t_len - 1)):
        next_width = float(states[t, 6]) + float(delta_g[t])
        next_abs01 = float(np.clip(next_width / gripper_max_m, 0.0, 1.0))
        out[t] = 1.0 if next_abs01 >= binary_threshold else 0.0

    if t_len == 1:
        curr_abs01 = float(np.clip(float(states[0, 6]) / gripper_max_m, 0.0, 1.0))
        out[0] = 1.0 if curr_abs01 >= binary_threshold else 0.0
    elif t_len > 1:
        out[-1] = out[-2]
    return out


def compute_norm_stats(states: np.ndarray, actions: np.ndarray) -> dict:
    def _stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    pad = 32

    def pad_stats(s):
        out = {}
        for k, v in s.items():
            v7 = v[:7]
            out[k] = v7 + [0.0] * (pad - len(v7))
        return out

    return {
        "norm_stats": {
            "state": pad_stats(_stats(states)),
            "actions": pad_stats(_stats(actions)),
        }
    }


def _default_hf_metadata() -> bytes:
    """Minimal HuggingFace metadata for LeRobot v2.1 style columns."""
    info = {
        "features": {
            "image": {"_type": "Image"},
            "wrist_image": {"_type": "Image"},
            "state": {
                "_type": "Sequence",
                "feature": {"_type": "Value", "dtype": "float32"},
                "length": 7,
            },
            "actions": {
                "_type": "Sequence",
                "feature": {"_type": "Value", "dtype": "float32"},
                "length": 7,
            },
            "timestamp": {"_type": "Value", "dtype": "float32"},
            "frame_index": {"_type": "Value", "dtype": "int64"},
            "episode_index": {"_type": "Value", "dtype": "int64"},
            "index": {"_type": "Value", "dtype": "int64"},
            "task_index": {"_type": "Value", "dtype": "int64"},
        }
    }
    return json.dumps({"info": info}, ensure_ascii=False).encode("utf-8")


def _normalize_table_schema(table: pa.Table) -> tuple[pa.Table, bool]:
    """Normalize critical columns to LeRobot/OpenPI-compatible Arrow types.

    Returns:
        (normalized_table, changed)
    """
    changed = False

    # Image columns should be struct<bytes: binary, path: string>.
    # Some broken rewrites produce path:null, which breaks HF decoding to Image.
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    for col in ("image", "wrist_image"):
        idx = table.schema.get_field_index(col)
        if idx < 0:
            continue
        arr = table[col].combine_chunks()
        if not arr.type.equals(image_type):
            arr = arr.cast(image_type)
            table = table.set_column(idx, col, arr)
            changed = True

    # Vector columns should be fixed_size_list<float32>[7] for stable decoding.
    vec_type = pa.list_(pa.float32(), 7)
    for col in ("state", "actions"):
        idx = table.schema.get_field_index(col)
        if idx < 0:
            continue
        arr = table[col].combine_chunks()
        if not arr.type.equals(vec_type):
            arr = arr.cast(vec_type)
            table = table.set_column(idx, col, arr)
            changed = True

    return table, changed


def _write_actions_preserve_metadata(fpath: pathlib.Path, actions: np.ndarray) -> bool:
    """Rewrite actions column while preserving (or repairing) parquet metadata.

    Returns True if huggingface metadata was repaired, else False.
    """
    table = pq.read_table(fpath)
    field_idx = table.schema.get_field_index("actions")
    if field_idx < 0:
        raise KeyError(f"No 'actions' column in {fpath}")

    # Always write actions back as fixed_size_list<float32>[7] to match OpenPI loaders.
    actions_pa = pa.array(actions.tolist(), type=pa.list_(pa.float32(), 7))
    table = table.set_column(field_idx, "actions", actions_pa)

    table, _ = _normalize_table_schema(table)

    meta = dict(table.schema.metadata or {})
    repaired = False
    if b"huggingface" not in meta:
        meta[b"huggingface"] = _default_hf_metadata()
        repaired = True

    table = table.replace_schema_metadata(meta)
    pq.write_table(table, fpath)
    return repaired


def main():
    parser = argparse.ArgumentParser(description="Force gripper action to binary absolute 0/1.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="LeRobot dataset root dir")
    parser.add_argument("--binary-threshold", type=float, default=0.5, help="Threshold in [0,1]")
    parser.add_argument("--gripper-max-m", type=float, default=0.08, help="Gripper full width in meters")
    parser.add_argument("--delta-threshold", type=float, default=0.12, help="Max abs value to classify as delta meters")
    parser.add_argument("--write-norm-stats", action="store_true", help="Recompute and write norm_stats.json")
    parser.add_argument("--norm-stats-out", type=str, default=None, help="Output path for norm_stats.json")
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    files = sorted(dataset_dir.glob("data/chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under: {dataset_dir}/data/chunk-*/*.parquet")

    print(f"Dataset: {dataset_dir}")
    print(f"Found {len(files)} episodes")

    counts = {
        "binary_kept": 0,
        "abs01_cont_binarized": 0,
        "delta_converted": 0,
        "unknown_fallback": 0,
        "hf_metadata_repaired": 0,
    }

    all_states = []
    all_actions = []

    for fpath in tqdm(files, desc="Converting"):
        table = pq.read_table(fpath, columns=["state", "actions"])
        states = np.array(table["state"].to_pylist(), dtype=np.float32)
        actions = np.array(table["actions"].to_pylist(), dtype=np.float32)

        if states.ndim != 2 or states.shape[1] < 7 or actions.ndim != 2 or actions.shape[1] < 7:
            raise ValueError(f"Unexpected shape in {fpath}: state={states.shape}, actions={actions.shape}")

        g = actions[:, 6].astype(np.float32)
        mode = classify_gripper_action(g, args.delta_threshold)

        if mode == "binary":
            new_g = (g >= args.binary_threshold).astype(np.float32)
            counts["binary_kept"] += 1
        elif mode == "abs01_cont":
            new_g = (g >= args.binary_threshold).astype(np.float32)
            counts["abs01_cont_binarized"] += 1
        elif mode == "delta_m":
            new_g = convert_delta_to_abs01_binary(
                states=states,
                delta_g=g,
                gripper_max_m=args.gripper_max_m,
                binary_threshold=args.binary_threshold,
            )
            counts["delta_converted"] += 1
        else:
            # Fallback: clamp to [0,1] then binarize.
            g01 = np.clip(g, 0.0, 1.0)
            new_g = (g01 >= args.binary_threshold).astype(np.float32)
            counts["unknown_fallback"] += 1

        actions[:, 6] = new_g
        repaired = _write_actions_preserve_metadata(fpath, actions)
        if repaired:
            counts["hf_metadata_repaired"] += 1

        all_states.append(states)
        all_actions.append(actions)

    print("\nConversion summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    merged_actions = np.concatenate(all_actions, axis=0)
    unique_gripper = sorted(set(np.round(merged_actions[:, 6], 6).tolist()))
    print(f"\nAction[6] unique after conversion (first 8): {unique_gripper[:8]}")
    print(f"Action[6] min={merged_actions[:,6].min():.6f} max={merged_actions[:,6].max():.6f}")

    if args.write_norm_stats:
        merged_states = np.concatenate(all_states, axis=0)
        norm_stats = compute_norm_stats(merged_states, merged_actions)
        if args.norm_stats_out is None:
            raise ValueError("--write-norm-stats requires --norm-stats-out")
        out_path = pathlib.Path(args.norm_stats_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bak = out_path.with_suffix(".json.bak")
        if out_path.exists() and not bak.exists():
            shutil.copy(out_path, bak)
            print(f"Backed up original to {bak}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(norm_stats, f, indent=2)
        print(f"Saved norm_stats to {out_path}")


if __name__ == "__main__":
    main()
