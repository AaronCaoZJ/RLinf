"""
Fix parquet metadata: replace '_type': 'List' with '_type': 'Sequence'
to be compatible with the datasets library used by lerobot 0.1.0 / openpi.
"""

import glob
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def fix_features(features: dict) -> dict:
    """Recursively replace _type 'List' with 'Sequence'."""
    result = {}
    for k, v in features.items():
        if isinstance(v, dict):
            fixed = fix_features(v)
            if fixed.get("_type") == "List":
                fixed["_type"] = "Sequence"
            result[k] = fixed
        else:
            result[k] = v
    return result


def fix_parquet_file(path: str) -> bool:
    """Fix a single parquet file. Returns True if modified."""
    table = pq.read_table(path)
    meta = table.schema.metadata or {}

    hf_meta_raw = meta.get(b"huggingface")
    if hf_meta_raw is None:
        return False

    hf_meta = json.loads(hf_meta_raw.decode())
    features = hf_meta.get("info", {}).get("features", {})

    # Check if any List type exists
    has_list = any(
        v.get("_type") == "List"
        for v in features.values()
        if isinstance(v, dict)
    )
    if not has_list:
        return False

    # Fix features
    hf_meta["info"]["features"] = fix_features(features)
    new_meta = {**meta, b"huggingface": json.dumps(hf_meta).encode()}
    new_schema = table.schema.with_metadata(new_meta)
    new_table = table.cast(new_schema)
    pq.write_table(new_table, path)
    return True


def fix_dataset(dataset_dir: str):
    parquet_files = sorted(glob.glob(
        os.path.join(dataset_dir, "data", "**", "*.parquet"), recursive=True
    ))
    if not parquet_files:
        print(f"No parquet files found in {dataset_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files, fixing...")
    modified = 0
    for i, f in enumerate(parquet_files):
        if fix_parquet_file(f):
            modified += 1
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(parquet_files)}")

    print(f"Done. Modified {modified}/{len(parquet_files)} files.")


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace1/zhijun/mg_dataset/blockpap_cleaned_mimicgen"
    print(f"Fixing dataset at: {dataset_dir}")
    fix_dataset(dataset_dir)
