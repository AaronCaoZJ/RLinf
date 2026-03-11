#!/usr/bin/env python3
"""
Upload a local LeRobot v2.1 dataset to HuggingFace Hub.
Token is read from the HF_TOKEN environment variable.

Usage:
    python hf_upload.py --repo your-username/repo-name
    python hf_upload.py --dataset /other/path --repo your-username/repo-name --private
"""

import os
import argparse
from huggingface_hub import HfApi


def upload_dataset(dataset_path: str, repo_id: str, private: bool = False):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable not set.")

    api = HfApi(token=token)

    print(f"Creating repo: {repo_id} (private={private})")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    print(f"Uploading {dataset_path} → {repo_id} ...")
    api.upload_large_folder(
        folder_path=dataset_path,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/workspace1/zhijun/mg_dataset/blockpap_cleaned_mimicgen",
        help="Local dataset directory to upload",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="aaroncaozj/BlockPAP-v1_MimicGen",
        help="HuggingFace repo id",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"[ERROR] Dataset directory not found: {args.dataset}")
        return

    upload_dataset(args.dataset, args.repo, private=args.private)


if __name__ == "__main__":
    main()
