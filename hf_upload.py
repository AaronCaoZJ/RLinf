#!/usr/bin/env python3
"""
Upload a local directory to HuggingFace Hub (dataset or model).
Token is read from the HF_TOKEN environment variable.

Usage:
    # Upload dataset (default)
    python hf_upload.py --repo your-username/repo-name
    python hf_upload.py --path /other/path --repo your-username/repo-name --private

    # Upload model checkpoint
    python hf_upload.py --path /workspace1/zhijun/RLinf/logs/20260314-06:23:04/test_new_gripper_data/checkpoints/global_step_25000 \
                        --repo aaroncaozj/pi05-blockpap-step25000 --type model --private
"""

import os
import argparse
from huggingface_hub import HfApi


def upload_folder(local_path: str, repo_id: str, repo_type: str = "dataset", private: bool = False, path_in_repo: str = None):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable not set.")

    api = HfApi(token=token)

    print(f"Creating repo: {repo_id} (type={repo_type}, private={private})")
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    dest = f"{repo_id}/{path_in_repo}" if path_in_repo else repo_id
    print(f"Uploading {local_path} → {dest} ...")
    if repo_type == "dataset":
        api.upload_large_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
    else:
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
        )

    prefix = "datasets" if repo_type == "dataset" else "models" if repo_type == "model" else repo_type
    print(f"\nDone! Available at: https://huggingface.co/{prefix}/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload a local folder to HuggingFace Hub")
    parser.add_argument(
        "--path",
        type=str,
        default="/workspace1/zhijun/mg_dataset/blockpap_cleaned_mimicgen",
        help="Local directory to upload",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="aaroncaozj/BlockPAP-v1_MimicGen",
        help="HuggingFace repo id (e.g. aaroncaozj/my-model)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Repo type: dataset (default) or model",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default=None,
        help="Subdirectory inside the repo (e.g. global_step_25000). If omitted, uploads to repo root.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"[ERROR] Directory not found: {args.path}")
        return

    upload_folder(args.path, args.repo, repo_type=args.type, private=args.private, path_in_repo=args.path_in_repo)


if __name__ == "__main__":
    main()
