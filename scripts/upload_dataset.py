"""Universal uploader to ModelScope Dataset Hub (Supports any Folder or File).

Explicit, zero-hardcoding CLI tool to upload any local directory or file to a
ModelScope Dataset repository.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Ensure repository root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload any local directory or file to ModelScope Dataset Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=not bool(os.getenv("MODELSCOPE_REPO_ID")),
        default=os.getenv("MODELSCOPE_REPO_ID"),
        help="Target ModelScope Dataset Repo ID (e.g. Fang001/rgbdt-grounding-dataset).",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=not bool(os.getenv("MODELSCOPE_API_TOKEN")),
        default=os.getenv("MODELSCOPE_API_TOKEN"),
        help="ModelScope SDK Access Token (can also be set via MODELSCOPE_API_TOKEN).",
    )
    parser.add_argument(
        "--path",
        "--local-path",
        "--data-dir",
        dest="local_path",
        type=Path,
        required=True,
        help="Path to the local directory or file to upload (Required).",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default="",
        help="Target path inside the remote repository (default: root of repo).",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload dataset assets",
        help="Commit message for the upload.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    target_path = args.local_path.resolve()
    if not target_path.exists():
        print(f"Error: Local path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)

    print("Authenticating with ModelScope...")
    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        print(
            "Error: 'modelscope' package is not installed. Run 'pip install modelscope' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HubApi()
    api.login(args.token)

    if target_path.is_dir():
        print(f"Uploading directory [{target_path}] to [{args.repo_id}] (remote path: '{args.path_in_repo}')...")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(target_path),
            path_in_repo=args.path_in_repo,
            repo_type="dataset",
            commit_message=args.commit_message,
        )
    elif target_path.is_file():
        path_in_repo = args.path_in_repo or target_path.name
        print(f"Uploading single file [{target_path}] to [{args.repo_id}] (remote path: '{path_in_repo}')...")
        api.upload_file(
            repo_id=args.repo_id,
            path_or_fileobj=str(target_path),
            path_in_repo=path_in_repo,
            repo_type="dataset",
            commit_message=args.commit_message,
        )

    print(f"✅ Successfully uploaded to ModelScope: https://modelscope.cn/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
