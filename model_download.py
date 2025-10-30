# model_download.py

#!/usr/bin/env python
"""
Download all registered base models into the `base_models/<model-name>` folders.
The downloader will read model list in model/config.py

Example usage:
    python model_download.py
    python model_download.py --models gemma-3-1b-it Qwen2.5-1.5B

Before you download, please go to Hugging face and search for the certain model, eg. gemma-3-1b-it
Make sure you have been granted access to that model

- Go to https://huggingface.co/settings/tokens, add a new token
- Type "huggingface-cli login" in your terminal and paste your token
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from model.config import ModelConfigRegistry, ModelRuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download registered base models.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model names to download. Defaults to all registered models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target directory already exists.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token (falls back to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partial downloads when possible (default Hugging Face behavior).",
    )
    parser.add_argument(
        "--local-dir-use-symlinks",
        action="store_true",
        help="Store files as symlinks (Hugging Face Hub optimization).",
    )
    return parser.parse_args()


def ensure_directory(path: Path, force: bool) -> None:
    if path.exists():
        if force:
            return
        print(f"[skip] {path} already exists. Use --force to overwrite.")
        raise FileExistsError(f"Directory {path} already exists.")
    path.mkdir(parents=True, exist_ok=True)


def download_model(config: ModelRuntimeConfig, args: argparse.Namespace) -> None:
    target_dir = config.local_model_dir
    try:
        ensure_directory(target_dir, force=args.force)
    except FileExistsError:
        return

    print(f"[download] {config.name} -> {target_dir}")
    snapshot_download(
        repo_id=config.hf_model_id,
        local_dir=target_dir,
        token=args.token,
        force_download=args.force,   
    )
    print(f"[done] {config.name}")


def main() -> None:
    args = parse_args()
    available = ModelConfigRegistry.available_models()

    if args.models:
        missing = [name for name in args.models if name not in available]
        if missing:
            raise KeyError(
                f"Unknown model(s): {missing}. "
                f"Registered models: {list(available.keys())}"
            )
        selected = {name: available[name] for name in args.models}
    else:
        selected = available

    if not selected:
        print("No models registered; nothing to download.")
        return

    for config in selected.values():
        download_model(config, args)


if __name__ == "__main__":
    main()