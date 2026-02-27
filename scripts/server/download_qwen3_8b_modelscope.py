from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--cache-dir", default="/data/zhangyuyun/models/models")
    parser.add_argument(
        "--target-dir",
        default="/data/zhangyuyun/models/models/Qwen/Qwen3-8B",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    target_dir = Path(args.target_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "ModelScope is not installed. Please install with: uv pip install modelscope"
        ) from exc

    model_dir = snapshot_download(args.model_id, cache_dir=str(cache_dir))
    model_path = Path(model_dir).resolve()

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() or target_dir.is_symlink():
        if target_dir.resolve() != model_path:
            if target_dir.is_symlink():
                target_dir.unlink()
            else:
                # Keep existing files if they already contain model weights
                pass

    if not target_dir.exists() and not target_dir.is_symlink():
        os.symlink(model_path, target_dir)

    print(f"Model downloaded to: {model_path}")
    print(f"Stable path: {target_dir}")


if __name__ == "__main__":
    main()
