from __future__ import annotations

import argparse
import atexit
from datetime import datetime
import sys
from pathlib import Path
from typing import IO, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in map(str, sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .sft_training import (
        _build_manual_configuration,
        run_manual_training,
    )
except ImportError:
    from sft_training import (  # type: ignore
        _build_manual_configuration,
        run_manual_training,
    )

LOG_FILE_HANDLE: Optional[IO[str]] = None
_CURRENT_LOG_PATH: Optional[Path] = None


def _sanitize_filename_component(value: str) -> str:
    sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)
    return sanitized or "unknown"


def configure_output_streams(
    script_stem: str,
    model_name: str,
    tag: Optional[str] = None,
) -> Path:
    global LOG_FILE_HANDLE, _CURRENT_LOG_PATH

    if LOG_FILE_HANDLE is not None and _CURRENT_LOG_PATH is not None:
        return _CURRENT_LOG_PATH

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    components = [script_stem]
    if tag:
        components.append(_sanitize_filename_component(tag))
    components.append(_sanitize_filename_component(model_name))
    components.append(timestamp)

    log_filename = "_".join(components) + ".log"
    log_path = logs_dir / log_filename

    log_file = open(log_path, mode="a", encoding="utf-8", buffering=1)

    LOG_FILE_HANDLE = log_file
    _CURRENT_LOG_PATH = log_path

    sys.stdout = log_file  # type: ignore[assignment]
    sys.stderr = log_file  # type: ignore[assignment]

    def _close_log_file() -> None:
        try:
            if not log_file.closed:
                log_file.flush()
        finally:
            try:
                log_file.close()
            except Exception:
                pass

    atexit.register(_close_log_file)
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Server-oriented entry point for SFT fine-tuning.")
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-1b-it",
        help="Model identifier (default: gemma-3-1b-it). If no slash is present, google/<model> is assumed.",
    )
    parser.add_argument(
        "--log-tag",
        type=str,
        default=None,
        help="Optional tag appended to the generated log filename.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only print the resolved configuration without launching training.",
    )
    args = parser.parse_args()

    log_path = configure_output_streams("sft_server", args.model, args.log_tag)
    sys.__stdout__.write(f"[sft_server] Logging redirected to {log_path}\n")
    sys.__stdout__.flush()
    print(f"[Logging] STDOUT and STDERR redirected to {log_path}")

    model_cfg, training_cfg = _build_manual_configuration(args.model)
    print("==" * 60)
    print(f"[SFT Server Config] Model: {model_cfg}\n")
    print(f"[SFT Server Config] Training: {training_cfg}")
    print("==" * 60 + "\n")

    if args.dry_run:
        print("[sft_server] Dry run requested; skipping training execution.")
        return

    print("[sft_server] Starting supervised fine-tuning run...")
    run_manual_training(model_cfg, training_cfg)
    print("[sft_server] Training run completed.")


if __name__ == "__main__":
    main()