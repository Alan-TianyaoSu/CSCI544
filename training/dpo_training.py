# training/dpo_training.py
"""
Manual configuration entry-point for DPO fine-tuning.

Examples:
- python ./training/dpo_training.py                                  # print current config
- python ./training/dpo_training.py --run                            # run DPO with the default config
- python ./training/dpo_training.py --model gemma-3-1b-it --run      # run DPO with the selected model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from transformers.trainer_utils import get_last_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in map(str, sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))

from model import (
    AdaLoraRuntimeConfig,
    ModelConfigRegistry,
    ModelRuntimeConfig,
    DPOTrainingConfig
)
from rlhf import run_dpo_training


def _build_manual_configuration(model_name: str) -> Tuple[ModelRuntimeConfig, DPOTrainingConfig]:
    hf_model_id = model_name if "/" in model_name else f"google/{model_name}"

    model_config = ModelRuntimeConfig(
        name=model_name,
        hf_model_id=hf_model_id,
        dtype="bfloat16",
        tokenizer_kwargs={"use_fast": True},
        model_kwargs={"trust_remote_code": False},
        adalora=AdaLoraRuntimeConfig(
            target_modules=None,
            init_rank=64,
            target_rank=64,
            beta1=0.85,
            beta2=0.85,
            tinit=0,
            tfinal=0,
            delta_t=1,
            dropout=0.05,
            scaling_init=1e-5,
            use_rslora=True,
            inference_mode=False,
        ),
    )
    ModelConfigRegistry.register(model_config)

    output_dir = Path("checkpoints") / "dpo" / model_name
    if output_dir.exists():
        last_checkpoint = get_last_checkpoint(str(output_dir))
    else:
        last_checkpoint = None

    training_config = DPOTrainingConfig(
        train_file=Path("dataset/RLHF/train.npz"),
        eval_file=Path("dataset/RLHF/validation.npz"),
        output_dir=output_dir,
        adapter_name="adalora-dpo",
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        max_seq_length=1024,
        max_prompt_length=512,
        beta=0.1,
        loss_type="sigmoid",
        label_smoothing=0.0,
        save_total_limit=3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        report_to=["none"],
        world_size=1,
        resume_checkpoint=Path(last_checkpoint) if last_checkpoint else None,
    )

    return model_config, training_config


def run_manual_training(model_config: ModelRuntimeConfig, training_config: DPOTrainingConfig) -> None:
    metrics = run_dpo_training(model_config, training_config)
    if metrics:
        print("[Manual DPO Evaluation Metrics]")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview or execute the manual DPO configuration.")
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-1b-it",
        help="Model identifier (default: gemma-3-1b-it). If no slash is present, google/<model> is assumed.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute DPO using the preset configuration.",
    )

    args = parser.parse_args()

    model_cfg, training_cfg = _build_manual_configuration(args.model)
    print('==' * 60)
    print(f"[Manual Config] Model: {model_cfg}\n")
    print(f"[Manual Config] Training: {training_cfg}")
    print('==' * 60 + '\n')

    if args.run:
        run_manual_training(model_cfg, training_cfg)