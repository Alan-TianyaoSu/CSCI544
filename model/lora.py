# model/lora.py

from __future__ import annotations

from dataclasses import asdict
from inspect import signature
from pathlib import Path
from typing import Optional

import torch
from peft import AdaLoraConfig, PeftModel, get_peft_model

try:
    from peft import TaskType  # peft>=0.11
except ImportError:  # peft<=0.10
    from peft.utils import TaskType

from model.config import AdaLoraRuntimeConfig


def ensure_adapter_directory(adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)


def _filter_kwargs_for_config(config_cls, kwargs: dict) -> dict:
    valid_params = {
        name
        for name in signature(config_cls.__init__).parameters.keys()
        if name != "self"
    }
    return {key: value for key, value in kwargs.items() if key in valid_params}


def build_adalora_model(
    base_model: torch.nn.Module,
    *,
    adapter_dir: Path,
    adapter_name: str,
    adalora_config: AdaLoraRuntimeConfig,
    total_step: Optional[int] = None,
) -> PeftModel:
    """
    Wrap a base model with an Adaptive LoRA adapter using externally supplied
    configuration values.
    """
    ensure_adapter_directory(adapter_dir)

    config_dict = asdict(adalora_config)
    required_numeric_fields = [
        "init_rank",
        "beta1",
        "beta2",
        "delta_t",
        "dropout",
    ]
    for field_name in required_numeric_fields:
        if config_dict.get(field_name) is None:
            raise ValueError(
                f"AdaLoRA configuration is missing required field: '{field_name}'"
            )

    init_rank = config_dict["init_rank"]
    target_rank = config_dict.get("target_rank") or init_rank

    base_kwargs = {
        "task_type": TaskType.CAUSAL_LM,
        "target_modules": config_dict.get("target_modules"),
        "init_r": init_rank,
        "target_r": target_rank,
        "lora_alpha": target_rank,
        "lora_dropout": config_dict.get("dropout"),
        "beta1": config_dict.get("beta1"),
        "beta2": config_dict.get("beta2"),
        "tinit": config_dict.get("tinit") or 0,
        "tfinal": config_dict.get("tfinal") or 0,
        "deltaT": config_dict.get("delta_t"),
        "scaling_init": config_dict.get("scaling_init"),
        "use_rslora": config_dict.get("use_rslora"),
        "inference_mode": config_dict.get("inference_mode") or False,
    }
    if total_step is not None:
        base_kwargs["total_step"] = total_step

    filtered_kwargs = {key: value for key, value in base_kwargs.items() if value is not None}
    adalora_kwargs = _filter_kwargs_for_config(AdaLoraConfig, filtered_kwargs)
    adalora_instance = AdaLoraConfig(**adalora_kwargs)

    peft_model = get_peft_model(base_model, adalora_instance, adapter_name=adapter_name)
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()
    return peft_model


def load_peft_adapter_if_available(
    model: torch.nn.Module,
    adapter_path: Path,
    adapter_name: str,
) -> torch.nn.Module:
    candidate_paths = []
    if adapter_path.exists():
        candidate_paths.append(adapter_path)
    nested_path = adapter_path / adapter_name
    if nested_path.exists():
        candidate_paths.append(nested_path)

    for path in candidate_paths:
        if (path / "adapter_config.json").exists():
            print(f"[LoRA] Loading existing adapter from: {path}")
            return PeftModel.from_pretrained(
                model,
                path,
                adapter_name=adapter_name,
                is_trainable=True,
            )

    print(f"[LoRA] No existing adapter found at {adapter_path}. Starting fresh.")
    return model