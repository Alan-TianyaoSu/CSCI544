# model/config.py

"""
Configurable registry for supported models with manual override hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


@dataclass(frozen=True)
class AdaLoraRuntimeConfig:
    """
    Container for AdaLoRA hyper-parameters.
    """
    target_modules: Optional[Sequence[str]] = None
    init_rank: Optional[int] = None
    target_rank: Optional[int] = None
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    tinit: Optional[int] = None
    tfinal: Optional[int] = None
    delta_t: Optional[int] = None
    dropout: Optional[float] = None
    scaling_init: Optional[float] = None
    use_rslora: Optional[bool] = None
    inference_mode: Optional[bool] = None


@dataclass(frozen=True)
class SFTTrainingConfig:
    """
    Container for SFT run-time overrides.
    """
    train_file: Path
    eval_file: Optional[Path] = None
    output_dir: Path = Path("checkpoints/sft")
    adapter_name: str = "adalora-sft"
    learning_rate: Optional[float] = None
    num_train_epochs: Optional[float] = None
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    max_seq_length: int = 1024
    resume_checkpoint: Optional[Path] = None
    save_total_limit: Optional[int] = None
    logging_steps: Optional[int] = None
    eval_strategy: Optional[str] = None
    save_strategy: Optional[str] = None
    warmup_ratio: Optional[float] = None
    warmup_steps: Optional[int] = None
    weight_decay: Optional[float] = None
    bf16: Optional[bool] = None
    tf32: Optional[bool] = None
    gradient_checkpointing: Optional[bool] = None
    report_to: Optional[Union[str, Sequence[str]]] = None
    steps_per_epoch: Optional[int] = None
    max_steps: Optional[int] = None
    world_size: Optional[int] = None
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    logging_strategy: Optional[str] = None
    save_only_model: Optional[bool] = None
    load_best_model_at_end: Optional[bool] = None
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None


@dataclass
class ModelRuntimeConfig:
    """
    Container describing a model target.
    """
    name: str
    hf_model_id: str
    dtype: Optional[str] = "float32"
    base_model_root: Path = Path("base_models")
    adapter_root: Path = Path("adapters")
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    adalora: Optional[AdaLoraRuntimeConfig] = None

    @property
    def adapter_dir(self) -> Path:
        return self.adapter_root / self.name

    @property
    def local_model_dir(self) -> Path:
        return self.base_model_root / self.name

    def resolve_model_source(self) -> str:
        local_dir = self.local_model_dir
        if local_dir.exists() and any(local_dir.iterdir()):
            return str(local_dir)
        return self.hf_model_id

    def build_tokenizer_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self.tokenizer_kwargs)
        kwargs.setdefault("cache_dir", str(self.local_model_dir))
        return kwargs

    def build_model_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self.model_kwargs)
        kwargs.setdefault("cache_dir", str(self.local_model_dir))
        return kwargs


class ModelConfigRegistry:
    """
    Central registry for model configurations.
    """

    _registry: Dict[str, ModelRuntimeConfig] = {}

    @classmethod
    def register(cls, config: ModelRuntimeConfig) -> None:
        cls._registry[config.name] = config

    @classmethod
    def available_models(cls) -> Dict[str, ModelRuntimeConfig]:
        return dict(cls._registry)

    @classmethod
    def get(cls, name: str) -> ModelRuntimeConfig:
        try:
            return cls._registry[name]
        except KeyError as exc:
            raise KeyError(
                f"Model '{name}' is not registered. "
                f"Available options: {list(cls._registry.keys())}"
            ) from exc


# *********************************************************************************
# Add model here, use model_download.py will download all the model in base_models/
# *********************************************************************************

_DEFAULT_MODELS = [
    ModelRuntimeConfig(
        name="gemma-3-1b-it",
        hf_model_id="google/gemma-3-1b-it",
    ),
    # Add more models..
    # ModelRuntimeConfig(
    #     name="gemma-3-1b-it",
    #     hf_model_id="google/gemma-3-1b-it",
    # ),
]

for preset in _DEFAULT_MODELS:
    ModelConfigRegistry.register(preset)