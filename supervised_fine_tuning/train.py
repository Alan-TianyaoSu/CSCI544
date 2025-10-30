# supervised_fine_tuning\train.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from model.config import ModelRuntimeConfig, SFTTrainingConfig
from model.lora import (
    build_adalora_model,
    ensure_adapter_directory,
    load_peft_adapter_if_available,
)
from .data_module import SFTDataModule
from .evaluate import evaluate_perplexity


def _determine_world_size(configured_world_size: Optional[int]) -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return configured_world_size or 1


def _load_tokenizer(model_source: str, model_config: ModelRuntimeConfig):
    tokenizer_kwargs = model_config.build_tokenizer_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    return tokenizer


def _load_base_model(model_source: str, model_config: ModelRuntimeConfig) -> torch.nn.Module:
    model_kwargs = model_config.build_model_kwargs()
    dtype_name = model_config.dtype or "float32"
    dtype = getattr(torch, dtype_name)
    return AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=dtype,
        **model_kwargs,
    )


def _resolve_training_plan(
    training_config: SFTTrainingConfig,
    train_dataset_size: int,
) -> Dict[str, Optional[float]]:
    num_epochs = training_config.num_train_epochs
    max_steps = training_config.max_steps
    per_device_train = training_config.per_device_train_batch_size
    if per_device_train is None:
        raise ValueError("per_device_train_batch_size must be specified in manual configuration.")

    grad_accumulation = training_config.gradient_accumulation_steps or 1
    per_device_eval = training_config.per_device_eval_batch_size or per_device_train
    world_size = _determine_world_size(training_config.world_size)
    steps_per_epoch_override = training_config.steps_per_epoch

    if max_steps is not None:
        total_steps = max_steps
        steps_per_epoch = None
    else:
        if num_epochs is None:
            raise ValueError(
                "num_train_epochs must be provided when max_steps is not set."
            )
        if steps_per_epoch_override is not None:
            steps_per_epoch = steps_per_epoch_override
        else:
            effective_batch = per_device_train * world_size * grad_accumulation
            if effective_batch <= 0:
                raise ValueError("Effective batch size must be greater than zero.")
            steps_per_epoch = math.ceil(train_dataset_size / effective_batch)
        total_steps = int(math.ceil(float(num_epochs) * steps_per_epoch))

    return {
        "num_train_epochs": float(num_epochs) if num_epochs is not None else None,
        "per_device_train_batch_size": per_device_train,
        "per_device_eval_batch_size": per_device_eval,
        "gradient_accumulation_steps": grad_accumulation,
        "world_size": world_size,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": max_steps,
        "total_steps": total_steps,
    }


def run_sft_training(
    model_config: ModelRuntimeConfig,
    training_config: SFTTrainingConfig,
) -> Dict[str, float]:
    model_source = model_config.resolve_model_source()
    tokenizer = _load_tokenizer(model_source, model_config)

    data_module = SFTDataModule(
        tokenizer=tokenizer,
        max_length=training_config.max_seq_length,
    )
    dataset = data_module.load_dataset_dict(
        train_path=training_config.train_file,
        eval_path=training_config.eval_file,
    )
    train_sample_count = len(dataset["train"])
    print(f"[Model training]: ✅ Loaded dataset (train samples: {train_sample_count})")

    training_plan = _resolve_training_plan(
        training_config=training_config,
        train_dataset_size=train_sample_count,
    )
    total_steps = training_plan["total_steps"]
    if total_steps is None:
        raise ValueError(
            "Unable to determine total optimizer steps. Please set max_steps or provide "
            "steps_per_epoch and num_train_epochs through manual configuration."
        )

    print(f"[Model training]: ✅ Resolved optimizer steps: {total_steps}")

    base_model = _load_base_model(model_source, model_config)
    if hasattr(base_model, "resize_token_embeddings"):
        base_model.resize_token_embeddings(len(tokenizer))

    if training_config.gradient_checkpointing and hasattr(base_model, "config"):
        base_model.config.use_cache = False

    adapter_root = model_config.adapter_dir
    adapter_subdir = adapter_root / training_config.adapter_name
    ensure_adapter_directory(adapter_root)

    if training_config.resume_checkpoint is not None:
        model = load_peft_adapter_if_available(
            base_model,
            adapter_path=adapter_subdir,
            adapter_name=training_config.adapter_name,
        )
    else:
        ensure_adapter_directory(adapter_subdir)
        if model_config.adalora is None:
            raise ValueError(
                f"No AdaLoRA configuration supplied for model '{model_config.name}'. "
                "Please define it in manual_train.py."
            )
        model = build_adalora_model(
            base_model=base_model,
            adapter_dir=adapter_root,
            adapter_name=training_config.adapter_name,
            adalora_config=model_config.adalora,
            total_step=int(total_steps),
        )

    if training_config.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    if training_config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    print(f"[Model training]: ✅ Success loading model and LoRA: {model_config.name}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args_kwargs: Dict[str, object] = {
        "output_dir": str(training_config.output_dir),
        "num_train_epochs": training_plan["num_train_epochs"],
        "per_device_train_batch_size": training_plan["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_plan["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_plan["gradient_accumulation_steps"],
        "learning_rate": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
        "logging_steps": training_config.logging_steps,
        "eval_strategy": training_config.eval_strategy,
        "save_strategy": training_config.save_strategy,
        "save_total_limit": training_config.save_total_limit,
        "bf16": training_config.bf16,
        "tf32": training_config.tf32,
        "gradient_checkpointing": training_config.gradient_checkpointing,
        "report_to": training_config.report_to if training_config.report_to is not None else "none",
        "warmup_ratio": training_config.warmup_ratio,
        "warmup_steps": training_config.warmup_steps,
        "eval_steps": training_config.eval_steps,
        "save_steps": training_config.save_steps,
        "logging_strategy": training_config.logging_strategy,
        "save_only_model": training_config.save_only_model,
        "load_best_model_at_end": training_config.load_best_model_at_end,
        "metric_for_best_model": training_config.metric_for_best_model,
        "greater_is_better": training_config.greater_is_better,
    }

    if training_config.gradient_checkpointing:
        training_args_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    max_steps = training_plan["max_steps"]
    training_args_kwargs["max_steps"] = max_steps if max_steps is not None else -1

    training_args_kwargs = {
        key: value
        for key, value in training_args_kwargs.items()
        if value is not None
    }

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    resume_from_checkpoint = (
        str(training_config.resume_checkpoint)
        if training_config.resume_checkpoint is not None
        else None
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    adapter_save_root = adapter_root
    adapter_save_dir = adapter_subdir
    ensure_adapter_directory(adapter_save_root)
    trainer.model.save_pretrained(adapter_save_root, save_embedding_layers=True)
    ensure_adapter_directory(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)

    metrics: Dict[str, float] = {}
    if dataset.get("validation") is not None:
        metrics = evaluate_perplexity(trainer)

    trainer.save_state()
    return metrics