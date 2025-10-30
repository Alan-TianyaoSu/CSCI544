"""
Log-likelihood winner selection for SFT or RLHF evaluation sets.

Usage:
    python evaluate/select_winner.py --model gemma-3-1b-it --task SFT
    python evaluate/select_winner.py --model gemma-3-1b-it --task RLHF
"""

import argparse
import gc
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel  # type: ignore
except ImportError:  # pragma: no cover
    PeftModel = None  # type: ignore


Sample = Tuple[str, str, str, int]


BASE_MODEL_ROOT = Path("base_models")
ADAPTER_ROOT = Path("adapters")
DATASET_PATHS = {
    "SFT": Path("dataset/SFT/test.npz"),
    "RLHF": Path("dataset/RLHF/test.npz"),
}
ADAPTER_NAMES = {
    "SFT": "adalora-sft",
    "RLHF": "adalora-rlhf",
}
SCORE_TYPE = "average"  # or "total"
APPEND_EOS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate response preference classification via log-likelihood."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model directory name inside base_models/ and adapters/.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("SFT", "RLHF"),
        required=True,
        help="Task split to evaluate.",
    )
    return parser.parse_args()


def resolve_paths(model_name: str, task: str) -> Tuple[Path, Path, Path]:
    base_model_dir = (BASE_MODEL_ROOT / model_name).resolve()
    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")

    adapter_name = ADAPTER_NAMES[task]
    adapter_dir = (ADAPTER_ROOT / model_name / adapter_name).resolve()

    dataset_path = DATASET_PATHS[task]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    return base_model_dir, adapter_dir, dataset_path


def load_dataset(task: str, dataset_path: Path) -> List[Sample]:
    with np.load(dataset_path, allow_pickle=True) as data:
        prompts = data["prompt"]

        if task == "SFT":
            responses_a = data["response_a"]
            responses_b = data["response_b"]
            labels = data["winner_label"]
        else:  # RLHF
            responses_a = data["response_winner"]
            responses_b = data["response_loser"]
            labels = np.ones_like(responses_a, dtype=np.int8)

    samples: List[Sample] = []
    for prompt, resp_a, resp_b, label in zip(prompts, responses_a, responses_b, labels):
        samples.append(
            (
                str(prompt),
                str(resp_a),
                str(resp_b),
                int(label),
            )
        )

    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    return samples


def prepare_tokenizer(base_model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), use_fast=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        # Create a dedicated pad token to mimic training-time setup.
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resize_embeddings_if_needed(model: AutoModelForCausalLM, tokenizer) -> None:
    model_vocab = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab = len(tokenizer)
    if model_vocab != tokenizer_vocab:
        model.resize_token_embeddings(tokenizer_vocab)


def load_base_model(
    base_model_dir: Path,
    device: torch.device,
    tokenizer,
) -> AutoModelForCausalLM:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        dtype=dtype,
    )
    resize_embeddings_if_needed(model, tokenizer)
    model.to(device)
    model.eval()
    return model


def attach_adapter(
    model: AutoModelForCausalLM,
    adapter_dir: Path,
) -> AutoModelForCausalLM:
    if PeftModel is None:
        raise ImportError(
            "peft is required to load adapters. Install via `pip install peft`."
        )
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    peft_model.eval()
    return peft_model


def truncate_to_max_length(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    max_length: int | None,
) -> Tuple[List[int], List[int], bool]:
    prompt_ids = list(prompt_ids)
    response_ids = list(response_ids)
    truncated = False

    if max_length is None:
        return prompt_ids, response_ids, truncated

    total_len = len(prompt_ids) + len(response_ids)
    if total_len <= max_length:
        return prompt_ids, response_ids, truncated

    truncated = True
    overflow = total_len - max_length

    if overflow > 0 and prompt_ids:
        if overflow >= len(prompt_ids):
            overflow -= len(prompt_ids)
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[overflow:]
            overflow = 0

    if overflow > 0 and response_ids:
        if overflow >= len(response_ids):
            response_ids = []
        else:
            response_ids = response_ids[overflow:]

    return prompt_ids, response_ids, truncated


def compute_response_score(
    model: AutoModelForCausalLM,
    tokenizer,
    prompt_ids: Sequence[int],
    response_text: str,
    device: torch.device,
    truncation_notifier: List[bool],
) -> float:
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if APPEND_EOS and tokenizer.eos_token_id is not None:
        if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
            response_ids = response_ids + [tokenizer.eos_token_id]

    prompt_ids_trimmed, response_ids_trimmed, truncated = truncate_to_max_length(
        prompt_ids,
        response_ids,
        getattr(model.config, "max_position_embeddings", None),
    )

    if truncated and not truncation_notifier[0]:
        print(
            "Warning: One or more samples exceed the model context window. "
            "Truncating from the left to fit the maximum length."
        )
        truncation_notifier[0] = True

    response_len = len(response_ids_trimmed)
    if response_len == 0:
        return float("-inf")

    input_ids = torch.tensor(
        [prompt_ids_trimmed + response_ids_trimmed],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]

    prompt_len = len(prompt_ids_trimmed)
    response_start_idx = max(prompt_len - 1, 0)
    response_end_idx = response_start_idx + response_len

    selected_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    response_token_log_probs = selected_log_probs[:, response_start_idx:response_end_idx]

    total_logprob = response_token_log_probs.sum().item()

    if SCORE_TYPE == "average":
        return total_logprob / response_len

    return total_logprob


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer,
    samples: Iterable[Sample],
    device: torch.device,
    desc: str,
) -> dict:
    preds: List[int] = []
    labels: List[int] = []
    margins: List[float] = []
    truncation_notifier = [False]

    for prompt, resp_a, resp_b, label in tqdm(samples, desc=desc, ncols=100):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        score_a = compute_response_score(
            model,
            tokenizer,
            prompt_ids,
            resp_a,
            device,
            truncation_notifier,
        )
        score_b = compute_response_score(
            model,
            tokenizer,
            prompt_ids,
            resp_b,
            device,
            truncation_notifier,
        )

        pred = 1 if score_a > score_b else 0
        preds.append(pred)
        labels.append(label)
        margins.append(score_a - score_b)

    metrics = {
        "num_samples": len(labels),
        "accuracy": accuracy_score(labels, preds),
        "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
    }
    return metrics


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n[{title}]")
    print(f"  Samples : {metrics['num_samples']}")
    print(
        f"  Accuracy: {metrics['accuracy']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )
    print(f"  Avg margin (score A - B): {metrics['avg_margin']:.4f}")


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    base_model_dir, adapter_dir, dataset_path = resolve_paths(args.model, args.task)
    samples = load_dataset(args.task, dataset_path)

    print(f"Loaded {len(samples)} examples from {dataset_path}")
    tokenizer = prepare_tokenizer(base_model_dir)
    device = pick_device()
    print(f"Using device: {device}")

    print("\nEvaluating base model...")
    base_model = load_base_model(base_model_dir, device, tokenizer)
    base_metrics = evaluate_model(
        base_model,
        tokenizer,
        samples,
        device=device,
        desc="Scoring (base)",
    )
    print_metrics("Base model", base_metrics)
    cleanup_model(base_model)

    adapter_exists = adapter_dir.is_dir()
    if adapter_exists and PeftModel is None:
        print(
            f"\nAdapter directory found at {adapter_dir}, "
            "but `peft` is not installed. Skipping adapter evaluation."
        )
        adapter_exists = False

    if adapter_exists:
        print(f"\nEvaluating adapter from {adapter_dir}...")
        adapted_model = load_base_model(base_model_dir, device, tokenizer)
        adapted_model = attach_adapter(adapted_model, adapter_dir)
        adapter_metrics = evaluate_model(
            adapted_model,
            tokenizer,
            samples,
            device=device,
            desc="Scoring (adapter)",
        )
        print_metrics("Adapter + base model", adapter_metrics)
        cleanup_model(adapted_model)
    else:
        print(f"\nNo adapter found at {adapter_dir}. Skipping adapter evaluation.")


if __name__ == "__main__":
    main()