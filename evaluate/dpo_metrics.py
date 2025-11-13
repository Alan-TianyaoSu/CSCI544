"""
Log-likelihood winner selection for DPO evaluation sets with artifact logging.

Usage:
    python evaluate/dpo_metrics.py --model gemma-3-1b-it
"""

import argparse
import csv
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel  # type: ignore
except ImportError:  # pragma: no cover
    PeftModel = None  # type: ignore

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False


Sample = Tuple[str, str, str, int, float]


BASE_MODEL_ROOT = Path("base_models")
ADAPTER_ROOT = Path("adapters")
DATASET_PATH = Path("dataset/DPO/test.npz")
ADAPTER_NAME = "adalora-dpo"
SCORE_TYPE = "average"  # or "total"
APPEND_EOS = True
RESULTS_ROOT = Path("results") / "preference_selection" / "DPO"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_for_path(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__").replace(":", "_").strip() or "default"


def create_result_dir(model_name: str) -> Tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_component = sanitize_for_path(model_name)
    result_dir = RESULTS_ROOT / model_component / timestamp
    ensure_directory(result_dir)
    return result_dir, timestamp


def to_serializable(value) -> Optional[object]:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def safe_float(value: float) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(converted):
        return converted
    return None


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            serializable_row = {key: to_serializable(value) for key, value in row.items()}
            file.write(json.dumps(serializable_row, ensure_ascii=False))
            file.write("\n")


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: to_serializable(row.get(field)) for field in fieldnames})


def plot_margin_distribution(sample_rows: List[dict], output_path: Path, weight_key: Optional[str] = None) -> None:
    if not MATPLOTLIB_AVAILABLE or not sample_rows:
        return
    margins: List[float] = []
    weights: List[float] = []
    for row in sample_rows:
        margin = row.get("margin")
        if margin is None:
            continue
        margins.append(margin)
        if weight_key is not None:
            weights.append(float(row.get(weight_key, 1.0)))
    if not margins:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    hist_weights = weights if weight_key is not None and len(weights) == len(margins) else None
    ax.hist(
        margins,
        bins=30,
        weights=hist_weights,
        color="#6a5acd",
        edgecolor="white",
        alpha=0.85,
    )
    ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.2, label="Zero margin")
    ax.set_title("Score margin distribution (winner - loser)")
    ax.set_xlabel("Margin")
    ax.set_ylabel("Weighted frequency" if hist_weights is not None else "Frequency")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_evaluation_artifacts(
    result_dir: Path,
    variant_name: str,
    metrics: dict,
    sample_rows: List[dict],
    weight_key: Optional[str] = None,
) -> None:
    variant_dir = result_dir / variant_name
    ensure_directory(variant_dir)
    write_json(variant_dir / "metrics.json", metrics)
    if sample_rows:
        write_csv(variant_dir / "samples.csv", sample_rows)
        write_jsonl(variant_dir / "samples.jsonl", sample_rows)
        plot_margin_distribution(
            sample_rows,
            variant_dir / "plots" / "margin_distribution.png",
            weight_key=weight_key,
        )
    else:
        print(f"[Info] No sample records for '{variant_name}', skipping CSV/plot export.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DPO preference classification via log-likelihood."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model directory name inside base_models/ and adapters/.",
    )
    return parser.parse_args()


def resolve_paths(model_name: str) -> Tuple[Path, Path, Path]:
    base_model_dir = (BASE_MODEL_ROOT / model_name).resolve()
    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")

    adapter_dir = (ADAPTER_ROOT / model_name / ADAPTER_NAME).resolve()

    dataset_path = DATASET_PATH.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    return base_model_dir, adapter_dir, dataset_path


def load_dataset(dataset_path: Path) -> List[Sample]:
    with np.load(dataset_path, allow_pickle=True) as data:
        prompts = data["prompt"]
        responses_winner = data["response_chosen"]
        responses_loser = data["response_rejected"]

        if "preference_weight" in data:
            weights = data["preference_weight"].astype(np.float32)
        else:
            weights = np.ones_like(responses_winner, dtype=np.float32)

    samples: List[Sample] = []
    for prompt, chosen, rejected, weight in zip(
        prompts, responses_winner, responses_loser, weights
    ):
        samples.append(
            (
                str(prompt),
                str(chosen),
                str(rejected),
                1,  # winner label always 1 for chosen vs rejected
                float(weight),
            )
        )

    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    return samples


def prepare_tokenizer(base_model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), use_fast=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
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
    max_length: Optional[int],
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
) -> Tuple[float, bool]:
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
        return float("-inf"), truncated

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
        score = total_logprob / response_len
    else:
        score = total_logprob
    return score, truncated


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer,
    samples: Iterable[Sample],
    device: torch.device,
    desc: str,
) -> Tuple[dict, List[dict]]:
    preds: List[int] = []
    labels: List[int] = []
    weights_full: List[float] = []
    margins: List[float] = []
    margin_weights: List[float] = []
    sample_records: List[dict] = []
    truncation_notifier = [False]
    skipped_samples = 0
    truncated_pairs = 0

    for sample_idx, (prompt, resp_win, resp_lose, label, weight) in enumerate(
        tqdm(samples, desc=desc, ncols=100),
        start=1,
    ):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        try:
            score_win, trunc_win = compute_response_score(
                model,
                tokenizer,
                prompt_ids,
                resp_win,
                device,
                truncation_notifier,
            )
            score_lose, trunc_lose = compute_response_score(
                model,
                tokenizer,
                prompt_ids,
                resp_lose,
                device,
                truncation_notifier,
            )
        except torch.cuda.OutOfMemoryError:
            skipped_samples += 1
            torch.cuda.empty_cache()
            print(
                f"\n[Warning] CUDA OOM when scoring sample #{sample_idx}. "
                "Skipping this sample."
            )
            continue

        truncated_pairs += int(trunc_win or trunc_lose)
        preds.append(1 if score_win > score_lose else 0)
        labels.append(label)
        weights_full.append(weight)

        margin_value: Optional[float] = None
        if np.isfinite(score_win) and np.isfinite(score_lose):
            margin_value = score_win - score_lose
            margins.append(margin_value)
            margin_weights.append(weight)

        sample_records.append(
            {
                "index": sample_idx,
                "prompt": prompt,
                "response_chosen": resp_win,
                "response_rejected": resp_lose,
                "label": label,
                "prediction": preds[-1],
                "score_chosen": safe_float(score_win),
                "score_rejected": safe_float(score_lose),
                "margin": safe_float(margin_value) if margin_value is not None else None,
                "weight": float(weight),
                "is_correct": bool(preds[-1] == label),
                "truncated_chosen": bool(trunc_win),
                "truncated_rejected": bool(trunc_lose),
            }
        )

    sample_weight_full = np.array(weights_full, dtype=np.float32) if weights_full else None
    metrics: dict = {
        "num_samples": len(labels),
        "accuracy": float(
            accuracy_score(labels, preds, sample_weight=sample_weight_full)
        )
        if labels
        else 0.0,
        "recall": float(
            recall_score(
                labels,
                preds,
                pos_label=1,
                zero_division=0,
                sample_weight=sample_weight_full,
            )
        )
        if labels
        else 0.0,
        "f1": float(
            f1_score(
                labels,
                preds,
                pos_label=1,
                zero_division=0,
                sample_weight=sample_weight_full,
            )
        )
        if labels
        else 0.0,
        "avg_margin": 0.0,
        "samples_with_valid_margin": len(margins),
        "truncated_pairs": truncated_pairs,
        "skipped_samples": skipped_samples,
        "sample_weight_sum": float(sample_weight_full.sum())
        if sample_weight_full is not None
        else float(len(labels)),
    }

    if margins:
        margins_np = np.asarray(margins, dtype=np.float32)
        if margin_weights:
            weights_np = np.asarray(margin_weights, dtype=np.float32)
            weight_sum = float(np.sum(weights_np))
            if weight_sum > 0:
                avg_margin = float(np.average(margins_np, weights=weights_np))
                pos_ratio = float(
                    np.average((margins_np > 0).astype(np.float32), weights=weights_np)
                )
                variance = float(
                    np.average((margins_np - avg_margin) ** 2, weights=weights_np)
                )
                margin_std = float(np.sqrt(max(variance, 0.0)))
            else:
                avg_margin = float(np.mean(margins_np))
                pos_ratio = float(np.mean(margins_np > 0))
                margin_std = float(np.std(margins_np))
        else:
            avg_margin = float(np.mean(margins_np))
            pos_ratio = float(np.mean(margins_np > 0))
            margin_std = float(np.std(margins_np))

        metrics.update(
            {
                "avg_margin": avg_margin,
                "margin_std": margin_std,
                "margin_min": float(np.min(margins_np)),
                "margin_max": float(np.max(margins_np)),
                "margin_median": float(np.median(margins_np)),
                "positive_margin_ratio": pos_ratio,
                "positive_margin_count": int(np.sum(margins_np > 0)),
            }
        )
    else:
        metrics.update(
            {
                "margin_std": None,
                "margin_min": None,
                "margin_max": None,
                "margin_median": None,
                "positive_margin_ratio": 0.0,
                "positive_margin_count": 0,
            }
        )

    return metrics, sample_records


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n[{title}]")
    print(f"  Samples : {metrics['num_samples']}")
    print(
        f"  Accuracy: {metrics['accuracy']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )
    print(f"  Avg margin (winner - loser): {metrics['avg_margin']:.4f}")
    print(f"  Valid margin samples: {metrics['samples_with_valid_margin']}")
    print(f"  Truncated pairs: {metrics['truncated_pairs']}")
    print(f"  Skipped samples (OOM): {metrics['skipped_samples']}")


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    base_model_dir, adapter_dir, dataset_path = resolve_paths(args.model)
    samples = load_dataset(dataset_path)

    result_dir, timestamp = create_result_dir(args.model)
    run_info = {
        "timestamp": timestamp,
        "model": args.model,
        "base_model_dir": str(base_model_dir),
        "adapter_dir": str(adapter_dir),
        "dataset_path": str(dataset_path),
        "score_type": SCORE_TYPE,
        "append_eos": APPEND_EOS,
        "num_samples_loaded": len(samples),
        "matplotlib_available": MATPLOTLIB_AVAILABLE,
    }
    write_json(result_dir / "run_info.json", run_info)

    print(f"Loaded {len(samples)} examples from {dataset_path}")
    tokenizer = prepare_tokenizer(base_model_dir)
    device = pick_device()
    print(f"Using device: {device}")
    print(f"[Artifacts] Results will be stored under: {result_dir}")

    print("\nEvaluating base model (policy)...")
    base_model = load_base_model(base_model_dir, device, tokenizer)
    base_metrics, base_samples = evaluate_model(
        base_model,
        tokenizer,
        samples,
        device=device,
        desc="Scoring (base)",
    )
    print_metrics("Base model", base_metrics)
    save_evaluation_artifacts(
        result_dir,
        "base_model",
        base_metrics,
        base_samples,
        weight_key="weight",
    )
    cleanup_model(base_model)

    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "run_directory": str(result_dir),
        "matplotlib_available": MATPLOTLIB_AVAILABLE,
        "variants": {"base_model": base_metrics},
    }

    adapter_exists = adapter_dir.is_dir()
    adapter_metrics = None
    if adapter_exists and PeftModel is None:
        print(
            f"\nAdapter directory found at {adapter_dir}, "
            "but `peft` is not installed. Skipping adapter evaluation."
        )
        adapter_exists = False
        summary["variants"]["adapter_model"] = {
            "status": "skipped",
            "reason": "`peft` not installed",
        }

    if adapter_exists:
        print(f"\nEvaluating adapter from {adapter_dir}...")
        adapted_model = load_base_model(base_model_dir, device, tokenizer)
        adapted_model = attach_adapter(adapted_model, adapter_dir)
        adapter_metrics, adapter_samples = evaluate_model(
            adapted_model,
            tokenizer,
            samples,
            device=device,
            desc="Scoring (adapter)",
        )
        print_metrics("Adapter + base model", adapter_metrics)
        save_evaluation_artifacts(
            result_dir,
            "adapter_model",
            adapter_metrics,
            adapter_samples,
            weight_key="weight",
        )
        cleanup_model(adapted_model)
        summary["variants"]["adapter_model"] = adapter_metrics
    elif "adapter_model" not in summary["variants"]:
        summary["variants"]["adapter_model"] = {
            "status": "not_found",
            "reason": f"No adapter directory found at {adapter_dir}",
        }

    write_json(result_dir / "summary.json", summary)
    print(f"\n[Summary] Artifact directory: {result_dir}")
    if not MATPLOTLIB_AVAILABLE:
        print("[Info] matplotlib is not installed; margin plots were not generated.")


if __name__ == "__main__":
    main()