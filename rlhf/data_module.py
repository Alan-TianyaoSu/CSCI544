# rlhf/data_module.py
"""
Data loading utilities for DPO training on pairwise preference datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value
from transformers import PreTrainedTokenizerBase


@dataclass
class DPODataModule:
    """
    Utility to load preference datasets for DPO training.

    Attributes:
        tokenizer: Optional Hugging Face tokenizer kept for downstream use.
        max_length: Maximum sequence length for prompt + response pairs.
        max_prompt_length: Maximum prompt-only length used by the trainer.
    """
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    max_length: int = 1024
    max_prompt_length: int = 512

    def _load_npz(self, path: Path) -> Dict[str, List[str]]:
        """
        Load NPZ data and convert numpy arrays to Python lists.
        """
        if not path.exists():
            raise FileNotFoundError(f"DPO data file not found: {path}")

        data = np.load(path, allow_pickle=True)
        records = {key: data[key].tolist() for key in data.files}
        return records

    def _build_dataset(self, records: Dict[str, List[str]]) -> Dataset:
        """
        Convert raw preference triples into a Hugging Face Dataset.
        """
        required_keys = {"prompt", "response_winner", "response_loser"}
        missing_keys = required_keys.difference(records.keys())
        if missing_keys:
            raise KeyError(
                f"DPO dataset is missing required keys: {sorted(missing_keys)}"
            )

        features = Features(
            {
                "prompt": Value("string"),
                "chosen": Value("string"),
                "rejected": Value("string"),
            }
        )

        combined_rows = [
            {"prompt": prompt, "chosen": winner, "rejected": loser}
            for prompt, winner, loser in zip(
                records["prompt"],
                records["response_winner"],
                records["response_loser"],
            )
        ]

        return Dataset.from_list(combined_rows, features=features)

    def load_dataset_dict(
        self,
        train_path: Path,
        eval_path: Optional[Path] = None,
    ) -> DatasetDict:
        """
        Load training (and optional evaluation) datasets.
        """
        train_records = self._load_npz(train_path)
        train_dataset = self._build_dataset(train_records)

        dataset_dict = {"train": train_dataset}

        if eval_path is not None:
            eval_records = self._load_npz(eval_path)
            eval_dataset = self._build_dataset(eval_records)
            dataset_dict["validation"] = eval_dataset

        return DatasetDict(dataset_dict)