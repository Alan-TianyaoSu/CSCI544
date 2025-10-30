'''
Data Split
- 1. Read each original split (train/validation/test) from Parquet and normalize the `winner` label to point to the correct response column.
- 2. For every record, construct `response_winner` and `response_loser` by comparing the normalized winner flag.
- 3. Sample 20% of each split to form the SFT subset; the remaining 80% becomes the RLHF subset, ensuring no prompt overlap.
- 4. Save SFT data (`prompt`, `response_winner`) to `dataset/SFT/<split>.parquet` and RLHF data (`prompt`, `response_winner`, `response_loser`) to `dataset/RLHF/<split>.parquet`.
- 5. Print a brief summary confirming the row counts assigned to each subset per split.

* IMPORTANT: Disable SAMPLE_COUNTS for training
'''

import pandas as pd
from pathlib import Path
import numpy as np

# --- Config ---
SFT_FRACTION = 0.20
RANDOM_SEED = 42

RAW_PATHS = {
    "train": Path("dataset_raw/train_split.parquet"),
    "validation": Path("dataset_raw/val_split.parquet"),
    "test": Path("dataset_raw/test_split.parquet"),
}

OUTPUT_DIRS = {
    "sft": Path("dataset/SFT"),
    "rlhf": Path("dataset/RLHF"),
}

SAMPLE_COUNTS = None
# Technical verification sample sizes
SAMPLE_COUNTS = {
    "train": 50,
    "validation": 20,
    "test": 20,
}

WINNER_ALIASES = {
    "a": "response_a",
    "response_a": "response_a",
    "model_a": "response_a",
    "b": "response_b",
    "response_b": "response_b",
    "model_b": "response_b",
}

def normalize_winner_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add response_winner / response_loser columns based on the `winner` flag."""
    winner_col = df["winner"].astype(str).str.strip().str.lower()

    # Ensure every winner label is recognized
    if not winner_col.isin(WINNER_ALIASES).all():
        bad_values = winner_col[~winner_col.isin(WINNER_ALIASES)].unique().tolist()
        raise ValueError(f"Unexpected winner labels encountered: {bad_values}")

    winner_is_a = winner_col.map(lambda x: WINNER_ALIASES[x] == "response_a")

    df = df.copy()
    df["response_winner"] = df["response_a"].where(winner_is_a, df["response_b"])
    df["response_loser"] = df["response_b"].where(winner_is_a, df["response_a"])
    df["winner_is_a"] = winner_is_a.astype(int)  # 1 if A wins, 0 if B wins
    return df

def split_sft_rlhf(df: pd.DataFrame, sft_fraction: float, seed: int):
    """Return (sft_df, rlhf_df) disjoint subsets."""
    sft_count = max(1, int(len(df) * sft_fraction))
    sft_indices = df.sample(n=sft_count, random_state=seed, replace=False).index

    sft_df = df.loc[sft_indices].copy()
    rlhf_df = df.drop(index=sft_indices).copy()

    return sft_df, rlhf_df

def save_npz(data_dict: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data_dict)

for split_name, parquet_path in RAW_PATHS.items():
    df = pd.read_parquet(parquet_path)
    df = normalize_winner_column(df)

    try:
        sample_count = SAMPLE_COUNTS.get(split_name)
    except Exception:
        sample_count = None

    if sample_count is not None and len(df) > sample_count:
        df = df.sample(n=sample_count, random_state=RANDOM_SEED, replace=False)

    sft_df, rlhf_df = split_sft_rlhf(df, SFT_FRACTION, RANDOM_SEED)

    # --------- Save SFT ----------
    if split_name == "test":
        # Test split: binary classification format
        sft_data = {
            "prompt": sft_df["prompt"].to_numpy(dtype=object),
            "response_a": sft_df["response_a"].to_numpy(dtype=object),
            "response_b": sft_df["response_b"].to_numpy(dtype=object),
            # winner_label: 1 -> response_a wins, 0 -> response_b wins
            "winner_label": sft_df["winner_is_a"].to_numpy(dtype=np.int8),
        }
    else:
        # Train / validation: standard supervised fine-tuning format
        sft_data = {
            "prompt": sft_df["prompt"].to_numpy(dtype=object),
            "response_winner": sft_df["response_winner"].to_numpy(dtype=object),
        }

    save_npz(sft_data, OUTPUT_DIRS["sft"] / f"{split_name}.npz")

    # --------- Save RLHF ----------
    rlhf_data = {
        "prompt": rlhf_df["prompt"].to_numpy(dtype=object),
        "response_winner": rlhf_df["response_winner"].to_numpy(dtype=object),
        "response_loser": rlhf_df["response_loser"].to_numpy(dtype=object),
    }
    save_npz(rlhf_data, OUTPUT_DIRS["rlhf"] / f"{split_name}.npz")

    print(
        f"{split_name:>10}: total={len(df):5d} | "
        f"SFT={len(sft_df):5d} | RLHF={len(rlhf_df):5d}"
    )