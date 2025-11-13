# rlhf/evaluate.py
"""
Helper routines for evaluating DPO checkpoints.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from trl import DPOTrainer


def evaluate_dpo(trainer: DPOTrainer) -> Dict[str, float]:
    """
    Evaluate the model using the wrapped DPO trainer.
    """
    metrics = trainer.evaluate()
    sanitized: Dict[str, float] = {}

    for key, value in metrics.items():
        if hasattr(value, "item"):
            sanitized[key] = float(value.item())
        elif isinstance(value, (float, int)):
            sanitized[key] = float(value)
        elif isinstance(value, np.ndarray):
            sanitized[key] = float(value.mean())
        else:
            try:
                sanitized[key] = float(value)
            except (TypeError, ValueError):
                continue

    return sanitized