# supervised_fine_tuning/evaluate.py

"""
Helper functions for evaluating SFT checkpoints.

Currently, we compute perplexity on the validation set. This module is designed
to be extended with BLEU, ROUGE, or custom metrics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from transformers import Trainer


def evaluate_perplexity(trainer: Trainer) -> Dict[str, float]:
    """
    Evaluate the model and derive perplexity from the loss.
    """
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss")

    if loss is not None:
        metrics["perplexity"] = float(np.exp(loss))
    else:
        metrics["perplexity"] = float("inf")

    return metrics