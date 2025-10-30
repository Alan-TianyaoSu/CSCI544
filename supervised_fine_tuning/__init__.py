"""
Supervised Fine-Tuning (SFT) package.

Exports the primary training entry point and utilities to integrate with CLI.
"""

from .train import run_sft_training, SFTTrainingConfig

__all__ = ["run_sft_training", "SFTTrainingConfig"]