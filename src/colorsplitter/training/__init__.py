"""Encoder training.

Delivered as working code: it runs, it resumes, and its checkpoints drop
straight into the inference pipeline. No training run is performed as part of
this refactor, and no new weights are produced.

See ``docs/training.md``.
"""

from __future__ import annotations

__all__ = ["GE2ELoss", "Ge2eBatchSampler", "TimbreDataset", "TrainConfig", "train"]


def __getattr__(name: str):
    if name in {"TimbreDataset", "Ge2eBatchSampler"}:
        from . import data

        return getattr(data, name)
    if name == "GE2ELoss":
        from .loss import GE2ELoss

        return GE2ELoss
    if name in {"TrainConfig", "train"}:
        from . import train as train_module

        return getattr(train_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
