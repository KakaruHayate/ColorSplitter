"""Training model construction.

The trained network must be byte-for-byte the same architecture as the one used
for inference, otherwise a freshly trained checkpoint cannot be dropped into the
pipeline. Rather than duplicate the definition, this module builds the inference
class directly and only adds what training needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ..models.voice_encoder import VoiceEncoder

__all__ = ["build_encoder", "load_checkpoint", "save_checkpoint"]


def build_encoder(
    *,
    device: Optional[str] = None,
    resume_from: Optional[Path] = None,
) -> VoiceEncoder:
    """Create an encoder for training.

    Without *resume_from* the weights are randomly initialised; the model class
    is the same one inference uses, so anything trained here loads there.
    """
    if resume_from is not None:
        return VoiceEncoder(device=device, weights_fpath=Path(resume_from))
    return VoiceEncoder(device=device, load_weights=False)


def load_checkpoint(
    path: Path,
    encoder: VoiceEncoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
    similarity: Optional[torch.nn.Module] = None,
) -> int:
    """Restore model (and optionally optimiser/loss) state; returns the step."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    encoder.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if similarity is not None:
        if "similarity_weight" in checkpoint:
            similarity.similarity_weight.data.copy_(checkpoint["similarity_weight"].reshape(1))
        if "similarity_bias" in checkpoint:
            similarity.similarity_bias.data.copy_(checkpoint["similarity_bias"].reshape(1))
    return int(checkpoint.get("step", 0))


def save_checkpoint(
    path: Path,
    *,
    encoder: VoiceEncoder,
    step: int,
    loss: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    similarity: Optional[torch.nn.Module] = None,
) -> Path:
    """Write a checkpoint in the historical layout.

    ``model_state`` is the only key inference needs; the rest is kept so a run
    can be resumed. ``cs weights pack`` strips it down for release.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": encoder.state_dict(),
        "step": int(step),
        "loss": float(loss),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if similarity is not None:
        payload["similarity_weight"] = similarity.similarity_weight.detach().reshape(1)
        payload["similarity_bias"] = similarity.similarity_bias.detach().reshape(1)
    torch.save(payload, path)
    return path
