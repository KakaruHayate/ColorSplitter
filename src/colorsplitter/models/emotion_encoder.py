"""Speech-emotion embedding via ``wav2vec2-large-robust-12-ft-emotion-msp-dim``.

Two independent problems were fixed here.

**Loading.** The original module called ``from_pretrained`` at import time, so
merely importing the package downloaded and instantiated a 600 MB model even if
the emotion feature was never used. Loading is now lazy, per instance.

**Fidelity.** The trunk is no longer provided by ``transformers``. It is
reproduced in :mod:`colorsplitter.models.wav2vec2` against the real checkpoint,
so the graph cannot drift when that library changes. The feature-extractor
preprocessing from the model card is also reproduced rather than delegated —
including the zero-mean/unit-variance normalisation, which is part of the model
path and not an optional nicety.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ..core.audio import load_waveform, resample_waveform
from .wav2vec2 import Wav2Vec2Config, Wav2Vec2ForSpeechClassification

__all__ = ["EmotionEncoder", "zero_mean_unit_var_norm"]

log = logging.getLogger(__name__)

EMOTION_SAMPLE_RATE = 16000
_NORM_EPS = 1e-7


def zero_mean_unit_var_norm(wav: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation.

    Reproduces ``Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm`` exactly,
    including the epsilon and the fact that it operates on the array's own dtype
    with population variance (``ddof=0``).
    """
    x = np.asarray(wav)
    return (x - x.mean()) / np.sqrt(x.var() + _NORM_EPS)


class EmotionEncoder:
    """Lazily-loaded emotion embedder.

    :param model_dir: directory holding ``config.json`` and the checkpoint.
    :param device: torch device, or ``None`` for CUDA-when-available.
    :param normalize: apply the model's input normalisation (leave on unless you
        are deliberately feeding pre-normalised audio).
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device | None = None,
        *,
        normalize: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.normalize = bool(normalize)
        self._device = device
        self._model: Wav2Vec2ForSpeechClassification | None = None
        self._config: Wav2Vec2Config | None = None
        self.load_report: dict = {}

    # --- device / loading ---------------------------------------------------

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self._device, str):
            self._device = torch.device(self._device)
        return self._device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"emotion model not found at {self.model_dir}. "
                "Run `cs weights fetch --only emotion` to download it."
            )
        self._config = Wav2Vec2Config.from_json(self.model_dir / "config.json")
        model, report = Wav2Vec2ForSpeechClassification.from_pretrained(self.model_dir)
        self.load_report = report
        log.info(
            "loaded emotion model from %s (%s, %d tensors ignored)",
            self.model_dir, report["source"], len(report["unexpected"]),
        )
        model.to(self.device)
        model.eval()
        self._model = model

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        assert self._config is not None
        return int(self._config.hidden_size)

    @property
    def label_names(self) -> list[str]:
        self._ensure_loaded()
        assert self._config is not None
        return list(self._config.names)

    # --- inference ----------------------------------------------------------

    @torch.no_grad()
    def embed_waveform(self, wav: np.ndarray, embeddings: bool = True) -> np.ndarray:
        """Embed a waveform already at 16 kHz. Returns shape ``(1, dim)``."""
        self._ensure_loaded()
        assert self._model is not None
        prepared = zero_mean_unit_var_norm(wav) if self.normalize else np.asarray(wav, dtype=np.float32)
        tensor = torch.from_numpy(np.asarray(prepared, dtype=np.float32)).reshape(1, -1).to(self.device)
        pooled, logits = self._model(tensor)
        out = pooled if embeddings else logits
        return out.detach().cpu().numpy()

    def embed_file(self, path: str | Path) -> np.ndarray:
        """Embed an audio file (decoded and resampled to 16 kHz)."""
        wav, sr = load_waveform(path)
        if sr != EMOTION_SAMPLE_RATE:
            wav = resample_waveform(wav, sr, EMOTION_SAMPLE_RATE)
        return self.embed_waveform(wav)

    def embed_files(self, paths: Sequence[str | Path], progress=None) -> np.ndarray:
        """Embed several files; returns ``(N, dim)``."""
        rows = []
        for i, path in enumerate(paths):
            rows.append(np.asarray(self.embed_file(path)).reshape(-1))
            if progress is not None:
                progress(i + 1, len(paths))
        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.stack(rows).astype(np.float32)


def default_model_dir(cache_dir: str | Path) -> Path:
    """Conventional location of the emotion model inside a weights cache."""
    return Path(cache_dir) / "emotion"


def read_config_names(model_dir: str | Path) -> list[str]:
    """Label names, readable without loading any weights."""
    path = Path(model_dir) / "config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    label2id = raw.get("label2id") or {}
    return [name for name, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
