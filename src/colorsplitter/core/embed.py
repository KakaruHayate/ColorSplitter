"""Embedding extraction.

Four encoders are available:

``timbre``
    The project's own speaker encoder, used for timbre-style separation. This
    is the default and the one the tool exists for.
``speaker``
    The upstream Resemblyzer ``pretrained.pt``. It separates *speaker identity*
    rather than timbre, which is a different question; useful when you need to
    tell singers apart.
``emotion``
    The wav2vec2 emotion model, kept from the original tool.
``mix``
    Concatenation of ``timbre`` and ``emotion``, for filtering references that
    have to match on both.

Performance notes: the original code decoded, preprocessed and embedded one
file at a time, holding everything in RAM only implicitly. Here decoding is
optionally multi-process, the encoder forward pass is batched over partial
utterances, and work is done in bounded chunks so that a many-thousand-file
dataset does not blow up memory. Every step is cached per file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .audio import preprocess_many
from .cache import EmbeddingCache, digest_of
from .modelzoo import Registry, resolve_weight
from .types import AudioDataset, EmbeddingSet

__all__ = ["ENCODERS", "EmbedConfig", "embed_dataset", "embed_paths"]

log = logging.getLogger(__name__)

ENCODERS = ("timbre", "speaker", "emotion", "mix")

#: ``(stage, done, total)`` — stage is a short human-readable label.
ProgressFn = Callable[[str, int, int], None]


@dataclass
class EmbedConfig:
    """Settings for :func:`embed_dataset`."""

    encoder: str = "timbre"
    weights_id: Optional[str] = None
    device: Optional[str] = None
    batch_size: int = 64
    workers: int = 1
    amp: bool = False
    trim_silences: object = "auto"
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    rate: float = 1.3
    min_coverage: float = 0.75
    #: Files processed per batch of decoding. Bounds peak memory.
    chunk: int = 256

    def __post_init__(self) -> None:
        if self.encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}, got {self.encoder!r}")
        if self.chunk < 1:
            raise ValueError("chunk must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


def _cache_digest(config: EmbedConfig, weights_key: str) -> str:
    return digest_of(
        "embeddings",
        config.encoder,
        weights_key,
        config.rate,
        config.min_coverage,
        str(config.trim_silences),
        config.amp,
    )


class _EncoderRunner:
    """Lazily constructs whichever encoder the config asks for."""

    def __init__(self, config: EmbedConfig, registry: Registry):
        self.config = config
        self.registry = registry
        self._voice = None
        self._emotion = None
        self._weights_key = "none"

    @property
    def weights_key(self) -> str:
        return self._weights_key

    def _resolve(self, purpose: str) -> Path:
        if self.config.weights_id:
            return resolve_weight(self.registry, self.config.weights_id, cache_dir=self.config.cache_dir)
        return resolve_weight(self.registry, purpose=purpose, cache_dir=self.config.cache_dir)

    def _timbre(self):
        if self._voice is None:
            from ..models.voice_encoder import VoiceEncoder

            path = self._resolve("timbre")
            self._weights_key = path.name
            self._voice = VoiceEncoder(
                device=self.config.device, weights_fpath=path, amp=self.config.amp
            )
            self._weights_key = f"{path.name}:{self._voice.device.type}"
        return self._voice

    def _emotion(self):
        if self._emotion is None:
            from ..models.emotion_encoder import EmotionEncoder
            from .modelzoo import fetch_emotion_model

            model_dir = fetch_emotion_model(self.registry, cache_dir=self.config.cache_dir)
            self._emotion = EmotionEncoder(model_dir, device=self.config.device)
            self._weights_key = f"{self._weights_key}+emotion"
        return self._emotion

    def embed_wavs(self, wavs: Sequence[np.ndarray], progress: Optional[ProgressFn]) -> np.ndarray:
        encoder = self.config.encoder
        blocks = []
        if encoder in ("timbre", "speaker", "mix"):
            blocks.append(
                self._timbre().embed_utterances(
                    list(wavs),
                    rate=self.config.rate,
                    min_coverage=self.config.min_coverage,
                    batch_size=self.config.batch_size,
                    progress=(lambda d, t: progress("encoding", d, t)) if progress else None,
                )
            )
        if encoder in ("emotion", "mix"):
            self._emotion()
            rows = [np.asarray(self._emotion.embed_waveform(w)).reshape(-1) for w in wavs]
            blocks.append(np.stack(rows).astype(np.float32) if rows else np.zeros((0, 0), np.float32))
        if not blocks:
            raise ValueError(f"encoder {encoder!r} produced nothing")
        return np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]


def embed_dataset(
    dataset: AudioDataset,
    config: Optional[EmbedConfig] = None,
    *,
    registry: Optional[Registry] = None,
    progress: Optional[ProgressFn] = None,
) -> EmbeddingSet:
    """Embed every item of *dataset*.

    Returns an :class:`EmbeddingSet` whose rows follow ``dataset.items`` order.
    Cached rows are reused; only files whose fingerprint changed are recomputed.
    """
    config = config or EmbedConfig()
    if registry is None:
        from .modelzoo import load_registry

        registry = load_registry()

    runner = _EncoderRunner(config, registry)
    # Touch the encoder once so the weights key is known before the cache opens.
    if config.encoder in ("timbre", "speaker", "mix"):
        runner._timbre()  # noqa: SLF001 - intentional warm-up
    if config.encoder in ("emotion", "mix"):
        runner._emotion()  # noqa: SLF001

    dim_hint = None
    cache: Optional[EmbeddingCache] = None
    if config.use_cache and config.cache_dir:
        cache_path = Path(config.cache_dir) / f"embeddings-{_cache_digest(config, runner.weights_key)}.npz"
        cache = EmbeddingCache(cache_path, dim=dim_hint)

    items = list(dataset.items)
    total = len(items)
    if progress:
        progress("cached", 0, total)

    resolved: dict[str, np.ndarray] = {}
    if cache is not None:
        for item in items:
            hit = cache.lookup(item.key, item.fingerprint())
            if hit is not None:
                resolved[item.key] = hit
    if progress:
        progress("cached", len(resolved), total)

    pending = [it for it in items if it.key not in resolved]
    for start in range(0, len(pending), config.chunk):
        block = pending[start : start + config.chunk]
        if progress:
            progress("decoding", start, len(pending))
        wavs = preprocess_many(
            [it.path for it in block],
            workers=config.workers,
            trim_silences=config.trim_silences,
        )
        vectors = runner.embed_wavs(wavs, progress)
        for item, vector in zip(block, vectors):
            vec = np.asarray(vector, dtype=np.float32).reshape(-1)
            resolved[item.key] = vec
            if cache is not None:
                cache.update(item.key, item.fingerprint(), vec)
        if cache is not None:
            cache.flush()
        if progress:
            progress("embedded", min(start + len(block), len(pending)), len(pending))

    if not items:
        return EmbeddingSet([], np.zeros((0, 0), np.float32), config.encoder, runner.weights_key)

    embeds = np.stack([resolved[it.key] for it in items]).astype(np.float32)
    if progress:
        progress("done", total, total)
    return EmbeddingSet(dataset.keys, embeds, config.encoder, runner.weights_key)


def embed_paths(
    paths: Sequence[Path],
    config: Optional[EmbedConfig] = None,
    *,
    registry: Optional[Registry] = None,
    progress: Optional[ProgressFn] = None,
    root: Optional[Path] = None,
) -> EmbeddingSet:
    """Convenience wrapper: build a dataset from *paths* and embed it."""
    from .pipeline import dataset_from_paths

    dataset = dataset_from_paths(paths, root=root)
    return embed_dataset(dataset, config, registry=registry, progress=progress)
