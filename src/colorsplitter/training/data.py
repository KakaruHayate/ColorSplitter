"""Training data: class discovery and GE2E batch construction.

The only structural assumption is the directory convention::

    <root>/<singer>_<timbre>/<anything>.{wav,m4a,flac,mp3,...}

A directory is a *class*. The class granularity is ``<singer>_<timbre>`` rather
than the singer alone, and that choice is the whole point of the tool: it makes
two timbres of one singer into two different classes, so the encoder is pushed
to separate them. Grouping by singer instead would actively train the timbre
differences away.

The sampler then exploits that structure: a batch is filled from a small pool of
singers, so several of its classes usually belong to the *same* singer. Those
pairs are hard negatives — acoustically close, different labels — which is where
the gradient actually is.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.audio import find_audio_files, preprocess_wav
from ..core.hparams import mel_n_channels, partials_n_frames
from ..core.pipeline import scan

__all__ = ["ClassSpec", "TimbreDataset", "Ge2eBatchSampler", "split_class_name"]

log = logging.getLogger(__name__)

DEFAULT_TIMBRE_NAME = "default"


def split_class_name(name: str) -> tuple[str, str]:
    """Split ``<singer>_<timbre>`` into ``(singer, timbre)``.

    Splits on the *last* underscore, so singers whose names contain hyphens or
    devices keep working. A directory with no underscore is treated as a singer
    with a single, unnamed timbre.
    """
    if "_" not in name:
        return name, DEFAULT_TIMBRE_NAME
    singer, timbre = name.rsplit("_", 1)
    if not singer or not timbre:
        return name, DEFAULT_TIMBRE_NAME
    return singer, timbre


@dataclass
class ClassSpec:
    """One directory of audio, i.e. one GE2E class."""

    name: str
    singer: str
    timbre: str
    paths: list[Path] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.paths)


class TimbreDataset:
    """Directory-of-classes dataset, loaded lazily."""

    def __init__(self, root: Path, *, min_clips: int = 1):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root is not a directory: {self.root}")

        self.classes: list[ClassSpec] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            paths = find_audio_files(child)
            if len(paths) < min_clips:
                if paths:
                    log.info("skipping %s: only %d clip(s)", child.name, len(paths))
                continue
            singer, timbre = split_class_name(child.name)
            self.classes.append(ClassSpec(child.name, singer, timbre, paths))

        if not self.classes:
            raise ValueError(
                f"no class directories with audio found under {self.root}. "
                "Expected <root>/<singer>_<timbre>/<audio>."
            )

    def __len__(self) -> int:
        return len(self.classes)

    @property
    def singers(self) -> list[str]:
        return sorted({c.singer for c in self.classes})

    def classes_by_singer(self) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = {}
        for index, spec in enumerate(self.classes):
            grouped.setdefault(spec.singer, []).append(index)
        return grouped

    def stats(self) -> dict:
        clips = sum(len(c) for c in self.classes)
        grouped = self.classes_by_singer()
        return {
            "classes": len(self.classes),
            "singers": len(grouped),
            "clips": clips,
            "clips_per_class_min": min(len(c) for c in self.classes),
            "clips_per_class_max": max(len(c) for c in self.classes),
            "timbres_per_singer_max": max(len(v) for v in grouped.values()),
        }

    def load_mel(self, path: Path, rng: random.Random) -> np.ndarray | None:
        """Preprocess a clip and cut a fixed-length mel window.

        Returns ``(partials_n_frames, mel_n_channels)``, or ``None`` if the clip
        cannot be read. Short clips are zero-padded rather than dropped, so a
        class with only one very short file still contributes.
        """
        from ..core.audio import wav_to_mel_spectrogram

        try:
            wav = preprocess_wav(path)
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill the run
            log.warning("cannot read %s: %s", path, exc)
            return None

        mel = wav_to_mel_spectrogram(wav)
        frames = mel.shape[0]
        if frames >= partials_n_frames:
            start = rng.randrange(0, frames - partials_n_frames + 1)
            window = mel[start : start + partials_n_frames]
        else:
            window = np.zeros((partials_n_frames, mel.shape[1]), dtype=np.float32)
            window[:frames] = mel
        return window.astype(np.float32)


class Ge2eBatchSampler:
    """Builds GE2E batches from a :class:`TimbreDataset`.

    :param speakers_per_batch: number of class slots (``N`` in GE2E).
    :param utterances_per_speaker: clips drawn per class (``M``).
    :param singers_per_batch: how many distinct singers the class slots are
        drawn from. Small values put several timbres of one singer in the same
        batch, which is the intended hard-negative construction; setting it
        equal to ``speakers_per_batch`` disables that and samples classes
        uniformly at random.
    """

    def __init__(
        self,
        dataset: TimbreDataset,
        *,
        speakers_per_batch: int = 8,
        utterances_per_speaker: int = 4,
        singers_per_batch: int = 4,
        seed: int | None = None,
    ):
        if speakers_per_batch < 2:
            raise ValueError("speakers_per_batch must be >= 2")
        if utterances_per_speaker < 1:
            raise ValueError("utterances_per_speaker must be >= 1")
        if singers_per_batch < 1:
            raise ValueError("singers_per_batch must be >= 1")

        self.dataset = dataset
        self.n_classes = speakers_per_batch
        self.m_clips = utterances_per_speaker
        self.singers_per_batch = min(singers_per_batch, speakers_per_batch)
        self._rng = random.Random(seed)
        self._by_singer = dataset.classes_by_singer()
        self._singer_pool = sorted(self._by_singer)

    def _pick_classes(self) -> list[int]:
        """Choose ``n_classes`` class indices, biased towards reusing singers.

        Distinct classes are exhausted across *all* chosen singers before any
        class is used twice, so a batch is only degenerate when the singer pool
        genuinely cannot supply enough different classes.
        """
        n_singers = min(self.singers_per_batch, len(self._singer_pool))
        singers = self._rng.sample(self._singer_pool, n_singers)

        chosen: list[int] = []
        # Pass 1: round-robin over the singers, taking only unused classes.
        while len(chosen) < self.n_classes:
            progressed = False
            for singer in singers:
                if len(chosen) >= self.n_classes:
                    break
                available = [c for c in self._by_singer[singer] if c not in chosen]
                if available:
                    chosen.append(self._rng.choice(available))
                    progressed = True
            if not progressed:
                break
        # Pass 2: the pool is exhausted; allow repeats rather than fail.
        while len(chosen) < self.n_classes:
            singer = self._rng.choice(singers)
            chosen.append(self._rng.choice(self._by_singer[singer]))
        return chosen[: self.n_classes]

    def sample_indices(self) -> list[list[tuple[int, int]]]:
        """Return ``[[(class_index, clip_index), ...], ...]`` shaped (N, M)."""
        classes = self._pick_classes()
        batch: list[list[tuple[int, int]]] = []
        for class_index in classes:
            spec = self.dataset.classes[class_index]
            n_available = len(spec.paths)
            if n_available >= self.m_clips:
                picks = self._rng.sample(range(n_available), self.m_clips)
            else:
                # Not enough clips: sample with replacement, but never repeat a
                # clip twice inside one class when it can be avoided.
                picks = list(range(n_available))
                while len(picks) < self.m_clips:
                    picks.append(self._rng.randrange(n_available))
            batch.append([(class_index, clip) for clip in picks])
        return batch

    def batches(self, steps: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(mels, labels)`` with shapes ``(N*M, F, C)`` and ``(N*M,)``."""
        for _ in range(steps):
            yield self.load_batch(self.sample_indices())

    def load_batch(self, indices: Sequence[Sequence[tuple[int, int]]]) -> tuple[np.ndarray, np.ndarray]:
        mels: list[np.ndarray] = []
        labels: list[int] = []
        for label, entries in enumerate(indices):
            for class_index, clip_index in entries:
                spec = self.dataset.classes[class_index]
                mel = self.dataset.load_mel(spec.paths[clip_index], self._rng)
                if mel is None:
                    mel = np.zeros((partials_n_frames, mel_n_channels), dtype=np.float32)
                mels.append(mel)
                labels.append(label)
        if not mels:
            raise RuntimeError("empty batch")
        return np.stack(mels), np.asarray(labels, dtype=np.int64)

    def same_singer_fraction(self, batch_indices: Sequence[Sequence[tuple[int, int]]]) -> float:
        """Share of class pairs in a batch that belong to the same singer.

        Exposed because it is the property worth asserting on: it is the whole
        reason this sampler exists.
        """
        singers = [self.dataset.classes[entries[0][0]].singer for entries in batch_indices]
        n = len(singers)
        if n < 2:
            return 0.0
        pairs = n * (n - 1) / 2
        same = sum(
            1 for i in range(n) for j in range(i + 1, n) if singers[i] == singers[j]
        )
        return same / pairs


def quick_scan(root: Path) -> dict:
    """Summarise a dataset without constructing the full index."""
    dataset = scan(root)
    return {"files": len(dataset), "root": str(dataset.root)}
