"""Plain data containers shared across the core pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "AudioDataset",
    "AudioItem",
    "ClusterResult",
    "EmbeddingSet",
    "ExportReport",
    "Projection",
]


@dataclass(frozen=True)
class AudioItem:
    """One audio file, identified by a stable path relative to the scan root."""

    path: Path
    key: str
    size: int
    mtime_ns: int
    duration: Optional[float] = None

    def fingerprint(self) -> str:
        """Content-ish identity used by the embedding cache."""
        h = hashlib.sha1()
        h.update(self.key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(self.size).encode("ascii"))
        h.update(b"\0")
        h.update(str(self.mtime_ns).encode("ascii"))
        return h.hexdigest()


@dataclass
class AudioDataset:
    """The result of scanning a directory tree."""

    root: Path
    items: list[AudioItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def keys(self) -> list[str]:
        return [it.key for it in self.items]

    @property
    def paths(self) -> list[Path]:
        return [it.path for it in self.items]

    def select(self, keys: Sequence[str]) -> "AudioDataset":
        wanted = set(keys)
        return AudioDataset(self.root, [it for it in self.items if it.key in wanted])


@dataclass
class EmbeddingSet:
    """Embeddings for a dataset, one row per :class:`AudioItem`."""

    keys: list[str]
    embeds: np.ndarray
    encoder: str
    weights_id: str

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def dim(self) -> int:
        return int(self.embeds.shape[1]) if self.embeds.ndim == 2 else 0


@dataclass
class ClusterResult:
    """Integer cluster assignment, one label per embedding row."""

    labels: np.ndarray
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_clusters(self) -> int:
        return int(len(np.unique(self.labels)))

    def sizes(self) -> dict[int, int]:
        uniq, counts = np.unique(self.labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(uniq, counts)}

    def copy(self) -> "ClusterResult":
        return ClusterResult(self.labels.copy(), self.method, dict(self.params))


@dataclass
class Projection:
    """Two-dimensional coordinates for display."""

    coords: np.ndarray
    method: str

    def __len__(self) -> int:
        return int(self.coords.shape[0])


@dataclass
class ExportReport:
    """Outcome of writing a clustered dataset back to disk."""

    dest: Path
    mode: str
    written: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    per_cluster: dict[int, int] = field(default_factory=dict)
