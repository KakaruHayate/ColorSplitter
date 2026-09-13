"""Cluster-label editing.

Every function here is pure: it takes the current labels and returns new ones.
That keeps the interactive editing in the WebUI testable without a browser, and
makes undo/redo a matter of keeping a stack of label arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np

__all__ = [
    "LabelHistory",
    "cluster_sizes",
    "compact",
    "merge_clusters",
    "relabel_points",
    "remove_cluster",
    "rename_cluster",
    "split_cluster",
]

Labels = np.ndarray


def _as_labels(labels) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    return arr.astype(np.int64, copy=True)


def cluster_sizes(labels) -> dict[int, int]:
    """Number of points per cluster id."""
    uniq, counts = np.unique(np.asarray(labels), return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}


def compact(labels) -> np.ndarray:
    """Renumber clusters to ``0..k-1``, ordered by their smallest current id.

    Keeps the export directory names stable and readable.
    """
    arr = _as_labels(labels)
    present = np.unique(arr)
    mapping = {int(old): new for new, old in enumerate(present)}
    return np.array([mapping[int(v)] for v in arr], dtype=np.int64)


def relabel_points(labels, indices: Sequence[int], target: int) -> np.ndarray:
    """Assign the given point *indices* to cluster *target*."""
    arr = _as_labels(labels)
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        return arr
    if idx.min() < 0 or idx.max() >= arr.shape[0]:
        raise IndexError("point index out of range")
    arr[idx] = int(target)
    return arr


def new_cluster_id(labels) -> int:
    """Smallest non-negative id not currently in use."""
    arr = np.asarray(labels)
    used = set(int(v) for v in np.unique(arr))
    candidate = 0
    while candidate in used:
        candidate += 1
    return candidate


def split_cluster(labels, cluster: int, indices: Sequence[int], new_id: Optional[int] = None) -> np.ndarray:
    """Move *indices* out of *cluster* into a (new) cluster."""
    arr = _as_labels(labels)
    target = new_cluster_id(arr) if new_id is None else int(new_id)
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        return arr
    if not np.all(arr[idx] == int(cluster)):
        raise ValueError("all split points must currently belong to the source cluster")
    arr[idx] = target
    return arr


def merge_clusters(labels, sources: Iterable[int], target: int) -> np.ndarray:
    """Fold every id in *sources* into *target*."""
    arr = _as_labels(labels)
    target = int(target)
    for src in sources:
        src = int(src)
        if src != target:
            arr[arr == src] = target
    return arr


def rename_cluster(labels, old: int, new: int) -> np.ndarray:
    """Change a cluster's id, merging into *new* if it already exists."""
    arr = _as_labels(labels)
    arr[arr == int(old)] = int(new)
    return arr


def remove_cluster(labels, cluster: int, *, reassign_to: Optional[int] = None) -> np.ndarray:
    """Delete a cluster, either moving its points elsewhere or dropping them.

    Points are dropped by assigning them ``-1``, matching the convention used
    by HDBSCAN for noise and understood by :func:`compact` / export.
    """
    arr = _as_labels(labels)
    mask = arr == int(cluster)
    arr[mask] = int(reassign_to) if reassign_to is not None else -1
    return arr


class LabelHistory:
    """Bounded undo/redo stack over label arrays."""

    def __init__(self, initial, limit: int = 100):
        self._limit = int(limit)
        self._past: list[np.ndarray] = []
        self._future: list[np.ndarray] = []
        self._current = _as_labels(initial)

    @property
    def current(self) -> np.ndarray:
        return self._current

    @property
    def can_undo(self) -> bool:
        return bool(self._past)

    @property
    def can_redo(self) -> bool:
        return bool(self._future)

    def push(self, labels) -> np.ndarray:
        """Record *labels* as the new state, invalidating any redo history."""
        new = _as_labels(labels)
        if np.array_equal(new, self._current):
            return self._current
        self._past.append(self._current)
        if len(self._past) > self._limit:
            self._past.pop(0)
        self._future.clear()
        self._current = new
        return self._current

    def undo(self) -> np.ndarray:
        if self._past:
            self._future.append(self._current)
            self._current = self._past.pop()
        return self._current

    def redo(self) -> np.ndarray:
        if self._future:
            self._past.append(self._current)
            self._current = self._future.pop()
        return self._current

    def reset(self, labels) -> np.ndarray:
        self._past.clear()
        self._future.clear()
        self._current = _as_labels(labels)
        return self._current
