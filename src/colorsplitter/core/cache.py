"""On-disk caches for embeddings and projections.

Embeddings are cached per file, keyed by a fingerprint of the source file plus
the encoder settings. That makes re-running after adding a handful of files
cost only those files, which is the difference between iterate-and-look and
walk-away-and-come-back.

Caches live in a dedicated directory, never inside the scanned dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

__all__ = ["ArrayCache", "EmbeddingCache", "digest_of"]

log = logging.getLogger(__name__)


def digest_of(*parts) -> str:
    """Stable short digest over arbitrary JSON-serialisable parts."""
    h = hashlib.sha1()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:20]


def _string_array(values) -> np.ndarray:
    """Unicode array, not an object array.

    ``np.load(allow_pickle=False)`` refuses to read object arrays, and the cache
    should not require unpickling a file from disk. A fixed-width unicode array
    round-trips natively and costs nothing here.
    """
    return np.asarray(list(values), dtype=str)


def _atomic_savez(path: Path, **arrays) -> None:
    """Write an ``.npz`` atomically.

    ``np.savez_compressed`` silently appends ``.npz`` when the name lacks it, so
    a ``<file>.tmp`` staging path would produce ``<file>.tmp.npz`` and the
    subsequent rename would fail. Writing through an open handle avoids that.
    """
    tmp = path.with_name(path.name + ".staging")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


class EmbeddingCache:
    """Row-per-file embedding cache.

    Stored as a single ``.npz`` holding parallel ``keys``/``fps`` arrays and the
    ``embeds`` matrix. A change in embedding dimension invalidates the file.
    """

    def __init__(self, path: Path, dim: int | None = None):
        self.path = Path(path)
        self.dim = dim
        self._keys: list[str] = []
        self._fps: dict[str, str] = {}
        self._rows: list[np.ndarray] = []
        self._index: dict[str, int] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                keys = [str(k) for k in data["keys"]]
                fps = [str(f) for f in data["fps"]]
                embeds = data["embeds"]
        except Exception as exc:  # noqa: BLE001 - a broken cache is not fatal
            log.warning("ignoring unreadable embedding cache %s: %s", self.path, exc)
            return
        if self.dim is not None and embeds.ndim == 2 and embeds.shape[1] != self.dim:
            log.info("embedding cache dimension changed, discarding %s", self.path)
            return
        self._keys = keys
        self._fps = dict(zip(keys, fps, strict=True))
        self._rows = [embeds[i] for i in range(embeds.shape[0])]
        self._index = {k: i for i, k in enumerate(keys)}

    def __len__(self) -> int:
        return len(self._keys)

    def lookup(self, key: str, fingerprint: str) -> np.ndarray | None:
        pos = self._index.get(key)
        if pos is None or self._fps.get(key) != fingerprint:
            return None
        return self._rows[pos]

    def update(self, key: str, fingerprint: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector)
        pos = self._index.get(key)
        if pos is None:
            self._index[key] = len(self._keys)
            self._keys.append(key)
            self._rows.append(vector)
        else:
            self._rows[pos] = vector
        self._fps[key] = fingerprint
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._rows:
            embeds = np.stack(self._rows).astype(np.float32)
        else:
            embeds = np.zeros((0, self.dim or 0), dtype=np.float32)
        _atomic_savez(
            self.path,
            keys=_string_array(self._keys),
            fps=_string_array([self._fps[k] for k in self._keys]),
            embeds=embeds,
        )
        self._dirty = False


class ArrayCache:
    """Whole-array cache, one ``.npz`` per digest."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, digest: str) -> Path:
        return self.directory / f"{digest}.npz"

    def get(self, digest: str) -> tuple[list[str], np.ndarray] | None:
        path = self._path(digest)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return [str(k) for k in data["keys"]], data["arr"]
        except Exception as exc:  # noqa: BLE001
            log.warning("ignoring unreadable cache %s: %s", path, exc)
            return None

    def put(self, digest: str, keys: list[str], arr: np.ndarray) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _atomic_savez(
            self._path(digest),
            keys=_string_array(keys),
            arr=np.asarray(arr),
        )
