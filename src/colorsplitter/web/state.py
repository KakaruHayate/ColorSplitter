"""Server-side session state.

Single-user by design: the tool is a local dataset workbench, not a service.
Everything the UI can do maps to a method here, and every mutation goes through
:class:`~colorsplitter.core.labels.LabelHistory` so undo/redo is uniform.

Audio files are addressed by opaque tokens derived from their key, so the
browser never learns a filesystem path.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..core import labels as labelops
from ..core.embed import EmbedConfig, embed_dataset
from ..core.modelzoo import Registry, load_registry
from ..core.pipeline import cluster_embeddings, export_clusters, project_embeddings, scan
from ..core.types import AudioDataset, ClusterResult, EmbeddingSet, ExportReport, Projection

__all__ = ["SessionState"]

log = logging.getLogger(__name__)


def media_token(key: str) -> str:
    """Opaque, stable per-file token used in ``/media/<token>``."""
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class SessionState:
    """Mutable state for one running instance."""

    cache_dir: Path
    registry_path: Optional[Path] = None
    embed_config: EmbedConfig = field(default_factory=EmbedConfig)

    dataset: Optional[AudioDataset] = None
    embeddings: Optional[EmbeddingSet] = None
    clusters: Optional[ClusterResult] = None
    projection: Optional[Projection] = None
    history: Optional[labelops.LabelHistory] = None
    export_report: Optional[ExportReport] = None

    projection_method: str = "tsne"
    cluster_method: str = "spectral"
    cluster_params: dict[str, Any] = field(default_factory=dict)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _media: dict[str, Path] = field(default_factory=dict, repr=False)

    # --- registry / config --------------------------------------------------

    @property
    def registry(self) -> Registry:
        with self._lock:
            if self.registry_path:
                return load_registry(self.registry_path)
            return load_registry()

    def weight_choices(self) -> list[dict]:
        entries = self.registry.encoders
        return [
            {
                "id": e.id,
                "purpose": e.purpose,
                "step": e.step,
                "default": e.is_default,
                "notes": e.notes,
            }
            for e in entries
        ]

    # --- loading ------------------------------------------------------------

    def set_dataset(self, dataset: AudioDataset) -> None:
        with self._lock:
            self.dataset = dataset
            self.embeddings = None
            self.clusters = None
            self.projection = None
            self.history = None
            self.export_report = None
            self._rebuild_media()

    def set_run(
        self,
        embeddings: EmbeddingSet,
        clusters: ClusterResult,
        projection: Projection,
        *,
        cluster_method: Optional[str] = None,
        cluster_params: Optional[dict] = None,
        projection_method: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.embeddings = embeddings
            self.clusters = clusters
            self.projection = projection
            self.history = labelops.LabelHistory(clusters.labels)
            if cluster_method:
                self.cluster_method = cluster_method
            if cluster_params is not None:
                self.cluster_params = dict(cluster_params)
            if projection_method:
                self.projection_method = projection_method

    def import_embeddings(self, embeds: np.ndarray, keys: Optional[Sequence[str]] = None) -> None:
        """Load an external embedding matrix, bypassing inference."""
        embeds = np.asarray(embeds, dtype=np.float32)
        if embeds.ndim != 2:
            raise ValueError("embedding matrix must be 2-D")
        n = embeds.shape[0]
        if keys is None:
            if self.dataset is not None and len(self.dataset) == n:
                keys = self.dataset.keys
            else:
                keys = [f"item_{i:05d}" for i in range(n)]
        if len(keys) != n:
            raise ValueError(f"got {len(keys)} keys for {n} embeddings")
        with self._lock:
            self.embeddings = EmbeddingSet(list(keys), embeds, "imported", "imported")
            self.clusters = None
            self.projection = None
            self.history = None
            self._rebuild_media()

    def _rebuild_media(self) -> None:
        self._media = {}
        if self.dataset is None:
            return
        for item in self.dataset.items:
            self._media[media_token(item.key)] = item.path

    def media_path(self, token: str) -> Optional[Path]:
        return self._media.get(token)

    # --- labels -------------------------------------------------------------

    @property
    def labels(self) -> Optional[np.ndarray]:
        return self.history.current if self.history is not None else None

    def _require(self) -> tuple[EmbeddingSet, labelops.LabelHistory]:
        if self.embeddings is None or self.history is None:
            raise RuntimeError("no embeddings loaded yet")
        return self.embeddings, self.history

    def assign(self, indices: Sequence[int], target: int) -> np.ndarray:
        _, history = self._require()
        return history.push(labelops.relabel_points(history.current, indices, int(target)))

    def create_cluster_from(self, indices: Sequence[int]) -> np.ndarray:
        """Move the given points into a brand new cluster id."""
        _, history = self._require()
        current = history.current
        return history.push(labelops.relabel_points(current, indices, labelops.new_cluster_id(current)))

    def split(self, cluster: int, indices: Sequence[int]) -> np.ndarray:
        _, history = self._require()
        return history.push(labelops.split_cluster(history.current, int(cluster), indices))

    def merge(self, sources: Sequence[int], target: int) -> np.ndarray:
        _, history = self._require()
        return history.push(labelops.merge_clusters(history.current, sources, int(target)))

    def rename(self, old: int, new: int) -> np.ndarray:
        _, history = self._require()
        return history.push(labelops.rename_cluster(history.current, int(old), int(new)))

    def remove(self, cluster: int, reassign_to: Optional[int] = None) -> np.ndarray:
        _, history = self._require()
        return history.push(labelops.remove_cluster(history.current, int(cluster), reassign_to=reassign_to))

    def compact(self) -> np.ndarray:
        _, history = self._require()
        return history.push(labelops.compact(history.current))

    def undo(self) -> np.ndarray:
        _, history = self._require()
        return history.undo()

    def redo(self) -> np.ndarray:
        _, history = self._require()
        return history.redo()

    # --- recompute ----------------------------------------------------------

    def recluster(self, **params) -> ClusterResult:
        embeddings, history = self._require()
        merged = {**self.cluster_params, **params}
        result = cluster_embeddings(
            embeddings.embeds,
            method=merged.pop("method", self.cluster_method),
            **merged,
        )
        with self._lock:
            self.clusters = result
            self.cluster_method = result.method
            self.cluster_params = dict(result.params)
            self.history = labelops.LabelHistory(result.labels)
        return result

    def reproject(self, method: str, **kwargs) -> Projection:
        embeddings, _ = self._require()
        projection = project_embeddings(
            embeddings.embeds,
            keys=embeddings.keys,
            method=method,
            cache_dir=self.cache_dir,
            **kwargs,
        )
        with self._lock:
            self.projection = projection
            self.projection_method = method
        return projection

    def export(self, dest: Path, mode: str = "copy") -> ExportReport:
        if self.dataset is None or self.history is None:
            raise RuntimeError("nothing to export yet")
        result = ClusterResult(self.history.current, self.cluster_method, dict(self.cluster_params))
        report = export_clusters(self.dataset, result, dest, mode=mode)
        with self._lock:
            self.export_report = report
        return report

    def embed_dataset(self, progress=None) -> EmbeddingSet:
        if self.dataset is None:
            raise RuntimeError("scan a directory first")
        return embed_dataset(
            self.dataset,
            self.embed_config,
            registry=self.registry,
            progress=progress,
        )

    def scan(self, root: Path) -> AudioDataset:
        dataset = scan(root)
        self.set_dataset(dataset)
        return dataset

    # --- views --------------------------------------------------------------

    def summary(self) -> dict:
        payload: dict[str, Any] = {
            "root": str(self.dataset.root) if self.dataset else None,
            "files": len(self.dataset) if self.dataset else 0,
            "embedding_dim": self.embeddings.dim if self.embeddings else 0,
            "encoder": self.embeddings.encoder if self.embeddings else None,
            "weights": self.embeddings.weights_id if self.embeddings else None,
            "cluster_method": self.cluster_method,
            "projection_method": self.projection_method,
            "cluster_params": self.cluster_params,
            "can_undo": bool(self.history and self.history.can_undo),
            "can_redo": bool(self.history and self.history.can_redo),
        }
        current = self.labels
        if current is not None:
            payload["n_clusters"] = int(len(np.unique(current[current >= 0])))
            payload["cluster_sizes"] = {str(k): v for k, v in labelops.cluster_sizes(current).items()}
            payload["noise"] = int((current < 0).sum())
        if self.export_report is not None:
            payload["export"] = {
                "dest": str(self.export_report.dest),
                "mode": self.export_report.mode,
                "written": self.export_report.written,
                "errors": len(self.export_report.errors),
            }
        return payload

    def points(self) -> list[dict]:
        """Flattened view of everything the scatter plot needs."""
        if self.embeddings is None:
            return []
        current = self.labels
        coords = self.projection.coords if self.projection is not None else None
        out: list[dict] = []
        for i, key in enumerate(self.embeddings.keys):
            out.append(
                {
                    "i": i,
                    "key": key,
                    "name": Path(key).name,
                    "cluster": int(current[i]) if current is not None else -1,
                    "x": float(coords[i, 0]) if coords is not None else 0.0,
                    "y": float(coords[i, 1]) if coords is not None else 0.0,
                    "token": media_token(key),
                    "playable": media_token(key) in self._media,
                }
            )
        return out
