"""Pipeline orchestration.

This module is the single definition of "what the tool does". The CLI and the
web app are both thin wrappers around it — there is no capability reachable from
one entry point and not the other, and nothing here blocks on stdin or opens a
plot window.

Every step takes and returns plain data (see :mod:`colorsplitter.core.types`),
so the whole pipeline is testable without audio, a GPU or a browser.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from .cache import ArrayCache, digest_of
from .embed import EmbedConfig, embed_dataset
from .modelzoo import Registry
from .types import AudioDataset, AudioItem, ClusterResult, ExportReport, Projection

__all__ = [
    "CLUSTER_METHODS",
    "cluster_embeddings",
    "dataset_from_paths",
    "export_clusters",
    "project_embeddings",
    "run",
    "scan",
]

log = logging.getLogger(__name__)

CLUSTER_METHODS = ("spectral", "umap_hdbscan")

ProgressFn = Callable[[str, int, int], None]


# --- scanning ---------------------------------------------------------------


def scan(root: str | Path, extensions: Iterable[str] | None = None) -> AudioDataset:
    """Recursively collect every audio file under *root*.

    No dataset layout is assumed: whatever directory you point at is walked, and
    each file becomes one item keyed by its path relative to the root. That is
    the only input contract this tool has.
    """
    from .audio import iter_audio_files

    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"input path does not exist: {base}")

    items: list[AudioItem] = []
    for path in iter_audio_files(base, extensions):
        try:
            stat = path.stat()
        except OSError:
            log.warning("skipping unreadable file: %s", path)
            continue
        try:
            key = str(path.relative_to(base))
        except ValueError:
            key = path.name
        items.append(AudioItem(path=path, key=key, size=stat.st_size, mtime_ns=stat.st_mtime_ns))

    items.sort(key=lambda it: it.key.lower())
    return AudioDataset(root=base, items=items)


def dataset_from_paths(paths: Sequence[str | Path], root: Path | None = None) -> AudioDataset:
    """Build a dataset from an explicit list of files (no scanning)."""
    resolved = [Path(p).expanduser().resolve() for p in paths]
    base = Path(root).resolve() if root else (Path(_common_prefix(resolved)) if resolved else Path("."))
    items = []
    for path in resolved:
        stat = path.stat()
        try:
            key = str(path.relative_to(base))
        except ValueError:
            key = path.name
        items.append(AudioItem(path=path, key=key, size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    items.sort(key=lambda it: it.key.lower())
    return AudioDataset(root=base, items=items)


def _common_prefix(paths: Sequence[Path]) -> str:
    if not paths:
        return "."
    parts = [p.parts for p in paths]
    shared: list[str] = []
    for chunk in zip(*parts, strict=False):
        if len(set(chunk)) == 1:
            shared.append(chunk[0])
        else:
            break
    return str(Path(*shared)) if shared else str(paths[0].parent)


# --- clustering / projection ------------------------------------------------


def cluster_embeddings(
    embeds: np.ndarray,
    *,
    method: str = "spectral",
    nmin: int = 1,
    mer_cos: float | None = None,
    max_num_spks: int = 14,
    min_cluster_size: int = 4,
    oracle_num: int | None = None,
    eigen_solver: str = "auto",
    **kwargs,
) -> ClusterResult:
    """Cluster an embedding matrix.

    The clustering algorithm is deliberately untouched; only ``max_num_spks``
    is now configurable instead of being pinned at 14, and ``mer_cos`` is
    actually passed through (the CLI used to accept it and drop it).
    """
    from .cluster import CommonClustering

    X = np.asarray(embeds, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("embeds must be a 2-D array")
    if method not in CLUSTER_METHODS:
        raise ValueError(f"method must be one of {CLUSTER_METHODS}, got {method!r}")

    params = {
        "nmin": int(nmin),
        "mer_cos": mer_cos,
        "max_num_spks": int(max_num_spks),
        "min_cluster_size": int(min_cluster_size),
        "oracle_num": oracle_num,
    }
    if method == "spectral":
        params["eigen_solver"] = eigen_solver
        clusterer = CommonClustering(
            cluster_type="spectral",
            mer_cos=mer_cos,
            min_cluster_size=min_cluster_size,
            min_num_spks=int(nmin),
            max_num_spks=int(max_num_spks),
            oracle_num=oracle_num,
            eigen_solver=eigen_solver,
            **kwargs,
        )
    else:
        clusterer = CommonClustering(
            cluster_type="umap_hdbscan",
            mer_cos=mer_cos,
            min_cluster_size=min_cluster_size,
            **kwargs,
        )

    labels = np.asarray(clusterer(X), dtype=np.int64)
    return ClusterResult(labels=labels, method=method, params=params)


def project_embeddings(
    embeds: np.ndarray,
    *,
    keys: Sequence[str] | None = None,
    method: str = "tsne",
    cache_dir: Path | None = None,
    use_cache: bool = True,
    **kwargs,
) -> Projection:
    """Reduce to 2D for display, reusing a cached projection when possible."""
    from .reduce import project

    X = np.asarray(embeds)
    cache = ArrayCache(Path(cache_dir) / "projections") if (use_cache and cache_dir) else None
    digest = digest_of("projection", method, list(kwargs.items()), int(X.shape[0]), X.shape[1])

    if cache is not None and keys is not None:
        hit = cache.get(digest)
        if hit is not None and list(hit[0]) == list(keys):
            return Projection(np.asarray(hit[1]), method)

    projection = project(X, method=method, **kwargs)
    if cache is not None and keys is not None:
        cache.put(digest, list(keys), projection.coords)
    return projection


# --- export -----------------------------------------------------------------


_NOISE_DIR = "noise"


def export_clusters(
    dataset: AudioDataset,
    result: ClusterResult,
    dest: str | Path,
    *,
    mode: str = "copy",
    prefix_keys: bool = False,
    progress: ProgressFn | None = None,
) -> ExportReport:
    """Write the audio into ``dest/<cluster>/``.

    ``mode="copy"`` duplicates the files and leaves the source untouched;
    ``mode="move"`` relocates them. The original tool had two separate scripts
    with these two behaviours and no way to tell them apart at a glance — here
    the choice is explicit and the default is the non-destructive one.
    """
    if mode not in ("copy", "move"):
        raise ValueError("mode must be 'copy' or 'move'")
    if len(result) != len(dataset):
        raise ValueError("cluster result and dataset length mismatch")

    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    report = ExportReport(dest=dest, mode=mode)
    seen: set[Path] = set()

    for index, item in enumerate(dataset.items):
        label = int(result.labels[index])
        name = f"{label:03d}" if label >= 0 else _NOISE_DIR
        target_dir = dest / name
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = item.path.name
        if prefix_keys:
            filename = f"{item.key.replace('/', '_')}"
        target = target_dir / filename
        if target in seen or target.exists():
            target = target_dir / f"{item.path.stem}__{item.fingerprint()[:8]}{item.path.suffix}"
        seen.add(target)

        try:
            if mode == "copy":
                shutil.copy2(item.path, target)
            else:
                shutil.move(str(item.path), str(target))
        except OSError as exc:
            report.errors.append(f"{item.key}: {exc}")
            continue

        report.written += 1
        report.per_cluster[label] = report.per_cluster.get(label, 0) + 1
        if progress is not None:
            progress("exporting", index + 1, len(dataset))

    report.skipped = len(dataset) - report.written
    return report


# --- convenience ------------------------------------------------------------


def run(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    embed_config: EmbedConfig | None = None,
    cluster_method: str = "spectral",
    nmin: int = 1,
    mer_cos: float | None = None,
    max_num_spks: int = 14,
    projection_method: str = "tsne",
    export: bool = False,
    export_mode: str = "copy",
    cache_dir: Path | None = None,
    registry: Registry | None = None,
    progress: ProgressFn | None = None,
) -> dict:
    """End-to-end run: scan → embed → cluster → project → (optionally) export."""
    config = embed_config or EmbedConfig(cache_dir=cache_dir)

    dataset = scan(input_dir)
    if progress:
        progress("scanned", len(dataset), len(dataset))
    if len(dataset) == 0:
        raise ValueError(f"no audio files found under {input_dir}")

    embeddings = embed_dataset(dataset, config, registry=registry, progress=progress)
    result = cluster_embeddings(
        embeddings.embeds,
        method=cluster_method,
        nmin=nmin,
        mer_cos=mer_cos,
        max_num_spks=max_num_spks,
    )
    projection = project_embeddings(
        embeddings.embeds,
        keys=embeddings.keys,
        method=projection_method,
        cache_dir=cache_dir,
    )

    report = None
    if export:
        report = export_clusters(
            dataset, result, output_dir or (Path(input_dir).parent / "colorsplitter-out"),
            mode=export_mode, progress=progress,
        )

    return {
        "dataset": dataset,
        "embeddings": embeddings,
        "clusters": result,
        "projection": projection,
        "export": report,
    }
