"""Pipeline behaviour: clustering, projection, export and cache reuse."""

from __future__ import annotations

import numpy as np
import pytest

from colorsplitter.core.cache import EmbeddingCache, digest_of
from colorsplitter.core.pipeline import (
    CLUSTER_METHODS,
    cluster_embeddings,
    export_clusters,
    project_embeddings,
    scan,
)
from colorsplitter.core.types import ClusterResult
from tests.conftest import write_wav


def test_cluster_methods_are_restricted_to_the_chosen_two() -> None:
    assert CLUSTER_METHODS == ("spectral", "umap_hdbscan")


def test_cluster_embeddings_returns_one_label_per_point(toy_embeds) -> None:
    result = cluster_embeddings(toy_embeds, method="spectral", nmin=2)
    assert isinstance(result, ClusterResult)
    assert len(result) == toy_embeds.shape[0]
    assert result.n_clusters >= 2
    assert sum(result.sizes().values()) == toy_embeds.shape[0]


def test_cluster_embeddings_recovers_separated_blocks_when_k_is_given(toy_embeds) -> None:
    """With the cluster count supplied, the three blocks must come back cleanly."""
    from sklearn.metrics import adjusted_rand_score

    truth = np.repeat([0, 1, 2], 25)
    result = cluster_embeddings(toy_embeds, method="spectral", oracle_num=3)
    assert adjusted_rand_score(truth, result.labels) == 1.0


def test_automatic_cluster_count_is_reasonable_on_separated_data(toy_embeds) -> None:
    """The eigen-gap heuristic estimates k; it is not an oracle.

    What is worth asserting is that it does not fragment well-separated data
    beyond recognition — not that it lands on exactly 3.
    """
    from sklearn.metrics import adjusted_rand_score

    truth = np.repeat([0, 1, 2], 25)
    result = cluster_embeddings(toy_embeds, method="spectral", nmin=3, max_num_spks=8)
    score = adjusted_rand_score(truth, result.labels)
    assert score >= 0.7, f"blocks not recovered: ARI={score:.3f}, k={result.n_clusters}"


def test_cluster_embeddings_rejects_bad_input(toy_embeds) -> None:
    with pytest.raises(ValueError):
        cluster_embeddings(toy_embeds, method="kmeans")
    with pytest.raises(ValueError):
        cluster_embeddings(toy_embeds.ravel(), method="spectral")


def test_cluster_params_are_recorded_for_reproducibility(toy_embeds) -> None:
    result = cluster_embeddings(toy_embeds, method="spectral", nmin=2, mer_cos=0.8, max_num_spks=9)
    assert result.params["nmin"] == 2
    assert result.params["mer_cos"] == 0.8
    assert result.params["max_num_spks"] == 9


def test_result_copy_is_independent(toy_embeds) -> None:
    result = cluster_embeddings(toy_embeds, method="spectral")
    clone = result.copy()
    clone.labels[0] = 99
    assert result.labels[0] != 99


def test_projection_has_two_columns(toy_embeds) -> None:
    projection = project_embeddings(toy_embeds, method="pca")
    assert projection.coords.shape == (toy_embeds.shape[0], 2)
    assert projection.method == "pca"


def test_projection_rejects_bad_method(toy_embeds) -> None:
    with pytest.raises(ValueError):
        project_embeddings(toy_embeds, method="umap_v9")


def test_projection_is_cached_and_reused(toy_embeds, tmp_path) -> None:
    keys = [f"k{i}" for i in range(toy_embeds.shape[0])]
    first = project_embeddings(toy_embeds, keys=keys, method="pca", cache_dir=tmp_path)
    second = project_embeddings(toy_embeds, keys=keys, method="pca", cache_dir=tmp_path)
    assert np.array_equal(first.coords, second.coords)
    assert any((tmp_path / "projections").iterdir())


# --- export ----------------------------------------------------------------


def _dataset_and_result(root, labels):
    dataset = scan(root)
    return dataset, ClusterResult(np.asarray(labels, dtype=np.int64), "spectral", {})


def test_export_copy_leaves_the_source_untouched(shuffled_audio_root, tmp_path) -> None:
    dataset, result = _dataset_and_result(shuffled_audio_root, [0, 1, 1])
    report = export_clusters(dataset, result, tmp_path / "out", mode="copy")
    assert report.mode == "copy"
    assert report.written == 3
    assert report.skipped == 0
    assert len(list(shuffled_audio_root.rglob("*.wav"))) == 3
    assert report.per_cluster == {0: 1, 1: 2}


def test_export_move_relocates_the_source(shuffled_audio_root, tmp_path) -> None:
    dataset, result = _dataset_and_result(shuffled_audio_root, [0, 1, 1])
    report = export_clusters(dataset, result, tmp_path / "out", mode="move")
    assert report.written == 3
    assert list(shuffled_audio_root.rglob("*.wav")) == []


def test_export_rejects_an_unknown_mode(shuffled_audio_root, tmp_path) -> None:
    dataset, result = _dataset_and_result(shuffled_audio_root, [0, 1, 1])
    with pytest.raises(ValueError):
        export_clusters(dataset, result, tmp_path / "out", mode="delete")


def test_export_rejects_a_length_mismatch(shuffled_audio_root, tmp_path) -> None:
    dataset, _ = _dataset_and_result(shuffled_audio_root, [0, 0, 0])
    bad = ClusterResult(np.array([0]), "spectral", {})
    with pytest.raises(ValueError):
        export_clusters(dataset, bad, tmp_path / "out")


def test_export_puts_noise_in_its_own_directory(shuffled_audio_root, tmp_path) -> None:
    dataset, result = _dataset_and_result(shuffled_audio_root, [0, -1, -1])
    export_clusters(dataset, result, tmp_path / "out", mode="copy")
    assert (tmp_path / "out" / "noise").is_dir()


def test_export_disambiguates_name_collisions(tmp_path) -> None:
    a = write_wav(tmp_path / "src" / "x" / "same.wav")
    b = write_wav(tmp_path / "src" / "y" / "same.wav")
    from colorsplitter.core.pipeline import dataset_from_paths

    dataset = dataset_from_paths([a, b], root=tmp_path / "src")
    result = ClusterResult(np.array([0, 0]), "spectral", {})
    report = export_clusters(dataset, result, tmp_path / "out", mode="copy")
    assert report.written == 2
    assert len(list((tmp_path / "out" / "000").iterdir())) == 2


# --- cache -----------------------------------------------------------------


def test_digest_is_order_sensitive_but_stable() -> None:
    assert digest_of("a", "b") == digest_of("a", "b")
    assert digest_of("a", "b") != digest_of("b", "a")


def test_embedding_cache_round_trips(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "c.npz", dim=4)
    cache.update("k", "fp", np.arange(4, dtype=np.float32))
    cache.flush()

    reopened = EmbeddingCache(tmp_path / "c.npz", dim=4)
    hit = reopened.lookup("k", "fp")
    assert hit is not None
    np.testing.assert_array_equal(hit, np.arange(4, dtype=np.float32))


def test_embedding_cache_misses_on_a_changed_fingerprint(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "c.npz", dim=4)
    cache.update("k", "fp", np.arange(4, dtype=np.float32))
    cache.flush()
    assert EmbeddingCache(tmp_path / "c.npz", dim=4).lookup("k", "different") is None


def test_embedding_cache_discards_a_dimension_change(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "c.npz", dim=4)
    cache.update("k", "fp", np.arange(4, dtype=np.float32))
    cache.flush()
    assert len(EmbeddingCache(tmp_path / "c.npz", dim=7)) == 0


def test_embedding_cache_tolerates_a_corrupt_file(tmp_path) -> None:
    path = tmp_path / "c.npz"
    path.write_bytes(b"not an npz")
    assert len(EmbeddingCache(path, dim=4)) == 0
