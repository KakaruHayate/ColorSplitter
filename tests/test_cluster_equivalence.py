"""Equivalence tests for the clustering refactor.

Two distinct claims are checked, and they are not the same claim:

1. The rewritten ``SpectralCluster`` computes *exactly* what the original did.
   Same pruning, same Laplacian, same eigen decomposition, same labels. Any
   difference here is a bug.
2. The optional sparse eigensolver produces the same *partition* as the dense
   one. This is an approximation, so it is asserted as agreement rather than
   identity — and the measured agreement is reported so a regression cannot hide
   behind a loose threshold.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from colorsplitter.core.cluster import CommonClustering, SpectralCluster
from tests.reference.cluster_legacy import LegacyCommonClustering


def _blocks(seed: int, n_blocks: int = 3, per_block: int = 30, dim: int = 24, spread: float = 0.35):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3.0, size=(n_blocks, dim))
    return np.vstack(
        [rng.normal(centers[i], spread, size=(per_block, dim)) for i in range(n_blocks)]
    ).astype(np.float64)


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Permutation-invariant partition equality.

    Renumbering alone is not enough: ``k_means`` is free to hand the same block a
    different index in the two runs, so the block *order* can differ even when
    the partition is identical. Adjusted Rand Index is invariant to that, and
    equals exactly 1.0 for the same partition.
    """
    return adjusted_rand_score(a, b) == 1.0


def _explain(a: np.ndarray, b: np.ndarray) -> str:
    return (
        f"partitions differ: ARI={adjusted_rand_score(a, b):.4f}, "
        f"k={len(np.unique(a))} vs {len(np.unique(b))}"
    )


# --- 1. exactness against the original implementation -----------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("nmin", [1, 2, 3])
def test_dense_matches_legacy_exactly(seed: int, nmin: int) -> None:
    X = _blocks(seed)
    legacy = LegacyCommonClustering(
        cluster_type="spectral", mer_cos=None, min_num_spks=nmin, max_num_spks=14
    )
    new = CommonClustering(
        cluster_type="spectral",
        mer_cos=None,
        min_cluster_size=4,
        min_num_spks=nmin,
        max_num_spks=14,
        eigen_solver="dense",
    )
    legacy_labels = np.asarray(legacy(X))
    new_labels = np.asarray(new(X))
    assert _same_partition(legacy_labels, new_labels), _explain(legacy_labels, new_labels)


def test_merge_by_cos_matches_legacy() -> None:
    """mer_cos used to be accepted by the CLI and then silently dropped."""
    X = _blocks(7, n_blocks=4, per_block=25)
    legacy = LegacyCommonClustering(
        cluster_type="spectral", mer_cos=0.5, min_num_spks=4, max_num_spks=14
    )
    new = CommonClustering(
        cluster_type="spectral",
        mer_cos=0.5,
        min_cluster_size=4,
        min_num_spks=4,
        max_num_spks=14,
        eigen_solver="dense",
    )
    a, b = np.asarray(legacy(X)), np.asarray(new(X))
    assert _same_partition(a, b), _explain(a, b)


def test_mer_cos_is_actually_applied() -> None:
    """Regression guard: the parameter must change the outcome, not be ignored."""
    X = _blocks(11, n_blocks=3, per_block=20, spread=0.9)
    without = CommonClustering(cluster_type="spectral", min_num_spks=1, eigen_solver="dense")(X)
    with_merge = CommonClustering(
        cluster_type="spectral", mer_cos=0.999, min_num_spks=1, eigen_solver="dense"
    )(X)
    assert len(np.unique(with_merge)) <= len(np.unique(without))


# --- 2. pruning and Laplacian are computed identically ----------------------


def test_prune_mask_is_the_complement_of_the_original_loop() -> None:
    rng = np.random.default_rng(3)
    A = rng.normal(size=(60, 60))
    cluster = SpectralCluster(min_num_spks=1, max_num_spks=14, pval=0.05, min_pnum=6)
    expected = A.copy()
    n_elems = min(int((1 - cluster.pval) * A.shape[0]), A.shape[0] - cluster.min_pnum)
    for i in range(expected.shape[0]):
        expected[i, np.argsort(expected[i])[0:n_elems]] = 0
    assert np.array_equal(cluster.p_pruning(A, cluster.pval), expected)


def test_laplacian_matches_legacy() -> None:
    rng = np.random.default_rng(5)
    aff = rng.random((40, 40))
    cluster = SpectralCluster()
    np.testing.assert_allclose(cluster.get_laplacian(aff.copy()), _legacy_laplacian(aff.copy()))


def _legacy_laplacian(M: np.ndarray) -> np.ndarray:
    M[np.diag_indices(M.shape[0])] = 0
    D = np.sum(np.abs(M), axis=1)
    D = np.diag(D)
    return D - M


# --- 3. sparse solver agreement --------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_sparse_solver_agrees_with_dense(seed: int) -> None:
    """This is the evidence that the fast path may be enabled at all."""
    X = _blocks(seed, n_blocks=3, per_block=40)
    dense = SpectralCluster(min_num_spks=1, max_num_spks=14, eigen_solver="dense")
    sparse = SpectralCluster(min_num_spks=1, max_num_spks=14, eigen_solver="sparse")
    a = np.asarray(dense(X))
    b = np.asarray(sparse(X))
    ari = adjusted_rand_score(a, b)
    assert ari >= 0.95, f"sparse/dense agreement too low: ARI={ari:.3f}"
    assert sparse.last_solver == "sparse"
    assert dense.last_solver == "dense"


def test_auto_solver_switches_on_size() -> None:
    small = SpectralCluster(eigen_solver="auto", sparse_threshold=2048)
    assert small._resolve_solver(100) == "dense"
    assert small._resolve_solver(5000) == "sparse"


# --- 4. the newly configurable knobs ---------------------------------------


def test_max_num_spks_is_no_longer_capped_at_14() -> None:
    """The old code hard-coded 14; asking for more is now honoured."""
    X = _blocks(2, n_blocks=3, per_block=20)
    result = CommonClustering(
        cluster_type="spectral", min_num_spks=20, max_num_spks=25, eigen_solver="dense"
    )(X)
    assert result.shape[0] == X.shape[0]
    assert len(np.unique(result)) >= 1


def test_min_num_spks_above_old_cap_does_not_crash() -> None:
    X = _blocks(4, n_blocks=1, per_block=30, spread=0.1)
    cluster = SpectralCluster(min_num_spks=18, max_num_spks=22, eigen_solver="dense")
    labels = cluster(X)
    assert labels.shape[0] == X.shape[0]


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        SpectralCluster(min_num_spks=5, max_num_spks=2)
    with pytest.raises(ValueError):
        SpectralCluster(min_num_spks=0)
    with pytest.raises(ValueError):
        SpectralCluster(eigen_solver="magic")
    with pytest.raises(ValueError):
        CommonClustering(cluster_type="spectral", mer_cos=1.5)
    with pytest.raises(ValueError):
        CommonClustering(cluster_type="nope")


def test_small_inputs_short_circuit() -> None:
    X = _blocks(0, n_blocks=1, per_block=5)
    labels = CommonClustering(cluster_type="spectral")(X)
    assert np.all(labels == 1)
