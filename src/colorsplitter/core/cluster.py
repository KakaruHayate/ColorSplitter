"""Clustering algorithms.

``SpectralCluster`` and ``UmapHdbscan`` are ported from 3D-Speaker
(Apache-2.0, https://github.com/alibaba-damo-academy/3D-Speaker) and their
algorithmic behaviour is deliberately preserved — this module is the part of
the project that must not change. What *is* new here:

* ``p_pruning`` is vectorised. ``np.argsort(A, axis=1)`` yields exactly the
  same per-row ordering as the original Python loop, so the pruned matrix is
  bit-identical while the O(N^2 log N) interpreter loop disappears.
* ``max_num_spks`` is no longer hard-coded to 14.
* An optional sparse eigensolver (``eigen_solver="sparse"``) builds the pruned
  affinity as a k-NN sparse matrix and asks ARPACK for the leading eigenpairs
  instead of running a full O(N^3) decomposition. It is opt-in; the dense path
  remains the reference-exact one. See ``tests/test_cluster_equivalence.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ["SpectralCluster", "UmapHdbscan", "CommonClustering"]

log = logging.getLogger(__name__)

#: Matrices at or below this size use the exact dense eigensolver under
#: ``eigen_solver="auto"``.
DEFAULT_SPARSE_THRESHOLD = 2048


class SpectralCluster:
    """Spectral clustering on the unnormalised Laplacian of an affinity matrix.

    Adapted from https://github.com/speechbrain/speechbrain via 3D-Speaker.

    :param min_num_spks: smallest candidate number of clusters.
    :param max_num_spks: largest candidate number of clusters. No longer capped
        at 14; keep it small relative to the sample count to stay cheap.
    :param pval: fraction of the largest similarities kept per row.
    :param min_pnum: never prune a row below this many kept entries.
    :param oracle_num: fix the cluster count instead of estimating it.
    :param eigen_solver: ``"dense"`` (exact, O(N^3)), ``"sparse"`` (ARPACK on a
        k-NN sparsified affinity) or ``"auto"``.
    :param sparse_threshold: sample count above which ``"auto"`` picks sparse.
    """

    def __init__(
        self,
        min_num_spks: int = 1,
        max_num_spks: int = 14,
        pval: float = 0.02,
        min_pnum: int = 6,
        oracle_num: Optional[int] = None,
        eigen_solver: str = "auto",
        sparse_threshold: int = DEFAULT_SPARSE_THRESHOLD,
    ):
        if min_num_spks < 1:
            raise ValueError("min_num_spks must be >= 1")
        if max_num_spks < min_num_spks:
            raise ValueError(
                f"max_num_spks ({max_num_spks}) must be >= min_num_spks ({min_num_spks})"
            )
        if eigen_solver not in {"auto", "dense", "sparse"}:
            raise ValueError(f"unknown eigen_solver: {eigen_solver!r}")

        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.min_pnum = min_pnum
        self.pval = pval
        self.k = oracle_num
        self.eigen_solver = eigen_solver
        self.sparse_threshold = sparse_threshold
        #: Populated on each call; useful for tests and progress reporting.
        self.last_num_spks: Optional[int] = None
        self.last_solver: Optional[str] = None

    def __call__(self, X, pval=None, oracle_num=None):
        sim_mat = self.get_sim_mat(X)
        keep = self.prune_mask(sim_mat, pval)
        pruned = np.where(keep, sim_mat, 0.0)
        sym = 0.5 * (pruned + pruned.T)
        emb, num_of_spk = self.get_spec_embs_from_matrix(sym, oracle_num)
        labels = self.cluster_embs(emb, num_of_spk)
        return labels

    # --- affinity -----------------------------------------------------------

    def get_sim_mat(self, X):
        return cosine_similarity(X, X)

    def _n_removed(self, n_rows: int, pval: float) -> int:
        n_elems = int((1 - pval) * n_rows)
        return max(0, min(n_elems, n_rows - self.min_pnum))

    def prune_mask(self, A: np.ndarray, pval=None) -> np.ndarray:
        """Boolean mask of the entries kept by ``p_pruning``.

        Row-wise ``argsort`` on a 2-D array is exactly what the original
        per-row loop did, minus the interpreter overhead.
        """
        if pval is None:
            pval = self.pval
        n_removed = self._n_removed(A.shape[0], pval)
        mask = np.ones(A.shape, dtype=bool)
        if n_removed <= 0:
            return mask
        drop = np.argsort(A, axis=1)[:, :n_removed]
        np.put_along_axis(mask, drop, False, axis=1)
        return mask

    def p_pruning(self, A, pval=None):
        """Return a copy of *A* with the smallest ``(1 - pval)`` fraction of each
        row replaced by zeros."""
        return A * self.prune_mask(A, pval)

    # --- Laplacian / eigen decomposition ------------------------------------

    @staticmethod
    def get_laplacian(M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        return D - M

    def _resolve_solver(self, n: int) -> str:
        if self.eigen_solver == "auto":
            return "dense" if n <= self.sparse_threshold else "sparse"
        return self.eigen_solver

    def _n_eig(self, oracle_num: Optional[int]) -> int:
        target = oracle_num if oracle_num is not None else self.k
        base = self.max_num_spks + 1
        return max(1, base, (target or 0) + 1)

    def get_spec_embs_from_matrix(self, sym_affinity: np.ndarray, oracle_num=None):
        """Spectral embedding of an already-symmetrised affinity matrix.

        Both solvers consume the *same* pruned affinity matrix, so the only
        difference between them is how the eigenpairs are computed.
        """
        n = sym_affinity.shape[0]
        solver = self._resolve_solver(n)
        k_oracle = oracle_num if oracle_num is not None else self.k

        if solver == "dense":
            laplacian = self.get_laplacian(sym_affinity)
            lambdas, eig_vecs = scipy.linalg.eigh(laplacian)
        else:
            laplacian = self._sparse_laplacian(sym_affinity)
            n_eig = min(self._n_eig(oracle_num), n - 1)
            lambdas, eig_vecs = spla.eigsh(laplacian, k=n_eig, which="SA")
            order = np.argsort(lambdas)
            lambdas, eig_vecs = lambdas[order], eig_vecs[:, order]

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            hi = min(self.max_num_spks + 1, len(lambdas))
            gap_list = self.getEigenGaps(lambdas[self.min_num_spks - 1 : hi])
            if not gap_list:
                num_of_spk = self.min_num_spks
            else:
                num_of_spk = int(np.argmax(gap_list)) + self.min_num_spks

        num_of_spk = int(min(max(num_of_spk, 1), eig_vecs.shape[1]))
        self.last_num_spks = num_of_spk
        self.last_solver = solver
        return eig_vecs[:, :num_of_spk], num_of_spk

    def get_spec_embs(self, L, k_oracle=None):
        """Kept for API compatibility; ``L`` is treated as a dense Laplacian."""
        laplacian = L.copy()
        if not np.allclose(laplacian, laplacian.T, atol=1e-8):
            laplacian = 0.5 * (laplacian + laplacian.T)
        lambdas, eig_vecs = scipy.linalg.eigh(laplacian)
        k_oracle = k_oracle if k_oracle is not None else self.k
        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            hi = min(self.max_num_spks + 1, len(lambdas))
            gap_list = self.getEigenGaps(lambdas[self.min_num_spks - 1 : hi])
            num_of_spk = (int(np.argmax(gap_list)) + self.min_num_spks) if gap_list else 1
        self.last_num_spks = int(num_of_spk)
        self.last_solver = "dense"
        return eig_vecs[:, :num_of_spk], num_of_spk

    @staticmethod
    def _sparse_laplacian(sym_affinity: np.ndarray) -> sp.csr_matrix:
        """Sparsify an already-pruned affinity matrix and form L = D - M.

        Note this still needs the dense N x N affinity in memory: ``p_pruning``
        keeps a fixed fraction of *every* row, so the structure is not k-NN
        sparse. The saving is the eigensolve, not the footprint.
        """
        M = sp.csr_matrix(sym_affinity)
        M.setdiag(0)
        M.eliminate_zeros()
        degrees = np.asarray(np.abs(M).sum(axis=1)).ravel()
        return (sp.diags(degrees) - M).tocsr()

    @staticmethod
    def cluster_embs(emb, k):
        _, labels, _ = k_means(emb, k, n_init="auto")
        return labels

    @staticmethod
    def getEigenGaps(eig_vals):
        return [float(eig_vals[i + 1]) - float(eig_vals[i]) for i in range(len(eig_vals) - 1)]


class UmapHdbscan:
    """UMAP projection followed by HDBSCAN.

    Reference: Siqi Zheng, Hongbin Suo. *Reformulating Speaker Diarization as
    Community Detection With Emphasis On Topological Structure*, ICASSP 2022.

    ``random_state`` is new and defaults to a fixed value: the upstream code
    produced a different partition on every run, which made results
    unreproducible. Pass ``random_state=None`` for the old behaviour.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        n_components: int = 60,
        min_samples: int = 20,
        min_cluster_size: int = 10,
        metric: str = "euclidean",
        random_state: Optional[int] = 42,
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.random_state = random_state

    def __call__(self, X):
        import hdbscan
        import umap

        n = X.shape[0]
        umap_X = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=0.0,
            n_components=max(2, min(self.n_components, n - 2)),
            metric=self.metric,
            random_state=self.random_state,
        ).fit_transform(X)
        return hdbscan.HDBSCAN(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
        ).fit_predict(umap_X)


class CommonClustering:
    """Cluster embeddings and return integer labels.

    :param cluster_type: ``"spectral"`` or ``"umap_hdbscan"``.
    :param mer_cos: cosine threshold above which two cluster centroids are
        merged. This is the parameter the CLI used to accept and silently drop.
    """

    def __init__(
        self,
        cluster_type: str,
        cluster_line: int = 10,
        mer_cos: Optional[float] = None,
        min_cluster_size: int = 4,
        **kwargs,
    ):
        self.cluster_type = cluster_type
        self.cluster_line = cluster_line
        self.min_cluster_size = min_cluster_size
        if mer_cos is not None:
            mer_cos = float(mer_cos)
            if not 0 < mer_cos <= 1:
                raise ValueError("mer_cos must be in (0, 1]")
        self.mer_cos = mer_cos

        if cluster_type == "spectral":
            self.cluster = SpectralCluster(**kwargs)
        elif cluster_type == "umap_hdbscan":
            for key in (
                "min_num_spks", "max_num_spks", "pval", "min_pnum",
                "oracle_num", "eigen_solver", "sparse_threshold",
            ):
                kwargs.pop(key, None)
            self.cluster = UmapHdbscan(min_cluster_size=min_cluster_size, **kwargs)
        else:
            raise ValueError(f"{cluster_type} is not currently supported.")

    def __call__(self, X):
        assert len(X.shape) == 2, "Shape of input should be [N, C]"
        if X.shape[0] < self.cluster_line:
            return np.ones(X.shape[0], dtype=int)

        labels = np.asarray(self.cluster(X))
        labels = self.filter_minor_cluster(labels, X, self.min_cluster_size)
        if self.mer_cos is not None:
            labels = self.merge_by_cos(labels, X, self.mer_cos)
        return labels

    def filter_minor_cluster(self, labels, x, min_cluster_size):
        cset = np.unique(labels)
        csize = np.array([(labels == i).sum() for i in cset])
        minor_idx = np.where(csize < min_cluster_size)[0]
        if len(minor_idx) == 0:
            return labels

        minor_cset = cset[minor_idx]
        major_idx = np.where(csize >= min_cluster_size)[0]
        major_cset = cset[major_idx]
        if len(major_cset) == 0:
            return labels
        major_center = np.stack([x[labels == i].mean(0) for i in major_cset])
        for i in range(len(labels)):
            if labels[i] in minor_cset:
                cos_sim = cosine_similarity(x[i][np.newaxis], major_center)
                labels[i] = major_cset[cos_sim.argmax()]
        return labels

    def merge_by_cos(self, labels, x, cos_thr):
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            cset = np.unique(labels)
            if len(cset) == 1:
                break
            centers = np.stack([x[labels == i].mean(0) for i in cset])
            affinity = cosine_similarity(centers, centers)
            affinity = np.triu(affinity, 1)
            idx = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[idx] < cos_thr:
                break
            c1, c2 = cset[np.array(idx)]
            labels[labels == c2] = c1
        return labels
