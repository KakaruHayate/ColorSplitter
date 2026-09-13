"""Verbatim copy of the clustering implementation as it shipped before the
refactor.

Kept only as the reference for ``tests/test_cluster_equivalence.py``. It is
never imported by the package: if it were, the thing being compared against
could drift with the thing being compared.

Source: ``modules/cluster.py`` at the commit this refactor branched from,
including the per-row pruning loop and the hard-coded ``max_num_spks=14``.
"""

from __future__ import annotations

import numpy as np
import scipy
from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity


class LegacySpectralCluster:
    def __init__(self, min_num_spks=1, max_num_spks=14, pval=0.02, min_pnum=6, oracle_num=None):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.min_pnum = min_pnum
        self.pval = pval
        self.k = oracle_num

    def __call__(self, X, pval=None, oracle_num=None):
        sim_mat = self.get_sim_mat(X)
        prunned_sim_mat = self.p_pruning(sim_mat, pval)
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)
        laplacian = self.get_laplacian(sym_prund_sim_mat)
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)
        labels = self.cluster_embs(emb, num_of_spk)
        return labels

    def get_sim_mat(self, X):
        return cosine_similarity(X, X)

    def p_pruning(self, A, pval=None):
        if pval is None:
            pval = self.pval
        n_elems = int((1 - pval) * A.shape[0])
        n_elems = min(n_elems, A.shape[0] - self.min_pnum)

        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=None):
        if k_oracle is None:
            k_oracle = self.k
        lambdas, eig_vecs = scipy.linalg.eigh(L)
        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1 : self.max_num_spks + 1]
            )
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks
        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        _, labels, _ = k_means(emb, k, n_init="auto")
        return labels

    def getEigenGaps(self, eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


class LegacyCommonClustering:
    def __init__(self, cluster_type, cluster_line=10, mer_cos=None, min_cluster_size=4, **kwargs):
        self.cluster_type = cluster_type
        self.cluster_line = cluster_line
        self.min_cluster_size = min_cluster_size
        self.mer_cos = mer_cos
        if self.cluster_type == "spectral":
            self.cluster = LegacySpectralCluster(**kwargs)
        else:
            raise ValueError("%s is not currently supported." % self.cluster_type)

    def __call__(self, X):
        assert len(X.shape) == 2, "Shape of input should be [N, C]"
        if X.shape[0] < self.cluster_line:
            return np.ones(X.shape[0], dtype=int)
        labels = self.cluster(X)
        labels = self.filter_minor_cluster(labels, X, self.min_cluster_size)
        if self.mer_cos is not None:
            labels = self.merge_by_cos(labels, X, self.mer_cos)
        return labels

    def filter_minor_cluster(self, labels, x, min_cluster_size):
        cset = np.unique(labels)
        csize = np.array([(labels == i).sum() for i in cset])
        minor_idx = np.where(csize < self.min_cluster_size)[0]
        if len(minor_idx) == 0:
            return labels
        minor_cset = cset[minor_idx]
        major_idx = np.where(csize >= self.min_cluster_size)[0]
        major_cset = cset[major_idx]
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
