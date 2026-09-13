"""Dimensionality reduction for display.

The reducers themselves are unchanged (t-SNE / UMAP as originally chosen). Two
things are added: a fixed ``random_state`` so a picture can be reproduced, and
PCA as a cheap preview option. Results are cached by the caller through
:class:`~colorsplitter.core.cache.ArrayCache`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import Projection

__all__ = ["PROJECTION_METHODS", "project"]

#: Supported ``method`` values for :func:`project`.
PROJECTION_METHODS = ("tsne", "umap", "pca")


def project(
    embeds: np.ndarray,
    method: str = "tsne",
    *,
    random_state: int | None = 42,
    **kwargs: Any,
) -> Projection:
    """Reduce *embeds* to two dimensions.

    :param method: ``"tsne"``, ``"umap"`` or ``"pca"``.
    :param kwargs: forwarded to the underlying reducer. Defaults mirror the
        original implementation (``TSNE(init="pca")``, ``UMAP()``).
    """
    X = np.asarray(embeds, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("embeds must be a 2-D array")
    if X.shape[0] < 3:
        raise ValueError("need at least 3 embeddings to project")

    if method == "tsne":
        from sklearn.manifold import TSNE

        # perplexity must stay below the sample count
        perplexity = kwargs.pop("perplexity", min(30.0, max(5.0, (X.shape[0] - 1) / 3.0)))
        coords = TSNE(
            init="pca", perplexity=perplexity, random_state=random_state, **kwargs
        ).fit_transform(X)
    elif method == "umap":
        import umap

        coords = umap.UMAP(random_state=random_state, **kwargs).fit_transform(X)
    elif method == "pca":
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=random_state, **kwargs).fit_transform(X)
    else:
        raise ValueError(f"unknown projection method: {method!r}")

    return Projection(np.asarray(coords, dtype=np.float64), method)
