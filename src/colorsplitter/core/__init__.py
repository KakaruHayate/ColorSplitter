"""Core capabilities: scanning, embedding, clustering, projection, export.

Submodules are imported explicitly by callers (``from colorsplitter.core import
pipeline``) so that importing the package never pulls in torch, and so that
``core`` and ``models`` cannot form an import cycle.
"""

from __future__ import annotations

__all__ = [
    "audio",
    "cache",
    "cluster",
    "labels",
    "modelzoo",
    "pipeline",
    "reduce",
    "types",
]
