"""ColorSplitter — timbre clustering and filtering for singing datasets.

The pipeline is: scan a directory of audio, embed each file with a speaker
encoder, cluster the embeddings, project them to 2D for review, edit the
clusters interactively, then export the result.
"""

from __future__ import annotations

__version__ = "2.0.0.dev0"

__all__ = ["__version__"]
