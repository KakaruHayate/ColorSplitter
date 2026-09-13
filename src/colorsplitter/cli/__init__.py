"""Command-line entry point."""

from __future__ import annotations

__all__ = ["main"]


def main(argv=None) -> int:
    from .main import main as _main

    return _main(argv)
