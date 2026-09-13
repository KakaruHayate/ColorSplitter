"""Workspace-local test bootstrap.

The shared interpreter's own pytest install is damaged, and repairing it means
writing outside the workspace. This keeps a working pytest in ``.tools/`` and
puts it at the front of ``sys.path`` before importing, so the broken copy is
never reached.

Usage::

    python scripts/run_tests.py -q -m "not slow" tests
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
for entry in (ROOT, TOOLS):
    if entry.exists():
        sys.path.insert(0, str(entry))

import pytest  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:] or ["tests"]
    raise SystemExit(pytest.main(args))
