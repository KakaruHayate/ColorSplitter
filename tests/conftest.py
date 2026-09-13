"""Shared fixtures.

Audio is generated rather than shipped: the suite must not depend on any
external dataset, and it must not encode anything about one.
"""

from __future__ import annotations

import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def write_wav(path: Path, *, seconds: float = 1.0, sr: int = 16000, freq: float = 220.0) -> Path:
    """Write a small mono 16-bit PCM wav using only the standard library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    signal = 0.3 * np.sin(2 * math.pi * freq * t) * np.hanning(n)
    pcm = (signal * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def synthetic_audio_root(tmp_path: Path) -> Path:
    """A directory tree shaped like ``<singer>_<timbre>/<clip>.wav``.

    Three singers; the first has three timbres (the hard-negative case), the
    others one each. Clips differ in pitch so embeddings are not degenerate.
    """
    root = tmp_path / "audio"
    layout = {
        "alpha_soft": 220.0,
        "alpha_bright": 330.0,
        "alpha_power": 440.0,
        "beta_soft": 200.0,
        "gamma_bright": 500.0,
    }
    for offset, (name, freq) in enumerate(layout.items()):
        for clip in range(3):
            write_wav(root / name / f"{name}_{clip}.wav", seconds=0.8, freq=freq + clip * 15)
    return root


@pytest.fixture
def shuffled_audio_root(tmp_path: Path) -> Path:
    """Same shape but nested one level deeper, plus multiple extensions."""
    root = tmp_path / "nested"
    write_wav(root / "s1_a" / "x" / "one.wav", freq=200.0)
    write_wav(root / "s1_b" / "y" / "two.wav", freq=300.0)
    write_wav(root / "s2_a" / "z" / "three.wav", freq=400.0)
    return root


@pytest.fixture
def toy_embeds() -> np.ndarray:
    """Three well-separated blocks of embeddings, deterministic.

    Built around three *directions*, not three offsets. The clustering is
    cosine-based, and cosine similarity is meaningless for vectors around the
    origin — a block centred on zero has essentially random internal angles, so
    a fixture written that way tests the fixture rather than the algorithm.
    """
    rng = np.random.default_rng(0)
    dim = 16
    directions = np.eye(dim)[:3] + 0.15 * rng.normal(size=(3, dim))
    blocks = [d * 4.0 + rng.normal(0.0, 0.25, size=(25, dim)) for d in directions]
    return np.vstack(blocks).astype(np.float64)


@pytest.fixture
def tiny_registry(tmp_path: Path):
    """A registry pointing at local files, so no network is touched."""
    import json

    (tmp_path / "a.pt").write_bytes(b"first local weight")
    (tmp_path / "b.pt").write_bytes(b"second local weight")
    registry = {
        "version": 2,
        "mirrors": ["https://hf-mirror.com"],
        "encoders": [
            {
                "id": "a",
                "file": "a.pt",
                "purpose": "timbre",
                "default": True,
                "step": 10,
                "urls": ["https://example.invalid/a.pt"],
                "sha256": None,
            },
            {
                "id": "b",
                "file": "b.pt",
                "purpose": "timbre",
                "default": False,
                "step": 20,
                "urls": [],
                "sha256": None,
            },
        ],
        "downloads": {
            "emotion": {
                "repo": "some/repo",
                "revision": "main",
                "target": "emotion",
                "files": ["config.json"],
            }
        },
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path
