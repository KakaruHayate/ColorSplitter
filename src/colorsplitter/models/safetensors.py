"""Minimal ``safetensors`` reader.

safetensors is a trivial format: an 8-byte little-endian header length, a JSON
header mapping tensor names to ``{dtype, shape, data_offsets}``, then one
contiguous little-endian payload starting right after the header. Reading it
needs nothing beyond numpy, so we can consume HF-hosted weights without the
``safetensors`` package and without ever unpickling a file we did not create.

Only the dtypes we actually encounter are supported; anything else raises rather
than silently reinterpreting bytes.
"""

from __future__ import annotations

import json
import mmap
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

__all__ = ["load_safetensors", "read_header", "iter_tensors"]

_DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}

_PREFIX = struct.Struct("<Q")


def _read_prefix(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (length,) = _PREFIX.unpack(raw)
        header = json.loads(fh.read(length).decode("utf-8"))
    header.pop("__metadata__", None)
    return int(length), header


def read_header(path: Path) -> dict:
    """Parse just the JSON header of a safetensors file."""
    return _read_prefix(Path(path))[1]


def iter_tensors(path: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` pairs, reading the payload through mmap.

    Each tensor is copied out of the mapping before being yielded, so no numpy
    view outlives the ``mmap`` and there is no risk of a ``BufferError`` on
    close. Peak memory is the model size, which is unavoidable once the tensors
    are meant to be resident.
    """
    path = Path(path)
    header_length, header = _read_prefix(path)
    payload_offset = 8 + header_length

    with open(path, "rb") as fh:
        payload = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for name, info in header.items():
                np_dtype = _DTYPES.get(info["dtype"])
                if np_dtype is None:
                    raise NotImplementedError(f"unsupported safetensors dtype {info['dtype']!r}")
                start, end = info["data_offsets"]
                block = payload[payload_offset + start : payload_offset + end]
                array = np.frombuffer(block, dtype=np_dtype).reshape(info["shape"]).copy()
                yield name, torch.from_numpy(array)
        finally:
            payload.close()


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Load every tensor into a plain dict."""
    return dict(iter_tensors(Path(path)))
