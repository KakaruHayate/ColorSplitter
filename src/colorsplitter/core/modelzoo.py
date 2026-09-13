"""Model registry and weight acquisition.

Weights are **not** stored in the git repository. ``models/registry.json``
records, for each weight, what it is, what it is for, how it was trained, its
step count, its SHA-256 and where to fetch it; everything is downloaded on
demand into a local cache.

Design notes
------------
* Weights used to be committed directly, which meant a weight swap looked like
  an ordinary source edit and could silently overwrite a good checkpoint. That
  is exactly what happened once; the registry plus hashes makes it visible.
* Downloads verify SHA-256 and resume, because these files are hundreds of
  megabytes.
* Hub mirror fallback: ``huggingface.co`` is unreachable from some networks
  (through a blocking proxy it fails outright), so every hub asset has an
  ordered candidate list — ``HF_ENDPOINT`` if set, then the configured mirrors.
  The host that last worked is remembered in the cache directory and tried
  first afterwards, so a broken primary costs one timeout rather than one per
  file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

__all__ = [
    "DEFAULT_MIRRORS",
    "Registry",
    "RegistryEntry",
    "default_cache_dir",
    "default_registry_path",
    "download_file",
    "fetch_emotion_model",
    "hf_endpoints",
    "load_registry",
    "pack_inference_weights",
    "resolve_weight",
    "sha256_of",
]

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]
_CHUNK = 1 << 20
_USER_AGENT = "colorsplitter"

#: Tried after the primary hub host when it is unreachable.
DEFAULT_MIRRORS = ("https://hf-mirror.com",)

#: Host used for the first attempt; kept short so a dead primary fails fast.
_PROBE_TIMEOUT = 15
_FULL_TIMEOUT = 120


# --- locations --------------------------------------------------------------


def default_cache_dir() -> Path:
    """Where downloaded weights live.

    Override with ``COLORSPLITTER_HOME``. Falls back to a per-user cache dir so
    the repository stays free of binaries.
    """
    env = os.environ.get("COLORSPLITTER_HOME")
    if env:
        return Path(env).expanduser() / "weights"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "colorsplitter" / "weights"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "colorsplitter" / "weights"


def default_registry_path() -> Path:
    """``models/registry.json`` shipped with the repo."""
    env = os.environ.get("COLORSPLITTER_REGISTRY")
    if env:
        return Path(env).expanduser()
    for candidate in (
        Path(__file__).resolve().parents[3] / "models" / "registry.json",
        Path.cwd() / "models" / "registry.json",
    ):
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[1] / "data" / "registry.json"


# --- hub host resolution ----------------------------------------------------


def _host_of(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0]


def _host_memory_file(cache_dir: Optional[Path]) -> Optional[Path]:
    if cache_dir is None:
        return None
    return Path(cache_dir) / ".hf_host"


def _remembered_host(cache_dir: Optional[Path]) -> Optional[str]:
    path = _host_memory_file(cache_dir)
    if path is None or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _remember_host(cache_dir: Optional[Path], url: str) -> None:
    path = _host_memory_file(cache_dir)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_host_of(url), encoding="utf-8")
    except OSError:
        pass


def _dedupe(items: Iterable[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def hf_endpoints(mirrors: Sequence[str] = ()) -> list[str]:
    """Hub base URLs to try, in order, for the first attempt of a run.

    ``HF_ENDPOINT`` wins when set (the conventional override), then the mirrors,
    then the upstream host last so that a working mirror is always preferred.
    Duplicates are collapsed.
    """
    candidates: list[str] = []
    env = os.environ.get("HF_ENDPOINT")
    if env:
        candidates.append(env.rstrip("/"))
    candidates.extend(m.rstrip("/") for m in mirrors)
    candidates.extend(DEFAULT_MIRRORS)
    upstream = "https://huggingface.co"
    ordered = [c for c in _dedupe(candidates) if c != upstream]
    ordered.append(upstream)
    return ordered


def _order_urls(urls: Iterable[str], cache_dir: Optional[Path]) -> list[str]:
    """Stable-sort candidate URLs so the last host that worked comes first."""
    urls = [u for u in urls if u]
    preferred = _remembered_host(cache_dir)
    if not preferred:
        return urls
    return sorted(urls, key=lambda u: 0 if _host_of(u) == preferred else 1)


# --- registry ---------------------------------------------------------------


@dataclass
class RegistryEntry:
    """One downloadable weight."""

    id: str
    file: str
    purpose: str
    urls: list[str] = field(default_factory=list)
    sha256: Optional[str] = None
    step: Optional[int] = None
    default: bool = False
    raw: dict = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        return self.default

    def local_candidates(self, cache_dir: Path) -> list[Path]:
        """Places a pre-placed file is honoured from, before any download.

        ``cache_dir`` comes first so that an explicitly provided weight overrides
        the one shipped in ``pretrain/``.
        """
        return [
            cache_dir / self.file,
            Path("pretrain") / self.file,
            Path(self.file),
        ]


@dataclass
class Registry:
    """Parsed ``registry.json``."""

    path: Path
    encoders: list[RegistryEntry]
    downloads: dict = field(default_factory=dict)
    mirrors: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def by_id(self, weight_id: str) -> RegistryEntry:
        for entry in self.encoders:
            if entry.id == weight_id:
                return entry
        known = ", ".join(e.id for e in self.encoders)
        raise KeyError(f"unknown weight id {weight_id!r}; known ids: {known}")

    def default_entry(self, purpose: Optional[str] = None) -> RegistryEntry:
        candidates = [e for e in self.encoders if purpose is None or e.purpose == purpose]
        for entry in candidates:
            if entry.is_default:
                return entry
        if candidates:
            return candidates[0]
        raise KeyError(f"no registered weight for purpose={purpose!r}")

    @property
    def emotion(self) -> dict:
        return dict(self.downloads.get("emotion", {}))

    def endpoints(self) -> list[str]:
        return hf_endpoints(self.mirrors)


def load_registry(path: Optional[Path] = None) -> Registry:
    """Load and validate ``registry.json``."""
    path = Path(path) if path else default_registry_path()
    if not path.exists():
        raise FileNotFoundError(f"registry not found at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    encoders = []
    for item in raw.get("encoders", []):
        urls = list(item.get("urls", []))
        if item.get("url"):
            urls.insert(0, item["url"])
        encoders.append(
            RegistryEntry(
                id=item["id"],
                file=item["file"],
                purpose=item.get("purpose", "timbre"),
                urls=urls,
                sha256=item.get("sha256"),
                step=item.get("step"),
                default=bool(item.get("default", False)),
                raw=item,
            )
        )
    return Registry(
        path=path,
        encoders=encoders,
        downloads=raw.get("downloads", {}),
        mirrors=list(raw.get("mirrors", [])),
        raw=raw,
    )


# --- hashing / download -----------------------------------------------------


def sha256_of(path: Path, progress: Optional[ProgressFn] = None) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
            done += len(block)
            if progress is not None:
                progress(done, total)
    return h.hexdigest()


def _http_get(url: str, dest: Path, resume_from: int = 0, timeout: int = _FULL_TIMEOUT) -> None:
    """Stream ``url`` into ``dest``, resuming when possible."""
    headers = {"User-Agent": _USER_AGENT}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response, open(dest, "ab") as out:
        expected = int(response.headers.get("Content-Length") or 0)
        if resume_from and expected and resume_from + expected != _total_size(response, resume_from):
            # Server ignored the Range header; drop the partial and start over.
            raise urllib.error.ContentTooShortError(f"server ignored Range for {url}", None)
        while True:
            block = response.read(_CHUNK)
            if not block:
                break
            out.write(block)


def _total_size(response, resume_from: int) -> int:
    content_range = response.headers.get("Content-Range") or ""
    if "/" in content_range:
        try:
            return int(content_range.rsplit("/", 1)[1])
        except ValueError:
            pass
    return resume_from + int(response.headers.get("Content-Length") or 0)


def download_file(
    urls: Iterable[str],
    dest: Path,
    *,
    sha256: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    attempts_per_url: int = 2,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download the first working URL into *dest*.

    Candidate URLs are reordered so the host that worked last is tried first,
    and the winner is recorded for subsequent calls in the same cache.
    """
    ordered = _order_urls(urls, cache_dir)
    if not ordered:
        raise ValueError(f"no download URL available for {Path(dest).name}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None

    for position, url in enumerate(ordered):
        for attempt in range(1, attempts_per_url + 1):
            partial = dest.with_suffix(dest.suffix + ".part")
            resume_from = partial.stat().st_size if partial.exists() else 0
            # Give the first candidate a short leash: if the primary host is
            # black-holed, we want to reach the mirror quickly.
            timeout = _PROBE_TIMEOUT if (position == 0 and attempt == 1) else _FULL_TIMEOUT
            try:
                log.info("downloading %s from %s (attempt %d)", dest.name, _host_of(url), attempt)
                _http_get(url, partial, resume_from, timeout)
                partial.replace(dest)
                if sha256:
                    actual = sha256_of(dest)
                    if actual.lower() != sha256.lower():
                        dest.unlink(missing_ok=True)
                        raise ValueError(
                            f"sha256 mismatch for {dest.name}: expected {sha256}, got {actual}"
                        )
                _remember_host(cache_dir, url)
                if progress is not None:
                    size = dest.stat().st_size
                    progress(size, size)
                return dest
            except Exception as exc:  # noqa: BLE001 - try the next URL/attempt
                last_error = exc
                log.warning("download failed via %s: %s", _host_of(url), exc)
    raise RuntimeError(f"could not download {dest.name}: {last_error}")


def _hub_urls(endpoints: Sequence[str], repo: str, revision: str, filename: str) -> list[str]:
    return [f"{base.rstrip('/')}/{repo}/resolve/{revision}/{filename}" for base in endpoints]


# --- resolution -------------------------------------------------------------


def resolve_weight(
    registry: Registry,
    weight_id: Optional[str] = None,
    *,
    purpose: str = "timbre",
    cache_dir: Optional[Path] = None,
    allow_download: bool = True,
    verify: bool = True,
) -> Path:
    """Return a local path for a registered weight, downloading it if needed.

    A file placed directly at ``<cache_dir>/<file>`` is used as-is, which keeps
    offline workflows possible.
    """
    entry = registry.by_id(weight_id) if weight_id else registry.default_entry(purpose)
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()

    for candidate in entry.local_candidates(cache_dir):
        if candidate.exists():
            if verify and entry.sha256:
                actual = sha256_of(candidate)
                if actual.lower() != entry.sha256.lower():
                    log.warning(
                        "local weight %s does not match the registered hash; using it anyway",
                        candidate,
                    )
            return candidate

    if not allow_download:
        raise FileNotFoundError(
            f"weight {entry.id!r} is not cached and downloads are disabled (looked in {cache_dir})"
        )

    urls = list(entry.urls)
    if not urls:
        urls = [f"{registry.path.parent / '..' / entry.file}"]
    return download_file(
        urls,
        cache_dir / entry.file,
        sha256=entry.sha256 if verify else None,
        cache_dir=cache_dir,
    )


def fetch_emotion_model(
    registry: Registry,
    *,
    cache_dir: Optional[Path] = None,
    allow_download: bool = True,
    progress: Optional[ProgressFn] = None,
) -> Path:
    """Download every file of the emotion model and return its directory."""
    spec = registry.emotion
    if not spec:
        raise KeyError("registry has no 'emotion' download entry")

    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    target = cache_dir / spec.get("target", "emotion")
    repo = spec["repo"]
    revision = spec.get("revision", "main")
    targets = list(spec.get("files", ["config.json", "preprocessor_config.json", "model.safetensors"]))

    missing = [name for name in targets if not (target / name).exists()]
    if missing and not allow_download:
        raise FileNotFoundError(f"emotion model incomplete at {target}: missing {missing}")

    endpoints = registry.endpoints()
    for name in missing:
        download_file(
            _hub_urls(endpoints, repo, revision, name),
            target / name,
            cache_dir=cache_dir,
            progress=progress,
        )
    return target


# --- packaging --------------------------------------------------------------


def pack_inference_weights(
    source: Path,
    dest: Path,
    *,
    weights_id: Optional[str] = None,
    step: Optional[int] = None,
) -> Path:
    """Strip everything but ``model_state`` from a training checkpoint.

    Training checkpoints carry optimiser state, which roughly doubles their size
    and is useless for inference. Requires torch.
    """
    import torch

    source, dest = Path(source), Path(dest)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"{source} does not look like a training checkpoint")

    payload = {"model_state": checkpoint["model_state"]}
    if weights_id is not None:
        payload["id"] = weights_id
    if step is not None:
        payload["step"] = int(step)
    elif "step" in checkpoint:
        payload["step"] = int(checkpoint["step"])

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".pt", dir=str(dest.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    torch.save(payload, tmp)
    shutil.move(str(tmp), str(dest))
    log.info("wrote inference weights to %s (%d bytes)", dest, dest.stat().st_size)
    return dest
