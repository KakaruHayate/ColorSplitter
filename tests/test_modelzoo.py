"""Weight registry, hub-mirror fallback and download plumbing.

Nothing here touches the network: URL construction, ordering and local
resolution are all testable on their own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from colorsplitter.core import modelzoo as M


# --- the shipped registry --------------------------------------------------


def test_shipped_registry_is_valid() -> None:
    registry = M.load_registry()
    assert registry.encoders, "registry must define at least one encoder"
    ids = [e.id for e in registry.encoders]
    assert len(ids) == len(set(ids)), "weight ids must be unique"


def test_shipped_registry_has_exactly_one_default_timbre_weight() -> None:
    registry = M.load_registry()
    defaults = [e for e in registry.encoders if e.purpose == "timbre" and e.is_default]
    assert len(defaults) == 1
    assert registry.default_entry("timbre").id == defaults[0].id


def test_shipped_registry_distinguishes_purposes() -> None:
    """`timbre` and `speaker` answer different questions and must both exist."""
    registry = M.load_registry()
    purposes = {e.purpose for e in registry.encoders}
    assert {"timbre", "speaker"} <= purposes


def test_shipped_registry_records_provenance_for_archived_weights() -> None:
    """The archived weight must be recoverable from the record alone."""
    registry = M.load_registry()
    entry = registry.default_entry("timbre")
    source = entry.raw.get("source") or {}
    assert source.get("blob"), "expected a git blob id so the weight can be recovered"
    assert len(source.get("sha256", "")) == 64
    assert source.get("bytes", 0) > 1_000_000


def test_shipped_registry_emotion_target_is_declared() -> None:
    registry = M.load_registry()
    emotion = registry.emotion
    assert emotion.get("repo")
    assert emotion.get("target")
    assert "model.safetensors" in emotion.get("files", [])


def test_unknown_weight_id_lists_the_known_ones() -> None:
    registry = M.load_registry()
    with pytest.raises(KeyError) as excinfo:
        registry.by_id("nope")
    assert "timbre-v1" in str(excinfo.value)


# --- hub mirror fallback ---------------------------------------------------


def test_mirror_is_tried_before_upstream(monkeypatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    endpoints = M.hf_endpoints(["https://hf-mirror.com"])
    assert endpoints[0] == "https://hf-mirror.com"
    assert endpoints.index("https://huggingface.co") > 0


def test_hf_endpoint_env_var_wins(monkeypatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://my.mirror")
    assert M.hf_endpoints(["https://hf-mirror.com"])[0] == "https://my.mirror"


def test_default_mirror_is_present_even_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert "https://hf-mirror.com" in M.hf_endpoints([])


def test_no_duplicate_endpoints(monkeypatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    endpoints = M.hf_endpoints(["https://hf-mirror.com", "https://hf-mirror.com"])
    assert len(endpoints) == len(set(endpoints))


def test_hub_urls_are_built_for_every_endpoint() -> None:
    registry = M.load_registry()
    urls = M._hub_urls(registry.endpoints(), "some/repo", "main", "model.safetensors")
    assert any("hf-mirror.com" in u for u in urls)
    assert any(u.startswith("https://huggingface.co/") for u in urls)
    assert all(u.endswith("/some/repo/resolve/main/model.safetensors") for u in urls)


def test_remembered_host_is_tried_first(tmp_path) -> None:
    urls = ["https://huggingface.co/r/f", "https://hf-mirror.com/r/f"]
    M._remember_host(tmp_path, urls[1])
    assert M._order_urls(urls, tmp_path)[0] == urls[1]
    assert M._host_memory_file(tmp_path).exists()


def test_ordering_is_a_noop_without_a_memory(tmp_path) -> None:
    urls = ["https://a.example/f", "https://b.example/f"]
    assert M._order_urls(urls, tmp_path) == urls


# --- resolution ------------------------------------------------------------


def test_resolve_weight_prefers_a_local_file(tiny_registry, tmp_path) -> None:
    registry = M.load_registry(tiny_registry)
    resolved = M.resolve_weight(registry, "a", cache_dir=tmp_path)
    assert resolved == tmp_path / "a.pt"


def test_resolve_weight_falls_back_to_the_default_entry(tiny_registry, tmp_path) -> None:
    registry = M.load_registry(tiny_registry)
    assert M.resolve_weight(registry, purpose="timbre", cache_dir=tmp_path).name == "a.pt"


def test_resolve_weight_refuses_to_download_when_disabled(tiny_registry, tmp_path) -> None:
    registry = M.load_registry(tiny_registry)
    with pytest.raises(FileNotFoundError):
        M.resolve_weight(registry, "b", cache_dir=tmp_path / "empty", allow_download=False)


def test_emotion_fetch_refuses_when_disabled(tiny_registry, tmp_path) -> None:
    registry = M.load_registry(tiny_registry)
    with pytest.raises(FileNotFoundError) as excinfo:
        M.fetch_emotion_model(registry, cache_dir=tmp_path, allow_download=False)
    assert "config.json" in str(excinfo.value)


def test_registry_without_emotion_entry_raises(tmp_path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"encoders": []}), encoding="utf-8")
    with pytest.raises(KeyError):
        M.fetch_emotion_model(M.load_registry(path), cache_dir=tmp_path)


def test_missing_registry_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        M.load_registry(tmp_path / "nope.json")


# --- hashing ---------------------------------------------------------------


def test_sha256_of_matches_hashlib(tmp_path) -> None:
    import hashlib

    path = tmp_path / "payload.bin"
    path.write_bytes(bytes(range(256)) * 8)
    assert M.sha256_of(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sha256_reports_progress(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"x" * 5000)
    seen = []
    M.sha256_of(path, progress=lambda done, total: seen.append((done, total)))
    assert seen and seen[-1][0] == seen[-1][1]


# --- cache dir override ---------------------------------------------------


def test_cache_dir_honours_the_env_var(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COLORSPLITTER_HOME", str(tmp_path))
    assert M.default_cache_dir() == tmp_path / "weights"


def test_cache_dir_is_per_user_by_default(monkeypatch) -> None:
    monkeypatch.delenv("COLORSPLITTER_HOME", raising=False)
    resolved = M.default_cache_dir()
    assert resolved.is_absolute()
    assert "colorsplitter" in str(resolved)
