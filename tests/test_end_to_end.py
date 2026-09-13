"""End-to-end runs, the CLI, and cache reuse.

A randomly initialised checkpoint is written into a throwaway weights cache so
the whole path — registry lookup, decoding, batching, clustering, projection,
export — executes without touching the network or any real model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from colorsplitter.cli.main import main  # noqa: E402
from colorsplitter.core.embed import EmbedConfig, embed_dataset  # noqa: E402
from colorsplitter.core.pipeline import (  # noqa: E402
    cluster_embeddings,
    export_clusters,
    project_embeddings,
    run,
    scan,
)

#: Name of the default weight in the registry; the fixture writes a file with
#: exactly this name so registry resolution finds it locally.
DEFAULT_WEIGHT_FILE = "timbre-v1.pt"


@pytest.fixture
def weights_home(tmp_path, monkeypatch) -> Path:
    """Place a synthetic checkpoint where the registry expects it."""
    from colorsplitter.models.voice_encoder import VoiceEncoder

    home = tmp_path / "cs-home"
    (home / "weights").mkdir(parents=True)
    model = VoiceEncoder(device="cpu", load_weights=False)
    torch.save({"model_state": model.state_dict(), "step": 1}, home / "weights" / DEFAULT_WEIGHT_FILE)
    monkeypatch.setenv("COLORSPLITTER_HOME", str(home))
    return home / "weights"


def _embed_config(cache_dir: Path) -> EmbedConfig:
    return EmbedConfig(
        encoder="timbre",
        device="cpu",
        batch_size=16,
        workers=1,
        trim_silences=False,
        cache_dir=cache_dir,
        chunk=64,
    )


# --- embedding -------------------------------------------------------------


@pytest.mark.slow
def test_embed_dataset_shape_and_order(synthetic_audio_root, weights_home) -> None:
    dataset = scan(synthetic_audio_root)
    embeddings = embed_dataset(dataset, _embed_config(weights_home))
    assert len(embeddings) == len(dataset)
    assert embeddings.keys == dataset.keys
    assert embeddings.embeds.shape == (len(dataset), 256)
    assert np.isfinite(embeddings.embeds).all()


@pytest.mark.slow
def test_embeddings_are_l2_normalised(synthetic_audio_root, weights_home) -> None:
    dataset = scan(synthetic_audio_root)
    embeds = embed_dataset(dataset, _embed_config(weights_home)).embeds
    norms = np.linalg.norm(embeds, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


@pytest.mark.slow
def test_batched_path_matches_single_utterance_path(synthetic_audio_root, weights_home) -> None:
    """The optimisation must not change the numbers it produces."""
    from colorsplitter.core.audio import preprocess_wav
    from colorsplitter.models.voice_encoder import VoiceEncoder

    dataset = scan(synthetic_audio_root)
    embeds = embed_dataset(dataset, _embed_config(weights_home)).embeds

    reference = VoiceEncoder(device="cpu", weights_fpath=weights_home / DEFAULT_WEIGHT_FILE)
    expected = np.stack(
        [reference.embed_utterance(preprocess_wav(item.path, trim_silences=False)) for item in dataset]
    )
    np.testing.assert_allclose(embeds, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_embedding_cache_is_reused(synthetic_audio_root, weights_home) -> None:
    dataset = scan(synthetic_audio_root)
    first = embed_dataset(dataset, _embed_config(weights_home))
    cache_files = list(weights_home.glob("embeddings-*.npz"))
    assert cache_files, "expected an embedding cache file"

    second = embed_dataset(dataset, _embed_config(weights_home))
    np.testing.assert_array_equal(first.embeds, second.embeds)


# --- full pipeline ---------------------------------------------------------


@pytest.mark.slow
def test_run_end_to_end(synthetic_audio_root, weights_home, tmp_path) -> None:
    result = run(
        synthetic_audio_root,
        output_dir=tmp_path / "out",
        embed_config=_embed_config(weights_home),
        cluster_method="spectral",
        nmin=2,
        projection_method="pca",
        export=True,
        cache_dir=weights_home,
    )
    assert len(result["dataset"]) == 15
    assert result["embeddings"].dim == 256
    assert len(result["clusters"]) == 15
    assert result["projection"].coords.shape == (15, 2)
    assert result["export"].written == 15
    assert (tmp_path / "out").is_dir()


@pytest.mark.slow
def test_run_rejects_an_empty_directory(tmp_path, weights_home) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no audio files"):
        run(empty, embed_config=_embed_config(weights_home), cache_dir=weights_home)


@pytest.mark.slow
def test_labels_can_be_edited_after_a_run(synthetic_audio_root, weights_home, tmp_path) -> None:
    """Edit-by-point then export: the flow the WebUI drives."""
    from colorsplitter.core.pipeline import dataset_from_paths
    from colorsplitter.core.types import ClusterResult

    dataset = scan(synthetic_audio_root)
    embeds = embed_dataset(dataset, _embed_config(weights_home)).embeds
    result = cluster_embeddings(embeds, method="spectral", nmin=2)

    from colorsplitter.core import labels as L

    edited = L.relabel_points(result.labels, [0, 1, 2], target=99)
    report = export_clusters(dataset, ClusterResult(edited, "spectral", {}), tmp_path / "edited", mode="copy")
    assert report.written == 15
    assert (tmp_path / "edited" / "099").is_dir()


# --- CLI -------------------------------------------------------------------


def test_cli_scan_reports_the_file_count(synthetic_audio_root, capsys) -> None:
    assert main(["scan", str(synthetic_audio_root), "--limit", "2"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out.splitlines()[0])
    assert payload["files"] == 15
    assert "one.wav" in out or "wav" in out


def test_cli_weights_list(capsys) -> None:
    assert main(["weights", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    ids = [entry["id"] for entry in payload]
    assert "timbre-v1" in ids
    defaults = [entry for entry in payload if entry["default"]]
    assert len(defaults) == 1


def test_cli_weights_list_honours_a_custom_registry(tiny_registry, capsys) -> None:
    assert main(["weights", "list", "--registry", str(tiny_registry)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [entry["id"] for entry in payload] == ["a", "b"]


@pytest.mark.slow
def test_cli_run_writes_a_cluster_csv(synthetic_audio_root, weights_home, tmp_path, capsys) -> None:
    out_dir = tmp_path / "cli-out"
    code = main(
        [
            "run", str(synthetic_audio_root),
            "--encoder", "timbre",
            "--device", "cpu",
            "--projection", "pca",
            "--cluster", "spectral",
            "--nmin", "2",
            "--workers", "1",
            "--trim-silences", "no",
            "--cache-dir", str(weights_home),
            "--output", str(out_dir),
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["files"] == 15
    # A randomly initialised encoder produces unstructured embeddings, so the
    # cluster count itself is not meaningful here. What matters is that the run
    # completed and wrote one row per file.
    assert summary["n_clusters"] >= 1

    csv_path = out_dir / "clusters.csv"
    assert csv_path.exists()
    header, *rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert header == "key,cluster,x,y"
    assert len(rows) == 15


@pytest.mark.slow
def test_cli_export_move_relocates(synthetic_audio_root, weights_home, tmp_path) -> None:
    out_dir = tmp_path / "moved"
    assert (
        main(
            [
                "run", str(synthetic_audio_root),
                "--device", "cpu", "--projection", "pca", "--nmin", "2",
                "--workers", "1", "--trim-silences", "no",
                "--cache-dir", str(weights_home),
                "--output", str(out_dir),
                "--export", "--export-mode", "move",
            ]
        )
        == 0
    )
    assert list(synthetic_audio_root.rglob("*.wav")) == []
    assert len(list(out_dir.rglob("*.wav"))) == 15
