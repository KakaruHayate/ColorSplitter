"""Training smoke test.

This is the acceptance check for "training is delivered as working code": a few
steps run to completion on generated audio, and the checkpoint it writes loads
back into the inference encoder. It is not a claim that a useful model comes out
of three steps on fifteen clips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from colorsplitter.training.train import TrainConfig, train  # noqa: E402


def _config(dataset: Path, output_dir: Path, steps: int = 3) -> TrainConfig:
    return TrainConfig(
        dataset=dataset,
        output_dir=output_dir,
        val_fraction=0.0,
        speakers_per_batch=3,
        utterances_per_speaker=2,
        singers_per_batch=3,
        max_steps=steps,
        lr=1e-3,
        device="cpu",
        seed=0,
        log_every=1,
        checkpoint_every=steps,
        val_every=1000,
        val_batches=1,
    )


@pytest.mark.slow
def test_training_runs_and_writes_a_usable_checkpoint(synthetic_audio_root, tmp_path) -> None:
    from colorsplitter.models.voice_encoder import VoiceEncoder

    output_dir = tmp_path / "run"
    summary = train(_config(synthetic_audio_root, output_dir))

    assert summary["steps"] == 3
    assert summary["dataset"]["classes"] == 5
    assert summary["device"] == "cpu"

    latest = output_dir / "latest.pt"
    assert latest.exists()
    assert (output_dir / "run.json").exists()
    assert (output_dir / "config.json").exists()

    # The acceptance criterion: inference can consume it directly.
    encoder = VoiceEncoder(device="cpu", weights_fpath=latest)
    assert sum(p.numel() for p in encoder.parameters()) > 0


@pytest.mark.slow
def test_training_loss_is_finite_and_recorded(synthetic_audio_root, tmp_path) -> None:
    summary = train(_config(synthetic_audio_root, tmp_path / "run", steps=4))
    assert summary["loss"] == summary["loss"]  # not NaN
    assert all(entry["loss"] == entry["loss"] for entry in summary["history"])


@pytest.mark.slow
def test_checkpointing_is_off_by_default_in_the_smoke_config(
    synthetic_audio_root, tmp_path
) -> None:
    """Only ``latest.pt`` and the final step file should be produced."""
    output_dir = tmp_path / "run"
    train(_config(synthetic_audio_root, output_dir, steps=3))
    assert sorted(p.name for p in output_dir.glob("*.pt")) == ["encoder_3.pt", "latest.pt"]


@pytest.mark.slow
def test_training_is_resumable(synthetic_audio_root, tmp_path) -> None:
    output_dir = tmp_path / "run"
    train(_config(synthetic_audio_root, output_dir, steps=2))

    resumed = _config(synthetic_audio_root, output_dir, steps=2)
    resumed.resume_from = output_dir / "latest.pt"
    summary = train(resumed)
    assert summary["steps"] == 4


def test_config_round_trips_through_yaml(tmp_path) -> None:
    import yaml

    source = {
        "dataset": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "out"),
        "speakers_per_batch": 4,
        "max_steps": 7,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    config = TrainConfig.from_yaml(path)
    assert config.speakers_per_batch == 4
    assert config.max_steps == 7
    assert config.dataset == tmp_path / "data"
    assert isinstance(config.output_dir, Path)


def test_config_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("not_a_real_option: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown training config keys"):
        TrainConfig.from_yaml(path)


def test_shipped_default_config_is_loadable() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src" / "colorsplitter" / "training" / "configs" / "default.yaml"
    )
    config = TrainConfig.from_yaml(path)
    assert config.speakers_per_batch >= 2
    assert config.utterances_per_speaker >= 2
    assert config.singers_per_batch <= config.speakers_per_batch
    assert config.dataset is None, "the shipped config must not hard-code a dataset path"


def test_training_requires_a_dataset() -> None:
    with pytest.raises(ValueError, match="dataset"):
        train(TrainConfig(dataset=None))
