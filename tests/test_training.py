"""Training data, the GE2E sampler's batch-shape guarantee, and the loss.

The property worth asserting hardest is the one the sampler exists for: within a
batch, classes drawn from the same singer should be common. Without that, the
encoder is trained on easy negatives and the timbre axis never gets learned.
"""

from __future__ import annotations

import numpy as np
import pytest

from colorsplitter.training.data import Ge2eBatchSampler, TimbreDataset, split_class_name

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional extra
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")


def ge2e_loss(**kwargs):
    """Import GE2ELoss lazily.

    ``colorsplitter.training.loss`` imports torch at module level, so importing
    it here would break collection of this file in a torch-free install — and
    the sampler tests in this module do not need torch at all.
    """
    from colorsplitter.training.loss import GE2ELoss

    return GE2ELoss(**kwargs)


# --- naming convention ------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("vocalist_Soft", ("vocalist", "Soft")),
        ("vocalist_v2_Belt", ("vocalist_v2", "Belt")),
        ("soloist", ("soloist", "default")),
        ("a_b_c", ("a_b", "c")),
        ("trailing_", ("trailing_", "default")),
        ("_leading", ("_leading", "default")),
    ],
)
def test_split_class_name(name, expected) -> None:
    assert split_class_name(name) == expected


def test_split_uses_the_last_underscore() -> None:
    """Singer names may contain hyphens; a mid-string underscore must survive."""
    assert split_class_name("my-singer_v2_Soft") == ("my-singer_v2", "Soft")


# --- dataset ---------------------------------------------------------------


def test_dataset_discovers_classes(synthetic_audio_root) -> None:
    dataset = TimbreDataset(synthetic_audio_root)
    assert len(dataset) == 5
    assert dataset.singers == ["alpha", "beta", "gamma"]


def test_dataset_stats(synthetic_audio_root) -> None:
    stats = TimbreDataset(synthetic_audio_root).stats()
    assert stats["classes"] == 5
    assert stats["singers"] == 3
    assert stats["clips"] == 15
    assert stats["timbres_per_singer_max"] == 3


def test_dataset_groups_timbres_by_singer(synthetic_audio_root) -> None:
    grouped = TimbreDataset(synthetic_audio_root).classes_by_singer()
    assert len(grouped["alpha"]) == 3
    assert len(grouped["beta"]) == 1


def test_dataset_ignores_directories_without_audio(tmp_path) -> None:
    (tmp_path / "empty_class").mkdir()
    from tests.conftest import write_wav

    write_wav(tmp_path / "real_one" / "a.wav")
    dataset = TimbreDataset(tmp_path)
    assert [c.name for c in dataset.classes] == ["real_one"]


def test_dataset_requires_a_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        TimbreDataset(tmp_path / "missing")


def test_dataset_requires_audio_somewhere(tmp_path) -> None:
    (tmp_path / "empty_class").mkdir()
    with pytest.raises(ValueError):
        TimbreDataset(tmp_path)


def test_load_mel_returns_a_fixed_window(synthetic_audio_root) -> None:
    import random

    from colorsplitter.core.hparams import mel_n_channels, partials_n_frames

    dataset = TimbreDataset(synthetic_audio_root)
    mel = dataset.load_mel(dataset.classes[0].paths[0], random.Random(0))
    assert mel is not None
    assert mel.shape == (partials_n_frames, mel_n_channels)
    assert mel.dtype == np.float32


def test_load_mel_pads_a_short_clip(tmp_path) -> None:
    import random

    from colorsplitter.core.hparams import partials_n_frames
    from tests.conftest import write_wav

    write_wav(tmp_path / "s_a" / "tiny.wav", seconds=0.2)
    dataset = TimbreDataset(tmp_path)
    mel = dataset.load_mel(dataset.classes[0].paths[0], random.Random(0))
    assert mel.shape[0] == partials_n_frames


def test_load_mel_survives_an_unreadable_file(tmp_path) -> None:
    import random

    (tmp_path / "s_a").mkdir()
    (tmp_path / "s_a" / "broken.wav").write_bytes(b"not audio")
    dataset = TimbreDataset(tmp_path)
    assert dataset.load_mel(dataset.classes[0].paths[0], random.Random(0)) is None


# --- sampler ---------------------------------------------------------------


def test_batch_shape_is_rectangular(synthetic_audio_root) -> None:
    dataset = TimbreDataset(synthetic_audio_root)
    sampler = Ge2eBatchSampler(
        dataset, speakers_per_batch=3, utterances_per_speaker=2, singers_per_batch=3, seed=1
    )
    mels, labels = sampler.load_batch(sampler.sample_indices())
    assert mels.shape[0] == 6
    assert labels.shape == (6,)
    assert sorted(set(labels.tolist())) == [0, 1, 2]


def test_each_class_contributes_exactly_m_utterances(synthetic_audio_root) -> None:
    dataset = TimbreDataset(synthetic_audio_root)
    sampler = Ge2eBatchSampler(
        dataset, speakers_per_batch=4, utterances_per_speaker=3, singers_per_batch=2, seed=2
    )
    indices = sampler.sample_indices()
    assert len(indices) == 4
    assert all(len(entries) == 3 for entries in indices)


def test_classes_within_a_batch_are_distinct(tmp_path) -> None:
    """Distinct classes must be exhausted before any is reused.

    Uses a dataset where every singer can supply enough timbres, because that is
    the condition under which the guarantee holds. The shared fixture has
    single-timbre singers, where repeats are unavoidable.
    """
    from tests.conftest import write_wav

    for singer in ("s1", "s2"):
        for timbre in ("a", "b", "c"):
            write_wav(tmp_path / f"{singer}_{timbre}" / "clip.wav")
    dataset = TimbreDataset(tmp_path)

    sampler = Ge2eBatchSampler(
        dataset, speakers_per_batch=3, utterances_per_speaker=2, singers_per_batch=2, seed=3
    )
    for _ in range(30):
        classes = [entries[0][0] for entries in sampler.sample_indices()]
        assert len(set(classes)) == len(classes)


def test_repeats_are_allowed_when_the_pool_cannot_supply_enough(tmp_path) -> None:
    """Degenerate input must still produce a usable batch, not an exception."""
    from tests.conftest import write_wav

    write_wav(tmp_path / "solo_only" / "a.wav")
    dataset = TimbreDataset(tmp_path)
    sampler = Ge2eBatchSampler(
        dataset, speakers_per_batch=3, utterances_per_speaker=2, singers_per_batch=1, seed=0
    )
    mels, labels = sampler.load_batch(sampler.sample_indices())
    assert mels.shape[0] == 6
    assert sorted(set(labels.tolist())) == [0, 1, 2]


def test_same_singer_pairs_are_deliberately_common(synthetic_audio_root) -> None:
    """The whole reason this sampler exists: hard negatives in every batch."""
    dataset = TimbreDataset(synthetic_audio_root)
    sampler = Ge2eBatchSampler(
        dataset, speakers_per_batch=3, utterances_per_speaker=2, singers_per_batch=1, seed=4
    )
    fractions = [
        sampler.same_singer_fraction(sampler.sample_indices()) for _ in range(50)
    ]
    assert np.mean(fractions) > 0.5, f"expected same-singer pairs, saw {np.mean(fractions):.2f}"


def test_widening_the_singer_pool_reduces_same_singer_pairs(synthetic_audio_root) -> None:
    dataset = TimbreDataset(synthetic_audio_root)
    narrow = Ge2eBatchSampler(
        dataset, speakers_per_batch=3, utterances_per_speaker=2, singers_per_batch=1, seed=5
    )
    wide = Ge2eBatchSampler(
        dataset, speakers_per_batch=3, utterances_per_speaker=2, singers_per_batch=3, seed=5
    )
    narrow_mean = np.mean([narrow.same_singer_fraction(narrow.sample_indices()) for _ in range(50)])
    wide_mean = np.mean([wide.same_singer_fraction(wide.sample_indices()) for _ in range(50)])
    assert narrow_mean > wide_mean


def test_sampler_handles_a_class_with_too_few_clips(tmp_path) -> None:
    from tests.conftest import write_wav

    write_wav(tmp_path / "s_one" / "only.wav")
    write_wav(tmp_path / "s_two" / "a.wav")
    write_wav(tmp_path / "s_two" / "b.wav")
    dataset = TimbreDataset(tmp_path)
    sampler = Ge2eBatchSampler(
        dataset, speakers_per_batch=2, utterances_per_speaker=4, singers_per_batch=2, seed=6
    )
    mels, labels = sampler.load_batch(sampler.sample_indices())
    assert mels.shape[0] == 8


def test_sampler_rejects_impossible_configurations(synthetic_audio_root) -> None:
    dataset = TimbreDataset(synthetic_audio_root)
    with pytest.raises(ValueError):
        Ge2eBatchSampler(dataset, speakers_per_batch=1)
    with pytest.raises(ValueError):
        Ge2eBatchSampler(dataset, utterances_per_speaker=0)
    with pytest.raises(ValueError):
        Ge2eBatchSampler(dataset, singers_per_batch=0)


def test_sampler_is_reproducible_for_a_given_seed(synthetic_audio_root) -> None:
    dataset = TimbreDataset(synthetic_audio_root)
    a = Ge2eBatchSampler(dataset, speakers_per_batch=3, utterances_per_speaker=2, seed=9)
    b = Ge2eBatchSampler(dataset, speakers_per_batch=3, utterances_per_speaker=2, seed=9)
    assert a.sample_indices() == b.sample_indices()


# --- loss ------------------------------------------------------------------


@requires_torch
def test_ge2e_rejects_a_flat_input() -> None:
    with pytest.raises(ValueError):
        ge2e_loss()(torch.randn(4, 8))


@requires_torch
def test_ge2e_rejects_degenerate_batches() -> None:
    with pytest.raises(ValueError):
        ge2e_loss()(torch.randn(1, 4, 8))
    with pytest.raises(ValueError):
        ge2e_loss()(torch.randn(4, 1, 8))


@requires_torch
def test_ge2e_loss_is_lower_when_classes_are_separable() -> None:
    torch.manual_seed(0)
    loss_fn = ge2e_loss()

    tight = torch.zeros(4, 3, 8)
    for k in range(4):
        tight[k] = torch.nn.functional.normalize(torch.randn(3, 8) * 0.01, dim=1) + torch.eye(8)[k] * 2
    scattered = torch.nn.functional.normalize(torch.randn(4, 3, 8), dim=2)

    assert float(loss_fn(tight)) < float(loss_fn(scattered))


@requires_torch
def test_ge2e_parameters_are_named_for_the_checkpoint() -> None:
    """The archived checkpoints store these two scalars under these names."""
    assert set(ge2e_loss().state_dict()) == {"similarity_weight", "similarity_bias"}


@requires_torch
def test_ge2e_gradients_reach_both_parameters() -> None:
    loss_fn = ge2e_loss()
    embeds = torch.nn.functional.normalize(torch.randn(4, 3, 8, requires_grad=True), dim=2)
    loss_fn(embeds).backward()
    assert loss_fn.similarity_weight.grad is not None
    assert torch.isfinite(loss_fn.similarity_weight.grad).all()
