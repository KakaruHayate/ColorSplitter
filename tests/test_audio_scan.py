"""Audio discovery and preprocessing.

The input contract is "point it at a directory" — no dataset layout, no
annotation files, no required folder names. These tests pin that contract, plus
the parts of the feature path that must stay numerically stable.
"""

from __future__ import annotations

import numpy as np
import pytest

from colorsplitter.core import audio
from colorsplitter.core.hparams import mel_n_channels, sampling_rate
from colorsplitter.core.pipeline import dataset_from_paths, scan
from tests.conftest import write_wav


def test_scan_finds_audio_recursively(shuffled_audio_root) -> None:
    dataset = scan(shuffled_audio_root)
    assert len(dataset) == 3
    assert all(item.path.exists() for item in dataset)


def test_scan_keys_are_relative_and_sorted(shuffled_audio_root) -> None:
    keys = scan(shuffled_audio_root).keys
    assert keys == sorted(keys, key=str.lower)
    assert not any(key.startswith("/") or ":" in key for key in keys)


def test_scan_ignores_non_audio_and_hidden(tmp_path) -> None:
    write_wav(tmp_path / "keep.wav")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "cover.jpg").write_bytes(b"\x00\x01")
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    write_wav(hidden / "skip.wav")

    keys = scan(tmp_path).keys
    assert keys == ["keep.wav"]


def test_scan_accepts_many_containers(tmp_path) -> None:
    """Extension filtering is by name; decoding is a separate concern."""
    for suffix in (".wav", ".m4a", ".mp3", ".flac", ".ogg", ".opus"):
        (tmp_path / f"clip{suffix}").write_bytes(b"stub")
    assert len(scan(tmp_path)) == 6


def test_scan_can_be_restricted_by_extension(tmp_path) -> None:
    for suffix in (".wav", ".m4a"):
        (tmp_path / f"clip{suffix}").write_bytes(b"stub")
    assert len(scan(tmp_path, extensions={".wav"})) == 1


def test_scan_missing_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        scan(tmp_path / "does-not-exist")


def test_scan_accepts_a_single_file(shuffled_audio_root) -> None:
    one = next(shuffled_audio_root.rglob("*.wav"))
    assert len(scan(one)) == 1


def test_iter_audio_files_does_not_follow_symlinked_dirs(tmp_path) -> None:
    real = tmp_path / "real"
    write_wav(real / "a.wav")
    loop = tmp_path / "loop"
    try:
        loop.symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    files = list(audio.iter_audio_files(tmp_path))
    assert len(files) == 1, "a symlink cycle must not duplicate or hang the scan"


def test_dataset_from_paths_keys_are_unique(tmp_path) -> None:
    a = write_wav(tmp_path / "one" / "same.wav")
    b = write_wav(tmp_path / "two" / "same.wav")
    dataset = dataset_from_paths([a, b])
    assert len(set(dataset.keys)) == 2


# --- feature path ----------------------------------------------------------


def test_preprocess_wav_produces_16k_mono(shuffled_audio_root) -> None:
    path = next(shuffled_audio_root.rglob("*.wav"))
    wav = audio.preprocess_wav(path, trim_silences=False)
    assert wav.dtype == np.float32
    assert wav.ndim == 1
    assert len(wav) > 0


def test_preprocess_wav_accepts_in_memory_arrays() -> None:
    sr = 8000
    t = np.arange(sr) / sr
    wav = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    out = audio.preprocess_wav(wav, source_sr=sr, trim_silences=False)
    assert abs(len(out) / len(wav) - sampling_rate / sr) < 0.05


def test_resample_is_a_noop_at_the_target_rate() -> None:
    wav = np.ones(1000, dtype=np.float32)
    assert np.array_equal(audio.resample_waveform(wav, sampling_rate, sampling_rate), wav)


def test_normalize_volume_raises_on_contradictory_flags() -> None:
    with pytest.raises(ValueError):
        audio.normalize_volume(np.ones(100), increase_only=True, decrease_only=True)


def test_normalize_volume_only_increases_when_asked() -> None:
    loud = np.full(1000, 0.9, dtype=np.float32)
    out = audio.normalize_volume(loud, increase_only=True)
    assert np.allclose(out, loud)


def test_mel_spectrogram_shape(shuffled_audio_root) -> None:
    path = next(shuffled_audio_root.rglob("*.wav"))
    wav = audio.preprocess_wav(path, trim_silences=False)
    mel = audio.wav_to_mel_spectrogram(wav)
    assert mel.ndim == 2
    assert mel.shape[1] == mel_n_channels
    assert mel.dtype == np.float32
    # 10 ms hop -> ~100 frames per second
    expected = int(len(wav) / sampling_rate * 100)
    assert abs(mel.shape[0] - expected) <= 2


def test_mel_is_deterministic(shuffled_audio_root) -> None:
    path = next(shuffled_audio_root.rglob("*.wav"))
    wav = audio.preprocess_wav(path, trim_silences=False)
    assert np.array_equal(
        audio.wav_to_mel_spectrogram(wav), audio.wav_to_mel_spectrogram(wav)
    )


def test_trim_silences_true_without_webrtcvad_is_explicit() -> None:
    """Asking for the reference preprocessing must not silently degrade."""
    if audio.vad_available():
        pytest.skip("webrtcvad is installed")
    with pytest.raises(RuntimeError, match="webrtcvad"):
        audio.trim_long_silences(np.zeros(16000, dtype=np.float32))


def test_trim_silences_auto_never_raises(shuffled_audio_root) -> None:
    path = next(shuffled_audio_root.rglob("*.wav"))
    assert len(audio.preprocess_wav(path, trim_silences="auto")) > 0


def test_preprocess_many_preserves_order(shuffled_audio_root) -> None:
    paths = sorted(shuffled_audio_root.rglob("*.wav"))
    got = audio.preprocess_many(paths, workers=1, trim_silences=False)
    assert len(got) == len(paths)
    for path, wav in zip(paths, got, strict=True):
        np.testing.assert_allclose(wav, audio.preprocess_wav(path, trim_silences=False))
