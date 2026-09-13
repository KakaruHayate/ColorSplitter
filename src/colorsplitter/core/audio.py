"""Audio discovery, decoding and feature extraction.

The preprocessing and mel-spectrogram code is inlined from Resemblyzer
(MIT, https://github.com/resemble-ai/Resemblyzer) — see ``NOTICE``. The calls
into ``librosa`` are kept argument-for-argument identical to upstream so the
resulting tensors match what existing checkpoints were trained on.

The only additions are: recursive multi-format scanning, a pluggable silence
trimmer, and optional multi-process decoding.
"""

from __future__ import annotations

import logging
import struct
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union

import numpy as np

from .hparams import (
    audio_norm_target_dBFS,
    int16_max,
    mel_n_channels,
    mel_window_length,
    mel_window_step,
    sampling_rate,
    vad_max_silence_length,
    vad_moving_average_width,
    vad_window_length,
)

__all__ = [
    "AUDIO_EXTENSIONS",
    "find_audio_files",
    "iter_audio_files",
    "load_waveform",
    "normalize_volume",
    "preprocess_wav",
    "resample_waveform",
    "trim_long_silences",
    "wav_to_mel_spectrogram",
    "vad_available",
]

log = logging.getLogger(__name__)

#: Containers decoded when scanning a directory tree.
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".wave", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}
)


def _vad_module():
    """Return the optional ``webrtcvad`` module, or ``None`` when unavailable."""
    try:  # pragma: no cover - depends on the install
        import webrtcvad  # type: ignore
    except Exception:  # noqa: BLE001 - any import failure means "not usable"
        return None
    return webrtcvad


def vad_available() -> bool:
    """Whether faithful silence trimming (``webrtcvad``) can be performed."""
    return _vad_module() is not None


def iter_audio_files(
    root: Union[str, Path],
    extensions: Optional[Iterable[str]] = None,
) -> Iterator[Path]:
    """Yield every audio file under *root*, recursively, in a stable order.

    Hidden files and directories (leading ``.``) are skipped. Symlinked
    directories are not followed, so a cyclic tree cannot hang the scan.
    """
    exts = frozenset(e.lower() for e in (extensions or AUDIO_EXTENSIONS))
    exts = frozenset(e if e.startswith(".") else f".{e}" for e in exts)
    base = Path(root).expanduser()
    if base.is_file():
        if base.suffix.lower() in exts:
            yield base
        return

    stack = [base]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except (PermissionError, OSError):
            log.warning("cannot read directory, skipping: %s", current)
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if not entry.is_symlink():
                    stack.append(entry)
            elif entry.suffix.lower() in exts:
                yield entry


def find_audio_files(
    root: Union[str, Path],
    extensions: Optional[Iterable[str]] = None,
) -> list[Path]:
    """Eager, sorted variant of :func:`iter_audio_files`."""
    return sorted(iter_audio_files(root, extensions), key=lambda p: str(p).lower())


# --- decoding ---------------------------------------------------------------


def load_waveform(
    fpath: Union[str, Path],
    source_sr: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Decode an audio file to a mono float32 waveform at its native rate.

    Returns ``(wav, sample_rate)``. Decoding is delegated to librosa so that
    every container librosa/audioread supports keeps working, matching the
    behaviour of the original pipeline.
    """
    import librosa

    wav, sr = librosa.load(str(fpath), sr=None, mono=True)
    if source_sr is not None:
        sr = source_sr
    return np.asarray(wav, dtype=np.float32), int(sr)


def resample_waveform(wav: np.ndarray, orig_sr: int, target_sr: int = sampling_rate) -> np.ndarray:
    """Resample *wav* from *orig_sr* to *target_sr* (no-op when already equal)."""
    if orig_sr == target_sr:
        return np.asarray(wav, dtype=np.float32)
    import librosa

    out = librosa.resample(
        np.asarray(wav, dtype=np.float32), orig_sr=int(orig_sr), target_sr=int(target_sr)
    )
    return np.asarray(out, dtype=np.float32)


def normalize_volume(
    wav: np.ndarray,
    target_dBFS: float = audio_norm_target_dBFS,
    increase_only: bool = False,
    decrease_only: bool = False,
) -> np.ndarray:
    """Peak-relative RMS normalisation. Verbatim from upstream Resemblyzer."""
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


def _moving_average(array: np.ndarray, width: int) -> np.ndarray:
    array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    ret = np.cumsum(array_padded, dtype=float)
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1:] / width


def trim_long_silences(wav: np.ndarray) -> np.ndarray:
    """Shorten voiced-region gaps using WebRTC VAD. Verbatim from upstream.

    Raises :class:`RuntimeError` when ``webrtcvad`` is not installed; callers
    that want to tolerate the missing dependency should use
    ``preprocess_wav(..., trim_silences="auto")``.
    """
    webrtcvad = _vad_module()
    if webrtcvad is None:
        raise RuntimeError(
            "webrtcvad is not installed, so silence trimming is unavailable. "
            "Install the 'vad' extra (`pip install colorsplitter[vad]`) to match "
            "the reference preprocessing, or pass trim_silences='none' to skip it."
        )

    from scipy.ndimage import binary_dilation

    samples_per_window = (vad_window_length * sampling_rate) // 1000
    if samples_per_window <= 0 or len(wav) < samples_per_window:
        return np.asarray(wav, dtype=np.float32)

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[: len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    vad = webrtcvad.Vad(mode=3)
    voice_flags = [
        vad.is_speech(
            pcm_wave[window_start * 2 : (window_start + samples_per_window) * 2],
            sample_rate=sampling_rate,
        )
        for window_start in range(0, len(wav), samples_per_window)
    ]
    audio_mask = _moving_average(np.array(voice_flags), vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    return wav[audio_mask == True]  # noqa: E712 - matches upstream


def preprocess_wav(
    fpath_or_wav: Union[str, Path, np.ndarray],
    source_sr: Optional[int] = None,
    *,
    trim_silences: Union[bool, str] = "auto",
) -> np.ndarray:
    """Resample, normalise and (optionally) silence-trim a waveform.

    :param fpath_or_wav: file path or an in-memory waveform.
    :param source_sr: sample rate of an in-memory waveform. Ignored for paths.
    :param trim_silences: ``True`` requires ``webrtcvad`` and reproduces the
        reference preprocessing exactly. ``False`` skips trimming. ``"auto"``
        (default) trims when ``webrtcvad`` is importable and otherwise
        continues with a warning, so the tool still runs on a minimal install.
    """
    if isinstance(fpath_or_wav, (str, Path)):
        wav, source_sr = load_waveform(fpath_or_wav)
    else:
        wav = np.asarray(fpath_or_wav, dtype=np.float32)

    if source_sr is not None:
        wav = resample_waveform(wav, source_sr, sampling_rate)

    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)

    if trim_silences == "auto":
        trim_silences = vad_available()
        if not trim_silences:
            log.warning(
                "webrtcvad not installed - skipping silence trimming. Embeddings will "
                "differ slightly from the reference preprocessing."
            )
    if trim_silences:
        wav = trim_long_silences(wav)

    return np.asarray(wav, dtype=np.float32)


def preprocess_many(
    paths: Sequence[Union[str, Path]],
    *,
    workers: int = 1,
    trim_silences: Union[bool, str] = "auto",
    chunk_size: int = 8,
) -> list[np.ndarray]:
    """Preprocess several files, optionally across processes.

    Decoding compressed containers is CPU-bound and single-threaded inside
    librosa, so parallelising it is the cheapest large win on the critical path.
    Results keep the order of *paths*.
    """
    path_strs = [str(p) for p in paths]
    if workers <= 1 or len(path_strs) < 2:
        return [preprocess_wav(p, trim_silences=trim_silences) for p in path_strs]

    from functools import partial

    worker = partial(_preprocess_one, trim_silences=trim_silences)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(worker, path_strs, chunksize=chunk_size))


def _preprocess_one(path_str: str, trim_silences: Union[bool, str] = "auto") -> np.ndarray:
    return preprocess_wav(path_str, trim_silences=trim_silences)


@lru_cache(maxsize=None)
def _mel_filterbank():
    import librosa

    return librosa.filters.mel(
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        n_mels=mel_n_channels,
    )


def wav_to_mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """Mel spectrogram of a *preprocessed* waveform.

    Argument-for-argument identical to upstream Resemblyzer so that the
    encoder sees exactly the features it was trained on.
    """
    import librosa

    frames = librosa.feature.melspectrogram(
        y=np.asarray(wav, dtype=np.float32),
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels,
    )
    return frames.astype(np.float32).T
