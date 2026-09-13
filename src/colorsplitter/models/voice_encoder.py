"""Speaker/timbre encoder.

The network definition is unchanged from the implementation this project has
always shipped (itself a copy of Resemblyzer's ``VoiceEncoder``, MIT) so that
every checkpoint in ``models/registry.json`` keeps loading as-is. Only the
imports and the inference path changed:

* the ``Resemblyzer`` package is no longer imported — hparams and mel features
  come from :mod:`colorsplitter.core.audio`;
* ``device`` now defaults to CUDA when available (the old default pinned the
  whole pipeline to CPU);
* :meth:`VoiceEncoder.embed_utterances` batches the partial-utterance forward
  passes instead of running one utterance at a time.

The batched path is mathematically identical to the per-utterance one: the
network L2-normalises each partial, the utterance embedding is the mean of its
partials, then L2-normalised again.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter as timer

import numpy as np
import torch
from torch import nn

from ..core.audio import wav_to_mel_spectrogram
from ..core.hparams import (
    mel_n_channels,
    mel_window_step,
    model_embedding_size,
    model_hidden_size,
    model_num_layers,
    partials_n_frames,
    sampling_rate,
)

__all__ = ["VoiceEncoder"]

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


class VoiceEncoder(nn.Module):
    """Maps mel spectrograms to L2-normalised speaker embeddings.

    :param device: torch device, or ``None`` to pick CUDA when available.
    :param weights_fpath: checkpoint path. Both ``{"model_state": ...}`` and a
        bare state dict are accepted.
    :param load_weights: set ``False`` to skip checkpoint loading entirely and
        start from a random initialisation (used when training from scratch).
    :param amp: run the forward pass under CUDA autocast (fp16). Off by default
        because it perturbs embeddings slightly.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        verbose: bool = False,
        weights_fpath: Path | str | None = None,
        amp: bool = False,
        load_weights: bool = True,
    ):
        super().__init__()

        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()
        self.amp = bool(amp)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        if load_weights:
            if weights_fpath is None:
                weights_fpath = Path(__file__).resolve().parent.joinpath("pretrained.pt")
            weights_fpath = Path(weights_fpath)
            if not weights_fpath.exists():
                raise FileNotFoundError(
                    f"Couldn't find the voice encoder checkpoint at {weights_fpath}. "
                    "Run `cs weights fetch` to download the registered weights."
                )

            start = timer()
            try:
                checkpoint = torch.load(weights_fpath, map_location="cpu", weights_only=True)
            except pickle.UnpicklingError:
                log.warning(
                    "checkpoint %s requires weights_only=False; "
                    "only load checkpoints from a trusted source",
                    weights_fpath.name,
                )
                checkpoint = torch.load(weights_fpath, map_location="cpu", weights_only=False)
            state = (
                checkpoint.get("model_state", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            missing, unexpected = self.load_state_dict(state, strict=False)
            if missing or unexpected:
                log.warning(
                    "checkpoint %s does not match the model exactly (missing=%d, unexpected=%d)",
                    weights_fpath.name,
                    len(missing),
                    len(unexpected),
                )
            if verbose:
                print(
                    f"Loaded the voice encoder model on {device.type} "
                    f"in {timer() - start:.2f} seconds."
                )

        self.to(device)

    def forward(self, mels: torch.FloatTensor) -> torch.FloatTensor:
        """Embed a batch of mel spectrograms of shape ``(B, frames, channels)``."""
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    # --- partial slicing ----------------------------------------------------

    @staticmethod
    def compute_partial_slices(
        n_samples: int, rate: float, min_coverage: float
    ) -> tuple[list[slice], list[slice]]:
        """Split an utterance into overlapping partial utterances.

        Unchanged from the original implementation: returned ranges may index
        past the end of the waveform, so pad the waveform to
        ``wav_slices[-1].stop`` before use.
        """
        assert 0 < min_coverage <= 1

        samples_per_frame = int(sampling_rate * mel_window_step / 1000)
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
        assert frame_step > 0, "The rate is too high"
        assert frame_step <= partials_n_frames, (
            "The rate is too low, it should be %f at least"
            % (sampling_rate / (samples_per_frame * partials_n_frames))
        )

        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices, wav_slices = mel_slices[:-1], wav_slices[:-1]

        return wav_slices, mel_slices

    def _mel_partials(
        self, wav: np.ndarray, rate: float, min_coverage: float
    ) -> np.ndarray:
        """Mel frames for every partial of one preprocessed waveform."""
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
        mel = wav_to_mel_spectrogram(wav)
        return np.array([mel[s] for s in mel_slices], dtype=np.float32)

    # --- inference ----------------------------------------------------------

    @torch.no_grad()
    def embed_utterance(
        self, wav: np.ndarray, return_partials: bool = False, rate: float = 1.3, min_coverage: float = 0.75
    ):
        """Embed a single preprocessed utterance.

        Kept for parity with the original implementation and used as the
        reference in tests.
        """
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        mel = wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        mels = torch.from_numpy(mels).to(self.device)
        partial_embeds = self(mels).float().cpu().numpy()

        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, wav_slices
        return embed

    @torch.no_grad()
    def embed_utterances(
        self,
        wavs: Sequence[np.ndarray],
        rate: float = 1.3,
        min_coverage: float = 0.75,
        batch_size: int = 64,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        """Embed many preprocessed utterances, batching the partials together.

        This is the hot path. The original code ran one utterance per forward
        pass, which left the GPU essentially idle.
        """
        if not wavs:
            return np.zeros((0, model_embedding_size), dtype=np.float32)

        parts: list[np.ndarray] = []
        counts: list[int] = []
        for wav in wavs:
            m = self._mel_partials(np.asarray(wav, dtype=np.float32), rate, min_coverage)
            parts.append(m)
            counts.append(int(m.shape[0]))

        all_mels = np.concatenate(parts, axis=0)
        partial = self._forward_batched(all_mels, batch_size, progress)

        out = np.empty((len(wavs), model_embedding_size), dtype=np.float32)
        start = 0
        for i, count in enumerate(counts):
            raw = partial[start : start + count].mean(axis=0)
            norm = np.linalg.norm(raw, 2)
            out[i] = raw / norm if norm > 0 else raw
            start += count
        return out

    def _forward_batched(
        self, mels: np.ndarray, batch_size: int, progress: ProgressFn | None = None
    ) -> np.ndarray:
        total = int(mels.shape[0])
        batch_size = max(1, int(batch_size))
        chunks: list[np.ndarray] = []
        use_amp = self.amp and self.device.type == "cuda"
        for start in range(0, total, batch_size):
            chunk = torch.from_numpy(mels[start : start + batch_size]).to(self.device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embeds = self(chunk)
            else:
                embeds = self(chunk)
            chunks.append(embeds.float().cpu().numpy())
            if progress is not None:
                progress(min(start + batch_size, total), total)
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, model_embedding_size), np.float32)

    @torch.no_grad()
    def embed_speaker(self, wavs: list[np.ndarray], **kwargs) -> np.ndarray:
        """Mean embedding of several utterances, L2-normalised."""
        raw_embed = np.mean(
            [self.embed_utterance(wav, return_partials=False, **kwargs) for wav in wavs], axis=0
        )
        return raw_embed / np.linalg.norm(raw_embed, 2)
