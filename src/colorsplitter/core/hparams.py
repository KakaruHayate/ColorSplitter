"""Audio / model hyper-parameters.

These values are kept **identical** to the upstream Resemblyzer defaults
(MIT, https://github.com/resemble-ai/Resemblyzer) so that embeddings produced
here stay numerically compatible with existing checkpoints.

Changing any value in this file invalidates every checkpoint in
``models/registry.json``.
"""

# --- Mel filterbank ---------------------------------------------------------
mel_window_length = 25  # milliseconds
mel_window_step = 10  # milliseconds
mel_n_channels = 40

# --- Audio ------------------------------------------------------------------
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance (1600 ms).
partials_n_frames = 160

# --- Voice activity detection ----------------------------------------------
# Window size of the VAD. Must be 10, 20 or 30 ms.
vad_window_length = 30
# Number of frames averaged together when smoothing the VAD output.
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment may contain.
vad_max_silence_length = 6

# --- Volume normalisation ---------------------------------------------------
audio_norm_target_dBFS = -30

# --- Model ------------------------------------------------------------------
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3

int16_max = (2**15) - 1
