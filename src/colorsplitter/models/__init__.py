"""Network definitions.

These are the only files that define on-disk model structure; every checkpoint
in ``models/registry.json`` is bound to them. Treat the tensor layouts here as a
frozen interface.

Everything is importable lazily so that ``import colorsplitter.models`` does not
require torch.
"""

from __future__ import annotations

__all__ = [
    "EmotionEncoder",
    "VoiceEncoder",
    "Wav2Vec2Config",
    "Wav2Vec2ForSpeechClassification",
    "load_safetensors",
]

_LAZY = {
    "VoiceEncoder": (".voice_encoder", "VoiceEncoder"),
    "EmotionEncoder": (".emotion_encoder", "EmotionEncoder"),
    "Wav2Vec2Config": (".wav2vec2", "Wav2Vec2Config"),
    "Wav2Vec2ForSpeechClassification": (".wav2vec2", "Wav2Vec2ForSpeechClassification"),
    "load_safetensors": (".safetensors", "load_safetensors"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0], __name__)
    return getattr(module, target[1])
