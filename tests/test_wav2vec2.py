"""Structural tests for the in-tree wav2vec2 implementation.

The point of these is that the graph is *pinned*. They check the things that
would otherwise fail silently: parameter naming against the real checkpoint's
convention, the weight-normalisation semantics, and the fact that one parameter
in the checkpoint is deliberately inert.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from colorsplitter.models.emotion_encoder import zero_mean_unit_var_norm  # noqa: E402
from colorsplitter.models.wav2vec2 import (  # noqa: E402
    Wav2Vec2Config,
    Wav2Vec2ForSpeechClassification,
    _PositionalConvEmbedding,
)

N_LAYERS = 3
CONV_DIM = (8,) * 7


def small_config() -> Wav2Vec2Config:
    return Wav2Vec2Config(
        hidden_size=16,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=CONV_DIM,
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=2,
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        num_labels=3,
    )


def expected_key_set(n_layers: int) -> set[str]:
    """The key set the real checkpoint uses, parameterised by depth.

    Derived from the published ``model.safetensors`` header of the emotion
    model: 7 feature-extractor stages, a feature projection, the unused encoder
    layer norm, the weight-normalised positional conv, 16 tensors per layer, and
    a 4-tensor regression head. 37 + 16 * layers + 4.
    """
    keys = set()
    for i in range(7):
        keys |= {
            f"wav2vec2.feature_extractor.conv_layers.{i}.conv.weight",
            f"wav2vec2.feature_extractor.conv_layers.{i}.conv.bias",
            f"wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.weight",
            f"wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.bias",
        }
    keys |= {
        "wav2vec2.feature_projection.layer_norm.weight",
        "wav2vec2.feature_projection.layer_norm.bias",
        "wav2vec2.feature_projection.projection.weight",
        "wav2vec2.feature_projection.projection.bias",
        "wav2vec2.encoder.layer_norm.weight",
        "wav2vec2.encoder.layer_norm.bias",
        "wav2vec2.encoder.pos_conv_embed.conv.weight_g",
        "wav2vec2.encoder.pos_conv_embed.conv.weight_v",
        "wav2vec2.encoder.pos_conv_embed.conv.bias",
    }
    for i in range(n_layers):
        prefix = f"wav2vec2.encoder.layers.{i}"
        for proj in ("k_proj", "v_proj", "q_proj", "out_proj"):
            keys |= {f"{prefix}.attention.{proj}.weight", f"{prefix}.attention.{proj}.bias"}
        keys |= {f"{prefix}.layer_norm.weight", f"{prefix}.layer_norm.bias"}
        keys |= {
            f"{prefix}.feed_forward.intermediate_dense.weight",
            f"{prefix}.feed_forward.intermediate_dense.bias",
            f"{prefix}.feed_forward.output_dense.weight",
            f"{prefix}.feed_forward.output_dense.bias",
        }
        keys |= {f"{prefix}.final_layer_norm.weight", f"{prefix}.final_layer_norm.bias"}
    keys |= {
        "classifier.dense.weight",
        "classifier.dense.bias",
        "classifier.out_proj.weight",
        "classifier.out_proj.bias",
    }
    return keys


def test_parameter_names_match_the_checkpoint_convention() -> None:
    model = Wav2Vec2ForSpeechClassification(small_config())
    assert set(model.state_dict().keys()) == expected_key_set(N_LAYERS)


def test_key_count_matches_the_published_checkpoint_formula() -> None:
    """At the real depth the count must come out at 233, plus 1 inert tensor."""
    keys = expected_key_set(12)
    assert len(keys) == 37 + 16 * 12 + 4 == 233


def test_masked_spec_embed_is_the_only_unexpected_tensor() -> None:
    """The checkpoint also stores a pre-training-only tensor we do not need."""
    model = Wav2Vec2ForSpeechClassification(small_config())
    state = model.state_dict()
    state["wav2vec2.masked_spec_embed"] = torch.zeros(16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == []
    assert sorted(unexpected) == ["wav2vec2.masked_spec_embed"]


def test_encoder_layer_norm_is_inert() -> None:
    """Verified against upstream: this parameter is never applied.

    If someone "fixes" the apparent omission by wiring it in, every emotion
    embedding silently changes. This test fails loudly instead.
    """
    model = Wav2Vec2ForSpeechClassification(small_config()).eval()
    waveform = torch.randn(1, 8000)

    with torch.no_grad():
        before, _ = model(waveform)
        model.wav2vec2.encoder.layer_norm.weight.fill_(0.0)
        model.wav2vec2.encoder.layer_norm.bias.fill_(3.0)
        after, _ = model(waveform)

    torch.testing.assert_close(before, after)


@pytest.mark.parametrize("kernel", [128, 129])
def test_weight_norm_uses_norm_over_all_dims_except_dim_2(kernel: int) -> None:
    """Verified against ``nn.utils.weight_norm`` itself, not from memory.

    The decisive check is the shape of ``weight_g``: upstream creates it as the
    shape of the norm, which collapses every dimension except ``dim`` — hence
    ``(1, 1, K)``. A ``(out_channels, 1, K)`` guess still broadcasts and still
    runs, and it silently breaks checkpoint loading instead.
    """
    import torch.nn as nn

    reference = nn.utils.weight_norm(
        nn.Conv1d(64, 64, kernel_size=kernel, padding=kernel // 2, groups=4, bias=True),
        name="weight",
        dim=2,
    )
    assert tuple(reference.weight_g.shape) == (1, 1, kernel)

    config = small_config()
    config.num_conv_pos_embeddings = kernel
    module = _PositionalConvEmbedding(config)
    with torch.no_grad():
        module.conv.weight_v.copy_(torch.randn_like(module.conv.weight_v))
        module.conv.weight_g.copy_(torch.rand_like(module.conv.weight_g) + 0.5)

    v, g = module.conv.weight_v, module.conv.weight_g
    manual = g * v / v.norm(2, dim=(0, 1), keepdim=True)
    torch.testing.assert_close(module.conv.weight, manual)
    torch.testing.assert_close(module.conv.weight, torch._weight_norm(v, g, 2))


def test_positional_embedding_shapes_are_stable() -> None:
    """Grouped conv with symmetric padding must preserve the frame count."""
    module = _PositionalConvEmbedding(small_config())
    x = torch.randn(2, 50, 16)
    out = module(x)
    assert out.shape == x.shape


def test_forward_shapes() -> None:
    model = Wav2Vec2ForSpeechClassification(small_config()).eval()
    pooled, logits = model(torch.randn(2, 9000))
    assert pooled.shape == (2, 16)
    assert logits.shape == (2, 3)


def test_pooling_is_the_mean_over_time() -> None:
    model = Wav2Vec2ForSpeechClassification(small_config()).eval()
    waveform = torch.randn(1, 6000)
    with torch.no_grad():
        trunk = model.wav2vec2(waveform)
        pooled, _ = model(waveform)
    torch.testing.assert_close(pooled, trunk.mean(dim=1))


def test_zero_mean_unit_var_norm_matches_the_reference_formula() -> None:
    rng = np.random.default_rng(0)
    x = (rng.normal(size=4000) * 3.0 + 1.5).astype(np.float32)
    expected = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    got = zero_mean_unit_var_norm(x)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
    assert abs(float(got.mean())) < 1e-5
    assert abs(float(got.var()) - 1.0) < 1e-2


def test_config_round_trips_a_real_shaped_config(tmp_path) -> None:
    raw = {
        "hidden_size": 1024,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "conv_dim": [512] * 7,
        "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
        "conv_stride": [5, 2, 2, 2, 2, 2, 2],
        "num_conv_pos_embeddings": 128,
        "num_conv_pos_embedding_groups": 16,
        "feat_extract_norm": "layer",
        "do_stable_layer_norm": True,
        "layer_norm_eps": 1e-05,
        "label2id": {"arousal": 0, "dominance": 1, "valence": 2},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = Wav2Vec2Config.from_json(path)
    assert config.num_labels == 3
    assert config.names == ["arousal", "dominance", "valence"]
    assert config.num_feat_extract_layers == 7
