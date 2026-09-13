"""A self-contained ``wav2vec2`` implementation for the emotion model.

Why this file exists
--------------------
The emotion model used to be instantiated through ``transformers``. That made
the package depend on a fast-moving library whose internal layout, class names
and default behaviour change between releases — so the model definition could
drift underneath us without a single line of our code changing, and pinning a
version to stop that drift is its own maintenance burden.

Instead the graph is reproduced here, once, against the real checkpoint. The
weights are fetched as a ``safetensors`` file and loaded by name, so the tensor
layout is pinned by us rather than by a library version.

Two details in here are easy to get wrong and fail *silently* if you do — both
were verified against the actual upstream source rather than from memory:

1. ``encoder.pos_conv_embed.conv`` uses weight normalisation with ``dim=2``,
   which in PyTorch means *the norm runs over every dimension except 2*. The
   checkpoint stores ``weight_g`` of shape ``(1, 1, 128)``, which only makes
   sense under that reading.
2. ``encoder.layer_norm`` exists in the checkpoint but is **never applied** by
   the upstream forward pass. It is constructed here so the state dict loads
   cleanly, and deliberately left out of the graph.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .safetensors import load_safetensors

__all__ = ["Wav2Vec2Config", "Wav2Vec2ForSpeechClassification", "Wav2Vec2Model"]

log = logging.getLogger(__name__)


@dataclass
class Wav2Vec2Config:
    """The subset of the HF config this graph needs."""

    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    conv_dim: tuple[int, ...] = (512,) * 7
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = True
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    feat_extract_norm: str = "layer"
    feat_extract_activation: str = "gelu"
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    do_stable_layer_norm: bool = True
    hidden_dropout: float = 0.1
    activation_dropout: float = 0.1
    attention_dropout: float = 0.1
    feat_proj_dropout: float = 0.1
    final_dropout: float = 0.1
    num_labels: int = 3
    names: list[str] = field(default_factory=list)

    @property
    def num_feat_extract_layers(self) -> int:
        return len(self.conv_dim)

    @classmethod
    def from_json(cls, path: Path) -> Wav2Vec2Config:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        label2id = raw.get("label2id") or {}
        names = [name for name, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
        return cls(
            hidden_size=int(raw.get("hidden_size", 1024)),
            num_hidden_layers=int(raw.get("num_hidden_layers", 12)),
            num_attention_heads=int(raw.get("num_attention_heads", 16)),
            intermediate_size=int(raw.get("intermediate_size", 4096)),
            conv_dim=tuple(raw["conv_dim"]),
            conv_kernel=tuple(raw["conv_kernel"]),
            conv_stride=tuple(raw["conv_stride"]),
            conv_bias=bool(raw.get("conv_bias", True)),
            num_conv_pos_embeddings=int(raw.get("num_conv_pos_embeddings", 128)),
            num_conv_pos_embedding_groups=int(raw.get("num_conv_pos_embedding_groups", 16)),
            feat_extract_norm=str(raw.get("feat_extract_norm", "layer")),
            feat_extract_activation=str(raw.get("feat_extract_activation", "gelu")),
            hidden_act=str(raw.get("hidden_act", "gelu")),
            layer_norm_eps=float(raw.get("layer_norm_eps", 1e-5)),
            do_stable_layer_norm=bool(raw.get("do_stable_layer_norm", True)),
            hidden_dropout=float(raw.get("hidden_dropout", 0.1)),
            activation_dropout=float(raw.get("activation_dropout", 0.1)),
            attention_dropout=float(raw.get("attention_dropout", 0.1)),
            feat_proj_dropout=float(raw.get("feat_proj_dropout", 0.1)),
            final_dropout=float(raw.get("final_dropout", 0.1)),
            num_labels=int(raw.get("num_labels", len(names) or 3)),
            names=names,
        )


def _activation(name: str):
    if name == "gelu":
        # Upstream maps "gelu" to the exact (erf-based) GELU, not the tanh
        # approximation. F.gelu defaults to approximate="none", i.e. exact.
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"unsupported activation {name!r}")


def _apply_weight_norm(v: torch.Tensor, g: torch.Tensor, dim: int) -> torch.Tensor:
    """PyTorch's weight norm: normalise over every dimension *except* ``dim``.

    ``torch._weight_norm`` is exactly this rule, and it is why the upstream
    ``weight_norm(conv, dim=2)`` produces a ``weight_g`` shaped ``(1, 1, K)``.
    """
    dims = [d for d in range(v.dim()) if d != dim]
    norm = v.norm(2, dim=dims, keepdim=True)
    return g * v / norm


class _SamePad(nn.Module):
    """Drops the trailing frame an even kernel adds via symmetric padding."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.num_pad_remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_pad_remove > 0:
            return hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class _LayerNormConvLayer(nn.Module):
    """Feature-extractor stage used when ``feat_extract_norm == "layer"``."""

    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0):
        super().__init__()
        in_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        out_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(out_dim, elementwise_affine=True)
        self._activation = _activation(config.feat_extract_activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states.transpose(-2, -1)).transpose(-2, -1)
        return self._activation(hidden_states)


class _FeatureEncoder(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        if config.feat_extract_norm != "layer":
            raise NotImplementedError(
                f"feat_extract_norm={config.feat_extract_norm!r} is not implemented; "
                "the emotion checkpoint uses 'layer'"
            )
        self.conv_layers = nn.ModuleList(
            _LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class _FeatureProjection(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.projection(self.layer_norm(hidden_states))
        return self.dropout(hidden_states)


class _WeightNormConv1d(nn.Module):
    """A grouped ``Conv1d`` carrying weight normalisation, HF-style.

    Two things are pinned here rather than inherited from an API:

    * **Parameter names** are ``weight_g`` / ``weight_v`` / ``bias``, matching the
      checkpoint. The modern ``nn.utils.parametrizations.weight_norm`` would name
      them ``parametrizations.weight.original0/1`` instead.
    * **``weight_g`` has shape ``(1, 1, K)``, not ``(out_channels, 1, K)``.**
      Upstream creates it as the shape of the norm, which collapses every
      dimension except ``dim``. Getting this wrong does not raise: the numbers
      still broadcast, and the checkpoint simply fails to load later.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        padding: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = padding

        self.weight_g = nn.Parameter(torch.empty(1, 1, kernel_size))
        self.weight_v = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_v, a=5**0.5)
        with torch.no_grad():
            self.weight_g.fill_(1.0)
        if self.bias is not None:
            bound = 1 / (self.in_channels * self.kernel_size) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        return _apply_weight_norm(self.weight_v, self.weight_g, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x,
            self.weight,
            self.bias,
            stride=1,
            padding=self.padding,
            groups=self.groups,
        )


class _PositionalConvEmbedding(nn.Module):
    """Grouped convolution carrying the relative-position signal."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        kernel = config.num_conv_pos_embeddings
        groups = config.num_conv_pos_embedding_groups
        self.conv = _WeightNormConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=kernel,
            groups=groups,
            padding=kernel // 2,
            bias=True,
        )
        self.same_pad = _SamePad(kernel)
        self._activation = _activation(config.feat_extract_activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.same_pad(hidden_states)
        hidden_states = self._activation(hidden_states)
        return hidden_states.transpose(1, 2)


class _Attention(nn.Module):
    """Eager multi-head self-attention, matching the upstream projection layout."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return self.out_proj(attn_output)


class _FeedForward(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self._activation = _activation(config.hidden_act)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._activation(self.intermediate_dense(hidden_states))
        hidden_states = self.output_dropout(
            self.output_dense(self.intermediate_dropout(hidden_states))
        )
        return hidden_states


class _EncoderLayerStableLayerNorm(nn.Module):
    """Pre-norm transformer layer.

    Note the shape of the residual: ``final_layer_norm`` normalises the input of
    the feed-forward branch, it is *not* a post-block norm. Reading it as the
    latter still runs and still produces plausible numbers — it just answers a
    different question than the one the checkpoint was trained on.
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.attention = _Attention(config.hidden_size, config.num_attention_heads, config.attention_dropout)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = _FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        return hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))


class _EncoderStableLayerNorm(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.pos_conv_embed = _PositionalConvEmbedding(config)
        # Present in the checkpoint, never applied upstream. Kept so that the
        # state dict loads without an "unexpected key" report, and excluded from
        # forward on purpose.
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            _EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.pos_conv_embed(hidden_states)
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Wav2Vec2Model(nn.Module):
    """Feature extractor + projection + transformer encoder."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        self.feature_extractor = _FeatureEncoder(config)
        self.feature_projection = _FeatureProjection(config)
        self.encoder = _EncoderStableLayerNorm(config)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """``input_values``: ``(B, samples)`` raw 16 kHz waveforms."""
        hidden_states = self.feature_extractor(input_values)
        hidden_states = self.feature_projection(hidden_states.transpose(1, 2))
        return self.encoder(hidden_states)


class _RegressionHead(nn.Module):
    """Arousal / dominance / valence regression head (upstream definition)."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dropout(torch.tanh(self.dense(x)))
        return self.out_proj(x)


class Wav2Vec2ForSpeechClassification(nn.Module):
    """The emotion model: ``wav2vec2`` trunk plus the regression head.

    ``forward`` returns ``(pooled_hidden_states, logits)`` to mirror the
    signature the original wrapper exposed; the pooled hidden states are the
    1024-d emotion embedding used by the pipeline.
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)

    def forward(self, input_values: torch.Tensor):
        hidden_states = self.wav2vec2(input_values)
        pooled = torch.mean(hidden_states, dim=1)
        return pooled, self.classifier(pooled)

    @classmethod
    def from_pretrained(cls, model_dir: Path) -> tuple[Wav2Vec2ForSpeechClassification, dict]:
        """Build the model and load a checkpoint from a HF-style directory.

        Accepts ``model.safetensors`` (preferred: no unpickling) or
        ``pytorch_model.bin``. Returns the model plus a load report.
        """
        model_dir = Path(model_dir)
        config = Wav2Vec2Config.from_json(model_dir / "config.json")
        model = cls(config)

        safetensors_path = model_dir / "model.safetensors"
        bin_path = model_dir / "pytorch_model.bin"
        if safetensors_path.exists():
            state = load_safetensors(safetensors_path)
            source = safetensors_path.name
        elif bin_path.exists():
            try:
                state = torch.load(bin_path, map_location="cpu", weights_only=True)
            except Exception:
                log.warning(
                    "%s requires weights_only=False; "
                    "only load checkpoints from a trusted source",
                    bin_path.name,
                )
                state = torch.load(bin_path, map_location="cpu", weights_only=False)
            state = state.get("state_dict", state) if isinstance(state, dict) else state
            source = bin_path.name
        else:
            raise FileNotFoundError(
                f"no model.safetensors or pytorch_model.bin in {model_dir}"
            )

        missing, unexpected = model.load_state_dict(state, strict=False)
        report = {
            "source": source,
            "missing": sorted(missing),
            "unexpected": sorted(unexpected),
        }
        if missing:
            raise RuntimeError(
                f"checkpoint {source} is missing {len(missing)} expected tensors, "
                f"first few: {sorted(missing)[:5]}"
            )
        if unexpected:
            log.debug("ignoring %d tensors not used by the graph: %s", len(unexpected), sorted(unexpected)[:5])
        return model, report
