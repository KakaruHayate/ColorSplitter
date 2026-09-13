# Weights

## The timbre weights are in `pretrain/`

They are committed to the repository (~17 MB each) so that a clone gives you a
working tool with no download step.

`models/registry.json` is the record. For each weight it states:

| field | meaning |
|---|---|
| `id` | the name you pass to `--weights` |
| `file` | filename in `pretrain/` (or the cache, for downloaded weights) |
| `purpose` | `timbre` or `speaker` — they answer different questions |
| `default` | which one `--encoder timbre` picks with no `--weights` |
| `step` | the training step the checkpoint itself reports |
| `sha256` | hash of the file, verified on load |
| `urls` | download locations (only for weights that are not committed) |
| `source` | provenance: where the weight came from, including the git blob id for archived ones |

## The weights

### `timbre-v1` — default

The weight this tool has always been used with. It separates timbre styles
well, which is the entire point.

Recovered from git history: an earlier commit overwrote it, and the registry
records the blob id (`e8560833…`) along with the SHA-256 of the original bytes so
that the recovery is verifiable rather than hopeful.

### `timbre-alt-v1`

An alternative checkpoint. It separates noticeably worse, which is why it is not
the default. The file is named for the step counter *inside* the checkpoint
(165000), because the previous filename claimed 1570000 and was simply wrong.

### `speaker-upstream-v1` — speaker identity

The upstream Resemblyzer encoder. This one separates *singers*, not registers —
a different question. Use it when you need to tell performers apart, for
instance when auditing a mixed dataset. Same architecture, so it drops into the
same encoder. Downloaded on first use.

### The emotion model

Not an encoder checkpoint: a `wav2vec2` model downloaded from the HuggingFace
hub, declared under `downloads.emotion` in the registry. Fetched as
`safetensors` rather than `pytorch_model.bin`, because it loads by tensor name
with no unpickling of a file we did not produce.

## Where they live

| weight | location |
|---|---|
| `timbre-v1`, `timbre-alt-v1` | `pretrain/` in the repository |
| `speaker-upstream-v1`, emotion model | per-user cache, downloaded on demand |

Cache directory:

| platform | default |
|---|---|
| Windows | `%LOCALAPPDATA%\colorsplitter\weights` |
| Linux / macOS | `$XDG_CACHE_HOME/colorsplitter/weights`, or `~/.cache/colorsplitter/weights` |
| either | override with `COLORSPLITTER_HOME` |

A file placed in the cache directory by hand is used as-is, so an offline or
air-gapped setup works: drop the files in, and no network is touched.

## Getting them

The timbre weights are already in `pretrain/` — no action needed. The other two
are fetched on demand:

```bash
cs weights list                     # what exists
cs weights fetch                    # download speaker + emotion
cs weights fetch --only emotion     # just the emotion model
```

## Mirrors

`huggingface.co` is unreachable from some networks — through a blocking proxy it
fails outright rather than being slow. Every hub asset therefore has an ordered
candidate list:

1. `HF_ENDPOINT`, if set,
2. the mirrors in `registry.json` (`https://hf-mirror.com` by default),
3. `huggingface.co` last.

The host that last worked is written to `.hf_host` in the cache and tried first
afterwards, so an unreachable primary costs one timeout rather than one per file.
The first candidate is given a deliberately short leash for the same reason.

Downloads resume from a partial file where the server supports range requests,
and a completed file is hashed before it is accepted. A hash mismatch deletes the
file and reports it — a quiet substitution of a different weight is the failure
mode worth guarding against.

If `sha256` is `null`, the asset has not been verified. That means "unchecked",
not "fine".
