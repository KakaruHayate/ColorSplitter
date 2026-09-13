# Weights

## Weights are not in this repository

They are downloaded on demand into a per-user cache and verified by SHA-256.

This is a deliberate reversal. Weights used to be committed directly, which
meant a weight swap looked like an ordinary source edit. That is how the better
of the two timbre checkpoints was silently overwritten by a commit whose message
only mentioned the README — recoverable, but only because git happened to keep
the blob, and not visible from the history at all.

`models/registry.json` is now the record. For each weight it states:

| field | meaning |
|---|---|
| `id` | the name you pass to `--weights` |
| `file` | filename in the cache |
| `purpose` | `timbre` or `speaker` — they answer different questions |
| `default` | which one `--encoder timbre` picks with no `--weights` |
| `step` | the training step the checkpoint itself reports |
| `sha256` | hash of the published asset; `null` until the assets are staged |
| `urls` | release asset locations, tried in order |
| `source` | provenance: where the weight came from, including the git blob id for archived ones |
| `notes` | what it is for, and its licence position |

## The weights

### `timbre-v1` — default

The weight this tool has always been used with. It separates timbre styles
well, which is the entire point.

Recovered from git history: an earlier commit overwrote it, and the registry
records the blob id (`e8560833…`) along with the SHA-256 of the original bytes so
that the recovery is verifiable rather than hopeful. `scripts/prepare_release.py`
performs the recovery and re-derives the hash.

**Licence position, stated plainly.** Its training data was used to build an
embedding model that feeds an unsupervised clustering step. That data's licence
constrains use in a *synthesis path*; this is not a synthesis model, it produces
no audio and is not a component of anything that generates a voice. This is
close to the line, and it is written here rather than glossed, because you should
be able to make that call yourself. If you would rather not, use `timbre-alt-v1`.

### `timbre-alt-v1` — the cautious alternative

Trained from a source with no such caveat. It separates noticeably worse, which
is why it is not the default. The file is named for the step counter *inside* the
checkpoint (165000), because the previous filename claimed 1570000 and was simply
wrong.

### `speaker-upstream-v1` — speaker identity

The upstream Resemblyzer encoder (MIT). This one separates *singers*, not
registers — a different question. Use it when you need to tell performers apart,
for instance when auditing a mixed dataset. Same architecture, so it drops into
the same encoder.

### The emotion model

Not an encoder checkpoint: a `wav2vec2` model downloaded from the HuggingFace
hub, declared under `downloads.emotion` in the registry. Fetched as
`safetensors` rather than `pytorch_model.bin`, because it loads by tensor name
with no unpickling of a file we did not produce.

## Where they live

| platform | default |
|---|---|
| Windows | `%LOCALAPPDATA%\colorsplitter\weights` |
| Linux / macOS | `$XDG_CACHE_HOME/colorsplitter/weights`, or `~/.cache/colorsplitter/weights` |
| either | override with `COLORSPLITTER_HOME` |

## Getting them

```bash
cs weights list                     # what exists and what it says about itself
cs weights fetch                    # everything
cs weights fetch --only emotion     # one thing
```

A file placed in the cache directory by hand is used as-is, so an offline or
air-gapped setup works: drop the files in, and no network is touched.

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

If `sha256` is `null`, the asset has not been staged yet and no verification
happens. That means "unchecked", not "fine".

## Publishing a release

```bash
python scripts/prepare_release.py            # stage assets and fill in hashes
python scripts/prepare_release.py --check    # verify what is already staged
```

The script recovers archived blobs, verifies them against the recorded hash
before touching anything, strips optimiser state, writes the assets to
`models/release/`, updates `models/registry.json` and emits a manifest for the
release notes.

The optimiser state is roughly half the file size and is no use for inference;
stripping it is why the published assets are smaller than the raw checkpoints.
