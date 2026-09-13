# Installation

## Requirements

* Python 3.10 or newer.
* A C++ build toolchain only if you install the `vad` extra (it needs to compile
  a small extension). Everything else ships as wheels.

## Install

```bash
git clone <this repository>
cd ColorSplitter
pip install -e ".[all]"
```

Or pick only what you need:

| extra | pulls in | needed for |
|---|---|---|
| *(none)* | numpy, scipy, scikit-learn, librosa, soundfile | scanning, clustering, projection, export |
| `cluster` | umap-learn, hdbscan | `--cluster umap_hdbscan`, `--projection umap` |
| `emotion` | torch | `--encoder emotion` and `--encoder mix` |
| `train` | torch | `cs train` |
| `web` | fastapi, uvicorn | `cs serve` |
| `vad` | webrtcvad-wheels | silence trimming that matches the reference preprocessing |
| `dev` | pytest, ruff, build | working on the project |

`pip install -e .` on its own is a real configuration, not a broken one. The
network definitions live in `models/` and are imported lazily, so the base
install covers scanning, clustering, projection, export and the served UI.
What it *cannot* do is embed, because the encoder is a torch module — that comes
with the `emotion` or `train` extra. In CI the torch-free install is exercised
explicitly, and the test suite asserts that `torch` really is absent from it.

## Audio decoders

Reading `.wav` works through libsndfile (bundled with `soundfile`). Compressed
containers — `.m4a`, `.mp3`, `.flac`, `.ogg`, `.opus` — go through `librosa`,
which uses `soundfile` where it can and falls back to `audioread`/`ffmpeg`
otherwise.

**Install `ffmpeg` and put it on `PATH`.** It is the difference between
"sometimes works" and "works". If a file cannot be decoded, the scan reports it
and carries on rather than aborting the run.

## Silence trimming

The reference preprocessing trims long silences with WebRTC VAD. Without the
`vad` extra, `trim_silences="auto"` skips the step and logs a warning, which
changes embeddings slightly relative to the shipped checkpoints. If you want the
exact reference behaviour, install `vad`. Passing `--trim-silences yes` without
it raises instead of silently degrading.

## Where things are stored

| what | where | override |
|---|---|---|
| downloaded weights | per-user cache directory | `COLORSPLITTER_HOME` |
| embedding + projection caches | alongside the weights | `--cache-dir` |
| training runs | `runs/<name>/` | `--out` |

Nothing is written into the directory you scan. Nothing is written into the
repository.

## Verifying the install

```bash
cs scan <some-directory>      # lists what it can see
cs weights list               # shows the registry
cs serve                      # http://127.0.0.1:8000
```

## Troubleshooting

**`Couldn't find the voice encoder checkpoint`** — the weight is not cached and
could not be downloaded. Run `cs weights fetch`. Behind a restricted network,
see [weights.md](weights.md) for the mirror behaviour and `HF_ENDPOINT`.

**`webrtcvad is not installed`** — either install the `vad` extra, or pass
`--trim-silences no` to accept the difference explicitly.

**A file is skipped during the scan** — decoding failed. The path is logged;
check that `ffmpeg` can read it.

**The web page is a 503 with `frontend not built`** — the bundle is missing from
`src/colorsplitter/web/static`. Build it with `cd frontend && npm install && npm run build`.
End users never need to do this; the bundle is committed.
