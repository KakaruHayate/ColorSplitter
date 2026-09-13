# ColorSplitter

[中文文档](README_CN.md)

Timbre clustering and filtering for single-speaker singing datasets.

Point it at a directory. It embeds every audio file it finds, clusters the
embeddings by timbre, shows you the result as an interactive scatter plot where
you can audition any point, fix up the clusters by hand, and then write the
files back out grouped by cluster.

Useful when preparing a dataset: splitting one singer's material into registers
or styles before training, or filtering out takes whose timbre does not match
the rest.

**A caveat worth stating.** This is speaker-verification machinery pointed at
singing. Singing timbre variation and voiceprint difference are related but not
the same thing, and the field has not settled the question. It works well enough
to be useful; it is not a solved problem.

---

## Install

```bash
pip install -e ".[all]"          # everything
pip install -e .                 # timbre encoder only, no torch
```

Optional extras: `cluster` (UMAP + HDBSCAN), `emotion`, `train`, `web`, `vad`
(faithful silence trimming), `dev`.

Reading `.m4a`, `.mp3`, `.flac` and friends needs a decoder. `ffmpeg` on `PATH`
covers everything; `libsndfile` (bundled with `soundfile`) covers most.

## Use

```bash
cs serve                          # browser UI on http://127.0.0.1:8000
```

or from the command line:

```bash
cs scan  ./my-singing-data                     # what did it find?
cs run   ./my-singing-data --nmin 2            # embed, cluster, write a CSV
cs run   ./my-singing-data --nmin 2 --export   # also write out by cluster
cs weights list                                # what weights exist?
```

The WebUI is the intended way to work: it is where you actually audition points
and correct clusters. Everything it does is also available from `cs`, so the
tool stays scriptable.

## How it works

```
scan → embed → cluster → project → review → export
```

* **scan** — walks the directory you name, recursively. No dataset layout is
  assumed: no annotation files, no required folder names. Only the audio matters.
* **embed** — each file becomes one vector. Four encoders are available:
  `timbre` (default), `speaker`, `emotion`, and `mix`.
* **cluster** — spectral clustering, or UMAP + HDBSCAN. These algorithms are
  fixed; see [docs/design.md](docs/design.md) for why, and for what did change.
* **project** — a 2D view for you to look at (t-SNE, UMAP or PCA).
* **review** — click any point to hear it, drag a box to select, reassign points
  between clusters, rename, merge, split, undo.
* **export** — copy (default) or move the files into `output/<cluster>/`.

## Weights

Weights are **not** stored in this repository; they are downloaded into a local
cache on first use and verified by SHA-256. `models/registry.json` is the list.

| id | purpose | notes |
|---|---|---|
| `timbre-v1` | timbre | default |
| `timbre-alt-v1` | timbre | alternative checkpoint, weaker separation |
| `speaker-upstream-v1` | speaker identity | upstream Resemblyzer encoder |

`timbre` and `speaker` answer different questions — separating one singer's
registers versus telling singers apart. Pick accordingly.

See [docs/weights.md](docs/weights.md) for provenance, caching and mirrors.

## Training

The encoder training code is in `src/colorsplitter/training/` and is complete
and runnable; no training run is performed as part of this repository. It expects
directories named `<singer>_<timbre>`, and its sampler deliberately fills each
batch with several timbres of the *same* singer, because those pairs are the
hard negatives that teach the model the timbre axis.

See [docs/training.md](docs/training.md).

## Documentation

| | |
|---|---|
| [installation.md](docs/installation.md) | environments, extras, decoders, troubleshooting |
| [cli.md](docs/cli.md) | every command and flag |
| [webui.md](docs/webui.md) | the interface, and the review workflow |
| [training.md](docs/training.md) | dataset layout, sampler, config, resuming |
| [weights.md](docs/weights.md) | registry, cache, mirrors, provenance |
| [design.md](docs/design.md) | architecture, what was kept, what was replaced |

## Licence

MIT — see [LICENSE](LICENSE). Third-party attribution is in [NOTICE](NOTICE).
