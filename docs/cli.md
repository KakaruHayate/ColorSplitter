# Command line

Every capability of the pipeline is available here, so runs are reproducible and
scriptable. The WebUI drives the same functions.

```
cs <command> [options]
```

`colorsplitter` is an alias for `cs`.

---

## `cs scan`

Lists the audio found under a directory, without loading any model.

```
cs scan ./my-singing-data --limit 50
```

| flag | default | meaning |
|---|---|---|
| `--limit` | 20 | how many paths to print |

Output is a JSON summary line followed by the file list. Paths are relative to
the scanned root, so output does not depend on where the repository lives.

---

## `cs run`

Scan, embed, cluster, project, and optionally export.

```
cs run ./my-singing-data \
    --encoder timbre \
    --cluster spectral \
    --nmin 2 \
    --projection tsne \
    --export --export-mode copy
```

### Input and output

| flag | default | meaning |
|---|---|---|
| `input` | — | directory to scan (positional) |
| `--output` | `<input>-colorsplitter` | where the CSV and exported files go |
| `--cache-dir` | per-user cache | embedding and projection cache location |
| `--no-cache` | off | ignore and do not write caches |

### Encoder

| flag | default | meaning |
|---|---|---|
| `--encoder` | `timbre` | `timbre`, `speaker`, `emotion`, `mix` |
| `--weights` | registry default | a weight id from `cs weights list` |
| `--device` | auto | `cpu`, `cuda`, … |
| `--batch-size` | 64 | partial utterances per forward pass |
| `--amp` | off | fp16 on CUDA; slightly lossy, off by default |
| `--trim-silences` | `auto` | `auto`, `yes`, `no` — see installation.md |

### Throughput

| flag | default | meaning |
|---|---|---|
| `--workers` | 1 | parallel audio decoders |
| `--chunk` | 256 | files decoded per batch; bounds peak memory |

Decoding compressed audio is CPU-bound and single-threaded inside the decoder, so
`--workers` is usually the single biggest lever on a large dataset. It is not the
default because process pools and notebook environments do not always get along.

### Clustering

| flag | default | meaning |
|---|---|---|
| `--cluster` | `spectral` | `spectral` or `umap_hdbscan` |
| `--nmin` | 1 | lower bound when estimating the cluster count |
| `--max-spks` | 14 | upper bound (used to be hard-coded to 14) |
| `--mer-cos` | off | merge clusters whose centroids exceed this cosine similarity |

`--mer-cos` was accepted and silently ignored before this version; it now does
what it says.

### Projection

| flag | default | meaning |
|---|---|---|
| `--projection` | `tsne` | `tsne`, `umap`, `pca` |

`pca` is instant and is the right choice when you only want to check that the
pipeline ran; `tsne` is the one to look at.

### Export

| flag | default | meaning |
|---|---|---|
| `--export` | off | write the clustered files |
| `--export-mode` | `copy` | `copy` leaves the source alone; `move` relocates it |

`copy` is the default because the alternative destroys data as a side effect of
a command you might run to look at something.

### Output

`clusters.csv` in the output directory, with columns `key,cluster,x,y`, plus a
JSON summary on stdout:

```json
{
  "files": 412,
  "embedding_dim": 256,
  "encoder": "timbre",
  "weights": "timbre-v1.pt:cuda",
  "cluster_method": "spectral",
  "n_clusters": 3,
  "cluster_sizes": {"0": 96, "1": 205, "2": 111},
  "csv": ".../clusters.csv"
}
```

---

## `cs serve`

Starts the WebUI. One process, one port, frontend included.

```
cs serve --input ./my-singing-data --port 8000
```

| flag | default | meaning |
|---|---|---|
| `--input` | none | preload this directory at startup |
| `--host` | `127.0.0.1` | bind address; localhost by default on purpose |
| `--port` | 8000 | port |
| `--cache-dir` | per-user cache | cache location |
| `--registry` | shipped registry | alternative `registry.json` |

See [webui.md](webui.md).

---

## `cs weights`

```
cs weights list
cs weights fetch                    # everything
cs weights fetch --only emotion     # just the emotion model
cs weights pack --source run/latest.pt --dest models/release/timbre-v2.pt --step 250000
```

| command | meaning |
|---|---|
| `list` | ids, purpose, step, hash prefix, licence notes |
| `fetch` | download and verify into the cache |
| `pack` | strip optimiser state from a training checkpoint for distribution |

`pack` is what turns a training checkpoint into a release asset: it keeps
`model_state` and drops the optimiser state, which roughly halves the file.

---

## `cs train`

```
cs train --dataset /path/to/dataset --config src/colorsplitter/training/configs/default.yaml
cs train --dataset /path/to/dataset --max-steps 200 --out runs/smoke
```

| flag | meaning |
|---|---|
| `--dataset` | root of `<singer>_<timbre>` directories |
| `--config` | YAML config; see [training.md](training.md) |
| `--out` | run directory |
| `--max-steps` | override the step count |

See [training.md](training.md).

---

## Environment variables

| variable | effect |
|---|---|
| `COLORSPLITTER_HOME` | base directory for the weight cache |
| `COLORSPLITTER_REGISTRY` | path to an alternative `registry.json` |
| `HF_ENDPOINT` | hub mirror base URL, tried before the built-in mirrors |

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | a run failed |
| 2 | bad usage, or a missing optional dependency |
