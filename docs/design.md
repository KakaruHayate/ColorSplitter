# Design

## The shape of the thing

```
src/colorsplitter/
  core/          audio, embed, cluster, reduce, labels, cache, modelzoo, pipeline
  models/        network definitions (voice encoder, wav2vec2, safetensors reader)
  cli/           command line
  web/           FastAPI app + the built frontend
  training/      encoder training
frontend/        frontend sources; the build output lands in web/static
models/          registry.json only -- no weight binaries
tests/           including the equivalence suite
```

Layering rules that are worth keeping:

* `core/pipeline.py` is the single definition of what the tool does. The CLI and
  the web app are thin shells over it; no capability is reachable from one and
  not the other.
* `core` imports `models` lazily. `import colorsplitter` does not pull in torch,
  which is what lets the package be installed and scripted without it.
* Cross-cutting state (the weights cache, the projection cache) lives in a
  dedicated cache directory, never inside the directory being scanned.

## What was kept

Only three things in the previous codebase were load-bearing, and all three were
kept:

1. **The clustering algorithms** (`core/cluster.py`). `SpectralCluster` and
   `UmapHdbscan`, ported from 3D-Speaker. These were chosen deliberately and are
   not to be changed.
2. **The encoder definition** (`models/voice_encoder.py`). Every checkpoint in
   existence is bound to this tensor layout, so it is effectively a frozen
   interface.
3. **The emotion model definition**, since it is the only thing that gives the
   `emotion` and `mix` encoders meaning.

Everything else — five top-level scripts, a pile of visualisation helpers, and
the standalone viewer — was replaced.

## What changed about the clustering, and why it is still the same algorithm

The algorithms are untouched. Parameter defaults (`pval`, `min_pnum`,
`n_components`, `min_cluster_size`) are untouched. What changed is how the same
arithmetic is executed:

* **`p_pruning` is vectorised.** `np.argsort(A, axis=1)` produces exactly the
  per-row ordering the original Python loop produced, so the pruned matrix is
  bit-identical while the interpreter loop over rows disappears.
* **`max_num_spks` is configurable.** It used to be hard-coded to 14, which made
  `nmin > 14` contradictory.
* **`mer_cos` is actually passed through.** The CLI accepted `--mer_cosine` and
  then dropped it on the floor: the value was read into a local variable and
  `mer_cos=None` was passed to the clusterer. It is now wired up, and validated
  (the old type declaration would have raised `TypeError` in the assertion had
  it ever been connected).
* **An optional sparse eigensolver.** With `eigen_solver="sparse"`, the pruned
  affinity is sparsified and ARPACK computes only the leading eigenpairs instead
  of a full O(N³) decomposition. It is opt-in; `"auto"` only selects it above
  2048 samples.

That last one is an *approximation*, so it is not asserted as equality.
`tests/test_cluster_equivalence.py` checks two separate things:

| claim | assertion |
|---|---|
| the dense path reproduces the original | exact label equality against a verbatim copy of the old implementation |
| the sparse path agrees with the dense path | Adjusted Rand Index ≥ 0.95, measured and reported |

If the second ever drops below the threshold the fast path is not fit to be
enabled, and the test says so rather than quietly tolerating it.

## What was replaced

| before | after |
|---|---|
| `splitter.py` etc., five scripts driven by `input()` and `plt.show()` | `core/pipeline.py`, pure functions; CLI and WebUI on top |
| `move_files.py` (copy) and `kick.py` (move) with no way to tell them apart | one `export` with an explicit `mode` |
| `clean_csv.py`, editing an annotation file in place | read-only validation and report; the tool no longer knows what an annotation file is |
| `load_npy.py`, drawing a plot from a `.npy` | import embeddings in the WebUI, then cluster and review as usual |
| a viewer needing a second server on :8080 plus CORS, building audio URLs from filesystem paths | one process, one port, audio streamed from an opaque token with range support |
| caching keyed by speaker directory, inside the scanned data | content-fingerprint cache in a dedicated directory |
| emotion model loaded at import time | loaded on demand |
| weights committed to git | resolved through `registry.json`, downloaded, hash-verified |

## Deliberate non-changes

* **No tuning.** The clustering and projection defaults were left exactly as they
  were. Changing them during a refactor would make it impossible to attribute a
  later difference in results to either the refactor or the tuning.
* **No ONNX export.** The encoder definition stays a torch module. Anything that
  needs a portable graph should be built on top of it rather than reshaping it.
* **No history rewrite.** The old weights are still in the git history. Removing
  them from the working tree is enough; rewriting history is a separate decision
  with separate consequences.

## Two details that are easy to get wrong

Both were verified against upstream source rather than recalled, because both
fail silently:

* In `models/wav2vec2.py`, the positional convolution uses weight normalisation
  with `dim=2`, which in PyTorch means *the norm runs over every dimension except
  dim 2*. The checkpoint's `weight_g` of shape `(1, 1, 128)` only makes sense
  under that reading.
* `wav2vec2.encoder.layer_norm` exists in the checkpoint but is **never applied**
  by the upstream forward pass. It is constructed here so the state dict loads
  cleanly, and deliberately kept out of the graph. `test_encoder_layer_norm_is_inert`
  perturbs it and asserts the output does not move, so that a well-meaning
  "fix" fails loudly instead of corrupting every embedding.
