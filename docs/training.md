# Training

The training code is complete and runnable. **No training run is performed as
part of this repository**, and no weights are produced by it here — but nothing
is stubbed out either. A short run works end to end, and the checkpoint it writes
loads straight back into the inference encoder, which is what
`tests/test_training_smoke.py` asserts.

```
src/colorsplitter/training/
  data.py     dataset discovery + the GE2E batch sampler
  model.py    model construction and checkpoint I/O
  loss.py     GE2E
  train.py    the loop
  configs/default.yaml
```

## Dataset layout

```
<root>/
  <singer>_<timbre>/
    anything.wav
    anything.m4a
    ...
```

A directory is a class. The name is split on the **last** underscore, so singer
names containing hyphens or version suffixes survive intact:

```
vocalist_Soft     ->  singer "vocalist",     timbre "Soft"
vocalist_v2_Belt  ->  singer "vocalist_v2",  timbre "Belt"
solo              ->  singer "solo",         timbre "default"
```

### Why the class is `<singer>_<timbre>` and not `<singer>`

This is the decision the whole training setup hangs on. If the class were the
singer, a batch would teach the model that *everything one singer does is the
same thing* — including their different registers. That trains the timbre axis
flat, which is the opposite of what this tool needs.

With the class at `<singer>_<timbre>`, two registers of one singer become two
labels. They are acoustically close and must be told apart: a hard negative,
which is exactly where the gradient is informative.

## The sampler

A GE2E batch is `N` classes × `M` utterances. This sampler fills those `N` slots
from a deliberately small pool of singers, so several slots usually belong to the
same singer.

| parameter | meaning |
|---|---|
| `speakers_per_batch` | `N` — class slots |
| `utterances_per_speaker` | `M` — clips per class |
| `singers_per_batch` | how many distinct singers the slots are drawn from |

`singers_per_batch` is the knob that matters. Set it to 1 and every batch is one
singer's timbres. Set it equal to `speakers_per_batch` and classes are sampled
uniformly — the hard-negative structure disappears. The default of 4 against 8
slots is a middle ground.

The class slots within a batch are always distinct, and a class with fewer clips
than `M` is sampled with replacement rather than dropped. Both properties are
pinned by tests.

## Configuration

```yaml
dataset: /path/to/dataset        # required
output_dir: runs/encoder
val_fraction: 0.1

speakers_per_batch: 8
utterances_per_speaker: 4
singers_per_batch: 4

max_steps: 200000
lr: 0.001
weight_decay: 0.0
decay_every: 5000
decay_rate: 0.8
grad_clip: 3.0
scheduler_floor: 0.00001

seed: 42
device: null
log_every: 50
checkpoint_every: 5000
val_every: 500
val_batches: 8
resume_from: null
stop_on_nonfinite: true
```

**About these defaults.** They are reasonable starting points carried over from
the lineage this code descends from, not reverse-engineered values. The original
training scripts are gone; nothing here claims to reproduce them exactly.
Optimiser and schedule are the first things to revisit when tuning.

Unknown keys are rejected rather than ignored, so a typo in a config file fails
loudly.

## Running

```bash
cs train --dataset /path/to/dataset \
         --config src/colorsplitter/training/configs/default.yaml

# a quick end-to-end check
cs train --dataset /path/to/dataset --max-steps 50 --out runs/smoke
```

Validation is split by class, stratified by singer, so a singer never appears on
both sides — a val set containing the same singer would only measure how well the
model memorised them.

Progress is logged every `log_every` steps with loss, validation loss and the
current learning rate. Training **stops on a non-finite loss** rather than
writing a broken checkpoint; set `stop_on_nonfinite: false` to skip bad steps
instead.

## Checkpoints

`latest.pt` is written every `checkpoint_every` steps and contains:

| key | contents |
|---|---|
| `model_state` | the encoder — all inference needs |
| `step`, `loss` | provenance |
| `optimizer_state` | for resuming |
| `similarity_weight`, `similarity_bias` | the two learned GE2E scalars |

Those last two are stored under the same names the archived checkpoints use, so
resuming from one needs no translation.

To resume:

```yaml
resume_from: runs/encoder/latest.pt
```

To turn a checkpoint into a distributable weight:

```bash
cs weights pack --source runs/encoder/latest.pt \
                --dest models/release/timbre-v2.pt \
                --step 250000
```

This keeps `model_state` and drops the optimiser state, roughly halving the file.
Then add it to `models/registry.json`, hash it with
`python scripts/prepare_release.py`, and it is available as `--weights <id>`.

## Relationship to the inference path

`training/model.py` builds the *same* class inference uses rather than a parallel
copy. A training-time divergence in architecture would be invisible until a
trained checkpoint failed to load, and there is no reason to risk that.
