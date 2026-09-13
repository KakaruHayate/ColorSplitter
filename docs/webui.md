# Web interface

```bash
cs serve --input /path/to/audio
```

Then open <http://127.0.0.1:8000>.

One process serves the API, the progress stream, the audio and the frontend. The
previous viewer needed a second static server on another port plus CORS, and
built audio URLs out of raw filesystem paths; here audio is addressed by an
opaque token and streamed from the same origin, with range support so seeking
works.

`--host` defaults to `127.0.0.1`. There is no authentication, and none is
pretended: this is a local dataset workbench, and binding it to a public
interface would be a mistake.

---

## The workflow

### 1. Scan

Enter a directory and press **Scan**. Nothing is assumed about its contents —
every audio file underneath is picked up, at any depth. The header shows the
root, the file count and, once known, the embedding dimension.

### 2. Run

Choose an encoder, a clustering method and a projection, then **Run pipeline**.
Progress streams in over Server-Sent Events: decoding, embedding, clustering and
projection are reported as separate stages, and the run can be abandoned without
killing the server.

Long runs are bounded in memory by processing the audio in chunks, so a
many-thousand-file directory does not have to fit in RAM.

### 3. Review

The scatter plot is where the actual work happens.

| action | result |
|---|---|
| **click a point** | selects it and plays its audio |
| **click the same point again** | pauses / resumes |
| **shift-click** | add to or remove from the selection |
| **drag** | box-select everything inside the rectangle |
| **Stop audio** | stops playback |

A single audio element is reused, so clicking through points never leaves two
sounds playing over each other.

### 4. Correct the clusters

The right-hand panel acts on the current selection:

| control | effect |
|---|---|
| **Move to cluster** | send the selected points to a specific cluster id |
| **New cluster** | send them to a fresh id |
| **Mark unassigned** | set them to "no cluster" |
| **Undo** / **Redo** | step through the edit history |
| **Renumber** | compact ids to `0..k-1`, ordered by smallest current id |

Per cluster, in the cluster list:

| control | effect |
|---|---|
| **Select** | select every point in that cluster |
| **Rename** | change its id, merging if the target already exists |
| **Merge** | fold it into another cluster |
| **Remove** | delete it, leaving its points unassigned |

Every edit is applied to a label array on the server and pushed onto a bounded
undo stack, so nothing is destructive and everything is reversible.

### 5. Re-cluster without re-embedding

Adjust `nmin`, `max_spks`, `mer_cos` or the method and press **Re-cluster**. The
embeddings are already computed, so this is fast and it discards your manual
edits — deliberately, since the labels it produces are new. Use it to find a
sensible starting point, then correct by hand.

Switching the projection (t-SNE / UMAP / PCA) re-renders from the same
embeddings.

### 6. Export

Pick a destination and a mode.

| mode | effect |
|---|---|
| **copy** | default; the scanned files are left untouched |
| **move** | relocates the files; asks for confirmation first |

Files land in `<destination>/<cluster>/`, with the cluster id zero-padded so the
directory listing sorts naturally. Points marked unassigned go to `noise/`. Name
collisions are disambiguated with a short content hash rather than overwriting.

**Download CSV** gives you the current labels and 2D coordinates without writing
anything to disk.

### Importing existing embeddings

If you already have an embedding matrix (`.npy`, or a headerless `.csv`), point
the panel at it and press **Load and cluster**. This skips inference entirely and
goes straight to clustering and review — useful for comparing embedding sources,
or for revisiting an analysis without re-running the encoder. If the row count
matches the scanned file list, the points keep their audio and remain
auditionable.

---

## API

The interface is a client of this API; anything the UI does can be scripted.

| method | path | purpose |
|---|---|---|
| `GET` | `/api/health` | liveness |
| `GET` | `/api/state` | summary plus every point |
| `GET` | `/api/summary` | summary only |
| `GET` | `/api/weights` | registry entries and their notes |
| `POST` | `/api/scan` | scan a directory |
| `POST` | `/api/run` | start a run; returns a task id |
| `GET` | `/api/tasks/{id}` | task status |
| `GET` | `/api/tasks/{id}/events` | SSE progress stream |
| `POST` | `/api/tasks/{id}/cancel` | request cancellation |
| `POST` | `/api/labels/assign` | move points, or into a new cluster |
| `POST` | `/api/labels/merge` | merge clusters |
| `POST` | `/api/labels/rename` | change a cluster id |
| `POST` | `/api/labels/split` | move points out into a new cluster |
| `POST` | `/api/labels/remove` | delete a cluster |
| `POST` | `/api/labels/compact` | renumber |
| `POST` | `/api/undo`, `/api/redo` | history |
| `POST` | `/api/recluster` | re-cluster the cached embeddings |
| `POST` | `/api/projection` | change the 2D projection |
| `POST` | `/api/import-embeddings` | load an external matrix |
| `POST` | `/api/export` | write files out |
| `GET` | `/api/export.csv` | download the current labels |
| `GET` | `/media/{token}` | audio stream (range-capable) |

Interactive API documentation is at `/docs`.

### A scripting example

```bash
curl -s -X POST localhost:8000/api/scan -H 'content-type: application/json' \
     -d '{"root": "/path/to/audio"}'

TASK=$(curl -s -X POST localhost:8000/api/run -H 'content-type: application/json' \
     -d '{"encoder": "timbre", "cluster": "spectral", "nmin": 2}' | jq -r .task_id)

curl -s localhost:8000/api/tasks/$TASK/events      # SSE progress

curl -s -X POST localhost:8000/api/labels/assign -H 'content-type: application/json' \
     -d '{"indices": [0, 1, 2], "target": 7}'

curl -s -X POST localhost:8000/api/export -H 'content-type: application/json' \
     -d '{"dest": "/path/to/output", "mode": "copy"}'
```
