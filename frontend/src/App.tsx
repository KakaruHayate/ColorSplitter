import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Scatter from './Scatter'
import { api, clusterColor, mediaUrl, watchTask } from './api'
import type { Point, Summary, WeightInfo } from './api'

const EMPTY_SUMMARY: Summary = {
  root: null,
  files: 0,
  embedding_dim: 0,
  encoder: null,
  weights: null,
  cluster_method: 'spectral',
  projection_method: 'tsne',
  cluster_params: {},
  can_undo: false,
  can_redo: false,
}

export default function App() {
  const [summary, setSummary] = useState<Summary>(EMPTY_SUMMARY)
  const [points, setPoints] = useState<Point[]>([])
  const [weights, setWeights] = useState<WeightInfo[]>([])

  const [root, setRoot] = useState('')
  const [encoder, setEncoder] = useState('timbre')
  const [weightId, setWeightId] = useState('')
  const [clusterMethod, setClusterMethod] = useState('spectral')
  const [projection, setProjection] = useState('tsne')
  const [nmin, setNmin] = useState(1)
  const [merCos, setMerCos] = useState('')
  const [maxSpks, setMaxSpks] = useState(14)
  const [workers, setWorkers] = useState(1)

  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [target, setTarget] = useState(0)
  const [activeToken, setActiveToken] = useState<string | null>(null)
  const [exportDest, setExportDest] = useState('')
  const [exportMode, setExportMode] = useState('copy')
  const [importPath, setImportPath] = useState('')

  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState({ stage: '', done: 0, total: 0 })
  const [message, setMessage] = useState<{ kind: 'info' | 'error'; text: string } | null>(null)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const unwatchRef = useRef<(() => void) | null>(null)

  const fail = (error: unknown) =>
    setMessage({ kind: 'error', text: error instanceof Error ? error.message : String(error) })
  const ok = (text: string) => setMessage({ kind: 'info', text })

  const refresh = useCallback(async () => {
    try {
      const state = await api.state()
      setSummary(state.summary)
      setPoints(state.points)
    } catch (error) {
      fail(error)
    }
  }, [])

  useEffect(() => {
    refresh()
    api.weights().then((payload) => setWeights(payload.weights)).catch(() => undefined)
  }, [refresh])

  useEffect(() => () => unwatchRef.current?.(), [])

  // --- playback -----------------------------------------------------------

  const play = useCallback((token: string) => {
    if (audioRef.current && activeToken === token) {
      if (audioRef.current.paused) {
        void audioRef.current.play()
      } else {
        audioRef.current.pause()
        setActiveToken(null)
      }
      return
    }
    audioRef.current?.pause()
    const audio = new Audio(mediaUrl(token))
    audio.onended = () => setActiveToken(null)
    audio.onerror = () => setActiveToken(null)
    audioRef.current = audio
    setActiveToken(token)
    void audio.play().catch(() => setActiveToken(null))
  }, [activeToken])

  // --- interactions -------------------------------------------------------

  const handlePick = useCallback(
    (index: number, additive: boolean) => {
      setSelected((previous) => {
        const next = additive ? new Set(previous) : new Set<number>()
        if (additive && previous.has(index)) {
          next.delete(index)
        } else {
          next.add(index)
        }
        return next
      })
      const point = points[index]
      if (point?.playable) play(point.token)
    },
    [points, play],
  )

  const handleBoxSelect = useCallback((indices: number[], additive: boolean) => {
    setSelected((previous) => (additive ? new Set([...previous, ...indices]) : new Set(indices)))
  }, [])

  const applySummary = (next: Summary) => {
    setSummary(next)
    if (next.n_clusters !== undefined) setTarget(Math.max(0, next.n_clusters - 1))
  }

  const guarded = async (action: () => Promise<void>) => {
    setBusy(true)
    setMessage(null)
    try {
      await action()
    } catch (error) {
      fail(error)
    } finally {
      setBusy(false)
    }
  }

  const doScan = () =>
    guarded(async () => {
      const result = await api.scan(root)
      ok(`Found ${result.files} audio files under ${result.root}`)
      setExportDest(`${result.root}-colorsplitter`)
      await refresh()
    })

  const doRun = () =>
    guarded(async () => {
      const payload = {
        root: root || null,
        encoder,
        weights: weightId || null,
        workers,
        cluster: clusterMethod,
        nmin,
        mer_cos: merCos === '' ? null : Number(merCos),
        max_spks: maxSpks,
        projection,
      }
      const { task_id: taskId } = await api.run(payload)
      setProgress({ stage: 'starting', done: 0, total: 0 })
      await new Promise<void>((resolve) => {
        unwatchRef.current?.()
        unwatchRef.current = watchTask(taskId, (event) => {
          if (event.type === 'progress' || event.type === 'start') {
            setProgress({
              stage: String(event.stage ?? ''),
              done: Number(event.done ?? 0),
              total: Number(event.total ?? 0),
            })
          }
          if (event.type === 'end' || event.type === 'closed') {
            if (event.status === 'error') fail(event.error ?? 'run failed')
            else ok('Run finished')
            resolve()
          }
        })
      })
      await refresh()
    })

  const labelAction = (action: () => Promise<Summary>) =>
    guarded(async () => {
      applySummary(await action())
      await refresh()
    })

  const clusters = useMemo(() => {
    const groups = new Map<number, number[]>()
    for (const point of points) {
      const bucket = groups.get(point.cluster)
      if (bucket) bucket.push(point.i)
      else groups.set(point.cluster, [point.i])
    }
    return [...groups.entries()].sort((a, b) => a[0] - b[0])
  }, [points])

  const selectionSize = selected.size
  const selectedIndices = useMemo(() => [...selected].sort((a, b) => a - b), [selected])

  const progressPercent =
    progress.total > 0 ? Math.min(100, Math.round((progress.done / progress.total) * 100)) : 0

  return (
    <div className="app">
      <header className="topbar">
        <h1>ColorSplitter</h1>
        <div className="topbar-meta">
          {summary.root ? <span title={summary.root}>root: {summary.root}</span> : <span>no directory loaded</span>}
          <span>{summary.files} files</span>
          {summary.embedding_dim > 0 && <span>{summary.embedding_dim}-d</span>}
          {summary.weights && <span>weights: {summary.weights}</span>}
          {summary.n_clusters !== undefined && <span>{summary.n_clusters} clusters</span>}
          {summary.noise ? <span>{summary.noise} unassigned</span> : null}
        </div>
      </header>

      <div className="layout">
        {/* ---------------- left: input and run ---------------- */}
        <aside className="panel">
          <section>
            <h2>Input</h2>
            <label>
              Directory
              <input
                value={root}
                placeholder="/path/to/audio"
                onChange={(event) => setRoot(event.target.value)}
              />
            </label>
            <div className="row">
              <button onClick={doScan} disabled={busy || !root}>
                Scan
              </button>
              <button className="primary" onClick={doRun} disabled={busy || (!root && summary.files === 0)}>
                Run pipeline
              </button>
            </div>
            <p className="hint">
              Point it at any directory — every audio file underneath is picked up, whatever the
              layout.
            </p>
          </section>

          <section>
            <h2>Encoder</h2>
            <label>
              Encoder
              <select value={encoder} onChange={(event) => setEncoder(event.target.value)}>
                <option value="timbre">timbre (default)</option>
                <option value="speaker">speaker (upstream)</option>
                <option value="emotion">emotion</option>
                <option value="mix">timbre + emotion</option>
              </select>
            </label>
            <label>
              Weight
              <select value={weightId} onChange={(event) => setWeightId(event.target.value)}>
                <option value="">registry default</option>
                {weights.map((weight) => (
                  <option key={weight.id} value={weight.id}>
                    {weight.id}
                    {weight.step ? ` · step ${weight.step}` : ''}
                    {weight.default ? ' · default' : ''}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Decode workers
              <input
                type="number"
                min={1}
                max={32}
                value={workers}
                onChange={(event) => setWorkers(Number(event.target.value))}
              />
            </label>
          </section>

          <section>
            <h2>Clustering</h2>
            <label>
              Method
              <select value={clusterMethod} onChange={(event) => setClusterMethod(event.target.value)}>
                <option value="spectral">spectral</option>
                <option value="umap_hdbscan">umap + hdbscan</option>
              </select>
            </label>
            <label>
              Min clusters (nmin)
              <input
                type="number"
                min={1}
                value={nmin}
                onChange={(event) => setNmin(Number(event.target.value))}
              />
            </label>
            <label>
              Max clusters
              <input
                type="number"
                min={2}
                value={maxSpks}
                onChange={(event) => setMaxSpks(Number(event.target.value))}
              />
            </label>
            <label>
              Merge above cosine
              <input
                value={merCos}
                placeholder="off"
                onChange={(event) => setMerCos(event.target.value)}
              />
            </label>
            <div className="row">
              <button
                disabled={busy || summary.embedding_dim === 0}
                onClick={() =>
                  labelAction(() =>
                    api
                      .recluster({
                        method: clusterMethod,
                        nmin,
                        mer_cos: merCos === '' ? null : Number(merCos),
                        max_spks: maxSpks,
                      })
                      .then((payload) => {
                        setPoints(payload.points)
                        return payload.summary
                      }),
                  )
                }
              >
                Re-cluster
              </button>
            </div>
            <label>
              Projection
              <select
                value={projection}
                onChange={(event) => {
                  const method = event.target.value
                  setProjection(method)
                  if (summary.embedding_dim > 0) {
                    labelAction(() =>
                      api.reproject(method).then((payload) => {
                        setPoints(payload.points)
                        return payload.summary
                      }),
                    )
                  }
                }}
              >
                <option value="tsne">t-SNE</option>
                <option value="umap">UMAP</option>
                <option value="pca">PCA</option>
              </select>
            </label>
          </section>

          <section>
            <h2>Import embeddings</h2>
            <label>
              .npy / .csv path
              <input
                value={importPath}
                placeholder="/path/to/embeddings.npy"
                onChange={(event) => setImportPath(event.target.value)}
              />
            </label>
            <div className="row">
              <button
                disabled={busy || !importPath}
                onClick={() =>
                  labelAction(() =>
                    api.importEmbeddings(importPath).then((payload) => {
                      setPoints(payload.points)
                      return payload.summary
                    }),
                  )
                }
              >
                Load and cluster
              </button>
            </div>
          </section>
        </aside>

        {/* ---------------- centre: plot ---------------- */}
        <main className="plot">
          <div className="plot-canvas">
            <Scatter
              points={points}
              selected={selected}
              activeToken={activeToken}
              onPick={handlePick}
              onBoxSelect={handleBoxSelect}
            />
          </div>
          <div className="plot-bar">
            <span>
              {points.length} points · {selectionSize} selected
            </span>
            <span className="hint">
              click a point to audition · shift-click to extend · drag to box-select
            </span>
            {activeToken && (
              <button
                onClick={() => {
                  audioRef.current?.pause()
                  setActiveToken(null)
                }}
              >
                Stop audio
              </button>
            )}
          </div>
          {progress.total > 0 && (
            <div className="progress">
              <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
              <span>
                {progress.stage} {progress.done}/{progress.total}
              </span>
            </div>
          )}
        </main>

        {/* ---------------- right: edit and export ---------------- */}
        <aside className="panel">
          <section>
            <h2>Selection</h2>
            <div className="row">
              <label>
                Cluster id
                <input
                  type="number"
                  value={target}
                  onChange={(event) => setTarget(Number(event.target.value))}
                />
              </label>
            </div>
            <div className="row">
              <button
                disabled={busy || selectionSize === 0}
                onClick={() => labelAction(() => api.assign(selectedIndices, target))}
              >
                Move to cluster
              </button>
              <button
                disabled={busy || selectionSize === 0}
                onClick={() => labelAction(() => api.assign(selectedIndices, 0, true))}
              >
                New cluster
              </button>
            </div>
            <div className="row">
              <button disabled={selectionSize === 0} onClick={() => setSelected(new Set())}>
                Clear
              </button>
              <button
                disabled={busy || selectionSize === 0}
                onClick={() => labelAction(() => api.assign(selectedIndices, -1))}
              >
                Mark unassigned
              </button>
            </div>
            <div className="row">
              <button disabled={busy || !summary.can_undo} onClick={() => labelAction(api.undo)}>
                Undo
              </button>
              <button disabled={busy || !summary.can_redo} onClick={() => labelAction(api.redo)}>
                Redo
              </button>
              <button disabled={busy || summary.embedding_dim === 0} onClick={() => labelAction(api.compact)}>
                Renumber
              </button>
            </div>
          </section>

          <section className="clusters">
            <h2>Clusters</h2>
            {clusters.length === 0 && <p className="hint">Nothing clustered yet.</p>}
            {clusters.map(([id, indices]) => (
              <div key={id} className={`cluster-row${selected.size > 0 && indices.every((i) => selected.has(i)) ? ' active' : ''}`}>
                <span className="swatch" style={{ background: clusterColor(id) }} />
                <span className="cluster-name">{id < 0 ? 'unassigned' : `cluster ${id}`}</span>
                <span className="count">{indices.length}</span>
                <button onClick={() => setSelected(new Set(indices))}>Select</button>
                <button
                  disabled={busy || id < 0}
                  onClick={() => {
                    const value = window.prompt(`Rename cluster ${id} to:`, String(id))
                    if (value === null) return
                    const parsed = Number(value)
                    if (!Number.isInteger(parsed)) {
                      fail('cluster ids must be integers')
                      return
                    }
                    labelAction(() => api.rename(id, parsed))
                  }}
                >
                  Rename
                </button>
                <button
                  disabled={busy || id < 0}
                  onClick={() => labelAction(() => api.remove(id, null))}
                >
                  Remove
                </button>
                <button
                  disabled={busy || id < 0 || clusters.length < 2}
                  onClick={() => {
                    const others = clusters.map(([other]) => other).filter((other) => other >= 0 && other !== id)
                    const value = window.prompt(`Merge cluster ${id} into which id?\nAvailable: ${others.join(', ')}`)
                    if (value === null) return
                    const parsed = Number(value)
                    if (!Number.isInteger(parsed)) {
                      fail('cluster ids must be integers')
                      return
                    }
                    labelAction(() => api.merge([id], parsed))
                  }}
                >
                  Merge
                </button>
              </div>
            ))}
          </section>

          <section>
            <h2>Export</h2>
            <label>
              Destination
              <input
                value={exportDest}
                placeholder="/path/to/output"
                onChange={(event) => setExportDest(event.target.value)}
              />
            </label>
            <label>
              Mode
              <select value={exportMode} onChange={(event) => setExportMode(event.target.value)}>
                <option value="copy">copy — leaves the source alone</option>
                <option value="move">move — relocates the files</option>
              </select>
            </label>
            {exportMode === 'move' && <p className="warn">Move relocates the originals out of the scanned directory.</p>}
            <div className="row">
              <button
                className="primary"
                disabled={busy || !exportDest || summary.embedding_dim === 0}
                onClick={() =>
                  guarded(async () => {
                    if (exportMode === 'move' && !window.confirm('Move the files? The originals will be relocated.')) {
                      return
                    }
                    const report = await api.exportTo(exportDest, exportMode)
                    ok(`Wrote ${report.written} files to ${report.dest}`)
                  })
                }
              >
                Export
              </button>
              <a className="button" href="/api/export.csv" download="clusters.csv">
                Download CSV
              </a>
            </div>
            {summary.export && (
              <p className="hint">
                last export: {summary.export.written} files, mode {summary.export.mode}
              </p>
            )}
          </section>

          {message && <div className={`message ${message.kind}`}>{message.text}</div>}
        </aside>
      </div>
    </div>
  )
}
