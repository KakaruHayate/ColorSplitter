export interface Point {
  i: number
  key: string
  name: string
  cluster: number
  x: number
  y: number
  token: string
  playable: boolean
}

export interface Summary {
  root: string | null
  files: number
  embedding_dim: number
  encoder: string | null
  weights: string | null
  cluster_method: string
  projection_method: string
  cluster_params: Record<string, unknown>
  can_undo: boolean
  can_redo: boolean
  n_clusters?: number
  cluster_sizes?: Record<string, number>
  noise?: number
  export?: { dest: string; mode: string; written: number; errors: number }
}

export interface WeightInfo {
  id: string
  purpose: string
  step: number | null
  default: boolean
}

export interface TaskSnapshot {
  id: string
  name: string
  status: 'pending' | 'running' | 'done' | 'error' | 'cancelled'
  stage: string
  done: number
  total: number
  percent: number
  error?: string | null
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail ?? detail
    } catch {
      /* keep the status text */
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export const api = {
  state: () => request<{ summary: Summary; points: Point[] }>('/api/state'),
  weights: () => request<{ weights: WeightInfo[] }>('/api/weights'),
  scan: (root: string) => request<{ files: number; root: string }>('/api/scan', {
    method: 'POST',
    body: JSON.stringify({ root }),
  }),
  run: (payload: Record<string, unknown>) => request<{ task_id: string }>('/api/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  assign: (indices: number[], target: number, newCluster = false) =>
    request<Summary>('/api/labels/assign', {
      method: 'POST',
      body: JSON.stringify({ indices, target, new_cluster: newCluster }),
    }),
  merge: (sources: number[], target: number) =>
    request<Summary>('/api/labels/merge', {
      method: 'POST',
      body: JSON.stringify({ sources, target }),
    }),
  rename: (oldId: number, newId: number) =>
    request<Summary>('/api/labels/rename', {
      method: 'POST',
      body: JSON.stringify({ old: oldId, new: newId }),
    }),
  remove: (cluster: number, reassignTo: number | null) =>
    request<Summary>('/api/labels/remove', {
      method: 'POST',
      body: JSON.stringify({ cluster, reassign_to: reassignTo }),
    }),
  compact: () => request<Summary>('/api/labels/compact', { method: 'POST' }),
  undo: () => request<Summary>('/api/undo', { method: 'POST' }),
  redo: () => request<Summary>('/api/redo', { method: 'POST' }),
  recluster: (payload: Record<string, unknown>) =>
    request<{ summary: Summary; points: Point[] }>('/api/recluster', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  reproject: (method: string) =>
    request<{ summary: Summary; points: Point[] }>('/api/projection', {
      method: 'POST',
      body: JSON.stringify({ method }),
    }),
  importEmbeddings: (path: string) =>
    request<{ summary: Summary; points: Point[] }>('/api/import-embeddings', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  exportTo: (dest: string, mode: string) =>
    request<{
      dest: string
      mode: string
      written: number
      skipped: number
      per_cluster: Record<string, number>
      errors: string[]
    }>('/api/export', { method: 'POST', body: JSON.stringify({ dest, mode }) }),
}

export const mediaUrl = (token: string) => `/media/${token}`

/** Subscribe to a task's progress stream. Returns an unsubscribe function. */
export function watchTask(
  taskId: string,
  onEvent: (event: Record<string, unknown>) => void,
): () => void {
  const source = new EventSource(`/api/tasks/${taskId}/events`)
  source.onmessage = (message) => {
    try {
      const payload = JSON.parse(message.data)
      onEvent(payload)
      if (payload.type === 'closed') source.close()
    } catch {
      /* ignore malformed frames */
    }
  }
  source.onerror = () => source.close()
  return () => source.close()
}

/** Deterministic, well-spread colour per cluster id. */
export function clusterColor(id: number): string {
  if (id < 0) return '#8a8a8a'
  const hue = (id * 137.508) % 360
  return `hsl(${hue.toFixed(0)}, 62%, 52%)`
}
