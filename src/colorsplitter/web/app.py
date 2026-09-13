"""Web interface.

One process, one port: this serves the JSON API, the Server-Sent Events progress
stream, the audio stream and the compiled frontend. The original viewer needed a
second ``http-server`` for audio plus CORS, and built audio URLs out of raw
filesystem paths; here audio is addressed by an opaque token and streamed from
the same origin with HTTP range support (so seeking works).

The API is a thin shell over :mod:`colorsplitter.core.pipeline` — no capability
lives only here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.pipeline import CLUSTER_METHODS
from ..core.reduce import PROJECTION_METHODS
from .state import SessionState
from .tasks import TaskManager

__all__ = ["create_app"]

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
_CHUNK = 1 << 18


# --- request models ---------------------------------------------------------


class ScanRequest(BaseModel):
    root: str


class RunRequest(BaseModel):
    root: Optional[str] = None
    encoder: str = "timbre"
    weights: Optional[str] = None
    device: Optional[str] = None
    batch_size: int = 64
    workers: int = 1
    amp: bool = False
    trim_silences: str = "auto"
    chunk: int = 256
    cluster: str = "spectral"
    nmin: int = 1
    mer_cos: Optional[float] = None
    max_spks: int = 14
    min_cluster_size: int = 4
    eigen_solver: str = "auto"
    projection: str = "tsne"


class AssignRequest(BaseModel):
    indices: list[int] = Field(default_factory=list)
    target: int = 0
    new_cluster: bool = False


class MergeRequest(BaseModel):
    sources: list[int] = Field(default_factory=list)
    target: int


class RenameRequest(BaseModel):
    old: int
    new: int


class SplitRequest(BaseModel):
    cluster: int
    indices: list[int] = Field(default_factory=list)


class RemoveRequest(BaseModel):
    cluster: int
    reassign_to: Optional[int] = None


class ReclusterRequest(BaseModel):
    method: Optional[str] = None
    nmin: Optional[int] = None
    mer_cos: Optional[float] = None
    max_spks: Optional[int] = None
    min_cluster_size: Optional[int] = None
    eigen_solver: Optional[str] = None


class ProjectionRequest(BaseModel):
    method: str = "tsne"


class ExportRequest(BaseModel):
    dest: str
    mode: str = "copy"


class ImportRequest(BaseModel):
    path: str
    keys: Optional[list[str]] = None


# --- app --------------------------------------------------------------------


def create_app(
    *,
    cache_dir: Path,
    registry_path: Optional[Path] = None,
    initial_root: Optional[Path] = None,
) -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="ColorSplitter", version="2.0.0")
    state = SessionState(cache_dir=Path(cache_dir), registry_path=registry_path)
    tasks = TaskManager()

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/weights")
    def weights() -> dict:
        try:
            return {"weights": state.weight_choices()}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"cannot read the weight registry: {exc}") from exc

    @app.get("/api/state")
    def get_state() -> dict:
        return {"summary": state.summary(), "points": state.points()}

    @app.get("/api/summary")
    def get_summary() -> dict:
        return state.summary()

    # --- tasks --------------------------------------------------------------

    @app.post("/api/scan")
    def do_scan(req: ScanRequest) -> dict:
        try:
            dataset = state.scan(Path(req.root))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"files": len(dataset), "root": str(dataset.root)}

    @app.post("/api/run")
    def do_run(req: RunRequest) -> dict:
        if state.dataset is None and not req.root:
            raise HTTPException(400, "scan a directory first, or pass root")
        if req.cluster not in CLUSTER_METHODS:
            raise HTTPException(400, f"cluster must be one of {CLUSTER_METHODS}")
        if req.projection not in PROJECTION_METHODS:
            raise HTTPException(400, f"projection must be one of {PROJECTION_METHODS}")

        from ..core import pipeline
        from ..core.embed import EmbedConfig

        config = EmbedConfig(
            encoder=req.encoder,
            weights_id=req.weights,
            device=req.device,
            batch_size=req.batch_size,
            workers=req.workers,
            amp=req.amp,
            trim_silences={"auto": "auto", "yes": True, "no": False}.get(req.trim_silences, "auto"),
            cache_dir=state.cache_dir,
            chunk=req.chunk,
        )

        def work(task, cancel):
            if req.root:
                state.scan(Path(req.root))
            state.embed_config = config
            report = tasks.progress_fn(task)

            embeddings = state.embed_dataset(progress=report)
            if cancel.is_set():
                return None
            task.stage = "clustering"
            clusters = pipeline.cluster_embeddings(
                embeddings.embeds,
                method=req.cluster,
                nmin=req.nmin,
                mer_cos=req.mer_cos,
                max_num_spks=req.max_spks,
                min_cluster_size=req.min_cluster_size,
                eigen_solver=req.eigen_solver,
            )
            if cancel.is_set():
                return None
            task.stage = "projecting"
            projection = pipeline.project_embeddings(
                embeddings.embeds,
                keys=embeddings.keys,
                method=req.projection,
                cache_dir=state.cache_dir,
            )
            state.set_run(
                embeddings,
                clusters,
                projection,
                cluster_method=req.cluster,
                cluster_params=clusters.params,
                projection_method=req.projection,
            )
            return {"n_clusters": clusters.n_clusters, "sizes": clusters.sizes()}

        task = tasks.submit("run", work)
        return {"task_id": task.id}

    @app.get("/api/tasks/{task_id}")
    def task_status(task_id: str) -> dict:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "unknown task")
        return task.snapshot()

    @app.post("/api/tasks/{task_id}/cancel")
    def task_cancel(task_id: str) -> dict:
        if not tasks.cancel(task_id):
            raise HTTPException(404, "unknown task")
        return {"cancelled": True}

    @app.get("/api/tasks/{task_id}/events")
    async def task_events(task_id: str, request: Request) -> StreamingResponse:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "unknown task")

        async def stream():
            cursor = 0
            while True:
                if await request.is_disconnected():
                    return
                while cursor < len(task.events):
                    event = task.events[cursor]
                    cursor += 1
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                if task.status in ("done", "error", "cancelled"):
                    yield f"data: {json.dumps({'type': 'closed', **task.snapshot()}, default=str)}\n\n"
                    return
                await asyncio.sleep(0.2)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- label editing ------------------------------------------------------

    def _edit(payload: dict) -> dict:
        try:
            state.assign([] if False else [], 0) if False else None  # no-op guard
            return {"summary": state.summary(), **payload}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/labels/assign")
    def labels_assign(req: AssignRequest) -> dict:
        try:
            if req.new_cluster:
                state.create_cluster_from(req.indices)
            else:
                state.assign(req.indices, req.target)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return state.summary()

    @app.post("/api/labels/merge")
    def labels_merge(req: MergeRequest) -> dict:
        try:
            state.merge(req.sources, req.target)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return state.summary()

    @app.post("/api/labels/rename")
    def labels_rename(req: RenameRequest) -> dict:
        try:
            state.rename(req.old, req.new)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return state.summary()

    @app.post("/api/labels/split")
    def labels_split(req: SplitRequest) -> dict:
        try:
            state.split(req.cluster, req.indices)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return state.summary()

    @app.post("/api/labels/remove")
    def labels_remove(req: RemoveRequest) -> dict:
        try:
            state.remove(req.cluster, req.reassign_to)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return state.summary()

    @app.post("/api/labels/compact")
    def labels_compact() -> dict:
        try:
            state.compact()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return state.summary()

    @app.post("/api/undo")
    def undo() -> dict:
        try:
            state.undo()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return state.summary()

    @app.post("/api/redo")
    def redo() -> dict:
        try:
            state.redo()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return state.summary()

    # --- recompute ----------------------------------------------------------

    @app.post("/api/recluster")
    def recluster(req: ReclusterRequest) -> dict:
        params = {k: v for k, v in req.model_dump().items() if v is not None}
        try:
            state.recluster(**params)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"summary": state.summary(), "points": state.points()}

    @app.post("/api/projection")
    def reprojection(req: ProjectionRequest) -> dict:
        if req.method not in PROJECTION_METHODS:
            raise HTTPException(400, f"projection must be one of {PROJECTION_METHODS}")
        try:
            state.reproject(req.method)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"summary": state.summary(), "points": state.points()}

    # --- import / export ----------------------------------------------------

    @app.post("/api/import-embeddings")
    def import_embeddings(req: ImportRequest) -> dict:
        path = Path(req.path)
        if not path.exists():
            raise HTTPException(404, f"not found: {path}")
        try:
            embeds = np.load(path) if path.suffix != ".csv" else np.loadtxt(path, delimiter=",", skiprows=1)
            state.import_embeddings(embeds, req.keys)
            state.reproject(state.projection_method)
            result = state.recluster(nmin=1, mer_cos=None, max_spks=14)
        except Exception as exc:  # noqa: BLE001 - reported to the client
            raise HTTPException(400, f"cannot import embeddings: {exc}") from exc
        return {"summary": state.summary(), "points": state.points(), "n_clusters": result.n_clusters}

    @app.post("/api/export")
    def do_export(req: ExportRequest) -> dict:
        if req.mode not in ("copy", "move"):
            raise HTTPException(400, "mode must be 'copy' or 'move'")
        try:
            report = state.export(Path(req.dest), req.mode)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {
            "dest": str(report.dest),
            "mode": report.mode,
            "written": report.written,
            "skipped": report.skipped,
            "per_cluster": {str(k): v for k, v in report.per_cluster.items()},
            "errors": report.errors[:20],
        }

    @app.get("/api/export.csv")
    def export_csv() -> Response:
        if state.embeddings is None or state.history is None:
            raise HTTPException(409, "nothing to export yet")
        lines = ["key,cluster,x,y"]
        coords = state.projection.coords if state.projection is not None else None
        current = state.labels
        for i, key in enumerate(state.embeddings.keys):
            x = float(coords[i, 0]) if coords is not None else 0.0
            y = float(coords[i, 1]) if coords is not None else 0.0
            lines.append(f'"{key}",{int(current[i])},{x},{y}')
        return Response(
            "\n".join(lines) + "\n",
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="clusters.csv"'},
        )

    # --- media --------------------------------------------------------------

    @app.get("/media/{token}")
    def media(token: str, request: Request) -> Response:
        path = state.media_path(token)
        if path is None or not path.exists():
            raise HTTPException(404, "unknown media token")

        size = path.stat().st_size
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = request.headers.get("range")
        if not range_header:
            return FileResponse(path, media_type=media_type)

        try:
            unit, _, spec = range_header.partition("=")
            if unit.strip().lower() != "bytes":
                raise ValueError
            start_s, _, end_s = spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
        except ValueError:
            raise HTTPException(416, "malformed Range header") from None
        end = min(end, size - 1)
        if start > end or start >= size:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

        def body():
            remaining = end - start + 1
            with open(path, "rb") as fh:
                fh.seek(start)
                while remaining > 0:
                    block = fh.read(min(_CHUNK, remaining))
                    if not block:
                        break
                    remaining -= len(block)
                    yield block

        return StreamingResponse(
            body(),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(end - start + 1),
            },
        )

    # --- static frontend ----------------------------------------------------

    if STATIC_DIR.exists() and any(STATIC_DIR.iterdir()):
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    else:  # pragma: no cover - only before the frontend is built
        @app.get("/")
        def placeholder() -> JSONResponse:
            return JSONResponse(
                {
                    "error": "frontend not built",
                    "hint": "cd frontend && npm install && npm run build",
                    "api": "/docs",
                },
                status_code=503,
            )

    if initial_root is not None:
        try:
            state.scan(Path(initial_root))
        except FileNotFoundError:
            log.warning("initial root does not exist: %s", initial_root)

    return app
