"""Background task registry with progress reporting.

Long operations (scanning, decoding, embedding, clustering) run in a worker
thread so the HTTP layer stays responsive. Each task keeps an append-only event
log; the SSE endpoint tails it. That is deliberately simpler than a message bus
and makes the progress stream trivially testable.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = ["Task", "TaskManager"]


@dataclass
class Task:
    """One unit of background work."""

    id: str
    name: str
    status: str = "pending"  # pending | running | done | error | cancelled
    stage: str = ""
    done: int = 0
    total: int = 0
    error: Optional[str] = None
    result: Any = None
    events: list[dict] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "stage": self.stage,
            "done": self.done,
            "total": self.total,
            "error": self.error,
            "percent": round(100.0 * self.done / self.total, 1) if self.total else 0.0,
        }


class TaskManager:
    """Spawns tasks on worker threads and tracks their progress."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def create(self, name: str) -> Task:
        task = Task(id=uuid.uuid4().hex[:12], name=name)
        with self._lock:
            self._tasks[task.id] = task
            self._cancel[task.id] = threading.Event()
        return task

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        event = self._cancel.get(task_id)
        if event is None:
            return False
        event.set()
        return True

    def cancellation_event(self, task_id: str) -> threading.Event:
        return self._cancel.get(task_id, threading.Event())

    def _notify(self, task: Task, kind: str, payload: dict) -> None:
        task.events.append({"type": kind, **payload})

    def start(self, task: Task, work: Callable[[Task, threading.Event], Any]) -> Task:
        """Run *work(task, cancel_event)* on a daemon thread."""

        def runner() -> None:
            cancel = self.cancellation_event(task.id)
            task.status = "running"
            self._notify(task, "start", task.snapshot())
            try:
                task.result = work(task, cancel)
                if cancel.is_set():
                    task.status = "cancelled"
                else:
                    task.status = "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                task.status = "error"
                task.error = f"{type(exc).__name__}: {exc}"
                task.events.append({"type": "traceback", "text": traceback.format_exc()})
            finally:
                self._notify(task, "end", task.snapshot())

        threading.Thread(target=runner, name=f"cs-{task.id}", daemon=True).start()
        return task

    def submit(self, name: str, work: Callable[[Task, threading.Event], Any]) -> Task:
        """Create and start a task in one call."""
        return self.start(self.create(name), work)

    def progress_fn(self, task: Task) -> Callable[[str, int, int], None]:
        """Build a ``(stage, done, total)`` callback bound to *task*."""

        def report(stage: str, done: int, total: int) -> None:
            task.stage = stage
            task.done = int(done)
            task.total = int(total)
            self._notify(task, "progress", task.snapshot())

        return report
