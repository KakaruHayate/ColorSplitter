"""Training loop.

The schedule defaults below (Adam, lr 1e-3, StepLR every 5000 steps by 0.8) are
reasonable starting points carried over from the lineage this training code
descends from, *not* recovered facts — the original scripts are gone, so nothing
here claims archaeological accuracy. Treat them as tunable.

The loop is deliberately small and readable rather than wrapped in a trainer
framework: it has to be runnable, resumable and testable, and it is not the
bottleneck of anything.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .data import Ge2eBatchSampler, TimbreDataset
from .loss import GE2ELoss
from .model import build_encoder, load_checkpoint, save_checkpoint

__all__ = ["TrainConfig", "train"]

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Everything a run needs. Also serialisable for reproducibility."""

    # --- data ---
    dataset: Path | None = None
    output_dir: Path = Path("runs/encoder")
    val_fraction: float = 0.1

    # --- batch shape ---
    speakers_per_batch: int = 8
    utterances_per_speaker: int = 4
    singers_per_batch: int = 4

    # --- optimisation ---
    max_steps: int = 200_000
    lr: float = 1e-3
    weight_decay: float = 0.0
    decay_every: int = 5_000
    decay_rate: float = 0.8
    grad_clip: float = 3.0
    scheduler_floor: float = 1e-5

    # --- bookkeeping ---
    seed: int = 42
    device: str | None = None
    log_every: int = 50
    checkpoint_every: int = 5_000
    val_every: int = 500
    val_batches: int = 8
    resume_from: Path | None = None

    # --- guards ---
    #: Abort rather than write a broken checkpoint if the loss goes non-finite.
    stop_on_nonfinite: bool = True

    def to_json(self, path: Path) -> None:
        payload = asdict(self)
        payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in payload.items()}
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown training config keys: {sorted(unknown)}")
        for key in ("dataset", "output_dir", "resume_from"):
            if raw.get(key) is not None:
                raw[key] = Path(raw[key])
        return cls(**raw)


def _split_classes(dataset: TimbreDataset, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split class indices, stratified by singer so validation is not easy."""
    if not 0 <= val_fraction < 0.5:
        raise ValueError("val_fraction must be in [0, 0.5)")
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _singer, indices in sorted(dataset.classes_by_singer().items()):
        if len(indices) == 1:
            train_idx.extend(indices)
            continue
        shuffled = rng.permutation(indices)
        n_val = max(1, int(round(len(indices) * val_fraction)))
        if val_fraction <= 0:
            n_val = 0
        val_idx.extend(int(i) for i in shuffled[:n_val])
        train_idx.extend(int(i) for i in shuffled[n_val:])
    if not train_idx:
        train_idx = list(range(len(dataset)))
    return sorted(train_idx), sorted(val_idx)


class _Subset:
    """View over a subset of a dataset's classes, for the validation split."""

    def __init__(self, dataset: TimbreDataset, indices: list[int]):
        self._dataset = dataset
        self._indices = indices
        self.classes = [dataset.classes[i] for i in indices]
        self.root = dataset.root

    def classes_by_singer(self):
        grouped: dict[str, list[int]] = {}
        for local, spec in enumerate(self.classes):
            grouped.setdefault(spec.singer, []).append(local)
        return grouped

    def load_mel(self, path, rng):
        return self._dataset.load_mel(path, rng)

    def __len__(self) -> int:
        return len(self.classes)


def _evaluate(encoder, loss_fn, sampler: Ge2eBatchSampler, batches: int, device) -> float:
    encoder.eval()
    values = []
    with torch.no_grad():
        for mels, labels in sampler.batches(batches):
            embeds = encoder(torch.from_numpy(mels).to(device))
            shaped = _shape_embeddings(embeds, labels)
            values.append(float(loss_fn(shaped).item()))
    encoder.train()
    return float(np.mean(values)) if values else float("nan")


def _shape_embeddings(embeds: torch.Tensor, labels: np.ndarray) -> torch.Tensor:
    """Reorder a flat batch into ``(N, M, D)`` for the GE2E loss."""
    n_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_classes)
    if len(set(counts.tolist())) != 1:
        raise ValueError(f"GE2E needs a rectangular batch, got class counts {counts.tolist()}")
    m = int(counts[0])
    return embeds.reshape(n_classes, m, -1)


def train(config: TrainConfig) -> dict:
    """Run training. Returns a summary dict; also writes ``run.json``."""
    if config.dataset is None:
        raise ValueError("TrainConfig.dataset is required")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device) if config.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = TimbreDataset(Path(config.dataset))
    stats = dataset.stats()
    log.info("dataset: %s", stats)
    train_idx, val_idx = _split_classes(dataset, config.val_fraction, config.seed)
    train_view = _Subset(dataset, train_idx)
    val_view = _Subset(dataset, val_idx) if val_idx else None

    train_sampler = Ge2eBatchSampler(
        train_view,
        speakers_per_batch=config.speakers_per_batch,
        utterances_per_speaker=config.utterances_per_speaker,
        singers_per_batch=config.singers_per_batch,
        seed=config.seed,
    )
    val_sampler = (
        Ge2eBatchSampler(
            val_view,
            speakers_per_batch=config.speakers_per_batch,
            utterances_per_speaker=config.utterances_per_speaker,
            singers_per_batch=config.singers_per_batch,
            seed=config.seed + 1,
        )
        if val_view is not None
        else None
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(output_dir / "config.json")

    encoder = build_encoder(device=str(device), resume_from=config.resume_from)
    loss_fn = GE2ELoss().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(loss_fn.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.decay_every, gamma=config.decay_rate
    )

    start_step = 0
    if config.resume_from is not None:
        start_step = load_checkpoint(Path(config.resume_from), encoder, optimizer, loss_fn)
        for _ in range(start_step):
            scheduler.step()
        log.info("resumed from %s at step %d", config.resume_from, start_step)

    encoder.train()
    history: list[dict] = []
    running: list[float] = []
    started = time.time()
    last_val: float | None = None
    completed = start_step

    for step, (mels, labels) in enumerate(train_sampler.batches(config.max_steps), start=start_step + 1):
        embeds = encoder(torch.from_numpy(mels).to(device))
        shaped = _shape_embeddings(embeds, labels)
        loss = loss_fn(shaped)

        if not math.isfinite(float(loss)):
            if config.stop_on_nonfinite:
                raise RuntimeError(f"non-finite loss at step {step}; refusing to continue")
            log.warning("non-finite loss at step %d, skipping", step)
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(loss_fn.parameters()), config.grad_clip
            )
        optimizer.step()
        scheduler.step()
        for group in optimizer.param_groups:
            if group["lr"] < config.scheduler_floor:
                group["lr"] = config.scheduler_floor

        running.append(float(loss.item()))
        completed = step

        if step % config.val_every == 0 and val_sampler is not None:
            last_val = _evaluate(encoder, loss_fn, val_sampler, config.val_batches, device)

        if step % config.log_every == 0:
            recent = float(np.mean(running[-config.log_every :]))
            lr_now = optimizer.param_groups[0]["lr"]
            record = {
                "step": step,
                "loss": recent,
                "val_loss": last_val,
                "lr": lr_now,
                "elapsed_s": round(time.time() - started, 1),
            }
            history.append(record)
            log.info(
                "step %d  loss %.4f  val %s  lr %.2e  %.0fs",
                step, recent,
                f"{last_val:.4f}" if last_val is not None else "-",
                lr_now, record["elapsed_s"],
            )

        if config.checkpoint_every and step % config.checkpoint_every == 0:
            save_checkpoint(
                output_dir / "latest.pt",
                encoder=encoder, step=step, loss=float(np.mean(running[-config.log_every :])),
                optimizer=optimizer, similarity=loss_fn,
            )
            save_checkpoint(
                output_dir / f"encoder_{step}.pt",
                encoder=encoder, step=step, loss=float(np.mean(running[-config.log_every :])),
            )

    final_loss = float(np.mean(running[-config.log_every :])) if running else float("nan")
    save_checkpoint(
        output_dir / "latest.pt",
        encoder=encoder, step=completed, loss=final_loss,
        optimizer=optimizer, similarity=loss_fn,
    )
    summary = {
        "steps": completed,
        "loss": final_loss,
        "val_loss": last_val,
        "dataset": stats,
        "train_classes": len(train_idx),
        "val_classes": len(val_idx),
        "device": str(device),
        "output_dir": str(output_dir),
        "elapsed_s": round(time.time() - started, 1),
        "history": history,
    }
    (output_dir / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
