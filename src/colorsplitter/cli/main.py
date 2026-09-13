"""Command-line interface.

Every capability of the pipeline is reachable from here, which is what makes
the tool scriptable and CI-testable. The WebUI wraps the same functions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from ..core.embed import EmbedConfig
from ..core.modelzoo import (
    default_cache_dir,
    fetch_emotion_model,
    load_registry,
    pack_inference_weights,
    resolve_weight,
)

log = logging.getLogger("colorsplitter")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="directory to scan for audio")
    parser.add_argument("--encoder", default="timbre", choices=["timbre", "speaker", "emotion", "mix"])
    parser.add_argument("--weights", default=None, help="registered weight id (see `weights list`)")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="partial utterances per forward pass")
    parser.add_argument("--workers", type=int, default=1, help="parallel audio decoders")
    parser.add_argument("--chunk", type=int, default=256, help="files decoded per batch")
    parser.add_argument("--amp", action="store_true", help="fp16 forward pass on CUDA (slightly lossy)")
    parser.add_argument("--trim-silences", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None, help="cache directory (defaults to a per-user dir)")
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not write caches")
    parser.add_argument("--registry", type=Path, default=None, help="path to registry.json")


def _embed_config(args: argparse.Namespace) -> EmbedConfig:
    trim = {"auto": "auto", "yes": True, "no": False}[args.trim_silences]
    cache_dir = None if args.no_cache else (args.cache_dir or default_cache_dir())
    return EmbedConfig(
        encoder=args.encoder,
        weights_id=args.weights,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        amp=args.amp,
        trim_silences=trim,
        use_cache=not args.no_cache,
        cache_dir=cache_dir,
        chunk=args.chunk,
    )


def _progress_printer(enabled: bool = True):
    state: dict[str, int] = {}

    def report(stage: str, done: int, total: int) -> None:
        if not enabled:
            return
        total = max(total, 1)
        step = max(1, total // 20)
        if done < total and done - state.get(stage, 0) < step:
            return
        state[stage] = done
        sys.stderr.write(f"\r{stage:<10} {done:>7}/{total}")
        sys.stderr.flush()

    return report


def _write_csv(path: Path, cluster_result, projection, keys) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "cluster", "x", "y"])
        for i, key in enumerate(keys):
            writer.writerow(
                [key, int(cluster_result.labels[i]), float(projection.coords[i, 0]), float(projection.coords[i, 1])]
            )


# --- subcommands ------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    from ..core.pipeline import scan

    dataset = scan(args.input)
    print(json.dumps({"root": str(dataset.root), "files": len(dataset)}, ensure_ascii=False))
    for item in dataset.items[: args.limit]:
        print(item.key)
    if len(dataset) > args.limit:
        print(f"... and {len(dataset) - args.limit} more")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from ..core.pipeline import run

    registry = load_registry(args.registry) if args.registry else None
    out_dir = args.output or (args.input.parent / f"{args.input.name}-colorsplitter")
    report = _progress_printer()
    result = run(
        args.input,
        output_dir=out_dir,
        embed_config=_embed_config(args),
        cluster_method=args.cluster,
        nmin=args.nmin,
        mer_cos=args.mer_cos,
        max_num_spks=args.max_spks,
        projection_method=args.projection,
        export=args.export,
        export_mode=args.export_mode,
        cache_dir=None if args.no_cache else (args.cache_dir or default_cache_dir()),
        registry=registry,
        progress=report,
    )
    sys.stderr.write("\n")

    clusters = result["clusters"]
    csv_path = out_dir / "clusters.csv"
    _write_csv(csv_path, clusters, result["projection"], result["embeddings"].keys)

    summary = {
        "files": len(result["dataset"]),
        "embedding_dim": result["embeddings"].dim,
        "encoder": result["embeddings"].encoder,
        "weights": result["embeddings"].weights_id,
        "cluster_method": clusters.method,
        "n_clusters": clusters.n_clusters,
        "cluster_sizes": clusters.sizes(),
        "csv": str(csv_path),
    }
    if result["export"] is not None:
        summary["export"] = {
            "dest": str(result["export"].dest),
            "mode": result["export"].mode,
            "written": result["export"].written,
            "errors": len(result["export"].errors),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("the web interface needs the 'web' extra: pip install 'colorsplitter[web]'", file=sys.stderr)
        return 2

    from ..web.app import create_app

    app = create_app(
        cache_dir=args.cache_dir or default_cache_dir(),
        registry_path=args.registry,
        initial_root=args.input,
    )
    print(f"ColorSplitter WebUI on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_weights(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry) if args.registry else load_registry()
    cache_dir = args.cache_dir or default_cache_dir()

    if args.action == "list":
        payload = [
            {
                "id": entry.id,
                "purpose": entry.purpose,
                "step": entry.step,
                "default": entry.is_default,
                "sha256": (entry.sha256 or "")[:16],
            }
            for entry in registry.encoders
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.action == "fetch":
        for entry in registry.encoders:
            if args.only and entry.id != args.only:
                continue
            path = resolve_weight(registry, entry.id, cache_dir=cache_dir)
            print(f"{entry.id}: {path}")
        if not args.only or args.only == "emotion":
            try:
                print(f"emotion: {fetch_emotion_model(registry, cache_dir=cache_dir)}")
            except Exception as exc:  # noqa: BLE001
                print(f"emotion: FAILED ({exc})", file=sys.stderr)
                return 1
        return 0

    if args.action == "pack":
        if not args.source:
            print("--source is required for `weights pack`", file=sys.stderr)
            return 2
        out = pack_inference_weights(args.source, args.dest, weights_id=args.weights, step=args.step)
        print(str(out))
        return 0

    raise SystemExit(f"unknown weights action: {args.action}")


def cmd_train(args: argparse.Namespace) -> int:
    from ..training.train import TrainConfig, train

    config = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    if args.dataset:
        config.dataset = Path(args.dataset)
    if args.out:
        config.output_dir = Path(args.out)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    train(config)
    return 0


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colorsplitter",
        description="Timbre clustering and filtering for singing datasets.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="list the audio files found under a directory")
    p_scan.add_argument("input", type=Path)
    p_scan.add_argument("--limit", type=int, default=20)
    p_scan.set_defaults(func=cmd_scan)

    p_run = sub.add_parser("run", help="scan, embed, cluster and optionally export")
    _add_common(p_run)
    p_run.add_argument("--output", type=Path, default=None)
    p_run.add_argument("--cluster", default="spectral", choices=["spectral", "umap_hdbscan"])
    p_run.add_argument("--nmin", type=int, default=1, help="minimum number of clusters")
    p_run.add_argument("--mer-cos", type=float, default=None, help="merge clusters above this cosine similarity")
    p_run.add_argument("--max-spks", type=int, default=14, help="upper bound when estimating the cluster count")
    p_run.add_argument("--projection", default="tsne", choices=["tsne", "umap", "pca"])
    p_run.add_argument("--export", action="store_true", help="write the clustered files to disk")
    p_run.add_argument("--export-mode", default="copy", choices=["copy", "move"])
    p_run.set_defaults(func=cmd_run)

    p_serve = sub.add_parser("serve", help="start the web interface")
    p_serve.add_argument("--input", type=Path, default=None, help="preload this directory")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--cache-dir", type=Path, default=None)
    p_serve.add_argument("--registry", type=Path, default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_weights = sub.add_parser("weights", help="inspect, download or repackage weights")
    p_weights.add_argument("action", choices=["list", "fetch", "pack"])
    p_weights.add_argument("--only", default=None, help="limit `fetch` to one weight id")
    p_weights.add_argument("--source", type=Path, default=None, help="checkpoint to strip (for `pack`)")
    p_weights.add_argument("--dest", type=Path, default=Path("models") / "packed.pt")
    p_weights.add_argument("--weights", default=None)
    p_weights.add_argument("--step", type=int, default=None)
    p_weights.add_argument("--cache-dir", type=Path, default=None)
    p_weights.add_argument("--registry", type=Path, default=None)
    p_weights.set_defaults(func=cmd_weights)

    p_train = sub.add_parser("train", help="train a timbre encoder (see docs/training.md)")
    p_train.add_argument("--dataset", type=Path, default=None)
    p_train.add_argument("--config", type=Path, default=None)
    p_train.add_argument("--out", type=Path, default=None)
    p_train.add_argument("--max-steps", type=int, default=None)
    p_train.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
