#!/usr/bin/env python3
"""Produce the weight release assets from what is archived in git history.

Why this exists: a checkpoint was once overwritten by an ordinary-looking
commit, and the only reason it was recoverable is that git kept the blob. The
registry records each weight's blob id so that this can be re-run at any time.

What it does, per registered weight:

1. recovers the archived blob out of git (``git cat-file blob <sha>``),
2. strips the optimiser state, which roughly halves the file and is useless for
   inference,
3. writes it to ``models/release/`` with a name that reflects its actual step
   count,
4. computes the SHA-256 and writes it back into ``models/registry.json`` so the
   published asset is verifiable,
5. prints a manifest to attach to a release.

Usage::

    python scripts/prepare_release.py            # build the assets
    python scripts/prepare_release.py --check    # verify hashes only, no writes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "models" / "registry.json"
RELEASE_DIR = ROOT / "models" / "release"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recover_blob(blob: str, dest: Path) -> Path:
    """Materialise a git blob. Binary-safe: the shell handles the redirection."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as handle:
        result = subprocess.run(
            ["git", "cat-file", "blob", blob],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        raise RuntimeError(f"git cat-file failed for {blob}: {result.stderr.decode().strip()}")
    if dest.stat().st_size == 0:
        raise RuntimeError(f"recovered blob {blob} is empty")
    return dest


def strip_optimiser_state(source: Path, dest: Path, weight_id: str, step: int | None) -> Path:
    """Keep ``model_state`` only. Requires torch."""
    from colorsplitter.core.modelzoo import pack_inference_weights

    return pack_inference_weights(source, dest, weights_id=weight_id, step=step)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify existing assets, write nothing")
    parser.add_argument("--keep-optimizer", action="store_true", help="do not strip optimiser state")
    parser.add_argument("--out", type=Path, default=RELEASE_DIR)
    args = parser.parse_args(argv)

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT / "src"))

    manifest: list[dict] = []
    changed = False

    for entry in registry["encoders"]:
        source = entry.get("source") or {}
        kind = source.get("kind")
        target = args.out / entry["file"]

        if args.check:
            if not target.exists():
                print(f"MISSING  {entry['file']}")
                continue
            actual = sha256_of(target)
            recorded = entry.get("sha256")
            status = "ok" if recorded == actual else "MISMATCH"
            print(f"{status:<9}{entry['file']}  {actual}")
            continue

        if kind == "git-blob":
            blob = source["blob"]
            print(f"{entry['id']}: recovering blob {blob[:12]}…")
            with tempfile.TemporaryDirectory() as tmp:
                raw = recover_blob(blob, Path(tmp) / "raw.pt")
                expected = source.get("sha256")
                if expected:
                    actual = sha256_of(raw)
                    if actual != expected:
                        raise RuntimeError(
                            f"{entry['id']}: blob hash mismatch (expected {expected}, got {actual}). "
                            "The recorded provenance is wrong; refusing to publish."
                        )
                    print(f"  provenance verified ({raw.stat().st_size} bytes)")
                if args.keep_optimizer:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(raw, target)
                else:
                    strip_optimiser_state(raw, target, entry["id"], entry.get("step"))

        elif kind == "upstream":
            if target.exists():
                print(f"{entry['id']}: {target.name} already present")
            else:
                print(f"{entry['id']}: upstream weight not staged (download it with `cs weights fetch`)")
                continue
        else:
            print(f"{entry['id']}: nothing to do (source kind {kind!r})")
            continue

        digest = sha256_of(target)
        size = target.stat().st_size
        if entry.get("sha256") != digest:
            entry["sha256"] = digest
            changed = True
        manifest.append(
            {
                "id": entry["id"],
                "file": entry["file"],
                "bytes": size,
                "sha256": digest,
                "step": entry.get("step"),
                "default": entry.get("default", False),
            }
        )
        print(f"  -> {target} ({size} bytes) sha256={digest[:16]}…")

    if args.check:
        return 0

    if changed:
        REGISTRY_PATH.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"updated {REGISTRY_PATH.relative_to(ROOT)}")

    manifest_path = args.out / "MANIFEST.json"
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest_path.relative_to(ROOT)}")
    print("\nrelease assets to upload:")
    for item in manifest:
        print(f"  {item['file']}  {item['bytes']:>10} bytes  {item['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
