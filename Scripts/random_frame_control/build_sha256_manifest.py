#!/usr/bin/env python3
"""Build a deterministic SHA-256 index for the shareable control package."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUT = PACKAGE / "SHA256_MANIFEST.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role(relative_path: Path) -> str:
    first = relative_path.parts[0]
    if first == "trajectories":
        return "analyzed_trajectory_or_cif"
    if first == "inputs":
        return "immutable_input_snapshot"
    if first == "results":
        return "analysis_result_or_validation_receipt"
    if first == "figures":
        return "generated_figure"
    if first == "notebooks":
        return "executed_notebook"
    if first == "scripts":
        return "analysis_or_validation_code"
    if relative_path.suffix.lower() == ".xlsx":
        return "colleague_handoff_workbook"
    return "documentation"


def included_files() -> list[Path]:
    paths: list[Path] = []
    for path in PACKAGE.rglob("*"):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(PACKAGE)
        if "__pycache__" in relative.parts or ".ipynb_checkpoints" in relative.parts:
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(PACKAGE).as_posix())


def main() -> int:
    rows = []
    for path in included_files():
        relative = path.relative_to(PACKAGE)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": role(relative),
            }
        )

    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "size_bytes", "sha256", "role"]
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT)
    print(f"Indexed {len(rows):,} files in {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
