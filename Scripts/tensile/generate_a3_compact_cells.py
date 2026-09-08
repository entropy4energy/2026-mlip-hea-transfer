#!/usr/bin/env python3
"""Generate and verify the prospective A3 near-2,000-atom cells."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import generate_oriented_cells as base
from common import ROOT, atomic_json, atomic_tsv, load_json, read_tsv, sha256


CONFIG = ROOT / "config" / "compact_profile.json"


def geometry(orientation: str, repeats: tuple[int, int, int]):
    transform = np.diag(repeats) @ base.ORIENTATION_BASES[orientation]
    general = transform @ base.PRIMITIVE * base.LATTICE_PARAMETER_A
    lengths = np.linalg.norm(general, axis=1)
    cosines = general @ general.T / np.outer(lengths, lengths)
    if not np.allclose(cosines, np.eye(3), atol=1.0e-12):
        raise ValueError(f"compact/{orientation}: cell is not orthogonal")
    direction = general[0] / lengths[0]
    target = base.TENSILE_DIRECTIONS[orientation].copy()
    target /= np.linalg.norm(target)
    if not np.allclose(direction, target, atol=1.0e-12):
        raise ValueError(f"compact/{orientation}: tensile-axis mismatch")
    return transform, base.restricted_triclinic(general), base.enumerate_fractional_sites(transform), lengths


def build(output: Path) -> list[dict[str, object]]:
    manifest = output / "structure_manifest.tsv"
    receipt = output / "structure_generation_receipt.json"
    if manifest.exists() or receipt.exists():
        raise FileExistsError("refusing to overwrite the compact structure set")
    config = load_json(CONFIG)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for orientation, item in config["orientations"].items():
        repeats = tuple(int(value) for value in item["repeats"])
        transform, cell, fractional, lengths = geometry(orientation, repeats)
        natoms = int(round(abs(np.linalg.det(transform))))
        if natoms != int(item["expected_natoms"]) or natoms % 5:
            raise ValueError(f"compact/{orientation}: population contract failed")
        positions = fractional @ cell
        delta = fractional - fractional[0]
        delta -= np.rint(delta)
        distances = np.linalg.norm(delta @ cell, axis=1)
        nearest = float(np.min(distances[distances > 1.0e-8]))
        expected_nearest = base.LATTICE_PARAMETER_A * math.sqrt(3.0) / 2.0
        if not math.isclose(nearest, expected_nearest, abs_tol=1.0e-9):
            raise ValueError(f"compact/{orientation}: nearest-neighbor distance failed")
        for seed in (int(value) for value in config["chemical_seeds"]):
            types = np.repeat(np.arange(1, 6, dtype=int), natoms // 5)
            np.random.Generator(np.random.PCG64(seed)).shuffle(types)
            path = output / f"bcc_compact_{orientation}_seed{seed}.data"
            if path.exists():
                raise FileExistsError(path)
            base.write_data(path, "compact", orientation, seed, cell, positions, types)
            checked_n, checked_cell, checked_types = base.read_data(path)
            counts = np.bincount(checked_types, minlength=6)[1:]
            if checked_n != natoms or not np.array_equal(counts, np.repeat(natoms // 5, 5)):
                raise ValueError(f"{path}: population/composition read-back failed")
            if not np.allclose(checked_cell, cell, atol=1.0e-11):
                raise ValueError(f"{path}: cell read-back failed")
            rows.append(
                {
                    "structure_id": path.stem,
                    "profile": "compact",
                    "orientation": orientation,
                    "tensile_axis_cubic": f"[{orientation}]",
                    "chemical_seed": seed,
                    "natoms": natoms,
                    "n_Hf": int(counts[0]),
                    "n_Mo": int(counts[1]),
                    "n_Ta": int(counts[2]),
                    "n_Ti": int(counts[3]),
                    "n_Zr": int(counts[4]),
                    "lattice_parameter_A": base.LATTICE_PARAMETER_A,
                    "repeat_triplet": json.dumps(repeats),
                    "transform_matrix": json.dumps(transform.tolist(), separators=(",", ":")),
                    "cell_matrix_A": json.dumps(cell.tolist(), separators=(",", ":")),
                    "cell_spans_A": json.dumps(lengths.tolist(), separators=(",", ":")),
                    "minimum_periodic_span_A": float(np.min(lengths)),
                    "minimum_neighbor_distance_A": nearest,
                    "rng": "NumPy PCG64 exact-composition permutation",
                    "data_path": str(path.relative_to(ROOT)),
                    "data_sha256": sha256(path),
                }
            )
    if len(rows) != 9:
        raise ValueError("compact structure population is not nine")
    atomic_tsv(manifest, rows)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "structures": len(rows),
        "profile": "compact",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest),
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(receipt, payload)
    return rows


def verify(output: Path) -> list[dict[str, str]]:
    config = load_json(CONFIG)
    rows = read_tsv(output / "structure_manifest.tsv")
    expected = {
        (orientation, str(seed))
        for orientation in config["orientations"]
        for seed in config["chemical_seeds"]
    }
    if len(rows) != 9 or {(row["orientation"], row["chemical_seed"]) for row in rows} != expected:
        raise ValueError("compact structure grid is incomplete or duplicated")
    for row in rows:
        path = ROOT / row["data_path"]
        if sha256(path) != row["data_sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        natoms, _, types = base.read_data(path)
        counts = np.bincount(types, minlength=6)[1:]
        if natoms != int(row["natoms"]) or not np.array_equal(counts, np.repeat(natoms // 5, 5)):
            raise ValueError(f"population/composition mismatch: {path}")
        expected_natoms = int(config["orientations"][row["orientation"]]["expected_natoms"])
        if natoms != expected_natoms:
            raise ValueError(f"configured population mismatch: {path}")
    receipt = load_json(output / "structure_generation_receipt.json")
    manifest = output / "structure_manifest.tsv"
    if receipt.get("manifest_sha256") != sha256(manifest) or receipt.get("config_sha256") != sha256(CONFIG):
        raise ValueError("compact structure receipt binding failed")
    print(json.dumps({"status": "passed", "verified_compact_structures": len(rows)}, indent=2))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "structures_a3_compact")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.verify_only:
        verify(output)
    else:
        rows = build(output)
        verify(output)
        print(json.dumps({"status": "passed", "generated_compact_structures": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
