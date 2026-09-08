#!/usr/bin/env python3
"""Generate hash-recorded, exact-composition BCC/FCC HEA supercells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


SPECIES = ("Hf", "Mo", "Ta", "Ti", "Zr")
MASSES = (178.49, 95.95, 180.94788, 47.867, 91.224)
PRIMARY_SEEDS = (20260825, 20261834, 20262843, 20263852, 20264861)
MINIMUM_SPAN_A = 12.5


@dataclass(frozen=True)
class CellDefinition:
    phase: str
    natoms: int
    conventional_lattice_parameter_A: float
    primitive: np.ndarray
    transform: np.ndarray
    role: str


CELL_DEFINITIONS = {
    ("bcc", 2000): CellDefinition(
        phase="bcc",
        natoms=2000,
        conventional_lattice_parameter_A=3.45,
        primitive=np.asarray(
            [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]],
            dtype=float,
        ),
        transform=np.asarray([[0, 10, 10], [10, 0, 10], [10, 10, 0]], dtype=int),
        role="primary",
    ),
    ("fcc", 2000): CellDefinition(
        phase="fcc",
        natoms=2000,
        conventional_lattice_parameter_A=4.35,
        primitive=np.asarray(
            [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
            dtype=float,
        ),
        transform=np.asarray([[-6, 8, 8], [8, -10, 8], [9, 6, -8]], dtype=int),
        role="primary",
    ),
    ("bcc", 250): CellDefinition(
        phase="bcc",
        natoms=250,
        conventional_lattice_parameter_A=3.45,
        primitive=np.asarray(
            [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]],
            dtype=float,
        ),
        transform=np.asarray([[0, 5, 5], [5, 0, 5], [5, 5, 0]], dtype=int),
        role="cell_size_sensitivity",
    ),
    ("fcc", 250): CellDefinition(
        phase="fcc",
        natoms=250,
        conventional_lattice_parameter_A=4.35,
        primitive=np.asarray(
            [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
            dtype=float,
        ),
        transform=np.asarray([[-5, 4, 3], [2, -3, 5], [4, 5, -3]], dtype=int),
        role="cell_size_sensitivity",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enumerate_fractional_sites(transform: np.ndarray) -> np.ndarray:
    """Enumerate representatives of Z^3 / transform Z^3 in [0,1)^3."""

    determinant = int(round(abs(np.linalg.det(transform))))
    corners = np.asarray(
        [fraction @ transform for fraction in np.ndindex(2, 2, 2)], dtype=int
    )
    lower = corners.min(axis=0) - 1
    upper = corners.max(axis=0) + 1
    inverse = np.linalg.inv(transform)
    sites: list[np.ndarray] = []
    tolerance = 2.0e-10
    for n0 in range(int(lower[0]), int(upper[0]) + 1):
        for n1 in range(int(lower[1]), int(upper[1]) + 1):
            for n2 in range(int(lower[2]), int(upper[2]) + 1):
                fractional = np.asarray([n0, n1, n2], dtype=float) @ inverse
                if np.all(fractional >= -tolerance) and np.all(
                    fractional < 1.0 - tolerance
                ):
                    sites.append(np.mod(fractional, 1.0))
    result = np.asarray(sites, dtype=float)
    if result.shape != (determinant, 3):
        raise ValueError(
            f"supercell enumeration produced {result.shape[0]} sites; expected {determinant}"
        )
    order = np.lexsort((result[:, 2], result[:, 1], result[:, 0]))
    result = result[order]
    rounded = np.round(result, 12)
    if np.unique(rounded, axis=0).shape[0] != determinant:
        raise ValueError("duplicate fractional sites detected")
    return result


def restricted_triclinic(cell: np.ndarray) -> np.ndarray:
    """Rotate a right-handed row-vector cell into LAMMPS restricted form."""

    a, b, c = cell
    ax = float(np.linalg.norm(a))
    bx = float(np.dot(a, b) / ax)
    by_squared = float(np.dot(b, b) - bx * bx)
    if by_squared <= 0.0:
        raise ValueError("degenerate first two cell vectors")
    by = float(np.sqrt(by_squared))
    cx = float(np.dot(a, c) / ax)
    cy = float((np.dot(b, c) - bx * cx) / by)
    cz_squared = float(np.dot(c, c) - cx * cx - cy * cy)
    if cz_squared <= 0.0:
        raise ValueError("degenerate third cell vector")
    cz = float(np.sqrt(cz_squared))
    restricted = np.asarray([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])
    if np.linalg.det(restricted) <= 0.0:
        raise ValueError("restricted cell is not right handed")
    return restricted


def perpendicular_spans(cell: np.ndarray) -> np.ndarray:
    volume = abs(float(np.linalg.det(cell)))
    return np.asarray(
        [
            volume / np.linalg.norm(np.cross(cell[1], cell[2])),
            volume / np.linalg.norm(np.cross(cell[0], cell[2])),
            volume / np.linalg.norm(np.cross(cell[0], cell[1])),
        ]
    )


def minimum_periodic_distance(positions: np.ndarray, cell: np.ndarray) -> float:
    shifts = np.asarray(list(np.ndindex(3, 3, 3)), dtype=float) - 1.0
    translated = np.concatenate([positions + shift @ cell for shift in shifts], axis=0)
    tree = cKDTree(translated)
    distances, _ = tree.query(positions, k=2)
    return float(np.min(distances[:, 1]))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_lammps_data(
    path: Path,
    definition: CellDefinition,
    seed: int,
    cell: np.ndarray,
    positions: np.ndarray,
    atom_types: np.ndarray,
) -> None:
    lines = [
        f"# {definition.phase.upper()} HfMoTaTiZr; n={definition.natoms}; seed={seed}",
        "",
        f"{definition.natoms} atoms",
        "5 atom types",
        "",
        f"0.0 {cell[0, 0]:.16g} xlo xhi",
        f"0.0 {cell[1, 1]:.16g} ylo yhi",
        f"0.0 {cell[2, 2]:.16g} zlo zhi",
        f"{cell[1, 0]:.16g} {cell[2, 0]:.16g} {cell[2, 1]:.16g} xy xz yz",
        "",
        "Masses",
        "",
    ]
    lines.extend(
        f"{index} {mass:.12g} # {species}"
        for index, (species, mass) in enumerate(zip(SPECIES, MASSES, strict=True), start=1)
    )
    lines.extend(["", "Atoms # atomic", ""])
    lines.extend(
        f"{atom_id} {int(atom_type)} {position[0]:.16g} {position[1]:.16g} {position[2]:.16g}"
        for atom_id, (atom_type, position) in enumerate(
            zip(atom_types, positions, strict=True), start=1
        )
    )
    lines.append("")
    atomic_write_text(path, "\n".join(lines))


def read_back_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    bounds: dict[str, tuple[float, float]] = {}
    tilts: tuple[float, float, float] | None = None
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[-2:] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
            bounds[fields[-2][0]] = (float(fields[0]), float(fields[1]))
        if len(fields) == 6 and fields[-3:] == ["xy", "xz", "yz"]:
            tilts = (float(fields[0]), float(fields[1]), float(fields[2]))
    if set(bounds) != {"x", "y", "z"} or tilts is None:
        raise ValueError(f"could not reconstruct cell header from {path}")
    xy, xz, yz = tilts
    cell = np.asarray(
        [
            [bounds["x"][1] - bounds["x"][0], 0.0, 0.0],
            [xy, bounds["y"][1] - bounds["y"][0], 0.0],
            [xz, yz, bounds["z"][1] - bounds["z"][0]],
        ]
    )
    try:
        start = lines.index("Atoms # atomic") + 1
    except ValueError as error:
        raise ValueError(f"missing Atoms section in {path}") from error
    records = []
    for line in lines[start:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            break
        records.append((int(fields[0]), int(fields[1]), *(float(value) for value in fields[2:])))
    data = np.asarray(records, dtype=float)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(f"invalid atom records in {path}")
    atom_ids = data[:, 0].astype(int)
    atom_types = data[:, 1].astype(int)
    positions = data[:, 2:]
    return cell, atom_ids, atom_types, positions


def build_one(definition: CellDefinition, seed: int, output: Path) -> dict[str, object]:
    determinant = int(round(abs(np.linalg.det(definition.transform))))
    if determinant != definition.natoms:
        raise ValueError("transform determinant does not match requested atom count")
    if definition.natoms % len(SPECIES):
        raise ValueError("atom count must be divisible by five")
    fractional = enumerate_fractional_sites(definition.transform)
    general_cell = (
        definition.transform
        @ definition.primitive
        * definition.conventional_lattice_parameter_A
    )
    cell = restricted_triclinic(general_cell)
    positions = fractional @ cell
    per_species = definition.natoms // len(SPECIES)
    atom_types = np.repeat(np.arange(1, len(SPECIES) + 1, dtype=int), per_species)
    np.random.Generator(np.random.PCG64(seed)).shuffle(atom_types)

    structure_id = f"{definition.phase}_n{definition.natoms}_seed{seed}"
    data_path = output / f"{structure_id}.data"
    if data_path.exists():
        raise FileExistsError(f"refusing to overwrite existing structure: {data_path}")
    write_lammps_data(data_path, definition, seed, cell, positions, atom_types)

    read_cell, atom_ids, read_types, read_positions = read_back_data(data_path)
    if atom_ids.size != definition.natoms or not np.array_equal(
        np.sort(atom_ids), np.arange(1, definition.natoms + 1)
    ):
        raise ValueError(f"atom-ID population check failed for {data_path}")
    counts = np.bincount(read_types, minlength=6)[1:]
    if not np.array_equal(counts, np.repeat(per_species, 5)):
        raise ValueError(f"type-count check failed for {data_path}: {counts.tolist()}")
    if not np.all(np.isfinite(read_positions)) or not np.all(np.isfinite(read_cell)):
        raise ValueError(f"non-finite structure data in {data_path}")
    if np.linalg.det(read_cell) <= 0.0:
        raise ValueError(f"non-positive cell volume in {data_path}")
    fractional_read = read_positions @ np.linalg.inv(read_cell)
    wrapped = np.mod(fractional_read, 1.0)
    if np.unique(np.round(wrapped, 10), axis=0).shape[0] != definition.natoms:
        raise ValueError(f"duplicate sites detected after read-back in {data_path}")
    spans = perpendicular_spans(read_cell)
    if float(spans.min()) <= MINIMUM_SPAN_A:
        raise ValueError(f"shortest periodic span is too small in {data_path}: {spans.min():.6g}")
    minimum_distance = minimum_periodic_distance(read_positions, read_cell)
    expected_distance = definition.conventional_lattice_parameter_A * (
        np.sqrt(3.0) / 2.0 if definition.phase == "bcc" else 1.0 / np.sqrt(2.0)
    )
    if not np.isclose(minimum_distance, expected_distance, rtol=1.0e-9, atol=1.0e-9):
        raise ValueError(
            f"nearest-distance check failed for {data_path}: {minimum_distance} vs {expected_distance}"
        )

    lengths = np.linalg.norm(read_cell, axis=1)
    direction_cosines = read_cell @ read_cell.T / np.outer(lengths, lengths)
    return {
        "structure_id": structure_id,
        "role": definition.role,
        "phase": definition.phase,
        "natoms": definition.natoms,
        "chemical_seed": seed,
        "rng": "NumPy PCG64 exact-composition permutation",
        "n_Hf": int(counts[0]),
        "n_Mo": int(counts[1]),
        "n_Ta": int(counts[2]),
        "n_Ti": int(counts[3]),
        "n_Zr": int(counts[4]),
        "initial_conventional_lattice_parameter_A": definition.conventional_lattice_parameter_A,
        "transform_matrix": json.dumps(definition.transform.tolist(), separators=(",", ":")),
        "cell_matrix_A": json.dumps(read_cell.tolist(), separators=(",", ":")),
        "cell_lengths_A": json.dumps(lengths.tolist(), separators=(",", ":")),
        "cell_direction_cosines": json.dumps(direction_cosines.tolist(), separators=(",", ":")),
        "volume_A3": float(np.linalg.det(read_cell)),
        "volume_A3_per_atom": float(np.linalg.det(read_cell) / definition.natoms),
        "perpendicular_spans_A": json.dumps(spans.tolist(), separators=(",", ":")),
        "minimum_periodic_distance_A": minimum_distance,
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256(data_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-size-sensitivity",
        action="store_true",
        help="also build the fixed 250-atom first-seed cells",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "structure_manifest.csv"
    receipt_path = output / "structure_generation_receipt.json"
    if manifest_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite an existing structure manifest or receipt")

    rows: list[dict[str, object]] = []
    for phase in ("bcc", "fcc"):
        for seed in PRIMARY_SEEDS:
            rows.append(build_one(CELL_DEFINITIONS[(phase, 2000)], seed, output))
    if args.include_size_sensitivity:
        for phase in ("bcc", "fcc"):
            rows.append(build_one(CELL_DEFINITIONS[(phase, 250)], PRIMARY_SEEDS[0], output))

    fieldnames = list(rows[0])
    descriptor, temporary_name = tempfile.mkstemp(prefix=".structure_manifest.", dir=output)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, manifest_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    script_path = Path(__file__).resolve()
    receipt = {
        "schema_version": 1,
        "generator": str(script_path),
        "generator_sha256": sha256(script_path),
        "numpy_version": np.__version__,
        "rng": "NumPy PCG64",
        "species_order": list(SPECIES),
        "primary_seeds": list(PRIMARY_SEEDS),
        "minimum_required_perpendicular_span_A": MINIMUM_SPAN_A,
        "structures": len(rows),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "status": "passed",
    }
    atomic_write_text(receipt_path, json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
