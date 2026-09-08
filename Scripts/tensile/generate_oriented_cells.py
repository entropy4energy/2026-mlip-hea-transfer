#!/usr/bin/env python3
"""Generate and verify exact-composition, oriented BCC HfMoTaTiZr cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


SPECIES = ("Hf", "Mo", "Ta", "Ti", "Zr")
MASSES = (178.49, 95.95, 180.94788, 47.867, 91.224)
CHEMICAL_SEEDS = (20260825, 20262843, 20264861)
LATTICE_PARAMETER_A = 3.45
PRIMITIVE = np.asarray(
    [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]],
    dtype=float,
)

# Rows are supercell vectors expressed in the BCC primitive basis.  The first
# row is the tensile x axis.  All three bases are mutually orthogonal.
ORIENTATION_BASES = {
    "100": np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int),
    "110": np.asarray([[1, 1, 2], [1, -1, 0], [1, 1, 0]], dtype=int),
    "111": np.asarray([[1, 1, 1], [1, -1, 0], [1, 1, -2]], dtype=int),
}
TENSILE_DIRECTIONS = {
    "100": np.asarray([1.0, 0.0, 0.0]),
    "110": np.asarray([1.0, 1.0, 0.0]),
    "111": np.asarray([1.0, 1.0, 1.0]),
}
PROFILE_REPEATS = {
    "medium": {"100": (15, 15, 15), "110": (11, 11, 15), "111": (20, 10, 6)},
    "large": {"100": (20, 20, 20), "110": (14, 14, 20), "111": (23, 15, 8)},
    "xlarge": {"100": (25, 25, 25), "110": (18, 18, 25), "111": (30, 18, 10)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def enumerate_fractional_sites(transform: np.ndarray) -> np.ndarray:
    """Return representatives of Z^3 / transform Z^3 in [0,1)^3."""

    expected = int(round(abs(np.linalg.det(transform))))
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
    if result.shape != (expected, 3):
        raise ValueError(f"enumerated {result.shape[0]} sites, expected {expected}")
    order = np.lexsort((result[:, 2], result[:, 1], result[:, 0]))
    result = result[order]
    if np.unique(np.round(result, 12), axis=0).shape[0] != expected:
        raise ValueError("duplicate fractional sites")
    return result


def restricted_triclinic(cell: np.ndarray) -> np.ndarray:
    """Rotate a right-handed row-vector cell into LAMMPS restricted form."""

    a, b, c = cell
    ax = float(np.linalg.norm(a))
    bx = float(np.dot(a, b) / ax)
    by = math.sqrt(float(np.dot(b, b) - bx * bx))
    cx = float(np.dot(a, c) / ax)
    cy = float((np.dot(b, c) - bx * cx) / by)
    cz = math.sqrt(float(np.dot(c, c) - cx * cx - cy * cy))
    result = np.asarray([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])
    if np.linalg.det(result) <= 0.0:
        raise ValueError("cell is not right handed")
    return result


def transform_for(profile: str, orientation: str) -> np.ndarray:
    repeats = np.diag(PROFILE_REPEATS[profile][orientation])
    return repeats @ ORIENTATION_BASES[orientation]


def geometry(profile: str, orientation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = transform_for(profile, orientation)
    general_cell = transform @ PRIMITIVE * LATTICE_PARAMETER_A
    lengths = np.linalg.norm(general_cell, axis=1)
    cosines = general_cell @ general_cell.T / np.outer(lengths, lengths)
    if not np.allclose(cosines, np.eye(3), atol=1.0e-12):
        raise ValueError(f"{profile}/{orientation}: orientation cell is not orthogonal")
    x_direction = general_cell[0] / np.linalg.norm(general_cell[0])
    target = TENSILE_DIRECTIONS[orientation]
    target /= np.linalg.norm(target)
    if not np.allclose(x_direction, target, atol=1.0e-12):
        raise ValueError(f"{profile}/{orientation}: x axis does not match [{orientation}]")
    cell = restricted_triclinic(general_cell)
    fractional = enumerate_fractional_sites(transform)
    return transform, cell, fractional


def write_data(
    path: Path,
    profile: str,
    orientation: str,
    seed: int,
    cell: np.ndarray,
    positions: np.ndarray,
    types: np.ndarray,
) -> None:
    natoms = positions.shape[0]
    lines = [
        f"# BCC equiatomic HfMoTaTiZr; profile={profile}; x=[{orientation}]; seed={seed}",
        "",
        f"{natoms} atoms",
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
        for index, (species, mass) in enumerate(zip(SPECIES, MASSES, strict=True), 1)
    )
    lines.extend(["", "Atoms # atomic", ""])
    lines.extend(
        f"{atom_id} {int(atom_type)} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}"
        for atom_id, (atom_type, xyz) in enumerate(zip(types, positions, strict=True), 1)
    )
    lines.append("")
    atomic_text(path, "\n".join(lines))


def read_data(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(next(line.split()[0] for line in lines if line.strip().endswith(" atoms")))
    header = {line.split()[-2]: line.split() for line in lines if line.strip().endswith(("xlo xhi", "ylo yhi", "zlo zhi"))}
    tilt = next(line.split() for line in lines if line.strip().endswith("xy xz yz"))
    cell = np.asarray(
        [
            [float(header["xlo"][1]) - float(header["xlo"][0]), 0.0, 0.0],
            [float(tilt[0]), float(header["ylo"][1]) - float(header["ylo"][0]), 0.0],
            [float(tilt[1]), float(tilt[2]), float(header["zlo"][1]) - float(header["zlo"][0])],
        ]
    )
    start = lines.index("Atoms # atomic") + 1
    records: list[tuple[int, int, float, float, float]] = []
    for line in lines[start:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            break
        records.append((int(fields[0]), int(fields[1]), *(float(v) for v in fields[2:])))
    data = np.asarray(records, dtype=float)
    if data.shape != (natoms, 5):
        raise ValueError(f"{path}: expected {natoms} atom rows, found {data.shape}")
    if not np.array_equal(data[:, 0].astype(int), np.arange(1, natoms + 1)):
        raise ValueError(f"{path}: atom IDs are not contiguous")
    return natoms, cell, data[:, 1].astype(int)


def build(output: Path) -> list[dict[str, object]]:
    manifest = output / "structure_manifest.tsv"
    receipt = output / "structure_generation_receipt.json"
    if manifest.exists() or receipt.exists():
        raise FileExistsError("refusing to overwrite an existing structure set")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for profile in PROFILE_REPEATS:
        for orientation in ORIENTATION_BASES:
            transform, cell, fractional = geometry(profile, orientation)
            natoms = int(round(abs(np.linalg.det(transform))))
            if natoms % 5:
                raise ValueError(f"{profile}/{orientation}: atom count is not divisible by five")
            positions = fractional @ cell
            delta = fractional - fractional[0]
            delta -= np.rint(delta)
            distances = np.linalg.norm(delta @ cell, axis=1)
            minimum_distance = float(np.min(distances[distances > 1.0e-8]))
            expected_distance = LATTICE_PARAMETER_A * math.sqrt(3.0) / 2.0
            if not math.isclose(minimum_distance, expected_distance, abs_tol=1.0e-9):
                raise ValueError(f"{profile}/{orientation}: nearest-neighbor distance failed")
            for seed in CHEMICAL_SEEDS:
                types = np.repeat(np.arange(1, 6, dtype=int), natoms // 5)
                np.random.Generator(np.random.PCG64(seed)).shuffle(types)
                name = f"bcc_{profile}_{orientation}_seed{seed}.data"
                path = output / name
                if path.exists():
                    raise FileExistsError(path)
                write_data(path, profile, orientation, seed, cell, positions, types)
                checked_n, checked_cell, checked_types = read_data(path)
                counts = np.bincount(checked_types, minlength=6)[1:]
                if checked_n != natoms or not np.array_equal(counts, np.repeat(natoms // 5, 5)):
                    raise ValueError(f"{path}: population/composition read-back failed")
                if not np.allclose(checked_cell, cell, atol=1.0e-11):
                    raise ValueError(f"{path}: cell read-back failed")
                rows.append(
                    {
                        "structure_id": path.stem,
                        "profile": profile,
                        "orientation": orientation,
                        "tensile_axis_cubic": f"[{orientation}]",
                        "chemical_seed": seed,
                        "natoms": natoms,
                        "n_Hf": int(counts[0]),
                        "n_Mo": int(counts[1]),
                        "n_Ta": int(counts[2]),
                        "n_Ti": int(counts[3]),
                        "n_Zr": int(counts[4]),
                        "lattice_parameter_A": LATTICE_PARAMETER_A,
                        "repeat_triplet": json.dumps(PROFILE_REPEATS[profile][orientation]),
                        "transform_matrix": json.dumps(transform.tolist(), separators=(",", ":")),
                        "cell_matrix_A": json.dumps(cell.tolist(), separators=(",", ":")),
                        "minimum_periodic_span_A": float(np.min(np.diag(cell))),
                        "minimum_neighbor_distance_A": minimum_distance,
                        "rng": "NumPy PCG64 exact-composition permutation",
                        "data_path": f"structures/{name}",
                        "data_sha256": sha256(path),
                    }
                )
    fields = list(rows[0])
    text_lines: list[str] = []
    temporary_manifest = output / ".structure_manifest.tsv.build"
    with temporary_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_manifest, manifest)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "structures": len(rows),
        "profiles": list(PROFILE_REPEATS),
        "orientations": list(ORIENTATION_BASES),
        "chemical_seeds": list(CHEMICAL_SEEDS),
        "manifest": "structures/structure_manifest.tsv",
        "manifest_sha256": sha256(manifest),
        "generator_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_text(receipt, json.dumps(payload, indent=2) + "\n")
    return rows


def verify(output: Path) -> list[dict[str, str]]:
    manifest = output / "structure_manifest.tsv"
    receipt_path = output / "structure_generation_receipt.json"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 27:
        raise ValueError(f"expected 27 structures, found {len(rows)}")
    expected = {
        (profile, orientation, str(seed))
        for profile in PROFILE_REPEATS
        for orientation in ORIENTATION_BASES
        for seed in CHEMICAL_SEEDS
    }
    observed = {(row["profile"], row["orientation"], row["chemical_seed"]) for row in rows}
    if observed != expected:
        raise ValueError("structure factor grid is incomplete or duplicated")
    for row in rows:
        path = output / Path(row["data_path"]).name
        if sha256(path) != row["data_sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        natoms, _, types = read_data(path)
        counts = np.bincount(types, minlength=6)[1:]
        if natoms != int(row["natoms"]) or not np.array_equal(counts, np.repeat(natoms // 5, 5)):
            raise ValueError(f"population/composition mismatch: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("manifest_sha256") != sha256(manifest):
        raise ValueError("structure manifest hash differs from generation receipt")
    print(json.dumps({"status": "passed", "verified_structures": len(rows)}, indent=2))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.verify_only:
        verify(output)
    else:
        rows = build(output)
        verify(output)
        print(json.dumps({"status": "passed", "generated_structures": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
