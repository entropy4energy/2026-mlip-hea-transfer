#!/usr/bin/env python3
"""Compute preregistered, label-blind local-environment coverage distances."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[2]
STUDY = PROJECT / "work" / "study_data"
COVERAGE = PROJECT / "work" / "coverage"
TABLES = PROJECT / "tables"
TYPE_MAP = ("Hf", "Mo", "Ta", "Ti", "Zr")
RADIAL_CENTERS = np.arange(1.50, 6.00, 0.15, dtype=np.float64)
SIGMA_ANGSTROM = 0.15
CUTOFF_ANGSTROM = 6.00
MAX_ATOMS_PER_ELEMENT_PER_FRAME = 12
QUERY_WORKERS = 4
SPECIFICATION = {
    "schema_version": 1,
    "representation": "central-element-stratified chemical radial density",
    "global_type_map": list(TYPE_MAP),
    "radial_centers_angstrom": RADIAL_CENTERS.tolist(),
    "gaussian_sigma_angstrom": SIGMA_ANGSTROM,
    "cosine_cutoff_angstrom": CUTOFF_ANGSTROM,
    "descriptor_dimensions": len(TYPE_MAP) * len(RADIAL_CENTERS),
    "distance": "Euclidean within the same central element",
    "frame_variant": "independent (100-frame sensitivity stride); seed retained as one frame",
    "atom_selection": "up to 12 evenly spaced atom indices per element and frame",
    "energies_forces_virials_or_model_outputs_used": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_gzip_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def system_input_files(system: Path) -> list[Path]:
    files = [system / "type.raw", system / "type_map.raw"]
    for set_dir in sorted(system.glob("set.*")):
        files.extend((set_dir / "coord.npy", set_dir / "box.npy"))
    if any(not path.is_file() for path in files) or len(files) == 2:
        raise FileNotFoundError(f"Incomplete coordinate/cell inputs under {system}")
    return files


def system_input_hashes(system: Path) -> dict[str, str]:
    return {
        str(path.relative_to(system)): sha256(path)
        for path in system_input_files(system)
    }


def load_system(system: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    declared_type_map = tuple((system / "type_map.raw").read_text().split())
    if declared_type_map != TYPE_MAP:
        raise ValueError(f"Unexpected type map in {system}: {declared_type_map}")
    atom_types = np.asarray(np.loadtxt(system / "type.raw", dtype=int, ndmin=1))
    coordinates = []
    boxes = []
    for set_dir in sorted(system.glob("set.*")):
        coordinates.append(
            np.asarray(np.load(set_dir / "coord.npy", allow_pickle=False), dtype=np.float64)
        )
        boxes.append(
            np.asarray(np.load(set_dir / "box.npy", allow_pickle=False), dtype=np.float64)
        )
    coordinates_array = np.concatenate(coordinates, axis=0).reshape(-1, len(atom_types), 3)
    boxes_array = np.concatenate(boxes, axis=0).reshape(-1, 3, 3)
    if coordinates_array.shape[0] != boxes_array.shape[0]:
        raise ValueError(f"Coordinate/cell frame mismatch in {system}")
    if not np.isfinite(coordinates_array).all() or not np.isfinite(boxes_array).all():
        raise ValueError(f"Non-finite coordinate or cell in {system}")
    if set(np.unique(atom_types)) - set(range(len(TYPE_MAP))):
        raise ValueError(f"Unexpected atom type in {system}")
    return coordinates_array, boxes_array, atom_types


def evenly_spaced_atom_indices(atom_types: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    for element_index in range(len(TYPE_MAP)):
        candidates = np.flatnonzero(atom_types == element_index)
        if len(candidates) <= MAX_ATOMS_PER_ELEMENT_PER_FRAME:
            selected.extend(candidates.tolist())
            continue
        positions = np.rint(
            np.linspace(0, len(candidates) - 1, MAX_ATOMS_PER_ELEMENT_PER_FRAME)
        ).astype(int)
        chosen = candidates[positions]
        if len(np.unique(chosen)) != MAX_ATOMS_PER_ELEMENT_PER_FRAME:
            raise AssertionError("Even atom selection unexpectedly repeated an index")
        selected.extend(chosen.tolist())
    return np.asarray(selected, dtype=int)


def periodic_shift_grid(cell: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(cell)
    plane_heights = 1.0 / np.linalg.norm(inverse, axis=0)
    if np.any(~np.isfinite(plane_heights)) or np.any(plane_heights <= 0):
        raise ValueError("Invalid periodic cell")
    spans = np.ceil(CUTOFF_ANGSTROM / plane_heights).astype(int)
    axes = [np.arange(-span, span + 1, dtype=int) for span in spans]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, 3)


def frame_descriptors(
    coordinates: np.ndarray,
    cell: np.ndarray,
    atom_types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected atom indices and their chemical radial descriptors."""

    selected = evenly_spaced_atom_indices(atom_types)
    inverse = np.linalg.inv(cell)
    fractional = coordinates @ inverse
    shifts = periodic_shift_grid(cell)
    output = np.zeros(
        (len(selected), len(TYPE_MAP), len(RADIAL_CENTERS)), dtype=np.float64
    )
    for output_index, central_index in enumerate(selected):
        delta_fractional = fractional - fractional[central_index]
        delta_fractional -= np.floor(delta_fractional + 0.5)
        translated = delta_fractional[:, np.newaxis, :] + shifts[np.newaxis, :, :]
        distances = np.linalg.norm(translated @ cell, axis=2).reshape(-1)
        neighbour_types = np.repeat(atom_types, len(shifts))
        retained = (distances > 1.0e-10) & (distances < CUTOFF_ANGSTROM)
        retained_distances = distances[retained]
        retained_types = neighbour_types[retained]
        cutoff = 0.5 * (
            np.cos(np.pi * retained_distances / CUTOFF_ANGSTROM) + 1.0
        )
        radial_values = np.exp(
            -0.5
            * (
                (retained_distances[:, np.newaxis] - RADIAL_CENTERS[np.newaxis, :])
                / SIGMA_ANGSTROM
            )
            ** 2
        )
        radial_values *= cutoff[:, np.newaxis]
        np.add.at(output[output_index], retained_types, radial_values)
    return selected, output.reshape(len(selected), -1)


def compute_system_descriptors(system: Path) -> dict[str, np.ndarray]:
    coordinates, boxes, atom_types = load_system(system)
    descriptors = []
    frame_indices = []
    atom_indices = []
    element_indices = []
    for frame_index, (frame_coordinates, cell) in enumerate(
        zip(coordinates, boxes, strict=True)
    ):
        selected, frame_values = frame_descriptors(frame_coordinates, cell, atom_types)
        descriptors.append(frame_values.astype(np.float32))
        frame_indices.extend([frame_index] * len(selected))
        atom_indices.extend(selected.tolist())
        element_indices.extend(atom_types[selected].tolist())
    return {
        "descriptors": np.concatenate(descriptors, axis=0),
        "frame_index": np.asarray(frame_indices, dtype=np.int32),
        "atom_index": np.asarray(atom_indices, dtype=np.int32),
        "element_index": np.asarray(element_indices, dtype=np.int8),
    }


def descriptor_cache_paths(system: Path) -> tuple[Path, Path]:
    spec_hash = sha256_text(json.dumps(SPECIFICATION, sort_keys=True))[:12]
    system_hash = sha256_text(str(system.resolve()))[:20]
    stem = f"{system_hash}__{spec_hash}"
    cache_dir = COVERAGE / "descriptor_cache"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def load_or_compute_descriptors(system: Path) -> tuple[dict[str, np.ndarray], dict]:
    system = system.resolve()
    cache_path, metadata_path = descriptor_cache_paths(system)
    inputs = system_input_hashes(system)
    specification_hash = sha256_text(json.dumps(SPECIFICATION, sort_keys=True))
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("system_path") == str(system)
            and metadata.get("specification_sha256") == specification_hash
            and metadata.get("input_hashes") == inputs
            and metadata.get("cache_sha256") == sha256(cache_path)
        ):
            with np.load(cache_path, allow_pickle=False) as values:
                arrays = {name: values[name] for name in values.files}
            return arrays, metadata

    arrays = compute_system_descriptors(system)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(cache_path)
    metadata = {
        "schema_version": 1,
        "system_path": str(system),
        "specification_sha256": specification_hash,
        "input_hashes": inputs,
        "environment_count": int(len(arrays["descriptors"])),
        "descriptor_dimensions": int(arrays["descriptors"].shape[1]),
        "cache_file": cache_path.name,
        "cache_sha256": sha256(cache_path),
    }
    atomic_json(metadata_path, metadata)
    return arrays, metadata


def canonical_indices() -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    rows = read_csv(STUDY / "canonical_systems.csv")
    by_path = {str(Path(row["path"]).resolve()): row for row in rows}
    by_system_variant = {(row["system_id"], row["variant"]): row for row in rows}
    return by_path, by_system_variant


def coverage_variant_path(
    path: str,
    by_path: dict[str, dict[str, str]],
    by_system_variant: dict[tuple[str, str], dict[str, str]],
) -> Path:
    row = by_path[str(Path(path).resolve())]
    desired_variant = "seed" if row["variant"] == "seed" else "independent"
    resolved = by_system_variant[(row["system_id"], desired_variant)]["path"]
    return Path(resolved).resolve()


def build_element_trees(
    training_paths: list[Path],
    descriptor_records: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[int, cKDTree], dict[int, int]]:
    values: dict[int, list[np.ndarray]] = defaultdict(list)
    for path in training_paths:
        record = descriptor_records[str(path)]
        for element_index in range(len(TYPE_MAP)):
            mask = record["element_index"] == element_index
            if np.any(mask):
                values[element_index].append(record["descriptors"][mask])
    trees = {}
    counts = {}
    for element_index in range(len(TYPE_MAP)):
        if not values[element_index]:
            raise ValueError(f"No training environments for central {TYPE_MAP[element_index]}")
        stacked = np.concatenate(values[element_index], axis=0).astype(np.float64)
        trees[element_index] = cKDTree(stacked)
        counts[element_index] = int(len(stacked))
    return trees, counts


def summarize_distances(values: np.ndarray) -> dict[str, float | int]:
    return {
        "environment_count": int(len(values)),
        "coverage_distance_mean": float(np.mean(values)),
        "coverage_distance_median": float(np.median(values)),
        "coverage_distance_p90": float(np.percentile(values, 90)),
        "coverage_distance_minimum": float(np.min(values)),
        "coverage_distance_maximum": float(np.max(values)),
    }


def main() -> int:
    design = json.loads((STUDY / "study_design.json").read_text())
    by_path, by_system_variant = canonical_indices()
    fold_paths: dict[str, dict[str, list[Path]]] = {}
    all_paths: set[Path] = set()
    for fold_name, fold in design["folds"].items():
        fold_paths[fold_name] = {}
        for regime_name, regime in fold["regimes"].items():
            paths = sorted(
                {
                    coverage_variant_path(path, by_path, by_system_variant)
                    for path in regime["training_systems"]
                }
            )
            fold_paths[fold_name][regime_name] = paths
            all_paths.update(paths)
        test_paths = sorted(
            {
                coverage_variant_path(path, by_path, by_system_variant)
                for path in fold["evaluation"]["pocc_ood_test_systems"]
            }
        )
        fold_paths[fold_name]["test"] = test_paths
        all_paths.update(test_paths)

        training_ids = {
            by_path[str(path)]["system_id"]
            for regime_name in fold["regimes"]
            for path in fold_paths[fold_name][regime_name]
        }
        test_ids = {by_path[str(path)]["system_id"] for path in test_paths}
        if training_ids & test_ids:
            raise ValueError(f"Coverage train/test leakage in {fold_name}")

    descriptor_records: dict[str, dict[str, np.ndarray]] = {}
    cache_metadata = []
    for index, path in enumerate(sorted(all_paths), start=1):
        print(f"[{index:03d}/{len(all_paths):03d}] descriptor {path.name}", flush=True)
        arrays, metadata = load_or_compute_descriptors(path)
        descriptor_records[str(path)] = arrays
        cache_metadata.append(metadata)

    environment_rows: list[dict] = []
    system_rows: list[dict] = []
    training_environment_counts: dict[str, dict[str, dict[str, int]]] = {}
    for fold_name, paths_by_role in fold_paths.items():
        training_environment_counts[fold_name] = {}
        for regime_name in ("pure", "pure_pocc_seed", "pure_pocc_aimd"):
            training_paths = paths_by_role[regime_name]
            trees, element_counts = build_element_trees(training_paths, descriptor_records)
            training_environment_counts[fold_name][regime_name] = {
                TYPE_MAP[index]: count for index, count in element_counts.items()
            }
            for test_path in paths_by_role["test"]:
                canonical = by_path[str(test_path)]
                record = descriptor_records[str(test_path)]
                distances = np.empty(len(record["descriptors"]), dtype=np.float64)
                for element_index, tree in trees.items():
                    mask = record["element_index"] == element_index
                    distances[mask] = tree.query(
                        record["descriptors"][mask].astype(np.float64),
                        k=1,
                        workers=QUERY_WORKERS,
                    )[0]
                if not np.isfinite(distances).all():
                    raise ValueError(f"Non-finite coverage distance for {test_path}")
                base = {
                    "fold": fold_name,
                    "regime": regime_name,
                    "system_id": canonical["system_id"],
                    "parent_family": canonical["parent_family"],
                    "structure": canonical["structure"],
                    "hnf_group": canonical["hnf_group"],
                }
                system_rows.append({**base, **summarize_distances(distances)})
                for row_index, distance in enumerate(distances):
                    environment_rows.append(
                        {
                            **base,
                            "frame_index": int(record["frame_index"][row_index]),
                            "atom_index": int(record["atom_index"][row_index]),
                            "central_element": TYPE_MAP[
                                int(record["element_index"][row_index])
                            ],
                            "coverage_distance": float(distance),
                        }
                    )

    summary_by_key = {
        (row["fold"], row["regime"], row["system_id"]): row for row in system_rows
    }
    contrast_rows = []
    for fold_name, paths_by_role in fold_paths.items():
        for test_path in paths_by_role["test"]:
            system_id = by_path[str(test_path)]["system_id"]
            pure = summary_by_key[(fold_name, "pure", system_id)]
            for treatment in ("pure_pocc_seed", "pure_pocc_aimd"):
                augmented = summary_by_key[(fold_name, treatment, system_id)]
                pure_median = float(pure["coverage_distance_median"])
                augmented_median = float(augmented["coverage_distance_median"])
                contrast_rows.append(
                    {
                        "fold": fold_name,
                        "treatment": treatment,
                        "system_id": system_id,
                        "parent_family": pure["parent_family"],
                        "structure": pure["structure"],
                        "hnf_group": pure["hnf_group"],
                        "pure_median_distance": pure_median,
                        "augmented_median_distance": augmented_median,
                        "absolute_distance_reduction": pure_median - augmented_median,
                        "relative_distance_reduction": (
                            (pure_median - augmented_median) / pure_median
                            if pure_median > 0
                            else math.nan
                        ),
                    }
                )

    environment_path = COVERAGE / "environment_coverage.csv.gz"
    system_path = TABLES / "system_environment_coverage.csv"
    contrast_path = TABLES / "environment_coverage_contrasts.csv"
    atomic_gzip_csv(environment_path, environment_rows, list(environment_rows[0]))
    atomic_csv(system_path, system_rows, list(system_rows[0]))
    atomic_csv(contrast_path, contrast_rows, list(contrast_rows[0]))
    provenance = {
        "schema_version": 1,
        "specification": SPECIFICATION,
        "specification_sha256": sha256_text(json.dumps(SPECIFICATION, sort_keys=True)),
        "study_design_sha256": sha256(STUDY / "study_design.json"),
        "canonical_systems_sha256": sha256(STUDY / "canonical_systems.csv"),
        "fold_paths": {
            fold: {role: [str(path) for path in paths] for role, paths in roles.items()}
            for fold, roles in fold_paths.items()
        },
        "training_environment_counts_by_element": training_environment_counts,
        "descriptor_caches": cache_metadata,
        "outputs": {
            "environment_coverage": {
                "path": str(environment_path),
                "rows": len(environment_rows),
                "sha256": sha256(environment_path),
            },
            "system_environment_coverage": {
                "path": str(system_path),
                "rows": len(system_rows),
                "sha256": sha256(system_path),
            },
            "environment_coverage_contrasts": {
                "path": str(contrast_path),
                "rows": len(contrast_rows),
                "sha256": sha256(contrast_path),
            },
        },
    }
    atomic_json(COVERAGE / "coverage_provenance.json", provenance)
    print(json.dumps(provenance["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
