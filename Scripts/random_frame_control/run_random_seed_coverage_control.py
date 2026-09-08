#!/usr/bin/env python3
"""Build the matched-label random-frame structural-coverage control.

This program snapshots the exact frozen coverage inputs, exports the analyzed
trajectory frames in DeepMD and CIF forms, reconstructs the systematic seed
result, and evaluates two deterministic 10,000-draw random controls.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy
from scipy.spatial import cKDTree


PACKAGE = Path(__file__).resolve().parents[1]
FINAL_WORK = PACKAGE.parent
SOURCE_PROJECT = FINAL_WORK / "manuscript_pocc_mlf_2026-08-25"
SOURCE_STUDY = SOURCE_PROJECT / "work" / "study_data"
SOURCE_COVERAGE = SOURCE_PROJECT / "work" / "coverage"
SOURCE_TABLES = SOURCE_PROJECT / "tables"

INPUTS = PACKAGE / "inputs"
TRAJECTORIES = PACKAGE / "trajectories"
RESULTS = PACKAGE / "results"
CACHE = RESULTS / "distance_cache"

TYPE_MAP = ("Hf", "Mo", "Ta", "Ti", "Zr")
MASTER_RANDOM_SEED = 20260903
N_DRAWS = 10_000
FOLD_CODES = {"fcc_to_bcc": 1, "bcc_to_fcc": 2}
CONTROL_CODES = {"pooled_random_frames": 1, "one_per_trajectory": 2}
FOLD_LABELS = {"fcc_to_bcc": "FCC-to-BCC", "bcc_to_fcc": "BCC-to-FCC"}
TRAINING_PARENT = {"fcc_to_bcc": "fcc", "bcc_to_fcc": "bcc"}

EXPECTED_SOURCE_HASHES = {
    SOURCE_STUDY / "study_design.json": "89fcf35435e714788bfa73c6504620c323cb5ad9a96dc39475a4c6f1887b5bd2",
    SOURCE_STUDY / "canonical_systems.csv": "9e33e2288283f1f4cbe2c929c6de91981e5a6c24d973bb70fe277abec0fbd0ab",
    SOURCE_STUDY / "frame_curation_ledger.csv": "97422c91ed33b69a2e41afa1c81ec40f6feba71f6b532bcd4b3467e59c6e885e",
    SOURCE_COVERAGE / "compute_environment_coverage.py": "5af091ba4025d5af79364928c7b94cb36ab4de452fb2d151193e69a87fa158c4",
    SOURCE_COVERAGE / "coverage_provenance.json": "f4116606c3e3f53d34ae4dd815cfb44ff80ba20f11b84a944955f0e6ab4c4754",
    SOURCE_COVERAGE / "environment_coverage.csv.gz": "3da8f54091fc7a3a2b2fed717b77286f59096db60ec1de02383decc39ef3f866",
    SOURCE_TABLES / "system_environment_coverage.csv": "8cbb81d3eee6706e4bd32f80ecc5b8b616eda8309208298dc195fc8105a4c3b3",
    SOURCE_TABLES / "environment_coverage_contrasts.csv": "c1301c3b74714923faef954da552b7e1cdee342d757be64693b3c0ca7fd0db49",
}

SNAPSHOT_FILES = {
    SOURCE_STUDY / "study_design.json": INPUTS / "study_design.json",
    SOURCE_STUDY / "canonical_systems.csv": INPUTS / "canonical_systems.csv",
    SOURCE_STUDY / "frame_curation_ledger.csv": INPUTS / "frame_curation_ledger.csv",
    SOURCE_COVERAGE / "compute_environment_coverage.py": INPUTS / "compute_environment_coverage.py",
    SOURCE_COVERAGE / "coverage_provenance.json": INPUTS / "coverage_provenance.json",
    SOURCE_COVERAGE / "environment_coverage.csv.gz": INPUTS / "frozen_environment_coverage.csv.gz",
    SOURCE_TABLES / "system_environment_coverage.csv": INPUTS / "frozen_system_environment_coverage.csv",
    SOURCE_TABLES / "environment_coverage_contrasts.csv": INPUTS / "frozen_environment_coverage_contrasts.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_digest(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        file_hash = sha256(file_path)
        size = file_path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        count += 1
        total_bytes += size
    return digest.hexdigest(), count, total_bytes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_gzip_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, compresslevel=6, mtime=0
        ) as compressed_handle:
            with io.TextIOWrapper(compressed_handle, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == sha256(source):
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def load_deepmd_system(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    declared = tuple((path / "type_map.raw").read_text().split())
    if declared != TYPE_MAP:
        raise ValueError(f"Unexpected type map in {path}: {declared}")
    atom_types = np.asarray(np.loadtxt(path / "type.raw", dtype=int, ndmin=1))
    coordinates: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    for set_dir in sorted(path.glob("set.*")):
        coordinates.append(np.load(set_dir / "coord.npy", allow_pickle=False))
        boxes.append(np.load(set_dir / "box.npy", allow_pickle=False))
        energies.append(np.load(set_dir / "energy.npy", allow_pickle=False))
    if not coordinates:
        raise FileNotFoundError(f"No set.* arrays in {path}")
    coordinates_array = np.concatenate(coordinates).reshape(-1, len(atom_types), 3)
    boxes_array = np.concatenate(boxes).reshape(-1, 3, 3)
    energy_array = np.concatenate(energies).reshape(-1)
    if not (
        len(coordinates_array) == len(boxes_array) == len(energy_array)
        and np.isfinite(coordinates_array).all()
        and np.isfinite(boxes_array).all()
        and np.isfinite(energy_array).all()
    ):
        raise ValueError(f"Invalid DeepMD arrays in {path}")
    return coordinates_array, boxes_array, atom_types, energy_array


def cell_parameters(cell: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    a, b, c = (np.asarray(vector, dtype=float) for vector in cell)
    lengths = [float(np.linalg.norm(vector)) for vector in (a, b, c)]
    if min(lengths) <= 0:
        raise ValueError("Non-positive cell length")

    def angle(left: np.ndarray, right: np.ndarray) -> float:
        cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
        return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))

    alpha = angle(b, c)
    beta = angle(a, c)
    gamma = angle(a, b)
    volume = float(abs(np.linalg.det(cell)))
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("Non-positive cell volume")
    return lengths[0], lengths[1], lengths[2], alpha, beta, gamma, volume


def cif_text(
    system_id: str,
    output_index: int,
    original_output_index: int,
    coordinates: np.ndarray,
    cell: np.ndarray,
    atom_types: np.ndarray,
) -> tuple[str, tuple[float, float, float, float, float, float, float]]:
    a, b, c, alpha, beta, gamma, volume = cell_parameters(cell)
    fractional = coordinates @ np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    counts = Counter(TYPE_MAP[int(index)] for index in atom_types)
    formula = " ".join(f"{element}{counts[element]}" for element in TYPE_MAP if counts[element])
    block_name = "".join(character if character.isalnum() or character == "_" else "_" for character in system_id)
    lines = [
        f"data_{block_name}_frame_{output_index:04d}",
        "_audit_creation_method 'Frozen every-100th-frame POCC coverage-control export'",
        f"_chemical_formula_structural '{formula}'",
        f"_cell_length_a {a:.10f}",
        f"_cell_length_b {b:.10f}",
        f"_cell_length_c {c:.10f}",
        f"_cell_angle_alpha {alpha:.10f}",
        f"_cell_angle_beta {beta:.10f}",
        f"_cell_angle_gamma {gamma:.10f}",
        f"_cod_database_code 0",
        f"_publ_section_comment 'system_id={system_id}; canonical_frame={output_index}; original_frame={original_output_index}'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    seen: Counter[str] = Counter()
    for type_index, position in zip(atom_types, fractional, strict=True):
        element = TYPE_MAP[int(type_index)]
        seen[element] += 1
        lines.append(
            f"{element}{seen[element]} {element} "
            f"{position[0]:.12f} {position[1]:.12f} {position[2]:.12f} 1.0"
        )
    return "\n".join(lines) + "\n", (a, b, c, alpha, beta, gamma, volume)


def copy_directory_with_manifest(
    source: Path,
    destination: Path,
    system_id: str,
    source_rows: list[dict],
) -> None:
    for source_file in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        atomic_copy(source_file, destination_file)
        source_hash = sha256(source_file)
        copied_hash = sha256(destination_file)
        if source_hash != copied_hash:
            raise ValueError(f"Snapshot hash mismatch for {source_file}")
        source_rows.append(
            {
                "record_type": "trajectory_file",
                "system_id": system_id,
                "source_path": str(source_file.resolve()),
                "copied_path": destination_file.relative_to(PACKAGE).as_posix(),
                "bytes": source_file.stat().st_size,
                "sha256": source_hash,
            }
        )


def snapshot_inputs() -> None:
    print("Verifying and snapshotting frozen inputs", flush=True)
    source_rows: list[dict] = []
    for source, expected_hash in EXPECTED_SOURCE_HASHES.items():
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen source hash drift: {source}")
        destination = SNAPSHOT_FILES[source]
        atomic_copy(source, destination)
        source_rows.append(
            {
                "record_type": "frozen_context",
                "system_id": "",
                "source_path": str(source.resolve()),
                "copied_path": destination.relative_to(PACKAGE).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": actual_hash,
            }
        )

    canonical_rows = read_csv(INPUTS / "canonical_systems.csv")
    design = json.loads((INPUTS / "study_design.json").read_text())
    canonical_by_path = {str(Path(row["path"]).resolve()): row for row in canonical_rows}
    canonical_by_key = {(row["system_id"], row["variant"]): row for row in canonical_rows}

    pure_source_paths = design["folds"]["fcc_to_bcc"]["regimes"]["pure"]["training_systems"]
    pure_ids = sorted({canonical_by_path[str(Path(path).resolve())]["system_id"] for path in pure_source_paths})
    pocc_ids = sorted(
        {
            system_id
            for fold in design["folds"].values()
            for key in ("pocc_train_source_ids", "pocc_test_source_ids")
            for system_id in fold[key]
        }
    )
    if len(pure_ids) != 20 or len(pocc_ids) != 83:
        raise ValueError("Unexpected trajectory population")

    relevant_ids = pure_ids + pocc_ids
    trajectory_rows: list[dict] = []
    frame_rows: list[dict] = []
    for sequence, system_id in enumerate(relevant_ids, start=1):
        row = canonical_by_key[(system_id, "independent")]
        source = Path(row["path"]).resolve()
        family = row["family"]
        destination = TRAJECTORIES / "deepmd" / family / system_id
        print(f"  [{sequence:03d}/{len(relevant_ids):03d}] {system_id}", flush=True)
        copy_directory_with_manifest(source, destination, system_id, source_rows)

        source_digest, source_file_count, source_bytes = directory_digest(source)
        copied_digest, copied_file_count, copied_bytes = directory_digest(destination)
        if (source_digest, source_file_count, source_bytes) != (
            copied_digest,
            copied_file_count,
            copied_bytes,
        ):
            raise ValueError(f"Trajectory directory mismatch for {system_id}")

        roles: dict[str, str] = {}
        for fold_name, fold in design["folds"].items():
            if family == "PURE":
                roles[fold_name] = "pure_training"
            elif system_id in fold["pocc_train_source_ids"]:
                roles[fold_name] = "random_candidate_training"
            elif system_id in fold["pocc_validation_source_ids"]:
                roles[fold_name] = "validation_not_used_in_this_fold"
            elif system_id in fold["pocc_test_source_ids"]:
                roles[fold_name] = "held_out_test"
            else:
                roles[fold_name] = "not_used"

        coordinates, boxes, atom_types, energies = load_deepmd_system(destination)
        selection_rows = read_csv(destination / "frame_selection.csv")
        if len(selection_rows) != len(coordinates) or len(coordinates) != int(row["selected_frames"]):
            raise ValueError(f"Frame-selection mismatch for {system_id}")

        cif_directory = TRAJECTORIES / "cif" / family / system_id
        for frame_index, selection in enumerate(selection_rows):
            if int(selection["output_index"]) != frame_index:
                raise ValueError(f"Non-contiguous frame selection for {system_id}")
            original_index = int(selection["original_output_index"])
            cif_path = cif_directory / f"frame_{frame_index:04d}__source_{original_index:06d}.cif"
            text, parameters = cif_text(
                system_id,
                frame_index,
                original_index,
                coordinates[frame_index],
                boxes[frame_index],
                atom_types,
            )
            atomic_text(cif_path, text)
            a, b, c, alpha, beta, gamma, volume = parameters
            frame_rows.append(
                {
                    "system_id": system_id,
                    "family": family,
                    "parent_family": row["parent_family"],
                    "structure": row["structure"],
                    "hnf_group": row["hnf_group"],
                    "natoms": len(atom_types),
                    "canonical_frame_index": frame_index,
                    "original_output_index": original_index,
                    "source_frame_ordinal": selection["source_frame_ordinal"],
                    "selection_kind": selection["selection_kind"],
                    "source_target": selection["source_target"],
                    "trajectory_chain": selection["trajectory_chain"],
                    "fcc_to_bcc_role": roles["fcc_to_bcc"],
                    "bcc_to_fcc_role": roles["bcc_to_fcc"],
                    "energy_eV_not_used": float(energies[frame_index]),
                    "cell_a_angstrom": a,
                    "cell_b_angstrom": b,
                    "cell_c_angstrom": c,
                    "cell_alpha_degree": alpha,
                    "cell_beta_degree": beta,
                    "cell_gamma_degree": gamma,
                    "cell_volume_angstrom3": volume,
                    "deepmd_path": destination.relative_to(PACKAGE).as_posix(),
                    "cif_path": cif_path.relative_to(PACKAGE).as_posix(),
                    "cif_bytes": cif_path.stat().st_size,
                    "cif_sha256": sha256(cif_path),
                }
            )

        trajectory_rows.append(
            {
                "system_id": system_id,
                "family": family,
                "parent_family": row["parent_family"],
                "structure": row["structure"],
                "hnf_group": row["hnf_group"],
                "element": row["element"],
                "natoms": row["natoms"],
                "source_frames_full_trajectory": row["source_frames"],
                "analyzed_frames_temporally_thinned": row["selected_frames"],
                "fcc_to_bcc_role": roles["fcc_to_bcc"],
                "bcc_to_fcc_role": roles["bcc_to_fcc"],
                "source_path": str(source),
                "deepmd_path": destination.relative_to(PACKAGE).as_posix(),
                "cif_directory": cif_directory.relative_to(PACKAGE).as_posix(),
                "file_count": copied_file_count,
                "bytes": copied_bytes,
                "directory_sha256": copied_digest,
            }
        )

    provenance = json.loads((INPUTS / "coverage_provenance.json").read_text())
    metadata_by_path = {
        str(Path(metadata["system_path"]).resolve()): metadata
        for metadata in provenance["descriptor_caches"]
    }
    descriptor_rows: list[dict] = []
    descriptor_dir = INPUTS / "descriptor_cache"
    for system_id in relevant_ids:
        canonical = canonical_by_key[(system_id, "independent")]
        system_path = str(Path(canonical["path"]).resolve())
        metadata = metadata_by_path[system_path]
        source_cache = SOURCE_COVERAGE / "descriptor_cache" / metadata["cache_file"]
        if sha256(source_cache) != metadata["cache_sha256"]:
            raise ValueError(f"Descriptor-cache hash drift for {system_id}")
        for relative_name, expected_hash in metadata["input_hashes"].items():
            if sha256(Path(system_path) / relative_name) != expected_hash:
                raise ValueError(f"Descriptor input hash drift for {system_id}/{relative_name}")
        copied_cache = descriptor_dir / metadata["cache_file"]
        source_metadata = source_cache.with_suffix(".json")
        copied_metadata = descriptor_dir / source_metadata.name
        atomic_copy(source_cache, copied_cache)
        atomic_copy(source_metadata, copied_metadata)
        source_rows.append(
            {
                "record_type": "descriptor_cache",
                "system_id": system_id,
                "source_path": str(source_cache.resolve()),
                "copied_path": copied_cache.relative_to(PACKAGE).as_posix(),
                "bytes": source_cache.stat().st_size,
                "sha256": metadata["cache_sha256"],
            }
        )
        descriptor_rows.append(
            {
                "system_id": system_id,
                "family": canonical["family"],
                "parent_family": canonical["parent_family"],
                "cache_path": copied_cache.relative_to(PACKAGE).as_posix(),
                "metadata_path": copied_metadata.relative_to(PACKAGE).as_posix(),
                "environment_count": metadata["environment_count"],
                "descriptor_dimensions": metadata["descriptor_dimensions"],
                "specification_sha256": metadata["specification_sha256"],
                "cache_sha256": metadata["cache_sha256"],
            }
        )

    atomic_csv(
        INPUTS / "source_file_manifest.csv",
        source_rows,
        ["record_type", "system_id", "source_path", "copied_path", "bytes", "sha256"],
    )
    atomic_csv(
        RESULTS / "trajectory_manifest.csv",
        trajectory_rows,
        list(trajectory_rows[0]),
    )
    atomic_csv(RESULTS / "frame_manifest.csv", frame_rows, list(frame_rows[0]))
    atomic_csv(
        INPUTS / "descriptor_cache_manifest.csv",
        descriptor_rows,
        list(descriptor_rows[0]),
    )


def load_descriptor_records(system_ids: Iterable[str]) -> dict[str, dict[str, np.ndarray]]:
    manifest = {row["system_id"]: row for row in read_csv(INPUTS / "descriptor_cache_manifest.csv")}
    records: dict[str, dict[str, np.ndarray]] = {}
    for system_id in system_ids:
        row = manifest[system_id]
        cache_path = PACKAGE / row["cache_path"]
        if sha256(cache_path) != row["cache_sha256"]:
            raise ValueError(f"Packaged descriptor hash mismatch for {system_id}")
        with np.load(cache_path, allow_pickle=False) as values:
            record = {name: values[name] for name in values.files}
        required = {"descriptors", "frame_index", "atom_index", "element_index"}
        if set(record) != required or record["descriptors"].shape[1] != 150:
            raise ValueError(f"Unexpected descriptor arrays for {system_id}")
        records[system_id] = record
    return records


def build_candidate_rows(
    design: dict,
    frame_rows: list[dict[str, str]],
) -> list[dict]:
    frames_by_system: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        frames_by_system[row["system_id"]].append(row)
    output: list[dict] = []
    for fold_name in ("fcc_to_bcc", "bcc_to_fcc"):
        candidate_index = 0
        for system_code, system_id in enumerate(sorted(design["folds"][fold_name]["pocc_train_source_ids"])):
            frames = sorted(
                frames_by_system[system_id], key=lambda row: int(row["canonical_frame_index"])
            )
            for frame in frames:
                original_index = int(frame["original_output_index"])
                output.append(
                    {
                        "fold": fold_name,
                        "fold_label": FOLD_LABELS[fold_name],
                        "training_parent": TRAINING_PARENT[fold_name],
                        "candidate_index": candidate_index,
                        "candidate_system_code": system_code,
                        "system_id": system_id,
                        "parent_family": frame["parent_family"],
                        "structure": frame["structure"],
                        "hnf_group": frame["hnf_group"],
                        "natoms": frame["natoms"],
                        "canonical_frame_index": frame["canonical_frame_index"],
                        "original_output_index": original_index,
                        "source_frame_ordinal": frame["source_frame_ordinal"],
                        "systematic_first_frame": original_index == 0,
                        "source_target": frame["source_target"],
                        "trajectory_chain": frame["trajectory_chain"],
                        "deepmd_path": frame["deepmd_path"],
                        "cif_path": frame["cif_path"],
                        "cif_sha256": frame["cif_sha256"],
                    }
                )
                candidate_index += 1
    return output


def frozen_distance_lookup() -> dict[tuple[str, str, int, int, str], dict[str, float]]:
    lookup: dict[tuple[str, str, int, int, str], dict[str, float]] = defaultdict(dict)
    for row in read_gzip_csv(INPUTS / "frozen_environment_coverage.csv.gz"):
        key = (
            row["fold"],
            row["system_id"],
            int(row["frame_index"]),
            int(row["atom_index"]),
            row["central_element"],
        )
        lookup[key][row["regime"]] = float(row["coverage_distance"])
    return lookup


def make_test_arrays(
    fold_name: str,
    test_ids: list[str],
    records: dict[str, dict[str, np.ndarray]],
    distance_lookup: dict[tuple[str, str, int, int, str], dict[str, float]],
) -> dict:
    descriptors: list[np.ndarray] = []
    elements: list[np.ndarray] = []
    atom_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    baseline: list[float] = []
    frozen_seed: list[float] = []
    system_slices: list[tuple[int, int]] = []
    cursor = 0
    for system_id in test_ids:
        record = records[system_id]
        count = len(record["descriptors"])
        descriptors.append(record["descriptors"].astype(np.float64))
        elements.append(record["element_index"].astype(np.int8))
        atom_indices.append(record["atom_index"].astype(np.int32))
        frame_indices.append(record["frame_index"].astype(np.int32))
        for frame_index, atom_index, element_index in zip(
            record["frame_index"],
            record["atom_index"],
            record["element_index"],
            strict=True,
        ):
            key = (
                fold_name,
                system_id,
                int(frame_index),
                int(atom_index),
                TYPE_MAP[int(element_index)],
            )
            values = distance_lookup[key]
            baseline.append(values["pure"])
            frozen_seed.append(values["pure_pocc_seed"])
        system_slices.append((cursor, cursor + count))
        cursor += count
    return {
        "descriptors": np.concatenate(descriptors),
        "element_index": np.concatenate(elements),
        "atom_index": np.concatenate(atom_indices),
        "frame_index": np.concatenate(frame_indices),
        "baseline": np.asarray(baseline, dtype=np.float64),
        "frozen_seed": np.asarray(frozen_seed, dtype=np.float64),
        "system_slices": system_slices,
        "test_ids": test_ids,
    }


def candidate_distance_matrix(
    candidate_rows: list[dict],
    records: dict[str, dict[str, np.ndarray]],
    test: dict,
) -> np.ndarray:
    matrix = np.empty((len(candidate_rows), len(test["descriptors"])), dtype=np.float64)
    test_elements = test["element_index"]
    for candidate_position, candidate in enumerate(candidate_rows):
        record = records[candidate["system_id"]]
        frame_index = int(candidate["canonical_frame_index"])
        frame_mask = record["frame_index"] == frame_index
        distances = np.full(len(test["descriptors"]), np.inf, dtype=np.float64)
        for element_index in range(len(TYPE_MAP)):
            candidate_mask = frame_mask & (record["element_index"] == element_index)
            query_mask = test_elements == element_index
            candidate_values = record["descriptors"][candidate_mask].astype(np.float64)
            if not len(candidate_values) or not np.any(query_mask):
                raise ValueError("Missing central-element descriptor")
            tree = cKDTree(candidate_values)
            distances[query_mask] = tree.query(
                test["descriptors"][query_mask], k=1, workers=1
            )[0]
        if not np.isfinite(distances).all():
            raise ValueError("Non-finite candidate distance")
        matrix[candidate_position] = distances
    return matrix


def systematic_result(
    matrix: np.ndarray,
    candidates: list[dict],
    test: dict,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    systematic_indices = [
        int(row["candidate_index"]) for row in candidates if row["systematic_first_frame"]
    ]
    expected_systems = len({row["system_id"] for row in candidates})
    if len(systematic_indices) != expected_systems:
        raise ValueError("Systematic treatment is not exactly one frame per system")
    augmented = np.minimum(test["baseline"], np.min(matrix[systematic_indices], axis=0))
    maximum_environment_difference = float(np.max(np.abs(augmented - test["frozen_seed"])))
    system_reductions = []
    for start, stop in test["system_slices"]:
        pure_median = float(np.median(test["baseline"][start:stop]))
        augmented_median = float(np.median(augmented[start:stop]))
        system_reductions.append((pure_median - augmented_median) / pure_median)
    fold_statistic = float(np.median(system_reductions))
    return (
        np.asarray(systematic_indices, dtype=np.int16),
        np.asarray(system_reductions, dtype=np.float64),
        fold_statistic,
        maximum_environment_difference,
    )


def random_selections(
    fold_name: str,
    control: str,
    candidates: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    seed_sequence = np.random.SeedSequence(
        [MASTER_RANDOM_SEED, FOLD_CODES[fold_name], CONTROL_CODES[control]]
    )
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    number_of_candidates = len(candidates)
    system_codes = np.asarray([int(row["candidate_system_code"]) for row in candidates])
    number_of_systems = len(np.unique(system_codes))
    if control == "pooled_random_frames":
        selected = np.empty((N_DRAWS, number_of_systems), dtype=np.int16)
        for draw_index in range(N_DRAWS):
            selected[draw_index] = rng.choice(
                number_of_candidates, size=number_of_systems, replace=False
            )
    elif control == "one_per_trajectory":
        columns = []
        for system_code in range(number_of_systems):
            eligible = np.flatnonzero(system_codes == system_code)
            if not len(eligible):
                raise ValueError("Missing candidate system code")
            columns.append(rng.choice(eligible, size=N_DRAWS, replace=True))
        selected = np.column_stack(columns).astype(np.int16)
    else:
        raise ValueError(control)
    unique_systems = np.fromiter(
        (len(np.unique(system_codes[indices])) for indices in selected),
        dtype=np.int16,
        count=N_DRAWS,
    )
    return selected, unique_systems


def summarize_draws(
    matrix: np.ndarray,
    selections: np.ndarray,
    test: dict,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    number_of_systems = len(test["system_slices"])
    fold_statistics = np.empty(len(selections), dtype=np.float64)
    system_reductions = np.empty((len(selections), number_of_systems), dtype=np.float64)
    pure_medians = np.asarray(
        [np.median(test["baseline"][start:stop]) for start, stop in test["system_slices"]],
        dtype=np.float64,
    )
    for batch_start in range(0, len(selections), batch_size):
        batch_stop = min(batch_start + batch_size, len(selections))
        selected_distances = matrix[selections[batch_start:batch_stop]]
        augmented = np.minimum(
            test["baseline"][np.newaxis, :], np.min(selected_distances, axis=1)
        )
        batch_reductions = np.empty((batch_stop - batch_start, number_of_systems))
        for system_position, (start, stop) in enumerate(test["system_slices"]):
            augmented_median = np.median(augmented[:, start:stop], axis=1)
            batch_reductions[:, system_position] = (
                pure_medians[system_position] - augmented_median
            ) / pure_medians[system_position]
        system_reductions[batch_start:batch_stop] = batch_reductions
        fold_statistics[batch_start:batch_stop] = np.median(batch_reductions, axis=1)
    return fold_statistics, system_reductions


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def empirical_percentile(values: np.ndarray, observed: float) -> float:
    below = int(np.count_nonzero(values < observed))
    equal = int(np.count_nonzero(values == observed))
    return 100.0 * (below + 0.5 * equal) / len(values)


def interpretation(observed: float, values: np.ndarray, control: str) -> str:
    q975 = quantile(values, 0.975)
    p_value = (1 + int(np.count_nonzero(values >= observed))) / (len(values) + 1)
    if control == "pooled_random_frames":
        if observed > q975 and p_value < 0.05:
            return "systematic_one_per_system_exceeds_naive_pooled_random_reference"
        return "no_resolved_systematic_advantage_over_naive_pooled_random_reference"
    if quantile(values, 0.025) <= observed <= q975:
        return "first_frame_choice_is_within_one_per_trajectory_random_reference"
    return "first_frame_choice_is_outside_one_per_trajectory_95_percent_reference"


def run_analysis(reuse_distance_cache: bool) -> None:
    design = json.loads((INPUTS / "study_design.json").read_text())
    trajectory_rows = read_csv(RESULTS / "trajectory_manifest.csv")
    frame_rows = read_csv(RESULTS / "frame_manifest.csv")
    candidate_rows = build_candidate_rows(design, frame_rows)
    atomic_csv(RESULTS / "candidate_frames.csv", candidate_rows, list(candidate_rows[0]))

    all_ids = [row["system_id"] for row in trajectory_rows]
    records = load_descriptor_records(all_ids)
    distance_lookup = frozen_distance_lookup()
    frozen_contrasts = read_csv(INPUTS / "frozen_environment_coverage_contrasts.csv")

    draw_rows: list[dict] = []
    system_summary_rows: list[dict] = []
    control_summary_rows: list[dict] = []
    reconstruction_rows: list[dict] = []
    selection_arrays: dict[str, np.ndarray] = {}
    system_reduction_arrays: dict[str, np.ndarray] = {}
    distance_cache_files: list[dict] = []

    for fold_name in ("fcc_to_bcc", "bcc_to_fcc"):
        started = time.perf_counter()
        fold = design["folds"][fold_name]
        test_ids = sorted(fold["pocc_test_source_ids"])
        candidates = [row for row in candidate_rows if row["fold"] == fold_name]
        candidates.sort(key=lambda row: int(row["candidate_index"]))
        test = make_test_arrays(fold_name, test_ids, records, distance_lookup)
        cache_path = CACHE / f"{fold_name}_candidate_to_test_distances.npz"
        if reuse_distance_cache and cache_path.is_file():
            print(f"Loading {fold_name} distance cache", flush=True)
            with np.load(cache_path, allow_pickle=False) as cached:
                matrix = cached["candidate_distance"]
                cached_baseline = cached["baseline_distance"]
                cached_seed = cached["frozen_seed_distance"]
            if matrix.shape != (len(candidates), len(test["descriptors"])):
                raise ValueError(f"Stale distance cache shape for {fold_name}")
            np.testing.assert_array_equal(cached_baseline, test["baseline"])
            np.testing.assert_array_equal(cached_seed, test["frozen_seed"])
        else:
            print(
                f"Computing {fold_name}: {len(candidates)} candidate frames x "
                f"{len(test['descriptors'])} held-out environments",
                flush=True,
            )
            matrix = candidate_distance_matrix(candidates, records, test)
            CACHE.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(f".{cache_path.stem}.{os.getpid()}.tmp.npz")
            np.savez_compressed(
                temporary,
                candidate_distance=matrix,
                baseline_distance=test["baseline"],
                frozen_seed_distance=test["frozen_seed"],
            )
            temporary.replace(cache_path)
        distance_cache_files.append(
            {
                "fold": fold_name,
                "path": cache_path.relative_to(PACKAGE).as_posix(),
                "shape_candidates": matrix.shape[0],
                "shape_test_environments": matrix.shape[1],
                "bytes": cache_path.stat().st_size,
                "sha256": sha256(cache_path),
            }
        )

        systematic_indices, systematic_system_reductions, systematic_statistic, max_env_diff = systematic_result(
            matrix, candidates, test
        )
        frozen_system_reductions = np.asarray(
            [
                float(row["relative_distance_reduction"])
                for row in frozen_contrasts
                if row["fold"] == fold_name and row["treatment"] == "pure_pocc_seed"
            ]
        )
        frozen_fold_statistic = float(np.median(frozen_system_reductions))
        if max_env_diff > 1.0e-10 or abs(systematic_statistic - frozen_fold_statistic) > 1.0e-12:
            raise ValueError(f"Systematic seed reconstruction failed for {fold_name}")
        reconstruction_rows.append(
            {
                "fold": fold_name,
                "frozen_seed_reduction_fraction": frozen_fold_statistic,
                "recomputed_seed_reduction_fraction": systematic_statistic,
                "absolute_fold_difference": abs(systematic_statistic - frozen_fold_statistic),
                "maximum_environment_distance_difference": max_env_diff,
                "systematic_frame_count": len(systematic_indices),
                "status": "passed",
            }
        )
        selection_arrays[f"{fold_name}__systematic"] = systematic_indices
        system_reduction_arrays[f"{fold_name}__systematic"] = systematic_system_reductions

        for control in ("pooled_random_frames", "one_per_trajectory"):
            print(f"  {fold_name}: {control}, {N_DRAWS:,} draws", flush=True)
            selections, unique_systems = random_selections(fold_name, control, candidates)
            fold_statistics, system_reductions = summarize_draws(matrix, selections, test)
            key = f"{fold_name}__{control}"
            selection_arrays[key] = selections
            system_reduction_arrays[key] = system_reductions

            for draw_index, (statistic, unique_count) in enumerate(
                zip(fold_statistics, unique_systems, strict=True)
            ):
                draw_rows.append(
                    {
                        "fold": fold_name,
                        "fold_label": FOLD_LABELS[fold_name],
                        "control": control,
                        "draw_index": draw_index,
                        "coverage_reduction_fraction": statistic,
                        "coverage_reduction_percent": 100.0 * statistic,
                        "unique_training_systems": int(unique_count),
                    }
                )

            for system_position, system_id in enumerate(test_ids):
                values = system_reductions[:, system_position]
                observed = systematic_system_reductions[system_position]
                start, stop = test["system_slices"][system_position]
                system_summary_rows.append(
                    {
                        "fold": fold_name,
                        "fold_label": FOLD_LABELS[fold_name],
                        "control": control,
                        "system_id": system_id,
                        "held_out_environment_count": stop - start,
                        "pure_median_distance": float(np.median(test["baseline"][start:stop])),
                        "systematic_seed_reduction_fraction": observed,
                        "random_median_reduction_fraction": float(np.median(values)),
                        "random_q025_reduction_fraction": quantile(values, 0.025),
                        "random_q975_reduction_fraction": quantile(values, 0.975),
                        "systematic_empirical_percentile": empirical_percentile(values, observed),
                    }
                )

            p_value = (1 + int(np.count_nonzero(fold_statistics >= systematic_statistic))) / (
                len(fold_statistics) + 1
            )
            control_summary_rows.append(
                {
                    "fold": fold_name,
                    "fold_label": FOLD_LABELS[fold_name],
                    "training_parent": TRAINING_PARENT[fold_name],
                    "control": control,
                    "random_draws": N_DRAWS,
                    "labels_per_draw": selections.shape[1],
                    "candidate_frames": len(candidates),
                    "candidate_systems": len({row["system_id"] for row in candidates}),
                    "held_out_systems": len(test_ids),
                    "held_out_environments": len(test["descriptors"]),
                    "systematic_seed_reduction_fraction": systematic_statistic,
                    "systematic_seed_reduction_percent": 100.0 * systematic_statistic,
                    "random_median_reduction_fraction": float(np.median(fold_statistics)),
                    "random_median_reduction_percent": 100.0 * float(np.median(fold_statistics)),
                    "random_q025_reduction_fraction": quantile(fold_statistics, 0.025),
                    "random_q025_reduction_percent": 100.0 * quantile(fold_statistics, 0.025),
                    "random_q975_reduction_fraction": quantile(fold_statistics, 0.975),
                    "random_q975_reduction_percent": 100.0 * quantile(fold_statistics, 0.975),
                    "systematic_minus_random_median_fraction": systematic_statistic
                    - float(np.median(fold_statistics)),
                    "systematic_minus_random_median_percentage_points": 100.0
                    * (systematic_statistic - float(np.median(fold_statistics))),
                    "systematic_empirical_percentile": empirical_percentile(
                        fold_statistics, systematic_statistic
                    ),
                    "one_sided_p_random_ge_systematic": p_value,
                    "unique_systems_median": float(np.median(unique_systems)),
                    "unique_systems_q025": quantile(unique_systems.astype(float), 0.025),
                    "unique_systems_q975": quantile(unique_systems.astype(float), 0.975),
                    "unique_systems_minimum": int(np.min(unique_systems)),
                    "unique_systems_maximum": int(np.max(unique_systems)),
                    "interpretation_code": interpretation(
                        systematic_statistic, fold_statistics, control
                    ),
                }
            )
        print(f"Completed {fold_name} in {time.perf_counter() - started:.1f} s", flush=True)

    atomic_csv(
        RESULTS / "control_summary.csv", control_summary_rows, list(control_summary_rows[0])
    )
    atomic_gzip_csv(
        RESULTS / "random_draw_distribution.csv.gz", draw_rows, list(draw_rows[0])
    )
    atomic_csv(
        RESULTS / "system_random_control_summary.csv",
        system_summary_rows,
        list(system_summary_rows[0]),
    )
    atomic_csv(
        RESULTS / "systematic_seed_reconstruction.csv",
        reconstruction_rows,
        list(reconstruction_rows[0]),
    )
    atomic_csv(
        RESULTS / "distance_cache_manifest.csv",
        distance_cache_files,
        list(distance_cache_files[0]),
    )

    selections_path = RESULTS / "random_selections.npz"
    temporary = selections_path.with_name(f".{selections_path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **selection_arrays)
    temporary.replace(selections_path)
    system_values_path = RESULTS / "system_random_reductions.npz"
    temporary = system_values_path.with_name(
        f".{system_values_path.stem}.{os.getpid()}.tmp.npz"
    )
    np.savez_compressed(temporary, **system_reduction_arrays)
    temporary.replace(system_values_path)

    summary = {
        "schema_version": 1,
        "analysis": "matched-label random-frame structural-coverage control",
        "master_random_seed": MASTER_RANDOM_SEED,
        "rng": "NumPy PCG64 with SeedSequence([master_seed, fold_code, control_code])",
        "random_draws_per_fold_and_control": N_DRAWS,
        "coverage_frame_variant": "every-100th-frame temporal thinning",
        "descriptor_specification_sha256": json.loads(
            (INPUTS / "coverage_provenance.json").read_text()
        )["specification_sha256"],
        "controls": control_summary_rows,
        "systematic_reconstruction": reconstruction_rows,
        "artifacts": {
            "control_summary": {
                "path": "results/control_summary.csv",
                "sha256": sha256(RESULTS / "control_summary.csv"),
            },
            "draw_distribution": {
                "path": "results/random_draw_distribution.csv.gz",
                "rows": len(draw_rows),
                "sha256": sha256(RESULTS / "random_draw_distribution.csv.gz"),
            },
            "system_summary": {
                "path": "results/system_random_control_summary.csv",
                "rows": len(system_summary_rows),
                "sha256": sha256(RESULTS / "system_random_control_summary.csv"),
            },
            "random_selections": {
                "path": "results/random_selections.npz",
                "sha256": sha256(selections_path),
            },
            "system_random_reductions": {
                "path": "results/system_random_reductions.npz",
                "sha256": sha256(system_values_path),
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    atomic_json(RESULTS / "analysis_summary.json", summary)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-snapshot",
        action="store_true",
        help="Use the already copied inputs and trajectories.",
    )
    parser.add_argument(
        "--reuse-distance-cache",
        action="store_true",
        help="Reuse hash-checked candidate-to-test distance caches when present.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if not arguments.reuse_snapshot:
        snapshot_inputs()
    required = [
        INPUTS / "study_design.json",
        INPUTS / "descriptor_cache_manifest.csv",
        RESULTS / "trajectory_manifest.csv",
        RESULTS / "frame_manifest.csv",
    ]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Snapshot is incomplete; rerun without --reuse-snapshot")
    run_analysis(reuse_distance_cache=arguments.reuse_distance_cache)
    print(json.dumps(json.loads((RESULTS / "analysis_summary.json").read_text())["controls"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
