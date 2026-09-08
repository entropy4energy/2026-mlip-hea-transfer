#!/usr/bin/env python3
"""Join A11/A12 balanced replicas into four-temperature curves and paired slopes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


BRANCHES = {
    ("DPA2__bcc_parent", "bcc"),
    ("DPA3__bcc_parent", "bcc"),
    ("DPA4__bcc_parent", "bcc"),
    ("DPA4__fcc_parent", "fcc"),
}
TEMPERATURES = (300, 600, 900, 1200)
CHEMICAL_SEEDS = (20260825, 20262843, 20264861)
VELOCITY_SEEDS = (20260901, 20260903)
BOOTSTRAP_SEED = 20260827
BOOTSTRAP_DRAWS = 10000
VOLUME_FIELD = "volume_A3_per_atom_mean"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, values: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(values); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def verify_receipt(receipt_path: Path, replicas: Path, groups: Path) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed" or receipt.get("replicas_sha256") != sha256(replicas) or receipt.get("groups_sha256") != sha256(groups):
        raise ValueError(f"analysis receipt does not bind inputs: {receipt_path}")
    return receipt


def replica_key(row: dict[str, str]) -> tuple[str, str, int, int, int]:
    return (
        row["model_id"], row["simulated_phase"], int(row["target_temperature_K"]),
        int(row["chemical_seed"]), int(row["velocity_seed"]),
    )


def sample_draws(matrix: dict[tuple[int, int, int], float], rng: np.random.Generator) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    slopes = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    deltas = {(low, high): np.empty(BOOTSTRAP_DRAWS, dtype=float) for low, high in zip(TEMPERATURES, TEMPERATURES[1:])}
    temperatures = np.asarray(TEMPERATURES, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        chemical_indices = rng.integers(0, len(CHEMICAL_SEEDS), len(CHEMICAL_SEEDS))
        velocity_draws = [rng.integers(0, len(VELOCITY_SEEDS), len(VELOCITY_SEEDS)) for _ in chemical_indices]
        volume_by_temperature = []
        for temperature in TEMPERATURES:
            sampled = []
            for selection, chemical_index in enumerate(chemical_indices):
                chemical = CHEMICAL_SEEDS[int(chemical_index)]
                for velocity_index in velocity_draws[selection]:
                    velocity = VELOCITY_SEEDS[int(velocity_index)]
                    sampled.append(matrix[(temperature, chemical, velocity)])
            volume_by_temperature.append(float(np.mean(sampled)))
        values = np.asarray(volume_by_temperature)
        slopes[draw] = np.polyfit(temperatures, np.log(values), 1)[0]
        for index, transition in enumerate(zip(TEMPERATURES, TEMPERATURES[1:])):
            deltas[transition][draw] = values[index + 1] - values[index]
    return slopes, deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a11-replicas", type=Path, required=True)
    parser.add_argument("--a11-groups", type=Path, required=True)
    parser.add_argument("--a11-receipt", type=Path, required=True)
    parser.add_argument("--a12-replicas", type=Path, required=True)
    parser.add_argument("--a12-groups", type=Path, required=True)
    parser.add_argument("--a12-receipt", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    input_paths = [
        args.a11_replicas.resolve(), args.a11_groups.resolve(), args.a11_receipt.resolve(),
        args.a12_replicas.resolve(), args.a12_groups.resolve(), args.a12_receipt.resolve(),
    ]
    verify_receipt(input_paths[2], input_paths[0], input_paths[1])
    verify_receipt(input_paths[5], input_paths[3], input_paths[4])
    a11_replicas = [row for row in rows(input_paths[0]) if (row["model_id"], row["simulated_phase"]) in BRANCHES]
    a12_replicas = rows(input_paths[3])
    combined_replicas = a11_replicas + a12_replicas
    keyed = {replica_key(row): row for row in combined_replicas}
    expected = {
        (model, phase, temperature, chemical, velocity)
        for model, phase in BRANCHES for temperature in TEMPERATURES
        for chemical in CHEMICAL_SEEDS for velocity in VELOCITY_SEEDS
    }
    if len(combined_replicas) != 96 or len(keyed) != 96 or set(keyed) != expected:
        raise ValueError("combined A11/A12 replica matrix is not the exact 96-row population")
    a11_groups = [row for row in rows(input_paths[1]) if (row["model_id"], row["simulated_phase"]) in BRANCHES]
    a12_groups = rows(input_paths[4])
    combined_groups: list[dict[str, object]] = []
    for source, campaign in ((a11_groups, "A11"), (a12_groups, "A12")):
        for row in source:
            item: dict[str, object] = dict(row)
            item["source_campaign"] = campaign
            combined_groups.append(item)
    group_keys = {(str(row["model_id"]), str(row["simulated_phase"]), int(float(str(row["temperature_K"])))) for row in combined_groups}
    expected_groups = {(model, phase, temperature) for model, phase in BRANCHES for temperature in TEMPERATURES}
    if len(combined_groups) != 16 or group_keys != expected_groups:
        raise ValueError("combined group table is not the exact 16-row four-temperature grid")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in combined_replicas:
        grouped[(row["model_id"], row["simulated_phase"])].append(row)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    responses: list[dict[str, object]] = []
    for model_id, phase in sorted(BRANCHES):
        replica_group = grouped[(model_id, phase)]
        result: dict[str, object] = {
            "model_id": model_id, "architecture": model_id.split("__", 1)[0],
            "training_parent": phase, "simulated_phase": phase,
            "temperature_range_K": "300;600;900;1200", "planned_replicas_per_temperature": 6,
            "status": "blocked_incomplete_or_failed_temperature_matrix",
            "estimand": "paired hierarchical-bootstrap slope of ln(equal-replica mean volume) versus temperature",
        }
        complete = len(replica_group) == 24 and all(row["endpoint_status"] == "eligible" for row in replica_group)
        if complete:
            matrix = {
                (int(row["target_temperature_K"]), int(row["chemical_seed"]), int(row["velocity_seed"])): float(row[VOLUME_FIELD])
                for row in replica_group
            }
            volumes = np.asarray(
                [np.mean([matrix[(temperature, chemical, velocity)] for chemical in CHEMICAL_SEEDS for velocity in VELOCITY_SEEDS]) for temperature in TEMPERATURES]
            )
            slope, intercept = np.polyfit(np.asarray(TEMPERATURES, dtype=float), np.log(volumes), 1)
            predicted = slope * np.asarray(TEMPERATURES, dtype=float) + intercept
            total = float(np.sum((np.log(volumes) - np.mean(np.log(volumes))) ** 2))
            residual = float(np.sum((np.log(volumes) - predicted) ** 2))
            slope_draws, delta_draws = sample_draws(matrix, rng)
            adjacent_resolved = []
            for index, (low, high) in enumerate(zip(TEMPERATURES, TEMPERATURES[1:])):
                draws = delta_draws[(low, high)]
                difference = volumes[index + 1] - volumes[index]
                low_ci, high_ci = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
                result[f"volume_delta_{high}_minus_{low}_A3_per_atom"] = float(difference)
                result[f"volume_delta_{high}_minus_{low}_ci95_low"] = low_ci
                result[f"volume_delta_{high}_minus_{low}_ci95_high"] = high_ci
                adjacent_resolved.append(low_ci > 0.0)
            point_increasing = bool(np.all(np.diff(volumes) > 0.0))
            monotonic_status = (
                "resolved_increasing_all_adjacent_intervals_above_zero" if all(adjacent_resolved)
                else "increasing_point_estimates_with_unresolved_adjacent_contrast" if point_increasing
                else "nonmonotonic_point_estimates"
            )
            result.update(
                {
                    "status": "eligible", "monotonic_status": monotonic_status,
                    **{f"volume_A3_per_atom_{temperature}K": float(volumes[index]) for index, temperature in enumerate(TEMPERATURES)},
                    "volumetric_alpha_per_K": float(slope),
                    "volumetric_alpha_per_K_ci95_low": float(np.quantile(slope_draws, 0.025)),
                    "volumetric_alpha_per_K_ci95_high": float(np.quantile(slope_draws, 0.975)),
                    "isotropic_linear_alpha_per_K": float(slope / 3.0),
                    "isotropic_linear_alpha_per_K_ci95_low": float(np.quantile(slope_draws, 0.025) / 3.0),
                    "isotropic_linear_alpha_per_K_ci95_high": float(np.quantile(slope_draws, 0.975) / 3.0),
                    "ln_volume_fit_r_squared": 1.0 - residual / total if total > 0.0 else 1.0,
                }
            )
        responses.append(result)
    groups_path = args.groups.resolve(); response_path = args.response.resolve(); receipt_path = args.receipt.resolve()
    for path in (groups_path, response_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    atomic_csv(groups_path, sorted(combined_groups, key=lambda row: (str(row["model_id"]), str(row["simulated_phase"]), int(float(str(row["temperature_K"]))))))
    atomic_csv(response_path, responses)
    eligible = sum(row["status"] == "eligible" for row in responses)
    receipt = {
        "schema_version": 1, "status": "passed",
        "scientific_status": "complete_four_temperature_matrix" if eligible == 4 else "partial_with_explicit_blocks",
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in input_paths],
        "groups": str(groups_path), "groups_sha256": sha256(groups_path), "group_rows": len(combined_groups),
        "response": str(response_path), "response_sha256": sha256(response_path), "response_rows": len(responses),
        "eligible_response_rows": eligible,
        "builder": str(Path(__file__).resolve()), "builder_sha256": sha256(Path(__file__).resolve()),
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
