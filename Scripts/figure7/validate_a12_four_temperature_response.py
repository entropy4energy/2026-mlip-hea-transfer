#!/usr/bin/env python3
"""Independently replay the paired A11/A12 four-temperature response table."""

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


BRANCHES = (
    ("DPA2__bcc_parent", "bcc"), ("DPA3__bcc_parent", "bcc"),
    ("DPA4__bcc_parent", "bcc"), ("DPA4__fcc_parent", "fcc"),
)
TEMPERATURES = (300, 600, 900, 1200)
CHEMICAL = (20260825, 20262843, 20264861)
VELOCITY = (20260901, 20260903)
SEED = 20260827
DRAWS = 10000
VOLUME = "volume_A3_per_atom_mean"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(expected: float, observed: str, tolerance: float = 1.0e-11) -> bool:
    try:
        return math.isclose(expected, float(observed), rel_tol=0.0, abs_tol=tolerance)
    except ValueError:
        return False


def independent_draws(matrix: dict[tuple[int, int, int], float], rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    slope_draws = np.empty(DRAWS)
    delta_draws = [np.empty(DRAWS) for _ in range(3)]
    x = np.asarray(TEMPERATURES, dtype=float)
    for draw in range(DRAWS):
        chosen_chemistry = rng.integers(0, 3, size=3)
        chosen_velocity = [rng.integers(0, 2, size=2) for _ in chosen_chemistry]
        averages = []
        for temperature in TEMPERATURES:
            samples = []
            for selection in range(3):
                chemical = CHEMICAL[int(chosen_chemistry[selection])]
                for velocity_index in chosen_velocity[selection]:
                    samples.append(matrix[(temperature, chemical, VELOCITY[int(velocity_index)])])
            averages.append(float(np.mean(samples)))
        array = np.asarray(averages)
        slope_draws[draw] = np.polyfit(x, np.log(array), 1)[0]
        for index in range(3):
            delta_draws[index][draw] = array[index + 1] - array[index]
    return slope_draws, delta_draws


def atomic_json(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks = 0
    inputs = [Path(item["path"]).resolve() for item in receipt.get("inputs", [])]
    hashes = [item["sha256"] for item in receipt.get("inputs", [])]
    if len(inputs) != 6:
        errors.append("response receipt does not list six A11/A12 source files")
    for path, expected_hash in zip(inputs, hashes):
        checks += 2
        if not path.is_file() or sha256(path) != expected_hash:
            errors.append(f"input hash failed: {path}")
    groups_path = Path(str(receipt["groups"])).resolve(); response_path = Path(str(receipt["response"])).resolve()
    for name, path in (("groups", groups_path), ("response", response_path)):
        checks += 2
        if not path.is_file() or receipt.get(f"{name}_sha256") != sha256(path):
            errors.append(f"output hash failed: {name}")
    if errors:
        payload = {"schema_version": 1, "status": "failed", "checks": checks, "errors": errors}
        atomic_json(args.output.resolve(), payload); print(json.dumps(payload, indent=2)); raise SystemExit(1)

    a11_replicas = [row for row in rows(inputs[0]) if (row["model_id"], row["simulated_phase"]) in BRANCHES]
    a12_replicas = rows(inputs[3])
    replicas = a11_replicas + a12_replicas
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in replicas:
        grouped[(row["model_id"], row["simulated_phase"])].append(row)
    groups = rows(groups_path); responses = rows(response_path)
    response_by_key = {(row["model_id"], row["simulated_phase"]): row for row in responses}
    group_keys = {(row["model_id"], row["simulated_phase"], int(float(row["temperature_K"]))) for row in groups}
    expected_group_keys = {(model, phase, temperature) for model, phase in BRANCHES for temperature in TEMPERATURES}
    checks += 8
    if len(replicas) != 96 or len(groups) != 16 or group_keys != expected_group_keys or len(response_by_key) != 4:
        errors.append("four-temperature populations are not 96/16/4")

    rng = np.random.default_rng(SEED)
    for key in BRANCHES:
        population = grouped[key]
        response = response_by_key.get(key)
        checks += 6
        if response is None or len(population) != 24:
            errors.append(f"missing response population: {key}"); continue
        complete = all(row["endpoint_status"] == "eligible" for row in population)
        if (response["status"] == "eligible") != complete:
            errors.append(f"response eligibility mismatch: {key}")
        if not complete:
            if response.get("volumetric_alpha_per_K", ""):
                errors.append(f"blocked response leaked a slope: {key}")
            continue
        matrix = {
            (int(row["target_temperature_K"]), int(row["chemical_seed"]), int(row["velocity_seed"])): float(row[VOLUME])
            for row in population
        }
        expected_matrix_keys = {(temperature, chemical, velocity) for temperature in TEMPERATURES for chemical in CHEMICAL for velocity in VELOCITY}
        if set(matrix) != expected_matrix_keys:
            errors.append(f"paired matrix incomplete: {key}"); continue
        volumes = np.asarray([np.mean([matrix[(temperature, chemical, velocity)] for chemical in CHEMICAL for velocity in VELOCITY]) for temperature in TEMPERATURES])
        slope, intercept = np.polyfit(np.asarray(TEMPERATURES, dtype=float), np.log(volumes), 1)
        predicted = slope * np.asarray(TEMPERATURES, dtype=float) + intercept
        total = float(np.sum((np.log(volumes) - np.mean(np.log(volumes))) ** 2))
        residual = float(np.sum((np.log(volumes) - predicted) ** 2))
        slope_draws, delta_draws = independent_draws(matrix, rng)
        numeric = {
            "volumetric_alpha_per_K": slope,
            "volumetric_alpha_per_K_ci95_low": float(np.quantile(slope_draws, 0.025)),
            "volumetric_alpha_per_K_ci95_high": float(np.quantile(slope_draws, 0.975)),
            "isotropic_linear_alpha_per_K": slope / 3.0,
            "isotropic_linear_alpha_per_K_ci95_low": float(np.quantile(slope_draws, 0.025)) / 3.0,
            "isotropic_linear_alpha_per_K_ci95_high": float(np.quantile(slope_draws, 0.975)) / 3.0,
            "ln_volume_fit_r_squared": 1.0 - residual / total if total > 0.0 else 1.0,
        }
        for index, temperature in enumerate(TEMPERATURES):
            numeric[f"volume_A3_per_atom_{temperature}K"] = float(volumes[index])
        resolved = []
        for index, (low_temperature, high_temperature) in enumerate(zip(TEMPERATURES, TEMPERATURES[1:])):
            draws = delta_draws[index]
            low_ci = float(np.quantile(draws, 0.025)); high_ci = float(np.quantile(draws, 0.975))
            numeric[f"volume_delta_{high_temperature}_minus_{low_temperature}_A3_per_atom"] = float(volumes[index + 1] - volumes[index])
            numeric[f"volume_delta_{high_temperature}_minus_{low_temperature}_ci95_low"] = low_ci
            numeric[f"volume_delta_{high_temperature}_minus_{low_temperature}_ci95_high"] = high_ci
            resolved.append(low_ci > 0.0)
        expected_monotonic = (
            "resolved_increasing_all_adjacent_intervals_above_zero" if all(resolved)
            else "increasing_point_estimates_with_unresolved_adjacent_contrast" if np.all(np.diff(volumes) > 0.0)
            else "nonmonotonic_point_estimates"
        )
        if response.get("monotonic_status") != expected_monotonic:
            errors.append(f"monotonic decision mismatch: {key}")
        for field, expected in numeric.items():
            checks += 1
            if not close(float(expected), response.get(field, "")):
                errors.append(f"response replay mismatch: {key}/{field}")
    payload = {
        "schema_version": 1, "status": "passed" if not errors else "failed",
        "checks": checks, "errors": errors,
        "response_receipt": str(receipt_path), "response_receipt_sha256": sha256(receipt_path),
        "groups": str(groups_path), "response": str(response_path),
        "validator": str(Path(__file__).resolve()), "validator_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(args.output.resolve(), payload); print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
