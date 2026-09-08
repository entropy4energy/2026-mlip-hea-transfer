#!/usr/bin/env python3
"""Independently replay A12 raw volume/pressure endpoints and balanced bootstrap groups."""

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


CHEMICAL = (20260825, 20262843, 20264861)
VELOCITY = (20260901, 20260903)
DRAW_SEED = 20260827
DRAWS = 10000
FIELDS = (
    "temperature_K_mean", "pressure_GPa_mean", "volume_A3_per_atom_mean",
    "density_g_cm3_mean", "enthalpy_eV_per_atom_mean", "potential_energy_eV_per_atom_mean",
    "cell_a_A_mean", "cell_b_A_mean", "cell_c_A_mean", "cell_alpha_deg_mean",
    "cell_beta_deg_mean", "cell_gamma_deg_mean",
)
VOLUME_FIELD = "volume_A3_per_atom_mean"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def close(left: float, right: str, tolerance: float = 1.0e-11) -> bool:
    try:
        return math.isclose(left, float(right), rel_tol=0.0, abs_tol=tolerance)
    except ValueError:
        return False


def thermo_volume_pressure(path: Path, natoms: int) -> tuple[float, float, int]:
    data = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 10 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"invalid thermo sample: {path}")
        data.append(values)
    array = np.asarray(data)
    if array.size and array[0, 0] == 0.0 and array[0, 1] == 0.0:
        array = array[1:]
    if array.shape != (500, 10) or not np.allclose(array[:, 1], np.arange(1, 501) * 0.1, atol=1.0e-10, rtol=0.0):
        raise ValueError(f"incorrect A12 sample grid: {path}")
    if set(np.rint(array[:, 2]).astype(int)) != {natoms}:
        raise ValueError(f"atom population changed: {path}")
    return float(np.mean(array[:, 5] / natoms)), float(np.mean(array[:, 4] * 1.0e-4)), array.shape[0]


def bootstrap(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.empty(DRAWS, dtype=float)
    for draw in range(DRAWS):
        chemical_indices = rng.integers(0, 3, 3)
        subtotal = 0.0
        for chemical_index in chemical_indices:
            velocity_indices = rng.integers(0, 2, 2)
            subtotal += float(np.mean(matrix[int(chemical_index), velocity_indices]))
        output[draw] = subtotal / 3.0
    return output


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
    parser.add_argument("--analysis-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.analysis_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks = 0
    bindings = {}
    for name in ("plan", "stage_summary", "dependency_ledger", "replicas", "groups"):
        path = Path(str(receipt[name])).resolve()
        bindings[name] = path
        checks += 2
        if not path.is_file() or receipt.get(f"{name}_sha256") != sha256(path):
            errors.append(f"receipt binding failed: {name}")
    if errors:
        payload = {"schema_version": 1, "status": "failed", "checks": checks, "errors": errors}
        atomic_json(args.output.resolve(), payload); print(json.dumps(payload, indent=2)); raise SystemExit(1)
    plans = {row["task_id"]: row for row in rows(bindings["plan"], "\t")}
    stages = {row["task_id"]: row for row in rows(bindings["stage_summary"])}
    ledger = rows(bindings["dependency_ledger"], "\t")
    replicas = rows(bindings["replicas"])
    groups = rows(bindings["groups"])
    replica_by_key = {
        (row["model_id"], row["simulated_phase"], int(row["target_temperature_K"]), int(row["chemical_seed"]), int(row["velocity_seed"])): row
        for row in replicas
    }
    checks += 8
    if len(ledger) != 48 or len(replicas) != 48 or len(replica_by_key) != 48 or len(groups) != 8:
        errors.append("A12 analysis populations are not 48 replicas/eight groups")
    for item in ledger:
        key = (item["model_id"], item["simulated_phase"], int(float(item["temperature_K"])), int(item["chemical_seed"]), int(item["velocity_seed"]))
        replica = replica_by_key.get(key)
        checks += 6
        if replica is None:
            errors.append(f"missing replica: {key}"); continue
        task_id = item["production_task_id"]
        if not task_id:
            if replica["endpoint_status"] == "eligible" or any(replica.get(field, "") for field in FIELDS):
                errors.append(f"blocked dependency leaked an estimate: {key}")
            continue
        if task_id not in plans or task_id not in stages:
            errors.append(f"production task missing from validation: {task_id}"); continue
        if replica["endpoint_status"] == "eligible":
            try:
                volume, pressure, samples = thermo_volume_pressure(Path(plans[task_id]["output_dir"]) / "thermo_samples.dat", int(plans[task_id]["natoms"]))
                checks += samples + 2
                if not close(volume, replica[VOLUME_FIELD]):
                    errors.append(f"raw volume replay mismatch: {task_id}")
                if not close(pressure, replica["pressure_GPa_mean"]):
                    errors.append(f"raw pressure replay mismatch: {task_id}")
            except (OSError, ValueError) as error:
                errors.append(f"raw replay failed for {task_id}: {error}")

    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in replicas:
        grouped[(row["model_id"], row["simulated_phase"], int(row["target_temperature_K"]))].append(row)
    group_by_key = {(row["model_id"], row["simulated_phase"], int(float(row["temperature_K"]))): row for row in groups}
    rng = np.random.default_rng(DRAW_SEED)
    for key in sorted(grouped):
        population = grouped[key]
        group = group_by_key.get(key)
        checks += 5
        if group is None or len(population) != 6:
            errors.append(f"missing/incomplete group: {key}"); continue
        eligible = all(row["endpoint_status"] == "eligible" for row in population)
        if (group["status"] == "eligible") != eligible:
            errors.append(f"group eligibility mismatch: {key}")
        if not eligible:
            if any(group.get(field, "") for field in FIELDS):
                errors.append(f"blocked group leaked estimates: {key}")
            continue
        for field in FIELDS:
            matrix = np.asarray(
                [[float(next(row[field] for row in population if int(row["chemical_seed"]) == chemical and int(row["velocity_seed"]) == velocity)) for velocity in VELOCITY] for chemical in CHEMICAL]
            )
            draws = bootstrap(matrix, rng)
            expected_mean = float(np.mean(matrix))
            expected_low = float(np.quantile(draws, 0.025)); expected_high = float(np.quantile(draws, 0.975))
            checks += 3
            if not close(expected_mean, group[field]) or not close(expected_low, group[f"{field}_ci95_low"]) or not close(expected_high, group[f"{field}_ci95_high"]):
                errors.append(f"group/bootstrap replay mismatch: {key}/{field}")
    payload = {
        "schema_version": 1, "status": "passed" if not errors else "failed",
        "checks": checks, "errors": errors,
        "analysis_receipt": str(receipt_path), "analysis_receipt_sha256": sha256(receipt_path),
        "replicas": str(bindings["replicas"]), "groups": str(bindings["groups"]),
        "validator": str(Path(__file__).resolve()), "validator_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(args.output.resolve(), payload); print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
