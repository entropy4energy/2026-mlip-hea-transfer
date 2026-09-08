#!/usr/bin/env python3
"""Build receipt-chained NVT, NVE, and NPT plans for the two-phase campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path


CAMPAIGN_ID = "hea_two_phase_v1_20260827"
FIRST_CHEMICAL_SEED = 20260825
LAST_CHEMICAL_SEED = 20264861
NVE_TEMPERATURES = (300, 1200)
NVE_TIMESTEPS_PS = (0.0005, 0.001)
THERMAL_TEMPERATURES = (300, 600, 900, 1200)
VELOCITY_SEEDS = (20260901, 20260903, 20260907)
NVT_VELOCITY_SEED = 20260901
EQUILIBRATION_PS = 40.0
NVE_PRODUCTION_PS = 50.0
NPT_PRODUCTION_PS = 50.0
NPT_LENGTH_SENSITIVITY_PS = 100.0

BASE_FIELDS = (
    "task_id",
    "calculation",
    "model_id",
    "architecture",
    "training_parent",
    "simulated_phase",
    "chemical_seed",
    "velocity_seed",
    "temperature_K",
    "timestep_ps",
    "n_eq",
    "n_prod",
    "natoms",
    "structure_role",
    "model_path",
    "data_path",
    "input_path",
    "output_dir",
    "protocol_path",
    "model_manifest",
    "campaign_id",
)
CHAIN_FIELDS = (
    "parent_task_id",
    "parent_receipt_path",
    "parent_receipt_sha256",
    "parent_validation_path",
    "parent_validation_sha256",
    "timestep_selection_path",
    "timestep_selection_sha256",
    "timestep_selection_receipt_path",
    "timestep_selection_receipt_sha256",
    "timestep_selection_validation_path",
    "timestep_selection_validation_sha256",
    "physical_duration_ps",
    "length_sensitivity_role",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def atomic_rows(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(*BASE_FIELDS, *CHAIN_FIELDS),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_stage_validation(path: Path) -> tuple[dict[str, object], list[tuple[dict[str, str], dict[str, str]]]]:
    validation_path = path.resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("planned_tasks") != validation.get("validated_tasks"):
        raise ValueError(f"{validation_path}: validation lacks one row per planned task")
    plan_path = Path(str(validation["plan"])).resolve()
    summary_path = Path(str(validation["summary_csv"])).resolve()
    if validation.get("plan_sha256") != sha256(plan_path):
        raise ValueError(f"{validation_path}: upstream plan hash mismatch")
    if validation.get("summary_csv_sha256") != sha256(summary_path):
        raise ValueError(f"{validation_path}: upstream summary hash mismatch")
    plans = read_rows(plan_path, "\t")
    summaries = read_rows(summary_path)
    plans_by_task = {row["task_id"]: row for row in plans}
    summaries_by_task = {row["task_id"]: row for row in summaries}
    if len(plans_by_task) != len(plans) or set(plans_by_task) != set(summaries_by_task):
        raise ValueError(f"{validation_path}: upstream task population mismatch")
    return validation, [
        (plans_by_task[task_id], summaries_by_task[task_id])
        for task_id in sorted(plans_by_task)
    ]


def find_parent(
    validated: list[tuple[dict[str, str], dict[str, str]]],
    *,
    calculation: str,
    model_id: str,
    phase: str,
    chemical_seed: int,
    temperature: int | None = None,
    velocity_seed: int | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    matches: list[tuple[dict[str, str], dict[str, str]]] = []
    for plan, summary in validated:
        if (
            plan["calculation"] != calculation
            or plan["model_id"] != model_id
            or plan["simulated_phase"] != phase
            or int(plan["chemical_seed"]) != chemical_seed
        ):
            continue
        if temperature is not None and int(float(plan["temperature_K"])) != temperature:
            continue
        if velocity_seed is not None and int(plan["velocity_seed"]) != velocity_seed:
            continue
        matches.append((plan, summary))
    if len(matches) != 1:
        raise ValueError(
            "expected one upstream task for "
            f"{calculation}/{model_id}/{phase}/C{chemical_seed}/T{temperature}/V{velocity_seed}; "
            f"found {len(matches)}"
        )
    plan, summary = matches[0]
    if summary.get("status") != "passed":
        raise ValueError(f"required upstream task did not pass: {plan['task_id']}")
    return plan, summary


def verify_parent_data(plan: dict[str, str], output_name: str) -> dict[str, str]:
    root = Path(plan["output_dir"]).resolve()
    receipt_path = root / "completion_receipt.json"
    marker_path = root / "run.complete"
    index_path = root / "run_outputs.tsv"
    data_path = root / output_name
    for path in (receipt_path, marker_path, index_path, data_path):
        if not path.is_file():
            raise FileNotFoundError(f"required upstream artifact is missing: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed" or receipt.get("lammps_return_code") != 0:
        raise ValueError(f"upstream completion receipt did not pass: {receipt_path}")
    if marker.get("completion_receipt_sha256") != sha256(receipt_path):
        raise ValueError(f"upstream marker does not bind its receipt: {marker_path}")
    if receipt.get("output_index_sha256") != sha256(index_path):
        raise ValueError(f"upstream receipt does not bind its output index: {receipt_path}")
    index = {row["relative_path"]: row for row in read_rows(index_path, "\t")}
    if output_name not in index or index[output_name]["sha256"] != sha256(data_path):
        raise ValueError(f"upstream output index does not bind {data_path}")
    return {
        "parent_task_id": plan["task_id"],
        "parent_receipt_path": str(receipt_path),
        "parent_receipt_sha256": sha256(receipt_path),
        "data_path": str(data_path),
    }


def load_timestep_selection(
    selection_path: Path,
    receipt_path: Path,
    validation_path: Path,
) -> dict[tuple[str, str], float]:
    selection_path = selection_path.resolve()
    receipt_path = receipt_path.resolve()
    validation_path = validation_path.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed" or receipt.get("scientific_status") != "selected":
        raise ValueError("NVE timestep-selection receipt is not fully selected")
    if receipt.get("selection_sha256") != sha256(selection_path):
        raise ValueError("NVE timestep-selection receipt hash mismatch")
    if validation.get("status") != "passed":
        raise ValueError("independent NVE timestep-selection validation did not pass")
    if validation.get("selector_receipt_sha256") != sha256(receipt_path):
        raise ValueError("independent NVE selection validation does not bind its receipt")
    rows = read_rows(selection_path)
    selected: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["model_id"], row["simulated_phase"])
        if row["status"] != "selected" or not row["accepted_timestep_ps"]:
            raise ValueError(f"no accepted timestep for {key}")
        if key in selected:
            raise ValueError(f"duplicate accepted timestep for {key}")
        value = float(row["accepted_timestep_ps"])
        if value not in NVE_TIMESTEPS_PS:
            raise ValueError(f"unsupported accepted timestep {value} for {key}")
        selected[key] = value
    if len(selected) != 6:
        raise ValueError(f"expected six phase-native timestep selections, found {len(selected)}")
    return selected


def integral_steps(duration_ps: float, timestep_ps: float) -> int:
    steps = round(duration_ps / timestep_ps)
    if not math.isclose(steps * timestep_ps, duration_ps, abs_tol=1.0e-12):
        raise ValueError(f"duration {duration_ps} ps is not integral for dt={timestep_ps} ps")
    return steps


def task_row(
    *,
    calculation: str,
    model: dict[str, str],
    structure: dict[str, str],
    data_path: str,
    parent: dict[str, str],
    parent_validation: Path,
    input_path: Path,
    output_dir: Path,
    protocol: Path,
    model_manifest: Path,
    velocity_seed: int,
    temperature: int,
    timestep: float,
    n_eq: int | None,
    n_prod: int | None,
    duration_ps: float,
    length_role: str,
    selection_artifacts: dict[str, str],
) -> dict[str, object]:
    task_id = "__".join(
        (
            calculation,
            model["model_id"],
            structure["structure_id"],
            f"T{temperature}",
            f"V{velocity_seed}",
            f"dt{timestep}",
        )
    )
    return {
        "task_id": task_id,
        "calculation": calculation,
        "model_id": model["model_id"],
        "architecture": model["architecture"],
        "training_parent": model["training_parent"],
        "simulated_phase": structure["phase"],
        "chemical_seed": structure["chemical_seed"],
        "velocity_seed": velocity_seed,
        "temperature_K": temperature,
        "timestep_ps": timestep,
        "n_eq": n_eq if n_eq is not None else "NA",
        "n_prod": n_prod if n_prod is not None else "NA",
        "natoms": structure["natoms"],
        "structure_role": structure["role"],
        "model_path": model["frozen_model"],
        "data_path": data_path,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "protocol_path": str(protocol),
        "model_manifest": str(model_manifest),
        "campaign_id": CAMPAIGN_ID,
        "parent_task_id": parent["parent_task_id"],
        "parent_receipt_path": parent["parent_receipt_path"],
        "parent_receipt_sha256": parent["parent_receipt_sha256"],
        "parent_validation_path": str(parent_validation),
        "parent_validation_sha256": sha256(parent_validation),
        **selection_artifacts,
        "physical_duration_ps": duration_ps,
        "length_sensitivity_role": length_role,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("equilibrate_nvt", "nve", "equilibrate_npt", "npt"),
    )
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--upstream-validation", type=Path, required=True)
    parser.add_argument("--timestep-selection", type=Path)
    parser.add_argument("--timestep-selection-receipt", type=Path)
    parser.add_argument("--timestep-selection-validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    builder = Path(__file__).resolve()
    suite_root = builder.parents[1]
    model_manifest = args.model_manifest.resolve()
    structure_manifest = args.structure_manifest.resolve()
    protocol = args.protocol.resolve()
    campaign_root = args.campaign_root.resolve()
    upstream_validation = args.upstream_validation.resolve()
    output = args.output.resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite an existing plan or receipt")
    if campaign_root.name != CAMPAIGN_ID:
        raise ValueError(f"campaign root must end in {CAMPAIGN_ID}")

    models = read_rows(model_manifest, "\t")
    structures = read_rows(structure_manifest)
    expected_models = {
        f"{architecture}__{phase}_parent"
        for architecture in ("DPA2", "DPA3", "DPA4")
        for phase in ("bcc", "fcc")
    }
    if len(models) != 6 or {row["model_id"] for row in models} != expected_models:
        raise ValueError("model manifest is not the complete six-model phase set")
    for model in models:
        path = Path(model["frozen_model"])
        if not path.is_file() or sha256(path) != model["frozen_sha256"]:
            raise ValueError(f"frozen model hash mismatch: {model['model_id']}")
    primary = [row for row in structures if row["role"] == "primary" and int(row["natoms"]) == 2000]
    if len(primary) != 10:
        raise ValueError("structure manifest does not contain ten 2,000-atom primary cells")
    for structure in primary:
        path = Path(structure["data_path"])
        if not path.is_file() or sha256(path) != structure["data_sha256"]:
            raise ValueError(f"structure hash mismatch: {structure['structure_id']}")

    _, validated = load_stage_validation(upstream_validation)
    selection_artifacts = {field: "NA" for field in CHAIN_FIELDS[5:11]}
    selected_timesteps: dict[tuple[str, str], float] = {}
    if args.stage in {"equilibrate_npt", "npt"}:
        if not all(
            (
                args.timestep_selection,
                args.timestep_selection_receipt,
                args.timestep_selection_validation,
            )
        ):
            raise ValueError("NPT stages require all timestep-selection artifacts")
        selection_path = args.timestep_selection.resolve()
        selection_receipt = args.timestep_selection_receipt.resolve()
        selection_validation = args.timestep_selection_validation.resolve()
        selected_timesteps = load_timestep_selection(
            selection_path, selection_receipt, selection_validation
        )
        selection_artifacts = {
            "timestep_selection_path": str(selection_path),
            "timestep_selection_sha256": sha256(selection_path),
            "timestep_selection_receipt_path": str(selection_receipt),
            "timestep_selection_receipt_sha256": sha256(selection_receipt),
            "timestep_selection_validation_path": str(selection_validation),
            "timestep_selection_validation_sha256": sha256(selection_validation),
        }

    rows: list[dict[str, object]] = []
    native_models = sorted(models, key=lambda row: row["model_id"])
    for model in native_models:
        model_structures = sorted(
            (row for row in primary if row["phase"] == model["training_parent"]),
            key=lambda row: row["structure_id"],
        )
        if args.stage in {"equilibrate_nvt", "nve"}:
            model_structures = [
                row for row in model_structures if int(row["chemical_seed"]) == FIRST_CHEMICAL_SEED
            ]
        for structure in model_structures:
            phase = structure["phase"]
            chemical_seed = int(structure["chemical_seed"])
            if args.stage == "equilibrate_nvt":
                parent_plan, _ = find_parent(
                    validated,
                    calculation="relax_iso_coupled",
                    model_id=model["model_id"],
                    phase=phase,
                    chemical_seed=chemical_seed,
                )
                parent = verify_parent_data(parent_plan, "relaxed.data")
                for temperature in NVE_TEMPERATURES:
                    timestep = 0.0005
                    rows.append(
                        task_row(
                            calculation="equilibrate_nvt",
                            model=model,
                            structure=structure,
                            data_path=parent["data_path"],
                            parent=parent,
                            parent_validation=upstream_validation,
                            input_path=(suite_root / "inputs" / "11_nvt_equilibrate.in").resolve(),
                            output_dir=campaign_root / "stage_a_equilibrate_nvt" / model["model_id"] / structure["structure_id"] / f"T{temperature}",
                            protocol=protocol,
                            model_manifest=model_manifest,
                            velocity_seed=NVT_VELOCITY_SEED,
                            temperature=temperature,
                            timestep=timestep,
                            n_eq=integral_steps(EQUILIBRATION_PS, timestep),
                            n_prod=None,
                            duration_ps=EQUILIBRATION_PS,
                            length_role="timestep_gate_equilibration",
                            selection_artifacts=selection_artifacts,
                        )
                    )
            elif args.stage == "nve":
                for temperature in NVE_TEMPERATURES:
                    parent_plan, _ = find_parent(
                        validated,
                        calculation="equilibrate_nvt",
                        model_id=model["model_id"],
                        phase=phase,
                        chemical_seed=chemical_seed,
                        temperature=temperature,
                        velocity_seed=NVT_VELOCITY_SEED,
                    )
                    parent = verify_parent_data(parent_plan, "equilibrated.data")
                    for timestep in NVE_TIMESTEPS_PS:
                        rows.append(
                            task_row(
                                calculation="nve",
                                model=model,
                                structure=structure,
                                data_path=parent["data_path"],
                                parent=parent,
                                parent_validation=upstream_validation,
                                input_path=(suite_root / "inputs" / "10_nve_drift.in").resolve(),
                                output_dir=campaign_root / "stage_a_nve" / model["model_id"] / structure["structure_id"] / f"T{temperature}" / f"dt_{str(timestep).replace('.', 'p')}",
                                protocol=protocol,
                                model_manifest=model_manifest,
                                velocity_seed=NVT_VELOCITY_SEED,
                                temperature=temperature,
                                timestep=timestep,
                                n_eq=None,
                                n_prod=integral_steps(NVE_PRODUCTION_PS, timestep),
                                duration_ps=NVE_PRODUCTION_PS,
                                length_role="matched_timestep_gate",
                                selection_artifacts=selection_artifacts,
                            )
                        )
            elif args.stage == "equilibrate_npt":
                parent_plan, _ = find_parent(
                    validated,
                    calculation="relax_iso_coupled",
                    model_id=model["model_id"],
                    phase=phase,
                    chemical_seed=chemical_seed,
                )
                parent = verify_parent_data(parent_plan, "relaxed.data")
                timestep = selected_timesteps[(model["model_id"], phase)]
                for temperature in THERMAL_TEMPERATURES:
                    for velocity_seed in VELOCITY_SEEDS:
                        rows.append(
                            task_row(
                                calculation="equilibrate_npt",
                                model=model,
                                structure=structure,
                                data_path=parent["data_path"],
                                parent=parent,
                                parent_validation=upstream_validation,
                                input_path=(suite_root / "inputs" / "12_npt_equilibrate.in").resolve(),
                                output_dir=campaign_root / "stage_c_equilibrate_npt" / model["model_id"] / structure["structure_id"] / f"T{temperature}" / f"V{velocity_seed}",
                                protocol=protocol,
                                model_manifest=model_manifest,
                                velocity_seed=velocity_seed,
                                temperature=temperature,
                                timestep=timestep,
                                n_eq=integral_steps(EQUILIBRATION_PS, timestep),
                                n_prod=None,
                                duration_ps=EQUILIBRATION_PS,
                                length_role="primary_equilibration",
                                selection_artifacts=selection_artifacts,
                            )
                        )
            else:
                timestep = selected_timesteps[(model["model_id"], phase)]
                for temperature in THERMAL_TEMPERATURES:
                    for velocity_seed in VELOCITY_SEEDS:
                        parent_plan, _ = find_parent(
                            validated,
                            calculation="equilibrate_npt",
                            model_id=model["model_id"],
                            phase=phase,
                            chemical_seed=chemical_seed,
                            temperature=temperature,
                            velocity_seed=velocity_seed,
                        )
                        parent = verify_parent_data(parent_plan, "equilibrated.data")
                        extended = (
                            chemical_seed in {FIRST_CHEMICAL_SEED, LAST_CHEMICAL_SEED}
                            and velocity_seed == NVT_VELOCITY_SEED
                            and temperature in NVE_TEMPERATURES
                        )
                        duration = NPT_LENGTH_SENSITIVITY_PS if extended else NPT_PRODUCTION_PS
                        rows.append(
                            task_row(
                                calculation="npt",
                                model=model,
                                structure=structure,
                                data_path=parent["data_path"],
                                parent=parent,
                                parent_validation=upstream_validation,
                                input_path=(suite_root / "inputs" / "05_npt_thermo.in").resolve(),
                                output_dir=campaign_root / "stage_c_npt" / model["model_id"] / structure["structure_id"] / f"T{temperature}" / f"V{velocity_seed}",
                                protocol=protocol,
                                model_manifest=model_manifest,
                                velocity_seed=velocity_seed,
                                temperature=temperature,
                                timestep=timestep,
                                n_eq=None,
                                n_prod=integral_steps(duration, timestep),
                                duration_ps=duration,
                                length_role="extended_100ps" if extended else "primary_50ps",
                                selection_artifacts=selection_artifacts,
                            )
                        )

    expected_count = {
        "equilibrate_nvt": 12,
        "nve": 24,
        "equilibrate_npt": 360,
        "npt": 360,
    }[args.stage]
    if len(rows) != expected_count or len({str(row["task_id"]) for row in rows}) != len(rows):
        raise ValueError(
            f"{args.stage} plan is incomplete or has duplicate IDs: {len(rows)} rows"
        )
    if args.stage == "npt":
        extended = sum(row["length_sensitivity_role"] == "extended_100ps" for row in rows)
        if extended != 24:
            raise ValueError(f"expected 24 extended NPT tasks, found {extended}")

    atomic_rows(output, rows)
    receipt: dict[str, object] = {
        "schema_version": 2,
        "status": "frozen",
        "campaign_id": CAMPAIGN_ID,
        "stage": args.stage,
        "builder": str(builder),
        "builder_sha256": sha256(builder),
        "tasks": len(rows),
        "calculations": dict(Counter(str(row["calculation"]) for row in rows)),
        "model_ids": sorted({str(row["model_id"]) for row in rows}),
        "simulated_phases": sorted({str(row["simulated_phase"]) for row in rows}),
        "chemical_seeds": sorted({int(row["chemical_seed"]) for row in rows}),
        "velocity_seeds": sorted({int(row["velocity_seed"]) for row in rows}),
        "temperatures_K": sorted({int(row["temperature_K"]) for row in rows}),
        "timesteps_ps": sorted({float(row["timestep_ps"]) for row in rows}),
        "model_manifest": str(model_manifest),
        "model_manifest_sha256": sha256(model_manifest),
        "structure_manifest": str(structure_manifest),
        "structure_manifest_sha256": sha256(structure_manifest),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "upstream_validation": str(upstream_validation),
        "upstream_validation_sha256": sha256(upstream_validation),
        "plan": str(output),
        "plan_sha256": sha256(output),
    }
    if args.stage in {"equilibrate_npt", "npt"}:
        receipt.update(selection_artifacts)
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
