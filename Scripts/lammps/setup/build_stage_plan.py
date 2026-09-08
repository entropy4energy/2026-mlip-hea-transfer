#!/usr/bin/env python3
"""Build immutable, balanced task plans for the staged two-phase campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path


CAMPAIGN_ID = "hea_two_phase_v1_20260827"
FIRST_CHEMICAL_SEED = 20260825
NVE_TEMPERATURES = (300, 1200)
NVE_TIMESTEPS_PS = (0.0005, 0.001)
NVE_VELOCITY_SEED = 20260901
NVT_EQUILIBRATION_PS = 40.0
NVT_EQUILIBRATION_TIMESTEP_PS = 0.0005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
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


def task_row(
    *,
    calculation: str,
    model: dict[str, str],
    structure: dict[str, str],
    input_path: Path,
    output_dir: Path,
    protocol: Path,
    model_manifest: Path,
    velocity_seed: str = "NA",
    temperature: str = "NA",
    timestep: str = "NA",
    n_eq: str = "NA",
    n_prod: str = "NA",
) -> dict[str, object]:
    task_id = "__".join(
        part
        for part in (
            calculation,
            model["model_id"],
            structure["structure_id"],
            f"T{temperature}" if temperature != "NA" else "",
            f"dt{timestep}" if timestep != "NA" else "",
        )
        if part
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
        "n_eq": n_eq,
        "n_prod": n_prod,
        "natoms": structure["natoms"],
        "structure_role": structure["role"],
        "model_path": model["frozen_model"],
        "data_path": structure["data_path"],
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "protocol_path": str(protocol),
        "model_manifest": str(model_manifest),
        "campaign_id": CAMPAIGN_ID,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "preflight",
            "relax_iso",
            "relax_tri",
            "relax_size",
            "relax_iso_coupled",
            "relax_tri_coupled",
            "relax_size_coupled",
            "equilibrate_nvt",
            "nve",
        ),
    )
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_manifest = args.model_manifest.resolve()
    structure_manifest = args.structure_manifest.resolve()
    protocol = args.protocol.resolve()
    campaign_root = args.campaign_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite an existing plan or receipt")

    models = read_tsv(model_manifest)
    structures = read_csv(structure_manifest)
    if len(models) != 6 or {row["model_id"] for row in models} != {
        f"{architecture}__{parent}_parent"
        for architecture in ("DPA2", "DPA3", "DPA4")
        for parent in ("bcc", "fcc")
    }:
        raise ValueError("model manifest is not the complete six-model phase set")
    for model in models:
        model_path = Path(model["frozen_model"])
        if not model_path.is_file() or sha256(model_path) != model["frozen_sha256"]:
            raise ValueError(f"frozen model hash mismatch: {model['model_id']}")

    primary = [row for row in structures if row["role"] == "primary" and int(row["natoms"]) == 2000]
    size = [row for row in structures if row["role"] == "cell_size_sensitivity" and int(row["natoms"]) == 250]
    if len(primary) != 10 or len(size) != 2:
        raise ValueError("structure manifest does not contain the fixed 10 primary and 2 size cells")
    for structure in structures:
        data_path = Path(structure["data_path"])
        if not data_path.is_file() or sha256(data_path) != structure["data_sha256"]:
            raise ValueError(f"structure hash mismatch: {structure['structure_id']}")

    suite_root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, object]] = []
    if args.stage in {"preflight", "relax_iso", "relax_iso_coupled"}:
        calculation = args.stage
        input_name = {
            "preflight": "01_preflight.in",
            "relax_iso": "02_relax.in",
            "relax_iso_coupled": "02_relax_coupled.in",
        }[calculation]
        for model in sorted(models, key=lambda row: row["model_id"]):
            for structure in sorted(primary, key=lambda row: row["structure_id"]):
                output_dir = (
                    campaign_root
                    / f"stage_a_{calculation}"
                    / model["model_id"]
                    / structure["structure_id"]
                )
                rows.append(
                    task_row(
                        calculation=calculation,
                        model=model,
                        structure=structure,
                        input_path=(suite_root / "inputs" / input_name).resolve(),
                        output_dir=output_dir,
                        protocol=protocol,
                        model_manifest=model_manifest,
                    )
                )
    elif args.stage in {"relax_tri", "relax_tri_coupled"}:
        calculation = args.stage
        input_name = (
            "02_relax_coupled.in"
            if calculation == "relax_tri_coupled"
            else "02_relax.in"
        )
        for model in sorted(models, key=lambda row: row["model_id"]):
            for structure in sorted(
                (row for row in primary if int(row["chemical_seed"]) == FIRST_CHEMICAL_SEED),
                key=lambda row: row["structure_id"],
            ):
                rows.append(
                    task_row(
                        calculation=calculation,
                        model=model,
                        structure=structure,
                        input_path=(suite_root / "inputs" / input_name).resolve(),
                        output_dir=(
                            campaign_root
                            / f"stage_a_{calculation}"
                            / model["model_id"]
                            / structure["structure_id"]
                        ),
                        protocol=protocol,
                        model_manifest=model_manifest,
                    )
                )
    elif args.stage in {"relax_size", "relax_size_coupled"}:
        calculation = args.stage
        input_name = (
            "02_relax_coupled.in"
            if calculation == "relax_size_coupled"
            else "02_relax.in"
        )
        for model in sorted(models, key=lambda row: row["model_id"]):
            for structure in sorted(size, key=lambda row: row["structure_id"]):
                if model["training_parent"] != structure["phase"]:
                    continue
                rows.append(
                    task_row(
                        calculation=calculation,
                        model=model,
                        structure=structure,
                        input_path=(suite_root / "inputs" / input_name).resolve(),
                        output_dir=(
                            campaign_root
                            / f"stage_a_{calculation}"
                            / model["model_id"]
                            / structure["structure_id"]
                        ),
                        protocol=protocol,
                        model_manifest=model_manifest,
                    )
                )
    elif args.stage == "equilibrate_nvt":
        first_primary = [
            row for row in primary if int(row["chemical_seed"]) == FIRST_CHEMICAL_SEED
        ]
        for model in sorted(models, key=lambda row: row["model_id"]):
            for structure in sorted(first_primary, key=lambda row: row["structure_id"]):
                if model["training_parent"] != structure["phase"]:
                    continue
                relaxed_root = (
                    campaign_root
                    / "stage_a_relax_iso_coupled"
                    / model["model_id"]
                    / structure["structure_id"]
                )
                relaxed_data = relaxed_root / "relaxed.data"
                if not relaxed_data.is_file() or not (relaxed_root / "run.complete").is_file():
                    raise FileNotFoundError(
                        f"accepted relaxed input is unavailable: {relaxed_root}"
                    )
                equil_structure = dict(structure)
                equil_structure["data_path"] = str(relaxed_data)
                for temperature in NVE_TEMPERATURES:
                    rows.append(
                        task_row(
                            calculation="equilibrate_nvt",
                            model=model,
                            structure=equil_structure,
                            input_path=(suite_root / "inputs" / "11_nvt_equilibrate.in").resolve(),
                            output_dir=(
                                campaign_root
                                / "stage_a_equilibrate_nvt"
                                / model["model_id"]
                                / structure["structure_id"]
                                / f"T{temperature}"
                            ),
                            protocol=protocol,
                            model_manifest=model_manifest,
                            velocity_seed=str(NVE_VELOCITY_SEED),
                            temperature=str(temperature),
                            timestep=str(NVT_EQUILIBRATION_TIMESTEP_PS),
                            n_eq=str(
                                round(
                                    NVT_EQUILIBRATION_PS
                                    / NVT_EQUILIBRATION_TIMESTEP_PS
                                )
                            ),
                        )
                    )
    else:
        first_primary = [
            row for row in primary if int(row["chemical_seed"]) == FIRST_CHEMICAL_SEED
        ]
        for model in sorted(models, key=lambda row: row["model_id"]):
            for structure in sorted(first_primary, key=lambda row: row["structure_id"]):
                if model["training_parent"] != structure["phase"]:
                    continue
                equil_root = (
                    campaign_root
                    / "stage_a_equilibrate_nvt"
                    / model["model_id"]
                    / structure["structure_id"]
                )
                for temperature in NVE_TEMPERATURES:
                    equilibrated_root = equil_root / f"T{temperature}"
                    equilibrated_data = equilibrated_root / "equilibrated.data"
                    if not equilibrated_data.is_file() or not (
                        equilibrated_root / "run.complete"
                    ).is_file():
                        raise FileNotFoundError(
                            f"accepted NVT-equilibrated input is unavailable: {equilibrated_root}"
                        )
                    nve_structure = dict(structure)
                    nve_structure["data_path"] = str(equilibrated_data)
                    for timestep in NVE_TIMESTEPS_PS:
                        n_prod = round(50.0 / timestep)
                        dt_tag = str(timestep).replace(".", "p")
                        rows.append(
                            task_row(
                                calculation="nve",
                                model=model,
                                structure=nve_structure,
                                input_path=(suite_root / "inputs" / "10_nve_drift.in").resolve(),
                                output_dir=(
                                    campaign_root
                                    / "stage_a_nve"
                                    / model["model_id"]
                                    / structure["structure_id"]
                                    / f"T{temperature}"
                                    / f"dt_{dt_tag}"
                                ),
                                protocol=protocol,
                                model_manifest=model_manifest,
                                velocity_seed=str(NVE_VELOCITY_SEED),
                                temperature=str(temperature),
                                timestep=str(timestep),
                                n_prod=str(n_prod),
                            )
                        )

    if not rows:
        raise ValueError("stage plan contains no tasks")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate task IDs in stage plan")
    atomic_write_rows(output, rows)
    receipt = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "stage": args.stage,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "tasks": len(rows),
        "calculations": dict(Counter(str(row["calculation"]) for row in rows)),
        "model_ids": sorted({str(row["model_id"]) for row in rows}),
        "simulated_phases": sorted({str(row["simulated_phase"]) for row in rows}),
        "chemical_seeds": sorted({int(row["chemical_seed"]) for row in rows}),
        "model_manifest": str(model_manifest),
        "model_manifest_sha256": sha256(model_manifest),
        "structure_manifest": str(structure_manifest),
        "structure_manifest_sha256": sha256(structure_manifest),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "plan": str(output),
        "plan_sha256": sha256(output),
        "status": "frozen",
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{receipt_path.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, receipt_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
