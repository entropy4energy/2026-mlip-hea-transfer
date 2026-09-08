#!/usr/bin/env python3
"""Run and reduce one preregistered DeePMD evaluation job.

Raw ``dp test`` detail files are converted into frame-, system-, and
force-bin-level sufficient statistics.  The raw detail files are then removed
because the checkpoint, immutable system list, command, hashes, and reducer are
retained and can reproduce them.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
TRAINING = PROJECT / "work" / "training"
EVALUATION = PROJECT / "work" / "evaluation"
MANIFEST = EVALUATION / "evaluation_manifest.csv"
FORCE_BIN_EDGES = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0, np.inf], dtype=float)
EV_A3_TO_GPA = 160.21766208
DETAIL_SUFFIXES = (
    ".e.out",
    ".e_peratom.out",
    ".f.out",
    ".v.out",
    ".v_peratom.out",
    ".s.out",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def manifest_job(evaluation_id: str) -> dict[str, str]:
    matches = [row for row in read_csv(MANIFEST) if row["evaluation_id"] == evaluation_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row for {evaluation_id}, found {len(matches)}")
    return matches[0]


def output_is_complete(job: dict[str, str]) -> bool:
    output_dir = Path(job["output_dir"])
    complete_path = output_dir / "evaluation_complete.json"
    if not complete_path.is_file():
        return False
    try:
        complete = json.loads(complete_path.read_text())
        if complete["evaluation_id"] != job["evaluation_id"]:
            return False
        for label, record in complete["outputs"].items():
            path = output_dir / record["name"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                return False
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def numeric_rows(path: Path, expected_columns: int) -> np.ndarray:
    values = np.loadtxt(path, comments="#", dtype=np.float64)
    values = np.atleast_2d(values)
    if values.shape[1] != expected_columns:
        raise ValueError(
            f"{path} has {values.shape[1]} numeric columns; expected {expected_columns}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite detail value in {path}")
    return values


def header_paths(path: Path) -> list[str]:
    headers: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                headers.append(str(Path(line[1:].split(":", 1)[0].strip()).resolve()))
    return headers


def load_boxes(system_path: Path) -> np.ndarray:
    pieces = []
    for set_dir in sorted(system_path.glob("set.*")):
        path = set_dir / "box.npy"
        if path.is_file():
            piece = np.asarray(np.load(path), dtype=np.float64).reshape(-1, 9)
            pieces.append(piece)
    if not pieces:
        raise FileNotFoundError(f"No box.npy arrays under {system_path}")
    return np.concatenate(pieces, axis=0)


def summary(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(flat):
        return {
            "count": 0,
            "sum_sq": math.nan,
            "sum_abs": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
        }
    sum_sq = float(np.dot(flat, flat))
    sum_abs = float(np.abs(flat).sum())
    return {
        "count": int(flat.size),
        "sum_sq": sum_sq,
        "sum_abs": sum_abs,
        "rmse": math.sqrt(sum_sq / flat.size),
        "mae": sum_abs / flat.size,
    }


def bin_label(index: int) -> str:
    lower = FORCE_BIN_EDGES[index]
    upper = FORCE_BIN_EDGES[index + 1]
    if math.isinf(float(upper)):
        return f"[{lower:g}, inf)"
    return f"[{lower:g}, {upper:g})"


def parse_detail_outputs(
    detail_prefix: Path,
    system_map_path: Path,
    job: dict[str, str],
    output_dir: Path,
) -> dict[str, dict[str, str]]:
    systems = read_csv(system_map_path)
    expected_paths = [str(Path(row["path"]).resolve()) for row in systems]
    energy_path = Path(f"{detail_prefix}.e.out")
    force_path = Path(f"{detail_prefix}.f.out")
    virial_path = Path(f"{detail_prefix}.v.out")
    for path in (energy_path, force_path, virial_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing DeePMD detail output: {path}")
        observed_headers = header_paths(path)
        if observed_headers != expected_paths:
            raise ValueError(
                f"System header order mismatch in {path}: expected {expected_paths}, "
                f"observed {observed_headers}"
            )

    energy_all = numeric_rows(energy_path, 2)
    force_all = numeric_rows(force_path, 6)
    virial_all = numeric_rows(virial_path, 18)
    expected_frames = sum(int(row["frames"]) for row in systems)
    expected_force_rows = sum(int(row["frames"]) * int(row["natoms"]) for row in systems)
    if energy_all.shape[0] != expected_frames or virial_all.shape[0] != expected_frames:
        raise ValueError(
            f"Frame-row mismatch: expected {expected_frames}, energy={energy_all.shape[0]}, "
            f"virial={virial_all.shape[0]}"
        )
    if force_all.shape[0] != expected_force_rows:
        raise ValueError(
            f"Force-row mismatch: expected {expected_force_rows}, found {force_all.shape[0]}"
        )

    base_fields = {
        "evaluation_id": job["evaluation_id"],
        "run_id": job["run_id"],
        "architecture": job["architecture"],
        "training_fold": job["training_fold"],
        "regime": job["regime"],
        "checkpoint_variant": job["checkpoint_variant"],
        "evaluation_variant": job["evaluation_variant"],
    }
    frame_rows: list[dict] = []
    system_rows: list[dict] = []
    force_bin_rows: list[dict] = []
    frame_offset = 0
    force_offset = 0
    for system in systems:
        frames = int(system["frames"])
        natoms = int(system["natoms"])
        frame_slice = slice(frame_offset, frame_offset + frames)
        force_slice = slice(force_offset, force_offset + frames * natoms)
        energies = energy_all[frame_slice]
        virials = virial_all[frame_slice]
        forces = force_all[force_slice].reshape(frames, natoms, 6)
        frame_offset += frames
        force_offset += frames * natoms

        boxes = load_boxes(Path(system["path"]))
        if boxes.shape[0] != frames:
            raise ValueError(
                f"box.npy frame mismatch for {system['system_id']}: {boxes.shape[0]} != {frames}"
            )
        volumes = np.abs(np.linalg.det(boxes.reshape(-1, 3, 3)))
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
            raise ValueError(f"Invalid cell volume in {system['system_id']}")

        energy_error = (energies[:, 1] - energies[:, 0]) / natoms * 1000.0
        force_reference = forces[:, :, :3]
        force_error = forces[:, :, 3:] - force_reference
        force_vector_error = np.linalg.norm(force_error, axis=2)
        reference_force_magnitude = np.linalg.norm(force_reference, axis=2)
        virial_error = (virials[:, 9:] - virials[:, :9]) / natoms * 1000.0
        stress_error = (
            (virials[:, 9:] - virials[:, :9])
            / volumes[:, np.newaxis]
            * EV_A3_TO_GPA
        )

        e_summary = summary(energy_error)
        fc_summary = summary(force_error)
        fv_summary = summary(force_vector_error)
        v_summary = summary(virial_error)
        s_summary = summary(stress_error)
        metadata = {
            "system_id": system["system_id"],
            "evaluation_role": system["evaluation_role"],
            "evaluation_fold": system["evaluation_fold"],
            "family": system["family"],
            "parent_family": system["parent_family"],
            "structure": system["structure"],
            "hnf_group": system["hnf_group"],
            "element": system["element"],
        }
        system_rows.append(
            {
                **base_fields,
                **metadata,
                "frames": frames,
                "natoms": natoms,
                "energy_count": e_summary["count"],
                "energy_sum_sq_mev2_atom2": e_summary["sum_sq"],
                "energy_sum_abs_mev_atom": e_summary["sum_abs"],
                "energy_rmse_mev_atom": e_summary["rmse"],
                "energy_mae_mev_atom": e_summary["mae"],
                "force_component_count": fc_summary["count"],
                "force_component_sum_sq_ev2_a2": fc_summary["sum_sq"],
                "force_component_sum_abs_ev_a": fc_summary["sum_abs"],
                "force_component_rmse_ev_a": fc_summary["rmse"],
                "force_component_mae_ev_a": fc_summary["mae"],
                "force_vector_count": fv_summary["count"],
                "force_vector_sum_sq_ev2_a2": fv_summary["sum_sq"],
                "force_vector_sum_abs_ev_a": fv_summary["sum_abs"],
                "force_vector_rmse_ev_a": fv_summary["rmse"],
                "force_vector_mae_ev_a": fv_summary["mae"],
                "virial_component_count": v_summary["count"],
                "virial_component_sum_sq_mev2_atom2": v_summary["sum_sq"],
                "virial_component_sum_abs_mev_atom": v_summary["sum_abs"],
                "virial_component_rmse_mev_atom": v_summary["rmse"],
                "virial_component_mae_mev_atom": v_summary["mae"],
                "stress_component_count": s_summary["count"],
                "stress_component_sum_sq_gpa2": s_summary["sum_sq"],
                "stress_component_sum_abs_gpa": s_summary["sum_abs"],
                "stress_component_rmse_gpa": s_summary["rmse"],
                "stress_component_mae_gpa": s_summary["mae"],
            }
        )

        for frame_index in range(frames):
            fc_frame = summary(force_error[frame_index])
            fv_frame = summary(force_vector_error[frame_index])
            v_frame = summary(virial_error[frame_index])
            s_frame = summary(stress_error[frame_index])
            frame_rows.append(
                {
                    **base_fields,
                    **metadata,
                    "frame_index": frame_index,
                    "natoms": natoms,
                    "volume_angstrom3": volumes[frame_index],
                    "energy_error_mev_atom": energy_error[frame_index],
                    "energy_abs_error_mev_atom": abs(energy_error[frame_index]),
                    "force_component_count": fc_frame["count"],
                    "force_component_sum_sq_ev2_a2": fc_frame["sum_sq"],
                    "force_component_sum_abs_ev_a": fc_frame["sum_abs"],
                    "force_component_rmse_ev_a": fc_frame["rmse"],
                    "force_component_mae_ev_a": fc_frame["mae"],
                    "force_vector_count": fv_frame["count"],
                    "force_vector_sum_sq_ev2_a2": fv_frame["sum_sq"],
                    "force_vector_sum_abs_ev_a": fv_frame["sum_abs"],
                    "force_vector_rmse_ev_a": fv_frame["rmse"],
                    "force_vector_mae_ev_a": fv_frame["mae"],
                    "virial_component_count": v_frame["count"],
                    "virial_component_sum_sq_mev2_atom2": v_frame["sum_sq"],
                    "virial_component_sum_abs_mev_atom": v_frame["sum_abs"],
                    "virial_component_rmse_mev_atom": v_frame["rmse"],
                    "virial_component_mae_mev_atom": v_frame["mae"],
                    "stress_component_count": s_frame["count"],
                    "stress_component_sum_sq_gpa2": s_frame["sum_sq"],
                    "stress_component_sum_abs_gpa": s_frame["sum_abs"],
                    "stress_component_rmse_gpa": s_frame["rmse"],
                    "stress_component_mae_gpa": s_frame["mae"],
                    "reference_force_mean_ev_a": float(
                        reference_force_magnitude[frame_index].mean()
                    ),
                    "reference_force_max_ev_a": float(
                        reference_force_magnitude[frame_index].max()
                    ),
                }
            )

        bin_indices = np.digitize(
            reference_force_magnitude,
            FORCE_BIN_EDGES[1:-1],
            right=False,
        )
        for index in range(len(FORCE_BIN_EDGES) - 1):
            mask = bin_indices == index
            selected_components = force_error[mask]
            selected_vectors = force_vector_error[mask]
            selected_reference = reference_force_magnitude[mask]
            fc_bin = summary(selected_components)
            fv_bin = summary(selected_vectors)
            force_bin_rows.append(
                {
                    **base_fields,
                    **metadata,
                    "force_bin": bin_label(index),
                    "force_bin_lower_ev_a": FORCE_BIN_EDGES[index],
                    "force_bin_upper_ev_a": (
                        "inf"
                        if math.isinf(float(FORCE_BIN_EDGES[index + 1]))
                        else FORCE_BIN_EDGES[index + 1]
                    ),
                    "atom_count": int(mask.sum()),
                    "component_count": fc_bin["count"],
                    "component_sum_sq_ev2_a2": fc_bin["sum_sq"],
                    "component_sum_abs_ev_a": fc_bin["sum_abs"],
                    "component_rmse_ev_a": fc_bin["rmse"],
                    "component_mae_ev_a": fc_bin["mae"],
                    "vector_count": fv_bin["count"],
                    "vector_sum_sq_ev2_a2": fv_bin["sum_sq"],
                    "vector_sum_abs_ev_a": fv_bin["sum_abs"],
                    "vector_rmse_ev_a": fv_bin["rmse"],
                    "vector_mae_ev_a": fv_bin["mae"],
                    "reference_force_mean_ev_a": (
                        float(selected_reference.mean())
                        if selected_reference.size
                        else math.nan
                    ),
                }
            )

    frame_path = output_dir / "frame_metrics.csv.gz"
    system_path = output_dir / "system_metrics.csv"
    bin_path = output_dir / "force_magnitude_bin_metrics.csv"
    atomic_gzip_csv(frame_path, frame_rows, list(frame_rows[0]))
    atomic_csv(system_path, system_rows, list(system_rows[0]))
    atomic_csv(bin_path, force_bin_rows, list(force_bin_rows[0]))
    return {
        "frame_metrics": {
            "name": frame_path.name,
            "sha256": sha256(frame_path),
            "rows": str(len(frame_rows)),
        },
        "system_metrics": {
            "name": system_path.name,
            "sha256": sha256(system_path),
            "rows": str(len(system_rows)),
        },
        "force_magnitude_bin_metrics": {
            "name": bin_path.name,
            "sha256": sha256(bin_path),
            "rows": str(len(force_bin_rows)),
        },
    }


def remove_detail_outputs(prefix: Path) -> list[str]:
    removed = []
    for suffix in DETAIL_SUFFIXES:
        path = Path(f"{prefix}{suffix}")
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def verify_training_complete(job: dict[str, str]) -> None:
    run_dir = TRAINING / "runs" / job["run_id"]
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        raise RuntimeError(f"Training status is absent for {job['run_id']}")
    status = json.loads(status_path.read_text())
    if status.get("state") != "completed" or status.get("return_code") != 0:
        raise RuntimeError(f"Training is not successfully complete for {job['run_id']}: {status}")
    input_path = run_dir / "input.json"
    if sha256(input_path) != job["training_input_sha256"]:
        raise RuntimeError(f"Training input hash changed for {job['run_id']}")
    checkpoint = Path(job["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected final checkpoint is absent: {checkpoint}")


def run_job(job: dict[str, str], dp_executable: Path, keep_raw: bool) -> int:
    verify_training_complete(job)
    if output_is_complete(job):
        print(f"SKIP complete {job['evaluation_id']}", flush=True)
        return 0
    visible = [value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value]
    if len(visible) != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, found {visible}")
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "evaluation_status.json"
    detail_prefix = output_dir / "dp_detail"
    remove_detail_outputs(detail_prefix)
    command = [
        str(dp_executable),
        "--pt",
        "test",
        "-m",
        job["checkpoint_path"],
        "-f",
        job["systems_file"],
        "-n",
        "0",
        "-d",
        str(detail_prefix),
    ]
    version_result = subprocess.run(
        [str(dp_executable), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if version_result.returncode:
        raise RuntimeError(f"Unable to query DeePMD version: {version_result.stdout}")
    status = {
        "state": "running",
        "evaluation_id": job["evaluation_id"],
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "cuda_visible_devices": visible,
        "command": command,
    }
    atomic_json(status_path, status)
    log_path = output_dir / "dp_test.log"
    start = time.monotonic()
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            if "testing system" in line or "number of test data" in line:
                print(f"{job['evaluation_id']}: {line.strip()}", flush=True)
        return_code = process.wait()
    elapsed = time.monotonic() - start
    if return_code:
        status.update(
            {
                "state": "failed",
                "return_code": return_code,
                "finished_utc": utc_now(),
                "elapsed_seconds": elapsed,
            }
        )
        atomic_json(status_path, status)
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"dp test failed for {job['evaluation_id']}\n{tail}")

    outputs = parse_detail_outputs(
        detail_prefix,
        Path(job["system_map"]),
        job,
        output_dir,
    )
    removed = [] if keep_raw else remove_detail_outputs(detail_prefix)
    provenance = {
        "schema_version": 1,
        "evaluation_id": job["evaluation_id"],
        "completed_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "deePMD_version": version_result.stdout.strip(),
        "command": command,
        "job": job,
        "hashes": {
            "checkpoint": sha256(Path(job["checkpoint_path"])),
            "training_input": sha256(TRAINING / "runs" / job["run_id"] / "input.json"),
            "system_map": sha256(Path(job["system_map"])),
            "systems_file": sha256(Path(job["systems_file"])),
            "reducer": sha256(Path(__file__)),
            "dp_test_log": sha256(log_path),
        },
        "outputs": outputs,
        "raw_detail_files_removed": removed,
        "raw_detail_files_retained": bool(keep_raw),
        "force_magnitude_bins_eV_per_angstrom": [0.0, 0.5, 1.0, 2.0, 4.0, "inf"],
        "stress_conversion_GPa_per_eV_per_angstrom3": EV_A3_TO_GPA,
    }
    atomic_json(output_dir / "evaluation_complete.json", provenance)
    status.update(
        {
            "state": "completed",
            "return_code": 0,
            "finished_utc": provenance["completed_utc"],
            "elapsed_seconds": elapsed,
        }
    )
    atomic_json(status_path, status)
    print(f"COMPLETE {job['evaluation_id']} in {elapsed / 60:.1f} min", flush=True)
    return 0


def print_plan(evaluation_id: str | None) -> int:
    rows = read_csv(MANIFEST)
    if evaluation_id:
        rows = [row for row in rows if row["evaluation_id"] == evaluation_id]
    print(
        json.dumps(
            [
                {
                    "evaluation_id": row["evaluation_id"],
                    "checkpoint_exists": Path(row["checkpoint_path"]).is_file(),
                    "training_status": (
                        json.loads(
                            (TRAINING / "runs" / row["run_id"] / "run_status.json").read_text()
                        ).get("state")
                        if (TRAINING / "runs" / row["run_id"] / "run_status.json").is_file()
                        else "absent"
                    ),
                    "systems": int(row["system_count"]),
                    "frames": int(row["frame_count"]),
                    "complete": output_is_complete(row),
                }
                for row in rows
            ],
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--evaluation-id")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--evaluation-id", required=True)
    run_parser.add_argument("--dp-executable", type=Path, default=Path("dp"))
    run_parser.add_argument("--keep-raw", action="store_true")
    parse_parser = subparsers.add_parser("parse")
    parse_parser.add_argument("--evaluation-id", required=True)
    parse_parser.add_argument("--detail-prefix", type=Path, required=True)
    parse_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "plan":
        return print_plan(args.evaluation_id)
    job = manifest_job(args.evaluation_id)
    if args.command == "run":
        return run_job(job, args.dp_executable, args.keep_raw)
    if args.command == "parse":
        outputs = parse_detail_outputs(
            args.detail_prefix,
            Path(job["system_map"]),
            job,
            args.output_dir,
        )
        print(json.dumps(outputs, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
