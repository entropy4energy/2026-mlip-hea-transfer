#!/usr/bin/env python3
"""Render blue-marked SI tables from validated H100 tensile CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
BLUE_START = "[[BLUE]]"
BLUE_END = "[[/BLUE]]"
ENDPOINT_LABELS = {
    "tangent_modulus_GPa": "Tangent modulus (GPa)",
    "yield_0p2_GPa": "0.2% offset yield stress (GPa)",
    "uts_GPa": "Maximum stress (GPa)",
    "strain_at_uts": "Strain at maximum",
    "work_to_20pct_GJ_m3": "Work to 20% (GJ m⁻³)",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        output = list(csv.DictReader(handle))
    if not output:
        raise ValueError(f"{path}: empty table")
    return output


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def blue(value: object) -> str:
    text = str(value).replace("|", "\\|")
    return f"{BLUE_START}{text}{BLUE_END}"


def markdown_table(headers: list[str], values: list[list[object]]) -> str:
    output = [
        "| " + " | ".join(blue(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend(
        "| " + " | ".join(blue(value) for value in row) + " |" for row in values
    )
    return "\n".join(output)


def number(value: str, digits: int = 4) -> str:
    if value == "":
        return "unresolved"
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value {value!r}")
    return f"{result:.{digits}f}"


def interval(row: dict[str, str], endpoint: str, digits: int = 3) -> str:
    if row[f"{endpoint}_status"] != "estimated":
        return "unresolved"
    return (
        f"{number(row[f'{endpoint}_mean'], digits)} "
        f"[{number(row[f'{endpoint}_ci95_low'], digits)}, "
        f"{number(row[f'{endpoint}_ci95_high'], digits)}]"
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replicas",
        type=Path,
        default=PROJECT / "tables" / "lammps_h100_tension_replica_endpoints.csv",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=PROJECT / "tables" / "lammps_h100_tension_group_summaries.csv",
    )
    parser.add_argument(
        "--contrasts",
        type=Path,
        default=PROJECT / "tables" / "lammps_h100_tension_paired_contrasts.csv",
    )
    parser.add_argument(
        "--promotion-validation",
        type=Path,
        default=PROJECT / "work" / "lammps" / "h100_materials_science" / "lammps_h100_tension_promotion_validation.json",
    )
    parser.add_argument(
        "--snapshot-receipt",
        type=Path,
        default=PROJECT / "work" / "lammps" / "h100_materials_science" / "results_snapshot" / "snapshot_receipt.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "work" / "lammps" / "h100_materials_science" / "h100_si_generated_tables.md",
    )
    args = parser.parse_args()
    validation = load_json(args.promotion_validation.resolve())
    snapshot = load_json(args.snapshot_receipt.resolve())
    if validation.get("status") != "passed" or validation.get("errors") not in ([], None):
        raise ValueError("promotion validation did not pass cleanly")
    if snapshot.get("status") != "passed" or snapshot.get("runs_included") != 18:
        raise ValueError("snapshot receipt is not a passed 18-run population")

    replicas = rows(args.replicas.resolve())
    groups = rows(args.groups.resolve())
    contrasts = rows(args.contrasts.resolve())
    if (len(replicas), len(groups), len(contrasts)) != (18, 6, 45):
        raise ValueError("unexpected H100 SI table populations")

    replica_values = []
    for row in sorted(replicas, key=lambda item: int(item["task_id"])):
        replica_values.append(
            [
                f"[{row['orientation']}]",
                row["temperature_K"],
                row["chemical_seed"],
                row["velocity_seed"],
                number(row["tangent_modulus_GPa"], 4),
                number(row["yield_0p2_GPa"], 4),
                number(row["uts_GPa"], 4),
                number(row["strain_at_uts"], 5),
                number(row["work_to_20pct_GJ_m3"], 5),
            ]
        )
    group_values = []
    for row in sorted(groups, key=lambda item: (int(item["temperature_K"]), item["orientation"])):
        group_values.append(
            [
                f"[{row['orientation']}]",
                row["temperature_K"],
                row["n_realization_pairs"],
                interval(row, "tangent_modulus_GPa"),
                interval(row, "yield_0p2_GPa"),
                interval(row, "uts_GPa"),
                interval(row, "strain_at_uts", 4),
                interval(row, "work_to_20pct_GJ_m3", 4),
            ]
        )
    contrast_values = []
    for row in contrasts:
        context = (
            f"[{row['fixed_orientation']}]"
            if row["contrast_family"] == "temperature"
            else f"{row['fixed_temperature_K']} K"
        )
        if row["status"] == "estimated":
            summary = (
                f"{number(row['mean_difference_b_minus_a'], 4)} "
                f"[{number(row['ci95_low'], 4)}, {number(row['ci95_high'], 4)}]"
            )
        else:
            summary = "unresolved"
        contrast_values.append(
            [
                row["contrast_family"],
                context,
                row["difference_definition"],
                ENDPOINT_LABELS[row["endpoint"]],
                row["n_pairs_eligible"],
                summary,
            ]
        )

    parts = [
        f"## {BLUE_START}S13.2 Replica-level endpoints{BLUE_END}",
        "",
        (
            f"{BLUE_START}Table S13.1 retains every fixed chemical/velocity realization. "
            "Values shown here are rounded for reading; the unrounded source is "
            f"`tables/lammps_h100_tension_replica_endpoints.csv` (SHA-256 "
            f"`{digest(args.replicas.resolve())}`).{BLUE_END}"
        ),
        "",
        markdown_table(
            [
                "Direction",
                "T (K)",
                "Chemical seed",
                "Velocity seed",
                "Tangent modulus (GPa)",
                "0.2% offset yield (GPa)",
                "Maximum stress (GPa)",
                "Strain at maximum",
                "Work to 20% (GJ m⁻³)",
            ],
            replica_values,
        ),
        "",
        f"## {BLUE_START}S13.3 Orientation--temperature summaries{BLUE_END}",
        "",
        (
            f"{BLUE_START}Table S13.2 reports the mean and two-sided 95% Student-*t* "
            "interval in brackets across the three realization pairs. The exact source is "
            f"`tables/lammps_h100_tension_group_summaries.csv` (SHA-256 "
            f"`{digest(args.groups.resolve())}`).{BLUE_END}"
        ),
        "",
        markdown_table(
            [
                "Direction",
                "T (K)",
                "n",
                "Tangent modulus, mean [95% CI]",
                "0.2% offset yield, mean [95% CI]",
                "Maximum stress, mean [95% CI]",
                "Strain at maximum, mean [95% CI]",
                "Work to 20%, mean [95% CI]",
            ],
            group_values,
        ),
        "",
        f"## {BLUE_START}S13.4 Fixed paired contrasts{BLUE_END}",
        "",
        (
            f"{BLUE_START}Table S13.3 contains all predeclared temperature and loading-"
            "direction contrasts. Differences follow the displayed B-minus-A definition; "
            "intervals are pointwise, unadjusted and based on three paired values. The "
            "unrounded source and the 135 contributing paired values are in "
            f"`tables/lammps_h100_tension_paired_contrasts.csv` (SHA-256 "
            f"`{digest(args.contrasts.resolve())}`).{BLUE_END}"
        ),
        "",
        markdown_table(
            [
                "Family",
                "Fixed context",
                "Difference",
                "Endpoint",
                "Eligible pairs",
                "Mean difference [95% CI]",
            ],
            contrast_values,
        ),
        "",
    ]
    atomic_text(args.output.resolve(), "\n".join(parts))
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output.resolve()),
                "replica_rows": len(replicas),
                "group_rows": len(groups),
                "contrast_rows": len(contrasts),
                "blue_spans": "\n".join(parts).count(BLUE_START),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
