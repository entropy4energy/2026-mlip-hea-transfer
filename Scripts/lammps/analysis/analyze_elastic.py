#!/usr/bin/env python3
"""Reduce central finite-strain stresses to a symmetrized elastic tensor."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")
STRESS_COLUMNS = tuple(f"s{name}_GPa" for name in VOIGT)


def read_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("elastic_point.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            found = list(csv.DictReader(handle))
        if len(found) != 1:
            raise ValueError(f"expected one elastic row in {path}, found {len(found)}")
        found[0]["source"] = str(path)
        rows.extend(found)
    if not rows:
        raise ValueError(f"no elastic_point.csv files found below {root}")
    return rows


def linear_tensor(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = np.zeros((6, 6), dtype=float)
    intercepts = np.zeros((6, 6), dtype=float)
    r2 = np.zeros((6, 6), dtype=float)
    by_mode: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_mode[int(row["mode"])].append(row)
    if set(by_mode) != set(range(1, 7)):
        raise ValueError(f"elastic modes must be 1..6; found {sorted(by_mode)}")
    for mode in range(1, 7):
        mode_rows = by_mode[mode]
        strain = np.asarray([float(row["strain"]) for row in mode_rows])
        if strain.size < 2 or strain.min() >= 0.0 or strain.max() <= 0.0:
            raise ValueError(f"mode {mode} needs both positive and negative strain")
        for response, column in enumerate(STRESS_COLUMNS):
            stress = np.asarray([float(row[column]) for row in mode_rows])
            slope, intercept = np.polyfit(strain, stress, 1)
            predicted = slope * strain + intercept
            residual = float(np.sum((stress - predicted) ** 2))
            total = float(np.sum((stress - stress.mean()) ** 2))
            tensor[response, mode - 1] = slope
            intercepts[response, mode - 1] = intercept
            r2[response, mode - 1] = 1.0 - residual / total if total > 0.0 else 1.0
    return tensor, intercepts, r2


def central_tensor(rows: list[dict[str, str]], magnitude: float) -> np.ndarray:
    tensor = np.zeros((6, 6), dtype=float)
    tolerance = max(1.0e-12, magnitude * 1.0e-8)
    for mode in range(1, 7):
        positive = [
            row
            for row in rows
            if int(row["mode"]) == mode and abs(float(row["strain"]) - magnitude) < tolerance
        ]
        negative = [
            row
            for row in rows
            if int(row["mode"]) == mode and abs(float(row["strain"]) + magnitude) < tolerance
        ]
        if len(positive) != 1 or len(negative) != 1:
            raise ValueError(f"need one +/-{magnitude:g} pair for elastic mode {mode}")
        for response, column in enumerate(STRESS_COLUMNS):
            tensor[response, mode - 1] = (
                float(positive[0][column]) - float(negative[0][column])
            ) / (2.0 * magnitude)
    return tensor


def write_tensor(path: Path, tensor: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stress_response/strain_mode", *VOIGT])
        for label, row in zip(VOIGT, tensor, strict=True):
            writer.writerow([label, *(f"{value:.12g}" for value in row)])


def moduli(tensor: np.ndarray) -> dict[str, float | list[float] | bool]:
    c = tensor
    s = np.linalg.inv(c)
    bv = (c[0, 0] + c[1, 1] + c[2, 2] + 2.0 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9.0
    gv = (
        c[0, 0]
        + c[1, 1]
        + c[2, 2]
        - c[0, 1]
        - c[0, 2]
        - c[1, 2]
        + 3.0 * (c[3, 3] + c[4, 4] + c[5, 5])
    ) / 15.0
    br = 1.0 / (s[0, 0] + s[1, 1] + s[2, 2] + 2.0 * (s[0, 1] + s[0, 2] + s[1, 2]))
    gr = 15.0 / (
        4.0 * (s[0, 0] + s[1, 1] + s[2, 2])
        - 4.0 * (s[0, 1] + s[0, 2] + s[1, 2])
        + 3.0 * (s[3, 3] + s[4, 4] + s[5, 5])
    )
    bh = 0.5 * (bv + br)
    gh = 0.5 * (gv + gr)
    young = 9.0 * bh * gh / (3.0 * bh + gh)
    poisson = (3.0 * bh - 2.0 * gh) / (2.0 * (3.0 * bh + gh))
    c11 = float(np.mean(np.diag(c)[:3]))
    c12 = float(np.mean([c[0, 1], c[0, 2], c[1, 2]]))
    c44 = float(np.mean(np.diag(c)[3:]))
    eigenvalues = np.linalg.eigvalsh(c)
    return {
        "voigt_bulk_modulus_GPa": float(bv),
        "reuss_bulk_modulus_GPa": float(br),
        "hill_bulk_modulus_GPa": float(bh),
        "voigt_shear_modulus_GPa": float(gv),
        "reuss_shear_modulus_GPa": float(gr),
        "hill_shear_modulus_GPa": float(gh),
        "hill_young_modulus_GPa": float(young),
        "hill_poisson_ratio": float(poisson),
        "pugh_B_over_G": float(bh / gh),
        "universal_anisotropy_index": float(5.0 * gv / gr + bv / br - 6.0),
        "cubic_projection_C11_GPa": c11,
        "cubic_projection_C12_GPa": c12,
        "cubic_projection_C44_GPa": c44,
        "cubic_projection_cauchy_pressure_GPa": c12 - c44,
        "elastic_eigenvalues_GPa": [float(value) for value in eigenvalues],
        "positive_definite": bool(np.all(eigenvalues > 0.0)),
        "cubic_born_stable": bool(c11 - c12 > 0.0 and c11 + 2.0 * c12 > 0.0 and c44 > 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.input)
    architectures = {row["architecture"] for row in rows}
    if len(architectures) != 1:
        raise ValueError(f"mixed architectures in elastic points: {sorted(architectures)}")

    raw, intercepts, r2 = linear_tensor(rows)
    symmetric = 0.5 * (raw + raw.T)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_tensor(output / "elastic_tensor_raw_GPa.csv", raw)
    write_tensor(output / "elastic_tensor_symmetric_GPa.csv", symmetric)
    write_tensor(output / "fit_intercepts_GPa.csv", intercepts)
    write_tensor(output / "fit_r_squared.csv", r2)

    magnitudes = sorted({abs(float(row["strain"])) for row in rows if float(row["strain"]) != 0.0})
    central_results: dict[str, np.ndarray] = {}
    with (output / "elastic_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strain_magnitude", "frobenius_norm_GPa", "relative_change_from_smallest"])
        reference = None
        for magnitude in magnitudes:
            central_raw = central_tensor(rows, magnitude)
            central = 0.5 * (central_raw + central_raw.T)
            central_results[f"{magnitude:.10g}"] = central
            tag = f"{magnitude:.10g}".replace(".", "p")
            write_tensor(output / f"elastic_tensor_symmetric_strain_{tag}_GPa.csv", central)
            if reference is None:
                reference = central
            relative = np.linalg.norm(central - reference) / max(np.linalg.norm(reference), 1.0e-30)
            writer.writerow([f"{magnitude:.12g}", f"{np.linalg.norm(central):.12g}", f"{relative:.12g}"])

    asymmetry = np.linalg.norm(raw - raw.T) / max(np.linalg.norm(symmetric), 1.0e-30)
    result = {
        "architecture": next(iter(architectures)),
        "voigt_order": list(VOIGT),
        "strain_magnitudes": magnitudes,
        "fit_min_r_squared": float(r2.min()),
        "raw_tensor_antisymmetry_relative_frobenius": float(asymmetry),
        "maximum_residual_force_component_eV_per_A": max(
            float(row["max_force_component_eV_A"]) for row in rows
        ),
        **moduli(symmetric),
    }
    if len(magnitudes) > 1:
        smallest = central_results[f"{magnitudes[0]:.10g}"]
        largest = central_results[f"{magnitudes[-1]:.10g}"]
        result["largest_vs_smallest_strain_relative_tensor_change"] = float(
            np.linalg.norm(largest - smallest) / max(np.linalg.norm(smallest), 1.0e-30)
        )
    (output / "elastic_properties.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
