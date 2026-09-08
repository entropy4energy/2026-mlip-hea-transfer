#!/usr/bin/env python3
"""Fit a third-order Birch-Murnaghan equation of state to LAMMPS points."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

EV_A3_TO_GPA = 160.21766208


def birch_murnaghan_energy(volume, e0, v0, b0, b0_prime):
    eta = (v0 / volume) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * (
        (eta - 1.0) ** 3 * b0_prime
        + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def birch_murnaghan_pressure(volume, v0, b0, b0_prime):
    ratio = v0 / volume
    eta = ratio ** (2.0 / 3.0)
    return (
        1.5
        * b0
        * (ratio ** (7.0 / 3.0) - ratio ** (5.0 / 3.0))
        * (1.0 + 0.75 * (b0_prime - 4.0) * (eta - 1.0))
    )


def read_points(root: Path) -> list[dict[str, str]]:
    points: list[dict[str, str]] = []
    for path in sorted(root.rglob("eos_point.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"expected one EOS row in {path}, found {len(rows)}")
        row = rows[0]
        row["source"] = str(path)
        points.append(row)
    if len(points) < 5:
        raise ValueError(f"need at least five EOS points below {root}, found {len(points)}")
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    points = read_points(args.input)
    architectures = {row["architecture"] for row in points}
    if len(architectures) != 1:
        raise ValueError(f"mixed architectures in EOS points: {sorted(architectures)}")

    volumes = np.asarray([float(row["volume_A3_per_atom"]) for row in points])
    energies = np.asarray([float(row["energy_eV_per_atom"]) for row in points])
    pressures = np.asarray([float(row["pressure_GPa"]) for row in points])
    final_forces = np.asarray([float(row["max_force_component_eV_A"]) for row in points])
    order = np.argsort(volumes)
    volumes, energies, pressures, final_forces = (
        volumes[order],
        energies[order],
        pressures[order],
        final_forces[order],
    )
    points = [points[index] for index in order]
    if np.unique(np.round(volumes, 10)).size != volumes.size:
        raise ValueError("duplicate EOS volumes detected")

    imin = int(np.argmin(energies))
    v_guess = float(volumes[imin])
    e_guess = float(energies[imin])
    quadratic = np.polyfit(volumes - v_guess, energies - e_guess, 2)
    b_guess = float(np.clip(2.0 * quadratic[0] * v_guess, 0.05, 3.0))
    span = max(float(np.ptp(energies)), 1.0e-3)
    lower = [float(energies.min() - span), float(volumes.min()), 1.0e-6, 0.0]
    upper = [float(energies.max() + span), float(volumes.max()), 10.0, 12.0]
    params, covariance = curve_fit(
        birch_murnaghan_energy,
        volumes,
        energies,
        p0=[e_guess, v_guess, b_guess, 4.0],
        bounds=(lower, upper),
        maxfev=100000,
    )
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    e0, v0, b0, b0_prime = (float(value) for value in params)
    e_fit = birch_murnaghan_energy(volumes, *params)
    p_fit = birch_murnaghan_pressure(volumes, v0, b0, b0_prime) * EV_A3_TO_GPA

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (output / "eos_points_with_fit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scale",
                "volume_A3_per_atom",
                "energy_eV_per_atom",
                "energy_fit_eV_per_atom",
                "pressure_GPa",
                "pressure_fit_GPa",
                "source",
            ]
        )
        for row, volume, energy, fit_energy, pressure, fit_pressure in zip(
            points, volumes, energies, e_fit, pressures, p_fit, strict=True
        ):
            writer.writerow(
                [
                    row["scale"],
                    f"{volume:.12g}",
                    f"{energy:.12g}",
                    f"{fit_energy:.12g}",
                    f"{pressure:.12g}",
                    f"{fit_pressure:.12g}",
                    row["source"],
                ]
            )

    result = {
        "architecture": next(iter(architectures)),
        "equation": "third-order Birch-Murnaghan energy fit",
        "n_points": len(points),
        "equilibrium_energy_eV_per_atom": e0,
        "equilibrium_energy_standard_error_eV_per_atom": float(errors[0]),
        "equilibrium_volume_A3_per_atom": v0,
        "equilibrium_volume_standard_error_A3_per_atom": float(errors[1]),
        "bulk_modulus_eV_per_A3": b0,
        "bulk_modulus_GPa": b0 * EV_A3_TO_GPA,
        "bulk_modulus_standard_error_GPa": float(errors[2] * EV_A3_TO_GPA),
        "bulk_modulus_pressure_derivative": b0_prime,
        "bulk_modulus_pressure_derivative_standard_error": float(errors[3]),
        "energy_rmse_meV_per_atom": float(np.sqrt(np.mean((e_fit - energies) ** 2)) * 1000.0),
        "pressure_consistency_rmse_GPa": float(np.sqrt(np.mean((p_fit - pressures) ** 2))),
        "maximum_residual_force_component_eV_per_A": float(final_forces.max()),
        "fit_volume_bracket_A3_per_atom": [float(volumes.min()), float(volumes.max())],
    }
    (output / "eos_fit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
