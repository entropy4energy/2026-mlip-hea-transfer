#!/usr/bin/env bash
# Rebuild and independently validate all A21 Figure 7 data products.

set -Eeuo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_EXE="${PYTHON_EXE:-python3}"
cd -- "$PROJECT"

"$PYTHON_EXE" work/lammps/build_a21_figure7_tables.py \
    --a21-plan LAMMPS/figure7_recovery_20260901/plans/saiph_l40s_a21.tsv \
    --a19-validation work/lammps/lammps_a19_figure7_tables.validation.json \
    --a19-endpoints tables/lammps_a19_run_endpoints.csv \
    --a11-replicas tables/lammps_a5_npt_replicas_a11.csv \
    --a12-replicas tables/lammps_a12_npt_replicas.csv \
    --run-endpoints tables/lammps_a21_run_endpoints.csv \
    --groups tables/lammps_a21_four_temperature_groups.csv \
    --panel-groups tables/lammps_a21_panel_groups.csv \
    --response tables/lammps_a21_four_temperature_response.csv \
    --excluded-curve-audit tables/lammps_a21_excluded_dpa4_fcc_curve_audit.csv \
    --receipt work/lammps/lammps_a21_figure7_tables.receipt.json

"$PYTHON_EXE" work/lammps/validate_a21_figure7_tables.py \
    --a21-plan LAMMPS/figure7_recovery_20260901/plans/saiph_l40s_a21.tsv \
    --a19-validation work/lammps/lammps_a19_figure7_tables.validation.json \
    --a19-endpoints tables/lammps_a19_run_endpoints.csv \
    --a11-replicas tables/lammps_a5_npt_replicas_a11.csv \
    --a12-replicas tables/lammps_a12_npt_replicas.csv \
    --run-endpoints tables/lammps_a21_run_endpoints.csv \
    --groups tables/lammps_a21_four_temperature_groups.csv \
    --panel-groups tables/lammps_a21_panel_groups.csv \
    --response tables/lammps_a21_four_temperature_response.csv \
    --excluded-curve-audit tables/lammps_a21_excluded_dpa4_fcc_curve_audit.csv \
    --analysis-receipt work/lammps/lammps_a21_figure7_tables.receipt.json \
    --output work/lammps/lammps_a21_figure7_tables.validation.json

"$PYTHON_EXE" work/lammps/build_a21_manuscript_claims.py
"$PYTHON_EXE" work/manuscript/build_a21_manuscript_replacements.py

"$PYTHON_EXE" work/figures/build_lammps_physical_validation_figure_a21.py
"$PYTHON_EXE" work/figures/validate_lammps_physical_validation_figure_a21.py
"$PYTHON_EXE" work/figures/export_figure7_a21_data.py
