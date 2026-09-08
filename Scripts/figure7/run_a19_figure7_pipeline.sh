#!/usr/bin/env bash
# Rebuild, independently validate, plot, and export A19 Figure 7 data.

set -Eeuo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_EXE=python3
CAMPAIGN="${PROJECT}/LAMMPS/figure7_recovery_20260901"

"$PYTHON_EXE" "${PROJECT}/work/lammps/build_a19_figure7_tables.py" \
    --campaign-root "$CAMPAIGN" \
    --plan "${CAMPAIGN}/plans/skipjack_h100.tsv" \
    --a11-replicas "${PROJECT}/tables/lammps_a5_npt_replicas_a11.csv" \
    --a12-replicas "${PROJECT}/tables/lammps_a12_npt_replicas.csv" \
    --run-endpoints "${PROJECT}/tables/lammps_a19_run_endpoints.csv" \
    --groups "${PROJECT}/tables/lammps_a19_four_temperature_groups.csv" \
    --panel-groups "${PROJECT}/tables/lammps_a19_panel_groups.csv" \
    --response "${PROJECT}/tables/lammps_a19_four_temperature_response.csv" \
    --receipt "${PROJECT}/work/lammps/lammps_a19_figure7_tables.receipt.json"

"$PYTHON_EXE" "${PROJECT}/work/lammps/validate_a19_figure7_tables.py" \
    --campaign-root "$CAMPAIGN" \
    --plan "${CAMPAIGN}/plans/skipjack_h100.tsv" \
    --a11-replicas "${PROJECT}/tables/lammps_a5_npt_replicas_a11.csv" \
    --a12-replicas "${PROJECT}/tables/lammps_a12_npt_replicas.csv" \
    --run-endpoints "${PROJECT}/tables/lammps_a19_run_endpoints.csv" \
    --groups "${PROJECT}/tables/lammps_a19_four_temperature_groups.csv" \
    --panel-groups "${PROJECT}/tables/lammps_a19_panel_groups.csv" \
    --response "${PROJECT}/tables/lammps_a19_four_temperature_response.csv" \
    --analysis-receipt "${PROJECT}/work/lammps/lammps_a19_figure7_tables.receipt.json" \
    --output "${PROJECT}/work/lammps/lammps_a19_figure7_tables.validation.json"

"$PYTHON_EXE" "${PROJECT}/work/figures/build_lammps_physical_validation_figure_a19.py"
"$PYTHON_EXE" "${PROJECT}/work/figures/validate_lammps_physical_validation_figure_a19.py"
"$PYTHON_EXE" "${PROJECT}/work/figures/export_figure7_a19_data.py"
