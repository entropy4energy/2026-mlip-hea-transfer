# Scripts

The scripts that produced the tables in `Data/` and the inputs in `Inputs/`,
carried over from the study archive. They are the record of how each stage was
run, not a turnkey pipeline: paths to the raw campaign directories, which live
in the CHAOS archive rather than in this repository, are supplied through
environment variables or command-line arguments at the top of each file. Site
specific software locations have been replaced by environment variables
(`DEEPMD_BUILD_ROOT`, `DEEPMD_CONDA_ENV`, `CUDA_INSTALL_DIR`,
`NVHPC_INSTALL_DIR`, `PYTHON_EXE`, `TRAINING_RUNS`).

Cluster orchestration (queue workers, watchers, node launchers, per-machine
environment files) is not included; it has no meaning outside the machines the
campaign ran on.

## `deepmd/`

Training and static evaluation of the fifteen models.

- `runDP_restartable.py`, `run_one.sh`: one training run from its `input.json`,
  restartable from the last checkpoint.
- `evaluate_study_models.py`, `run_eval_one.sh`: `dp test` over a held-out
  system set, producing the per-frame and per-system metrics that reduce to
  `Data/model/`.

## `lammps/`

- `analysis/`: reduction of LAMMPS output to physical quantities.
  `analyze_eos.py`, `analyze_elastic.py`, `analyze_thermo.py`,
  `analyze_nve.py`, `analyze_transport.py`, `analyze_uniaxial.py`.
- `setup/`: cell generation, model freezing, stage planning and the run driver.
  `generate_phase_cells.py` writes the 2,000-atom decorations in
  `Inputs/structures/finite_temperature/`; `freeze_study_models.sh` and
  `freeze_phase_models.sh` freeze the deployment models from the training
  checkpoints; `build_stage_plan.py` and `build_dynamic_stage_plan.py` build
  the dependency-gated run plans; `run_lammps.sh` and `finalize_run.py` execute
  and close out one run.

## `figure7/`

The finite-temperature NPT campaign behind the thermal-response figure, its
equilibration gating, and the table builders, each paired with a validator
(`build_*` writes, `validate_*` checks the written tables against the run
records). `run_a21_figure7_pipeline.sh` is the current entry point;
`run_a19_*` is the earlier revision, kept because the run history references
it.

## `tensile/`

The uniaxial-tension campaign: oriented and compact cell generation,
equilibration gating, NVE validation, stage planning, analysis, promotion of
the accepted results, and the SI table renderer.

## `random_frame_control/`

The 10,000-draw random-frame coverage control.
`compute_environment_coverage.py` builds the descriptor cache and
nearest-training-environment distances;
`run_random_seed_coverage_control.py` runs the control;
`validate_random_seed_coverage_control.py` checks it;
`build_random_seed_coverage_workbook.py` and `..._notebook.py` produce the
workbook and notebook. `build_sha256_manifest.py` is the manifest generator
used across the archive.

## Figures

`Data/plot_ready/` covers F01-F04, F06, F08 and FS1. F05, F07, F09, FS2, FS3
and FS4 have no plot-ready table yet; F09 can be drawn from `Data/tensile/`,
FS2 from `Data/random_frame_control/`, and F07 from the figure7 table builders
above once their output tables are recovered from the archive.
