#!/usr/bin/env python3
"""A11 independent NPT-table replay with timestep zero excluded in memory."""

from __future__ import annotations

from pathlib import Path

import validate_npt_tables as raw
import validate_a5_npt_tables as frozen


A11_PROTOCOL_SHA256 = "326caa5a972c010b372544d47dea1fabecae2325210de135ef717042cf1bfaac"
_MATRIX = raw.matrix


def analysis_matrix(path: Path, columns: int):
    """Exclude only the raw endpoint-inclusive timestep-zero row in memory."""

    values = _MATRIX(path, columns)
    if values.shape[0] and values[0, 0] == 0.0 and values[0, 1] == 0.0:
        return values[1:, :]
    return values


def main() -> None:
    raw.matrix = analysis_matrix
    frozen.AMENDMENT_A9_PROTOCOL_SHA256 = A11_PROTOCOL_SHA256
    frozen.__file__ = __file__
    frozen.main()


if __name__ == "__main__":
    main()
