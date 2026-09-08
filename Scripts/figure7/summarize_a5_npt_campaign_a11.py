#!/usr/bin/env python3
"""A11 NPT reducer with the A10 endpoint-grid correction retained."""

from __future__ import annotations

from pathlib import Path

import summarize_npt_campaign as raw
import summarize_a5_npt_campaign as frozen


A11_PROTOCOL_SHA256 = "326caa5a972c010b372544d47dea1fabecae2325210de135ef717042cf1bfaac"
_READ_NUMERIC = raw.read_numeric


def analysis_arrays(path: Path, columns: tuple[str, ...]):
    """Exclude only the raw endpoint-inclusive timestep-zero row in memory."""

    values = _READ_NUMERIC(path, columns)
    steps = values.get("step")
    times = values.get("time_ps")
    if steps is not None and times is not None and steps.size and steps[0] == 0.0 and times[0] == 0.0:
        return {name: array[1:] for name, array in values.items()}
    return values


def main() -> None:
    raw.read_numeric = analysis_arrays
    frozen.AMENDMENT_A9_PROTOCOL_SHA256 = A11_PROTOCOL_SHA256
    frozen.__file__ = __file__
    frozen.main()


if __name__ == "__main__":
    main()
