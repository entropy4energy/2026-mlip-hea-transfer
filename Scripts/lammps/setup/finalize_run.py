#!/usr/bin/env python3
"""Create an immutable output index and completion/failure receipt for one run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


OWN_OUTPUTS = {
    "completion_receipt.json",
    "run.failed.json",
    "run.complete",
    "run_outputs.tsv",
}
NONFINITE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:[-+]?nan|[-+]?inf(?:inity)?)(?![A-Za-z0-9_])"
)
FATAL_PATTERNS = (
    re.compile(r"(?i)lost atoms"),
    re.compile(r"(?i)non-numeric"),
    re.compile(r"(?i)segmentation fault"),
    re.compile(r"(?i)cuda error"),
    re.compile(r"(?i)neighbor list overflow"),
    re.compile(r"(?i)^ERROR(?: on proc [0-9]+)?:", re.MULTILINE),
)
WARNING = re.compile(r"(?im)^WARNING:\s*(.+?)\s*$")
EXPECTED_WARNINGS = (
    re.compile(
        r"^Energy due to [0-9]+ extra global DOFs will be included in minimizer energies$"
    ),
    re.compile(
        r"^New thermo_style command, previous thermo_modify settings will be lost(?: \(.*\))?$"
    ),
)
CNA_DIAGNOSTIC_WARNINGS = (
    re.compile(r"^Too many common neighbors in CNA: [0-9]+x \(.*compute_cna_atom\.cpp:[0-9]+\)$"),
    re.compile(r"^Too many neighbors in CNA for [0-9]+ atoms \(.*compute_cna_atom\.cpp:[0-9]+\)$"),
)
WARNING_SUPPRESSION = re.compile(
    r"^Too many warnings: [0-9]+ vs [0-9]+\. All future warnings will be suppressed$"
)


def classify_warnings(unique_warnings: list[str]) -> tuple[list[str], list[str]]:
    """Return explicitly recognized and unhandled warning messages.

    CNA neighbor-limit messages affect only the optional fixed-cutoff
    structure diagnostic, not force evaluation or time integration.  The
    generic suppression message is recognized only when every other unique
    warning is already explicitly recognized, so it cannot conceal another
    warning type.
    """
    nonsuppression = [
        warning for warning in unique_warnings if not WARNING_SUPPRESSION.fullmatch(warning)
    ]
    recognized_without_suppression = [
        warning
        for warning in nonsuppression
        if any(
            pattern.fullmatch(warning)
            for pattern in (*EXPECTED_WARNINGS, *CNA_DIAGNOSTIC_WARNINGS)
        )
    ]
    suppression_is_scoped = len(recognized_without_suppression) == len(nonsuppression)
    recognized: list[str] = []
    unhandled: list[str] = []
    for warning in unique_warnings:
        if any(
            pattern.fullmatch(warning)
            for pattern in (*EXPECTED_WARNINGS, *CNA_DIAGNOSTIC_WARNINGS)
        ):
            recognized.append(warning)
        elif WARNING_SUPPRESSION.fullmatch(warning) and suppression_is_scoped:
            recognized.append(warning)
        else:
            unhandled.append(warning)
    return recognized, unhandled


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def scan_text(path: Path) -> tuple[list[str], list[str], bool]:
    text = path.read_text(encoding="utf-8", errors="replace")
    fatal = [pattern.pattern for pattern in FATAL_PATTERNS if pattern.search(text)]
    warnings = [match.strip() for match in WARNING.findall(text)]
    return fatal, warnings, bool(NONFINITE.search(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--return-code", type=int, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    manifest = args.manifest.resolve()
    if not root.is_dir() or manifest.parent != root or not manifest.is_file():
        raise ValueError("output and manifest do not describe one run directory")

    required_logs = [root / "screen.log", root / "log.lammps"]
    missing_logs = [str(path) for path in required_logs if not path.is_file()]
    fatal_patterns: dict[str, list[str]] = {}
    warnings: list[str] = []
    nonfinite_files: list[str] = []
    for path in required_logs:
        if not path.is_file():
            continue
        fatal, file_warnings, nonfinite = scan_text(path)
        if fatal:
            fatal_patterns[path.name] = fatal
        warnings.extend(file_warnings)
        if nonfinite:
            nonfinite_files.append(path.name)

    indexed: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in OWN_OUTPUTS or path.name.startswith(".run_outputs."):
            continue
        indexed.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    output_index = root / "run_outputs.tsv"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".run_outputs.", dir=root)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("relative_path", "size_bytes", "sha256"),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(indexed)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_index)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    unique_warnings = sorted(set(warnings))
    recognized_warnings, unhandled_warnings = classify_warnings(unique_warnings)
    passed = (
        args.return_code == 0
        and not missing_logs
        and not fatal_patterns
        and not nonfinite_files
        and not unhandled_warnings
    )
    receipt = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(root),
        "run_manifest": str(manifest),
        "run_manifest_sha256": sha256(manifest),
        "lammps_return_code": args.return_code,
        "missing_required_logs": missing_logs,
        "fatal_log_patterns": fatal_patterns,
        "nonfinite_log_files": nonfinite_files,
        "warning_count": len(warnings),
        "warning_messages": unique_warnings,
        "recognized_warning_count": len(recognized_warnings),
        "recognized_warning_messages": recognized_warnings,
        "unhandled_warning_count": len(unhandled_warnings),
        "unhandled_warning_messages": unhandled_warnings,
        "output_index": str(output_index),
        "output_index_sha256": sha256(output_index),
        "indexed_output_files": len(indexed),
        "indexed_output_bytes": sum(int(row["size_bytes"]) for row in indexed),
    }
    receipt_name = "completion_receipt.json" if passed else "run.failed.json"
    receipt_path = root / receipt_name
    atomic_write_text(receipt_path, json.dumps(receipt, indent=2) + "\n")
    if passed:
        marker = {
            "status": "passed",
            "completion_receipt": str(receipt_path),
            "completion_receipt_sha256": sha256(receipt_path),
        }
        atomic_write_text(root / "run.complete", json.dumps(marker, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
