"""Fail fast when a public release contains likely restricted data or secrets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.csv"

FORBIDDEN_EXTENSIONS = {
    ".dta", ".sav", ".parquet", ".feather", ".pkl", ".pickle",
    ".pt", ".pth", ".ckpt",
}
RISKY_NAME_PARTS = {
    "predictions_by_seed", "parish_predictions", "county_predictions",
    "outside_signal", "latent_states", "profile_loss", "checkpoint",
    "executed.ipynb",
}
TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".tex", ".bib", ".toml", ".cff",
    ".json", ".yaml", ".yml", ".csv", ".gitignore",
}
SENSITIVE_CSV_COLUMNS = {
    "weeklycasecount", "weeklytestcount", "cases", "tests", "total_cases",
    "total_tests", "first_week", "last_week", "raw_population_sum_tracts",
    "n_tracts", "observed", "prediction", "predicted", "y_true", "y_pred",
}
PUBLIC_CASE_FILE = ROOT / "data" / "external" / "cdc_neighbor_states_weekly_cases_2020.csv"


def release_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path != MANIFEST
        and "__pycache__" not in path.parts
    )


def scan_notebook(path: Path) -> list[str]:
    issues = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            issues.append(f"{path}: code cell {index} contains saved outputs")
        if cell.get("execution_count") is not None:
            issues.append(f"{path}: code cell {index} has an execution count")
    return issues


def scan_csv_schema(path: Path) -> list[str]:
    if path.resolve() == PUBLIC_CASE_FILE.resolve():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    risky = sorted({column.strip().lower() for column in header} & SENSITIVE_CSV_COLUMNS)
    return [f"{path}: restricted-looking CSV columns {risky}"] if risky else []


def scan_text(path: Path) -> list[str]:
    if path.name == "audit_public_release.py":
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "local Windows user path": re.compile(r"[A-Za-z]:\\Users\\", re.I),
        "local workspace path": re.compile(r"[A-Za-z]:\\Umich\\", re.I),
        "container path": re.compile(r"/(?:mnt/data|home)/", re.I),
        "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
        "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
        "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    }
    return [f"{path}: contains {label}" for label, pattern in patterns.items() if pattern.search(text)]


def audit() -> list[str]:
    issues = []
    for path in release_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            issues.append(f"{relative}: forbidden extension")
        lowered = relative.as_posix().lower()
        for part in RISKY_NAME_PARTS:
            if part in lowered:
                issues.append(f"{relative}: risky filename contains '{part}'")
        if path.suffix.lower() == ".ipynb":
            issues.extend(scan_notebook(relative if relative.is_absolute() else path))
        if path.suffix.lower() == ".csv":
            issues.extend(scan_csv_schema(path))
        if path.suffix.lower() in TEXT_EXTENSIONS or path.name == ".gitignore":
            issues.extend(scan_text(path))
    return issues


def write_manifest() -> None:
    rows = []
    for path in release_files():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append((path.relative_to(ROOT).as_posix(), path.stat().st_size, digest))
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    issues = audit()
    if issues:
        print("Public-release audit failed:")
        for issue in issues:
            print(f" - {issue}")
        raise SystemExit(1)
    if args.write_manifest:
        write_manifest()
        print(f"Wrote {MANIFEST}")
    print(f"Public-release audit passed for {len(release_files())} files.")


if __name__ == "__main__":
    main()
