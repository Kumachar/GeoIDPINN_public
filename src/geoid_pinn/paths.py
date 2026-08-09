"""Portable path discovery for scripts and notebooks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def project_root(start: Optional[Path] = None) -> Path:
    """Find the repository root by locating ``pyproject.toml``."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate pyproject.toml. Run from the repository or set an "
        "explicit root in the calling script."
    )


def data_root() -> Path:
    """Return the local restricted-data root.

    The tract-level Louisiana file is intentionally not distributed in this
    repository. Set ``LOUISIANA_DATA_ROOT`` to a directory containing the
    original ``Raw_data`` folder or the Louisiana ``.dta`` file.
    """
    configured = os.environ.get("LOUISIANA_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else project_root()


def results_root() -> Path:
    """Return the location for checked-in summaries and new experiment output."""
    configured = os.environ.get("GEOID_RESULTS_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else project_root() / "results"
    )


def public_prior_dir(scope: str = "all64") -> Path:
    """Return the bundled public geography/commuting prior directory."""
    base = project_root() / "data" / "public_priors"
    if scope.lower() == "top15":
        return base / "top15_subset"
    if scope.lower() != "all64":
        raise ValueError("scope must be 'all64' or 'top15'")
    return base
