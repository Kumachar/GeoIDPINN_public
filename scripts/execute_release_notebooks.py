"""Execute release notebook code cells without starting a Jupyter server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOKS = [
    ROOT / "notebooks" / "01_all64_model_comparison.ipynb",
    ROOT / "notebooks" / "02_spatial_prior_sensitivity.ipynb",
    ROOT / "notebooks" / "03_learned_matrix_comparison.ipynb",
    ROOT / "notebooks" / "04_paper_results_index.ipynb",
]


def execute(path: Path) -> None:
    from IPython.display import display
    import matplotlib.pyplot as plt

    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "display": display}
    original_directory = Path.cwd()
    os.chdir(ROOT)
    try:
        for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            exec(compile(source, f"{path.name}:cell-{cell_index}", "exec"), namespace)
    finally:
        plt.close("all")
        os.chdir(original_directory)
    print(f"PASS {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", type=Path)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.notebooks] if args.notebooks else DEFAULT_NOTEBOOKS
    for path in paths:
        execute(path)


if __name__ == "__main__":
    main()
