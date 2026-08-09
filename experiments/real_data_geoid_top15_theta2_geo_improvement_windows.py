from __future__ import annotations

import os
import re
import json
import math
import zipfile
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from geoid_pinn.priors import (
    build_nb_residual_pwccf_prior,
    permute_prior_offdiagonal,
)
from geoid_pinn.paths import data_root, project_root, public_prior_dir, results_root
try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 180)
np.set_printoptions(precision=4, suppress=True)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise RuntimeError("This notebook requires PyTorch. Install it with `pip install torch`.") from e

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, "torch threads:", torch.get_num_threads())

INITIAL_WORKING_DIR = Path(".").resolve()

def _has_prior_outputs(p: Path) -> bool:
    return (p / "outputs").is_dir() or (p / "louisiana_public_prior_data_bundle" / "outputs").is_dir()

def detect_project_root(start: Path) -> Path:
    """Find the project root on Windows/Jupyter.

    The notebook may start in a subfolder such as `.idea` or a notebook-specific
    working directory. We want the folder that contains `outputs/` and the raw
    data folder/zip.
    """
    candidates = []
    cur = start.resolve()
    candidates.extend([cur] + list(cur.parents))
    # Also check immediate children of the current directory and its parent.
    for base in [cur, cur.parent]:
        if base.exists() and base.is_dir():
            try:
                candidates.extend([p for p in base.iterdir() if p.is_dir()])
            except Exception:
                pass

    seen = set()
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            continue
        if c in seen:
            continue
        seen.add(c)
        if _has_prior_outputs(c):
            return c

    raise FileNotFoundError(
        "Could not find project root containing outputs/. "
        "Run `Path.cwd()` to see the current working directory, then set PROJECT_ROOT manually."
    )

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT_OVERRIDE"]).resolve() if os.environ.get("PROJECT_ROOT_OVERRIDE") else project_root(INITIAL_WORKING_DIR)
DATA_ROOT = data_root()
RESULTS_ROOT = results_root()
os.chdir(PROJECT_ROOT)
print("Initial working directory:", INITIAL_WORKING_DIR)
print("Detected project root:", PROJECT_ROOT)
RUN_MODE = os.environ.get("GEOID_ALL64_MODE", "paper64")  # smoke, pilot64, paper64

OUTPUT_DIR = RESULTS_ROOT / "real_data_geoid_top15_theta2_geo_improvement_windows_outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MATRIX_DIR = OUTPUT_DIR / "learned_matrices"
PROFILE_DIR = OUTPUT_DIR / "profile_loss"
BASELINE_DIR = OUTPUT_DIR / "baselines"
DATA_PRIOR_DIR = OUTPUT_DIR / "data_derived_priors"
for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, MATRIX_DIR, PROFILE_DIR, BASELINE_DIR, DATA_PRIOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# All-64 parish experiment.
# Scientific goal: test whether a closed Louisiana system reduces the need for an outside reservoir.
# Main model: cases_tests_nb + time-varying alpha + full 64-parish C matrix.
CONFIG_BY_MODE = {
    "smoke": dict(
        top_n=8, seeds=[0], epochs=8, warmup_epochs=2, hidden=12,
        splits=["final"], test_weeks=6,
        obs_models=["cases_tests_nb"],
        run_profile=False,
        profile_epochs=20,
        baseline_epochs=40,
        diag_values=[0.80],
    ),
    "pilot64": dict(
        top_n=64, seeds=[0], epochs=180, warmup_epochs=50, hidden=48,
        splits=["roll_2020-09-03", "roll_2020-10-15"],
        test_weeks=6,
        obs_models=["cases_tests_nb"],
        run_profile=False,
        profile_epochs=100,
        baseline_epochs=700,
        diag_values=[0.70, 0.80, 0.90, 0.95, 1.00],
    ),
    "paper64": dict(
        top_n=15, seeds=[0, 1, 2], epochs=450, warmup_epochs=120, hidden=64,
        splits=["roll_2020-07-23", "roll_2020-09-03", "roll_2020-10-15", "roll_2020-11-26"],
        test_weeks=6,
        obs_models=["cases_tests_nb"],
        run_profile=True,
        profile_epochs=160,
        baseline_epochs=1200,
        diag_values=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00],
    ),
}
if RUN_MODE not in CONFIG_BY_MODE:
    raise ValueError(f"Unknown GEOID_ALL64_MODE={RUN_MODE!r}; choose smoke, pilot64, or paper64.")
CFG = CONFIG_BY_MODE[RUN_MODE]
print("GEOID_ALL64_MODE:", RUN_MODE)
print(json.dumps(CFG, indent=2))

# Early/pre-vaccine window to reduce vaccine/variant confounding in the first real-data analysis.
START_DATE = pd.Timestamp("2020-03-05")
END_DATE = pd.Timestamp("2020-12-31")

# Weekly epidemiological constants. Sensitivity analysis can vary these later.
GAMMA_WEEK = 7 / 10       # approx. 10-day infectious period
MU_WEEK = 0.003           # small weekly mortality/removal-to-D channel; deaths are not observed in this experiment

# Loss weights. The model is a diagnostic inverse model, not a final forecasting system.
LOSS_WEIGHTS = dict(
    data=1.0,
    physics=2.0,
    ic=1.0,
    prior=0.20,
    alpha_smooth=0.03,
    state_smooth=0.001,
    external_l2=0.002,
    ar_l2=0.002,
)
PRIOR_SCALE = 0.10
EPS = 1e-8

def maybe_unzip_prior_bundle(root: Path) -> None:
    """Ensure `outputs/` exists in the project root.

    Handles both layouts:
    1. project_root/outputs/
    2. project_root/louisiana_public_prior_data_bundle/outputs/
    """
    if (root / "outputs").exists():
        return

    nested_outputs = root / "louisiana_public_prior_data_bundle" / "outputs"
    if nested_outputs.exists():
        print(f"Copying prior outputs from {nested_outputs} to {root / 'outputs'}")
        shutil.copytree(nested_outputs, root / "outputs", dirs_exist_ok=True)
        return

    candidates = list(root.glob("*prior*data*bundle*.zip")) + list(root.glob("louisiana_public_prior_data_bundle.zip"))
    if candidates:
        zip_path = candidates[0]
        print(f"Unzipping prior bundle: {zip_path}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)

    nested_outputs = root / "louisiana_public_prior_data_bundle" / "outputs"
    if (root / "outputs").exists():
        return
    if nested_outputs.exists():
        print(f"Copying prior outputs from {nested_outputs} to {root / 'outputs'}")
        shutil.copytree(nested_outputs, root / "outputs", dirs_exist_ok=True)
        return

    raise FileNotFoundError(
        f"Could not find outputs/. Current project root is {root}. "
        "Make sure the Louisiana prior bundle is unzipped so that either "
        "`outputs/` or `louisiana_public_prior_data_bundle/outputs/` exists."
    )


def find_louisiana_dta(root: Path) -> Path:
    """Find or extract the Louisiana weekly case Stata file.

    Supports common Windows layouts:
    - project_root/Raw_data.zip
    - project_root/raw/Raw_data.zip
    - project_root/raw/*.dta
    - project_root/Raw_data/*.dta
    """
    candidate_dirs = [
        root,
        root / "raw",
        root / "Raw_data",
        root / "Raw Data",
    ]

    direct_names = [
        "LousianaWeeklyCasesByCensusTract2022_date_20231001.dta",
        "LouisianaWeeklyCasesByCensusTract2022_date_20231001.dta",
    ]

    for d in candidate_dirs:
        for name in direct_names:
            p = d / name
            if p.exists():
                return p

    for d in candidate_dirs:
        if d.exists() and d.is_dir():
            dta_files = list(d.glob("*.dta"))
            for p in dta_files:
                if "Lousiana" in p.name or "Louisiana" in p.name:
                    return p
            if dta_files:
                return dta_files[0]

    zip_candidates = []
    for d in candidate_dirs:
        if d.exists():
            zip_candidates.extend(list(d.glob("*Raw_data*.zip")))
            zip_candidates.extend(list(d.glob("*raw_data*.zip")))
            zip_candidates.extend(list(d.glob("*Raw*.zip")))
    zip_candidates.extend([root / "Raw_data.zip", root / "raw" / "Raw_data.zip"])
    zip_candidates = [p for p in zip_candidates if p.exists()]

    if not zip_candidates:
        raise FileNotFoundError(
            f"Could not find Raw_data.zip or a Louisiana .dta file under {root}, {root/'raw'}, or {root/'Raw_data'}."
        )

    zip_path = zip_candidates[0]
    print(f"Extracting Louisiana DTA from {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".dta") and ("Lousiana" in n or "Louisiana" in n)]
        if not names:
            names = [n for n in z.namelist() if n.endswith(".dta")]
        if not names:
            raise FileNotFoundError("No .dta file found inside Raw_data zip.")
        name = names[0]
        z.extract(name, root)
    return root / name


def load_and_aggregate_louisiana(dta_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_stata(dta_path, convert_categoricals=False)
    required = ["parish", "tract_fips10", "dateforstartofweek", "weeklycasecount", "weeklytestcount"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")

    raw["parish_fips"] = raw["tract_fips10"].astype(str).str[:5].str.zfill(5)
    raw["week"] = pd.to_datetime(raw["dateforstartofweek"])
    raw["weeklycasecount"] = pd.to_numeric(raw["weeklycasecount"], errors="coerce").fillna(0).clip(lower=0)
    raw["weeklytestcount"] = pd.to_numeric(raw["weeklytestcount"], errors="coerce").fillna(0).clip(lower=0)

    pop_col = "tract_pop" if "tract_pop" in raw.columns else "totpop13_17"
    tract_pop = (raw[["parish_fips", "tract_fips10", pop_col]]
                 .drop_duplicates(subset=["tract_fips10"])
                 .assign(**{pop_col: lambda x: pd.to_numeric(x[pop_col], errors="coerce").fillna(0)})
                 .groupby("parish_fips", as_index=False)[pop_col].sum()
                 .rename(columns={pop_col: "population_raw_tract_sum"}))

    parish_names = raw[["parish_fips", "parish"]].drop_duplicates().sort_values("parish_fips")
    parish_meta = parish_names.merge(tract_pop, on="parish_fips", how="left")

    weekly = (raw.groupby(["parish_fips", "week"], as_index=False)
                 .agg(cases=("weeklycasecount", "sum"), tests=("weeklytestcount", "sum")))
    weekly = weekly.merge(parish_meta, on="parish_fips", how="left")
    weekly["positivity"] = weekly["cases"] / weekly["tests"].replace(0, np.nan)
    weekly["positivity"] = weekly["positivity"].replace([np.inf, -np.inf], np.nan)
    return raw, weekly, parish_meta

DTA_PATH = find_louisiana_dta(DATA_ROOT)
raw_df, weekly_df, parish_meta_raw = load_and_aggregate_louisiana(DTA_PATH)
print("Louisiana DTA:", DTA_PATH)
print("Raw shape:", raw_df.shape)
print("Weekly parish rows:", weekly_df.shape)
print("Parishes:", weekly_df["parish_fips"].nunique(), "Weeks:", weekly_df["week"].nunique())
weekly_df.head()

def build_case_test_matrices(weekly: pd.DataFrame, fips_order: List[str], start: pd.Timestamp, end: pd.Timestamp):
    sub = weekly[(weekly["week"] >= start) & (weekly["week"] <= end)].copy()
    all_weeks = pd.Index(sorted(sub["week"].unique()), name="week")
    cases = (sub.pivot_table(index="week", columns="parish_fips", values="cases", aggfunc="sum")
               .reindex(index=all_weeks, columns=fips_order).fillna(0.0))
    tests = (sub.pivot_table(index="week", columns="parish_fips", values="tests", aggfunc="sum")
               .reindex(index=all_weeks, columns=fips_order).fillna(0.0))
    return cases, tests

# Select parishes and prior directory.
# For top_n=64, use the full 64-parish prior matrices under outputs/.
# For smoke/top-n subset, use outputs/top15_subset when available.
if CFG["top_n"] >= 64:
    prior_dir = public_prior_dir("all64")
    meta_file = prior_dir / "parish_metadata_with_centroids.csv"
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing {meta_file}. Unzip the Louisiana public prior bundle.")
    meta_prior = pd.read_csv(meta_file)
else:
    top15_dir = public_prior_dir("top15")
    if top15_dir.exists() and (top15_dir / "top15_parish_metadata.csv").exists():
        prior_dir = top15_dir
        meta_prior = pd.read_csv(prior_dir / "top15_parish_metadata.csv")
    else:
        prior_dir = public_prior_dir("all64")
        meta_prior = pd.read_csv(prior_dir / "parish_metadata_with_centroids.csv")

meta_prior["parish_fips"] = meta_prior["parish_fips"].astype(str).str.zfill(5)
study_totals = (
    weekly_df.groupby("parish_fips", as_index=False)
    .agg(total_cases=("cases", "sum"), total_tests=("tests", "sum"))
)
meta_prior = meta_prior.merge(study_totals, on="parish_fips", how="left")
meta_prior = meta_prior.sort_values("total_cases", ascending=False).reset_index(drop=True)

selected_meta = meta_prior.head(CFG["top_n"]).copy()
selected_fips = selected_meta["parish_fips"].tolist()
selected_names = selected_meta["parish"].tolist()
print("Prior directory:", prior_dir)
print("Selected parishes:", len(selected_fips))
display(selected_meta[["parish_fips", "parish", "total_cases", "total_tests", "cenpop2020_population"]].head(20))

cases_df, tests_df = build_case_test_matrices(weekly_df, selected_fips, START_DATE, END_DATE)
pop_s = selected_meta.set_index("parish_fips").loc[selected_fips, "cenpop2020_population"].astype(float)

# Original top-15 outside reservoir: the other 49 Louisiana parishes.
all_fips = sorted(weekly_df["parish_fips"].astype(str).str.zfill(5).unique())
outside_fips = [fips for fips in all_fips if fips not in set(selected_fips)]
outside_cases_df, outside_tests_df = build_case_test_matrices(
    weekly_df, outside_fips, START_DATE, END_DATE
)
outside_pop = float(
    parish_meta_raw.set_index("parish_fips")
    .reindex(outside_fips)["population_raw_tract_sum"].fillna(0).sum()
)
outside_rate = outside_cases_df.sum(axis=1) / max(outside_pop, 1.0)
outside_smoothed = outside_rate.rolling(3, min_periods=1).mean().fillna(0.0)
outside_signal_series = outside_smoothed / max(float(outside_smoothed.mean()), 1e-8)
outside_signal = outside_signal_series.clip(lower=0, upper=5).to_numpy(dtype=np.float32)
outside_signal_df = pd.DataFrame({
    "week": cases_df.index,
    "outside_cases": outside_cases_df.sum(axis=1).to_numpy(dtype=float),
    "outside_rate_smoothed": outside_smoothed.to_numpy(dtype=float),
    "outside_signal": outside_signal,
})
outside_signal_df.to_csv(DATA_PRIOR_DIR / "top15_other49_outside_signal.csv", index=False)
print("Case matrix:", cases_df.shape, cases_df.index.min(), cases_df.index.max())
print("Outside Louisiana parishes:", len(outside_fips), "population approx:", outside_pop)
print("Outside signal range:", float(outside_signal.min()), float(outside_signal.max()))

def make_reporting_weights(cases: pd.DataFrame, tests: pd.DataFrame, pop: pd.Series) -> pd.DataFrame:
    tests_per_1000 = tests.div(pop, axis=1) * 1000
    # A conservative fixed offset: more tests implies higher ascertainment, but clipping prevents the offset from dominating.
    w = np.log1p(tests_per_1000)
    w = w / np.nanmean(w.to_numpy())
    w = w.clip(lower=0.35, upper=2.50).fillna(1.0)
    return w

report_weights_df = make_reporting_weights(cases_df, tests_df, pop_s)
positivity_df = cases_df / tests_df.replace(0, np.nan)

fig, ax = plt.subplots(figsize=(8, 3.0))
cases_df.sum(axis=1).plot(ax=ax)
ax.set_title("Total weekly cases in selected Louisiana parishes")
ax.set_ylabel("Cases")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "selected_total_cases.png", dpi=180, bbox_inches="tight")
plt.show()

def normalize_fips_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.astype(str).str.zfill(5)
    out.columns = out.columns.astype(str).str.zfill(5)
    return out


def row_normalize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    M = np.maximum(M, 0)
    rs = M.sum(axis=1, keepdims=True)
    bad = rs[:, 0] <= 0
    if bad.any():
        M[bad] = 1.0 / M.shape[1]
        rs = M.sum(axis=1, keepdims=True)
    return M / np.maximum(rs, EPS)


def impose_diag_mass(pattern: np.ndarray, diag_mass: float) -> np.ndarray:
    """Keep off-diagonal pattern but set each row's diagonal mass to diag_mass."""
    P = pattern.shape[0]
    d = float(np.clip(diag_mass, 0.0, 1.0))
    W = np.maximum(pattern.copy(), 0.0)
    np.fill_diagonal(W, 0.0)
    C = np.zeros_like(W)
    for i in range(P):
        off = W[i]
        total = off.sum()
        C[i, i] = d
        if d >= 1.0:
            continue
        if total <= 0:
            # If no off-diagonal structure, distribute uniformly over off-diagonal entries.
            off_mask = np.ones(P, dtype=bool)
            off_mask[i] = False
            C[i, off_mask] = (1.0 - d) / max(P - 1, 1)
        else:
            C[i] += (1.0 - d) * off / total
    return row_normalize(C)


def load_prior_csv(fname: str) -> Optional[np.ndarray]:
    p = prior_dir / fname
    if not p.exists():
        return None
    Cdf = normalize_fips_index(pd.read_csv(p, index_col=0))
    Cdf = Cdf.loc[selected_fips, selected_fips]
    return row_normalize(Cdf.to_numpy(dtype=np.float64))

priors: Dict[str, np.ndarray] = {}
for key, fname in {
    "identity": "C0_identity.csv",
    "distance_d80": "C0_distance_diag80_scale75km.csv",
    "adjacency_d80": "C0_adjacency_diag80.csv",
    "hybrid_d80": "C0_hybrid_distance_adjacency_diag80.csv",
    "wrong_d80": "C0_wrong_antidistance_diag80.csv",
}.items():
    C = load_prior_csv(fname)
    if C is not None:
        priors[key] = C

# Optional LODES orientation variants.
# Preferred: use raw commuting jobs matrix, rows=workplace, columns=home.
raw_commute_candidates = [
    "commute_jobs_work_by_home_LODES2019_JT01.csv",
    "commute_jobs_work_by_home_LODES2019_JT00.csv",
]
raw_commute = None
raw_commute_name = None
for fname in raw_commute_candidates:
    p = prior_dir / fname
    if p.exists():
        raw_commute_name = fname
        raw_df = normalize_fips_index(pd.read_csv(p, index_col=0))
        raw_commute = raw_df.loc[selected_fips, selected_fips].to_numpy(dtype=np.float64)
        break

if raw_commute is not None:
    W_work = raw_commute                         # rows=workplace/recipient, cols=home/source
    W_home = raw_commute.T                       # rows=home/reporting recipient, cols=workplace exposure source
    W_sym = 0.5 * (raw_commute + raw_commute.T)
    priors["lodes_work_d80"] = impose_diag_mass(W_work, 0.80)
    priors["lodes_home_d80"] = impose_diag_mass(W_home, 0.80)
    priors["lodes_sym_d80"] = impose_diag_mass(W_sym, 0.80)
    rng = np.random.default_rng(2026)
    perm = rng.permutation(W_home.shape[0])
    priors["lodes_home_permuted_d80"] = impose_diag_mass(W_home[:, perm], 0.80)
    print("Loaded raw LODES matrix:", raw_commute_name)
else:
    # Fallback: if only a row-normalized commute prior exists, approximate orientation variants with transpose.
    C_commute = None
    for fname in ["C0_commute_diag80_2019_JT01.csv", "C0_lodes_diag80.csv", "C0_commuting_diag80.csv"]:
        C = load_prior_csv(fname)
        if C is not None:
            C_commute = C
            raw_commute_name = fname
            break
    if C_commute is not None:
        priors["lodes_work_d80"] = C_commute
        priors["lodes_home_d80"] = row_normalize(C_commute.T)
        priors["lodes_sym_d80"] = row_normalize(0.5 * (C_commute + C_commute.T))
        rng = np.random.default_rng(2026)
        perm = rng.permutation(C_commute.shape[0])
        priors["lodes_home_permuted_d80"] = row_normalize(C_commute.T[:, perm])
        print("Loaded approximate LODES prior:", raw_commute_name)
    else:
        print("No LODES prior/raw commuting matrix found. LODES-orientation tests will be skipped.")

# Base pattern for diagonal-mass sweep: prefer LODES hybrid/home if available, otherwise hybrid geography.
base_pattern_key = "lodes_home_d80" if "lodes_home_d80" in priors else "hybrid_d80"
base_pattern = priors[base_pattern_key]
for d in CFG["diag_values"]:
    priors[f"diag_sweep_{base_pattern_key}_d{int(round(d*100)):02d}"] = impose_diag_mass(base_pattern, d)

# no_prior uses random C initialization with no prior penalty.
P_SELECTED = len(selected_fips)
priors["no_prior"] = np.ones((P_SELECTED, P_SELECTED), dtype=np.float64) / P_SELECTED

print("Available priors:")
for k, C in priors.items():
    print(f"{k:32s} diag={np.diag(C).mean():.3f} offdiag={(1-np.diag(C)).mean():.3f}")

def make_split_masks(weeks: pd.Index, split_name: str, test_weeks: int) -> Tuple[np.ndarray, np.ndarray, str]:
    T = len(weeks)
    if split_name == "final":
        train_end = max(T - test_weeks, 1)
    elif split_name.startswith("roll_"):
        date_str = split_name.replace("roll_", "")
        cutoff = pd.Timestamp(date_str)
        train_end = int(np.searchsorted(weeks.to_numpy(), cutoff.to_datetime64(), side="right"))
        train_end = min(max(train_end, 8), T - test_weeks)
    else:
        raise ValueError(split_name)
    train_mask = np.zeros(T, dtype=bool)
    test_mask = np.zeros(T, dtype=bool)
    train_mask[:train_end] = True
    test_mask[train_end:min(train_end + test_weeks, T)] = True
    label = f"train_to_{pd.Timestamp(weeks[train_end-1]).date()}_test_to_{pd.Timestamp(weeks[np.where(test_mask)[0][-1]]).date()}"
    return train_mask, test_mask, label

# Test split construction.
for split in CFG["splits"]:
    tr, te, label = make_split_masks(cases_df.index, split, CFG["test_weeks"])
    print(split, label, "train weeks", tr.sum(), "test weeks", te.sum())

# Reuse the exact training-only C0 matrices from the completed PWCCF run.
# This keeps every one-factor-at-a-time comparison paired on the same C0.
PWCCF_PRIOR_KEY = "nb_residual_pwccf_d80"
PWCCF_PERMUTED_KEY = "nb_residual_pwccf_permuted_d80"
PWCCF_SOURCE_DIR = RESULTS_ROOT / "real_data_geoid_all64_pwccf_outputs" / "data_derived_priors"
split_specific_priors: Dict[str, Dict[str, np.ndarray]] = {}

for split in CFG["splits"]:
    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "_", split)
    matrices = {}
    for prior_key in [PWCCF_PRIOR_KEY, PWCCF_PERMUTED_KEY]:
        path = PWCCF_SOURCE_DIR / f"C0_{prior_key}_{safe_split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run real_data_geoid_all64_pwccf_c0_experiment.py first."
            )
        frame = normalize_fips_index(pd.read_csv(path, index_col=0))
        matrices[prior_key] = row_normalize(
            frame.loc[selected_fips, selected_fips].to_numpy(dtype=np.float64)
        )
    split_specific_priors[split] = matrices

priors[PWCCF_PRIOR_KEY] = np.mean(
    [value[PWCCF_PRIOR_KEY] for value in split_specific_priors.values()], axis=0
)
priors[PWCCF_PERMUTED_KEY] = np.mean(
    [value[PWCCF_PERMUTED_KEY] for value in split_specific_priors.values()], axis=0
)
print("Loaded paired split-specific priors from:", PWCCF_SOURCE_DIR)

def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=DEVICE)

@dataclass
class FitData:
    cases: torch.Tensor
    report_w: torch.Tensor
    pop: torch.Tensor
    train_mask: torch.Tensor
    test_mask: torch.Tensor
    time_x: torch.Tensor
    external_signal: torch.Tensor
    weeks: List[pd.Timestamp]
    parish_names: List[str]
    parish_fips: List[str]
    obs_model: str
    split: str


def make_fit_data(obs_model: str, split: str) -> FitData:
    cases_np = cases_df.to_numpy(dtype=np.float32)
    pop_np = pop_s.to_numpy(dtype=np.float32)
    if obs_model == "cases_only":
        report_np = np.ones_like(cases_np, dtype=np.float32)
    elif obs_model in ["cases_tests_nb", "cases_test_offset"]:
        report_np = report_weights_df.to_numpy(dtype=np.float32)
    else:
        raise ValueError(obs_model)
    train_mask_np, test_mask_np, _ = make_split_masks(cases_df.index, split, CFG["test_weeks"])
    time_x_np = np.linspace(-1, 1, len(cases_df), dtype=np.float32).reshape(-1, 1)
    return FitData(
        cases=to_tensor(cases_np), report_w=to_tensor(report_np), pop=to_tensor(pop_np),
        train_mask=to_tensor(train_mask_np, dtype=torch.bool), test_mask=to_tensor(test_mask_np, dtype=torch.bool),
        time_x=to_tensor(time_x_np), external_signal=to_tensor(outside_signal),
        weeks=list(cases_df.index), parish_names=selected_names, parish_fips=selected_fips,
        obs_model=obs_model, split=split,
    )

class GeoIDDiagnosticPINN(nn.Module):
    def __init__(self, P: int, T: int, C0: np.ndarray, hidden: int,
                 alpha_mode: str = "time_varying", external_mode: str = "none",
                 theta_init: float = 20.0, theta_fixed: Optional[float] = None,
                 use_parish_reporting_intercept: bool = False, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.P = P
        self.T = T
        self.alpha_mode = alpha_mode
        self.external_mode = external_mode
        self.theta_is_fixed = theta_fixed is not None
        self.use_parish_reporting_intercept = bool(use_parish_reporting_intercept)

        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4 * P),
        )
        init_alpha = 0.60
        if alpha_mode == "static":
            self.raw_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(init_alpha))))
        elif alpha_mode == "time_varying":
            self.raw_alpha = nn.Parameter(
                torch.full((T,), math.log(math.expm1(init_alpha)))
            )
        else:
            raise ValueError(alpha_mode)

        self.raw_report_scale = nn.Parameter(torch.tensor(0.0))
        if self.use_parish_reporting_intercept:
            self.raw_parish_reporting_intercept = nn.Parameter(torch.zeros(P))
        else:
            self.raw_parish_reporting_intercept = None

        if self.theta_is_fixed:
            self.register_buffer("fixed_nb_theta", torch.tensor(float(theta_fixed)))
            self.raw_nb_theta = None
        else:
            target = max(float(theta_init) - 1e-3, 1e-4)
            self.raw_nb_theta = nn.Parameter(torch.tensor(math.log(math.expm1(target))))
            self.fixed_nb_theta = None

        if external_mode == "none":
            self.raw_external_scale = None
        elif external_mode == "outside":
            self.raw_external_scale = nn.Parameter(
                torch.full((P,), math.log(math.expm1(0.05)))
            )
        else:
            raise ValueError(external_mode)

        self.C_logits = nn.Parameter(torch.log(to_tensor(C0) + EPS))
        self.register_buffer("C0", to_tensor(C0))

    def C(self):
        return torch.softmax(self.C_logits, dim=1)

    def alpha(self):
        alpha = F.softplus(self.raw_alpha) + 1e-4
        if self.alpha_mode == "static":
            return alpha.repeat(self.T)
        return alpha

    def nb_theta(self):
        if self.theta_is_fixed:
            return self.fixed_nb_theta
        return F.softplus(self.raw_nb_theta) + 1e-3

    def parish_reporting_intercept(self):
        if not self.use_parish_reporting_intercept:
            return torch.zeros(self.P, device=DEVICE)
        # Centering separates relative parish reporting from the global scale.
        raw = self.raw_parish_reporting_intercept
        return raw - torch.mean(raw)

    def states(self, time_x):
        logits = self.net(time_x).reshape(-1, self.P, 4)
        states = torch.softmax(logits, dim=-1)
        return states[..., 0], states[..., 1], states[..., 2], states[..., 3]

    def forward(self, data: FitData):
        S, I, R, D = self.states(data.time_x)
        C = self.C()
        live = torch.clamp(S + I + R, min=1e-6)
        prevalence = I / live
        alpha_t = self.alpha()
        coupled = prevalence @ C.T
        if self.external_mode == "outside":
            ext_scale = F.softplus(self.raw_external_scale)[None, :]
            force_input = coupled + ext_scale * data.external_signal[:, None]
        else:
            ext_scale = torch.zeros((1, self.P), device=DEVICE)
            force_input = coupled
        force = alpha_t[:, None] * torch.clamp(force_input, min=1e-8)
        incidence_frac = torch.clamp(S * force, min=1e-10)
        report_scale = F.softplus(self.raw_report_scale) + 1e-4
        parish_intercept = self.parish_reporting_intercept()
        pred_cases = torch.clamp(
            incidence_frac
            * data.pop[None, :]
            * data.report_w
            * report_scale
            * torch.exp(parish_intercept[None, :]),
            min=1e-6,
        )
        return dict(
            S=S, I=I, R=R, D=D, C=C, alpha=alpha_t,
            incidence_frac=incidence_frac, pred_cases=pred_cases,
            report_scale=report_scale, nb_theta=self.nb_theta(),
            parish_reporting_intercept=parish_intercept,
            external_scale=ext_scale.squeeze(0),
        )


def negbin_nll_elementwise(mu, y, theta):
    mu = torch.clamp(mu, min=1e-6)
    theta = torch.clamp(theta, min=1e-4, max=1e5)
    return -(torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1.0)
             + theta * (torch.log(theta) - torch.log(theta + mu))
             + y * (torch.log(mu) - torch.log(theta + mu)))


def compute_losses(model: GeoIDDiagnosticPINN, data: FitData):
    out = model(data)
    S, I, R, D = out["S"], out["I"], out["R"], out["D"]
    inc = out["incidence_frac"]
    elem_nll = negbin_nll_elementwise(out["pred_cases"], data.cases, out["nb_theta"])
    data_loss = torch.mean(elem_nll[data.train_mask])

    rS = S[1:] - S[:-1] + inc[:-1]
    rI = I[1:] - I[:-1] - inc[:-1] + (GAMMA_WEEK + MU_WEEK) * I[:-1]
    rR = R[1:] - R[:-1] - GAMMA_WEEK * I[:-1]
    rD = D[1:] - D[:-1] - MU_WEEK * I[:-1]
    physics_loss = torch.mean(rS**2 + rI**2 + rR**2 + rD**2)

    first_cases = torch.clamp(
        data.cases[0] / torch.clamp(data.pop, min=1.0), min=1e-5, max=0.02
    )
    init_I = torch.clamp(3.0 * first_cases, min=1e-5, max=0.05)
    ic_loss = torch.mean(
        (I[0] - init_I)**2 + R[0]**2 + D[0]**2 + (S[0] - (1 - init_I))**2
    )
    prior_loss = torch.mean(((out["C"] - model.C0) / PRIOR_SCALE)**2)
    log_alpha = torch.log(out["alpha"])
    alpha_smooth = torch.mean((log_alpha[1:] - log_alpha[:-1])**2)
    state_smooth = torch.mean((I[1:] - I[:-1])**2 + (S[1:] - S[:-1])**2)
    external_l2 = (torch.mean(out["external_scale"]**2) if model.external_mode == "outside" else torch.tensor(0.0, device=DEVICE))

    total = (
        LOSS_WEIGHTS["data"] * data_loss
        + LOSS_WEIGHTS["physics"] * physics_loss
        + LOSS_WEIGHTS["ic"] * ic_loss
        + LOSS_WEIGHTS["prior"] * prior_loss
        + LOSS_WEIGHTS["alpha_smooth"] * alpha_smooth
        + LOSS_WEIGHTS["state_smooth"] * state_smooth
        + LOSS_WEIGHTS["external_l2"] * external_l2
    )
    return dict(
        total=total, data=data_loss, physics=physics_loss, ic=ic_loss,
        prior=prior_loss, alpha_smooth=alpha_smooth,
        state_smooth=state_smooth, external_l2=external_l2, elem_nll=elem_nll, out=out,
    )


from scipy.stats import nbinom


from geoid_pinn.significance import build_significance_ladder

ADAPTIVE_KEY = "circular_fdr_adaptive_diag"
adaptive_priors = {}
for split_number, split in enumerate(CFG["splits"]):
    train_mask, _, _ = make_split_masks(cases_df.index, split, CFG["test_weeks"])
    result = build_significance_ladder(
        cases=cases_df.to_numpy(dtype=np.float64),
        tests=tests_df.to_numpy(dtype=np.float64),
        population=pop_s.to_numpy(dtype=np.float64), train_mask=train_mask,
        labels=selected_fips, lags=(0, 1, 2),
        bootstrap_repetitions=100, circular_repetitions=199,
        stability_threshold=0.60, alpha=0.05, diagonal_mass=0.80,
        top_k=min(8, len(selected_fips) - 1), seed=42026 + split_number,
    )
    adaptive_priors[split] = result.adaptive_C0
    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "_", split)
    pd.DataFrame(
        result.adaptive_C0, index=selected_fips, columns=selected_fips
    ).to_csv(DATA_PRIOR_DIR / f"C0_{ADAPTIVE_KEY}_{safe_split}.csv")


THETA = 2.0
HUBER_BETA = 1.0
MIXED_RAW_FRACTION = 0.5
MIXED_WEIGHT_GRID = [0.0, 0.003, 0.01, 0.03, 0.1]
NLL_MARGIN = 0.25
VALIDATION_WEEKS = 6
MAX_INNER_ORIGINS = 2
MIN_INNER_TRAIN_WEEKS = 9
SELECTION_EPOCHS = {"smoke": 8, "pilot64": 80, "paper64": 120}[RUN_MODE]
INNER_BOOTSTRAP_REPS = {"smoke": 20, "pilot64": 60, "paper64": 100}[RUN_MODE]
INNER_CIRCULAR_REPS = {"smoke": 49, "pilot64": 99, "paper64": 199}[RUN_MODE]


def prediction_metrics(mu: np.ndarray, y: np.ndarray) -> dict:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    error = mu - y
    probability = THETA / (THETA + np.maximum(mu, 1e-8))
    lower = nbinom.ppf(0.05, THETA, probability)
    upper = nbinom.ppf(0.95, THETA, probability)
    return dict(
        mae=float(np.mean(np.abs(error))), mse=float(np.mean(error**2)),
        rmse=float(np.sqrt(np.mean(error**2))),
        wape=float(np.sum(np.abs(error)) / max(np.sum(y), 1.0)),
        log1p_mae=float(np.mean(np.abs(np.log1p(mu) - np.log1p(y)))),
        mean_bias=float(np.mean(error)),
        coverage90=float(np.mean((y >= lower) & (y <= upper))),
        interval_width90=float(np.mean(upper - lower)),
    )


def make_direct_data(split: str, train_end: int, eval_end: int, source_policy: str) -> FitData:
    base = make_fit_data("cases_tests_nb", split)
    train_mask = np.zeros(len(cases_df), dtype=bool)
    eval_mask = np.zeros(len(cases_df), dtype=bool)
    train_mask[:train_end] = True
    eval_mask[train_end:eval_end] = True
    signal = outside_signal.copy()
    if source_policy == "last_observed_frozen":
        signal[train_end:] = signal[train_end - 1]
    elif source_policy != "observed":
        raise ValueError(source_policy)
    return FitData(
        cases=base.cases, report_w=base.report_w, pop=base.pop,
        train_mask=to_tensor(train_mask, dtype=torch.bool),
        test_mask=to_tensor(eval_mask, dtype=torch.bool),
        time_x=base.time_x, external_signal=to_tensor(signal),
        weeks=base.weeks, parish_names=base.parish_names,
        parish_fips=base.parish_fips, obs_model=base.obs_model, split=base.split,
    )


def mixed_direct_loss(out, data: FitData):
    error = out["pred_cases"][data.train_mask] - data.cases[data.train_mask]
    parish_scale = torch.median(data.cases[data.train_mask], dim=0).values + 5.0
    raw_scale = torch.mean(data.cases[data.train_mask]) + 5.0
    parish_error = error / parish_scale[None, :]
    raw_error = error / raw_scale
    parish_huber = F.smooth_l1_loss(
        parish_error, torch.zeros_like(parish_error), beta=HUBER_BETA, reduction="mean"
    )
    raw_huber = F.smooth_l1_loss(
        raw_error, torch.zeros_like(raw_error), beta=HUBER_BETA, reduction="mean"
    )
    return (
        MIXED_RAW_FRACTION * raw_huber
        + (1.0 - MIXED_RAW_FRACTION) * parish_huber
    )


def train_direct_model(data: FitData, C0: np.ndarray, seed: int, weight: float, epochs: int):
    model = GeoIDDiagnosticPINN(
        P=len(selected_fips), T=len(cases_df), C0=C0, hidden=CFG["hidden"],
        theta_init=THETA, theta_fixed=THETA, external_mode="outside",
        use_parish_reporting_intercept=False, seed=seed,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    warmup = min(CFG["warmup_epochs"], max(2, epochs // 3))
    for epoch in range(epochs):
        model.C_logits.requires_grad_(epoch >= warmup)
        optimizer.zero_grad()
        losses = compute_losses(model, data)
        auxiliary = mixed_direct_loss(losses["out"], data)
        objective = losses["total"] + weight * auxiliary
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        losses = compute_losses(model, data)
        auxiliary = mixed_direct_loss(losses["out"], data)
    return model, losses, auxiliary


from geoid_pinn.significance import build_significance_ladder


def build_inner_prior(end: int, split: str, origin_number: int) -> np.ndarray:
    mask = np.zeros(len(cases_df), dtype=bool)
    mask[:end] = True
    result = build_significance_ladder(
        cases=cases_df.to_numpy(dtype=np.float64),
        tests=tests_df.to_numpy(dtype=np.float64),
        population=pop_s.to_numpy(dtype=np.float64), train_mask=mask,
        labels=selected_fips, lags=(0, 1, 2),
        bootstrap_repetitions=INNER_BOOTSTRAP_REPS,
        circular_repetitions=INNER_CIRCULAR_REPS,
        stability_threshold=0.60, alpha=0.05, diagonal_mass=0.80,
        top_k=min(8, len(selected_fips) - 1),
        seed=32026 + 101 * CFG["splits"].index(split) + origin_number,
    )
    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "_", split)
    pd.DataFrame(result.adaptive_C0, index=selected_fips, columns=selected_fips).to_csv(
        DATA_PRIOR_DIR / f"C0_inner_{safe_split}_end{end}.csv"
    )
    return result.adaptive_C0


def evaluate_direct(model, losses, data: FitData) -> dict:
    with torch.no_grad():
        mu = losses["out"]["pred_cases"][data.test_mask].cpu().numpy()
        y = data.cases[data.test_mask].cpu().numpy()
        nll = losses["elem_nll"][data.test_mask].mean().cpu().item()
    return dict(test_nll=float(nll), **prediction_metrics(mu, y))




MASK_HORIZON = 6
MAX_MASKED_ORIGINS = 3
MIN_MASK_PREFIX = 9
MASK_WEIGHT_GRID = [0.0, 0.01, 0.03, 0.1, 0.3]
IDENTITY_C0 = np.eye(len(selected_fips), dtype=np.float64)


def masked_origins(train_end: int) -> list[int]:
    return [
        train_end - MASK_HORIZON * offset
        for offset in range(1, MAX_MASKED_ORIGINS + 1)
        if train_end - MASK_HORIZON * offset >= MIN_MASK_PREFIX
    ]


def masked_origin_loss(out, data: FitData, train_end: int):
    values = []
    for origin in masked_origins(train_end):
        target = slice(origin, origin + MASK_HORIZON)
        error = out["pred_cases"][target] - data.cases[target]
        parish_scale = torch.median(data.cases[:origin], dim=0).values + 5.0
        raw_scale = torch.mean(data.cases[:origin]) + 5.0
        parish_error = error / parish_scale[None, :]
        raw_error = error / raw_scale
        values.append(
            (1.0 - MIXED_RAW_FRACTION) * F.smooth_l1_loss(
                parish_error, torch.zeros_like(parish_error),
                beta=HUBER_BETA, reduction="mean",
            )
            + MIXED_RAW_FRACTION * F.smooth_l1_loss(
                raw_error, torch.zeros_like(raw_error),
                beta=HUBER_BETA, reduction="mean",
            )
        )
    return torch.stack(values).mean() if values else torch.zeros((), device=DEVICE)


def train_variant(
    data: FitData, C0: np.ndarray, seed: int, mixed_weight: float,
    mask_weight: float, epochs: int, external_mode: str,
    fixed_identity: bool = False,
):
    train_end = int(data.train_mask.sum().item())
    model = GeoIDDiagnosticPINN(
        P=len(selected_fips), T=len(cases_df), C0=C0, hidden=CFG["hidden"],
        theta_init=THETA, theta_fixed=THETA, external_mode=external_mode,
        use_parish_reporting_intercept=False, seed=seed,
    ).to(DEVICE)
    if fixed_identity:
        model.C_logits.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=3e-3
    )
    warmup = min(CFG["warmup_epochs"], max(2, epochs // 3))
    for epoch in range(epochs):
        if not fixed_identity:
            model.C_logits.requires_grad_(epoch >= warmup)
        optimizer.zero_grad()
        losses = compute_losses(model, data)
        ordinary_mixed = mixed_direct_loss(losses["out"], data)
        masked = masked_origin_loss(losses["out"], data, train_end)
        objective = (
            losses["total"] + mixed_weight * ordinary_mixed + mask_weight * masked
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        losses = compute_losses(model, data)
        ordinary_mixed = mixed_direct_loss(losses["out"], data)
        masked = masked_origin_loss(losses["out"], data, train_end)
    return model, losses, ordinary_mixed, masked



# -----------------------------------------------------------------------------
# Updated Top-15 prior experiment
# -----------------------------------------------------------------------------
# Everything except C0 is held fixed at the completed Combined + masked-origin
# design. In particular, the mixed/masked weights are reused rather than tuned
# separately for each prior.

MATRIX_DIR = OUTPUT_DIR / "learned_matrices"
MATRIX_DIR.mkdir(parents=True, exist_ok=True)

unified_run_summary_path = (
    RESULTS_ROOT / "real_data_geoid_top15_theta2_updated_unified_outputs"
    / "run_summary.json"
)
if not unified_run_summary_path.exists():
    raise FileNotFoundError(
        "Run real_data_geoid_top15_theta2_updated_unified_experiment.py first; "
        "its nested-selected Combined + masked-origin weights are required."
    )
with open(unified_run_summary_path, "r", encoding="utf-8") as handle:
    unified_run_summary = json.load(handle)
MAIN_WEIGHTS = {
    split: tuple(float(value) for value in weights)
    for split, weights in unified_run_summary["observed_masked_weights"].items()
}
missing_weight_splits = sorted(set(CFG["splits"]) - set(MAIN_WEIGHTS))
if missing_weight_splits:
    raise ValueError(f"Missing main-model weights for splits: {missing_weight_splits}")

# Add the directly constructed hybrid LODES matrix used by the original Top-15
# experiment when it is present in the public prior bundle.
hybrid_lodes = load_prior_csv("C0_hybrid_lodes_diag80_2019_JT01.csv")
if hybrid_lodes is not None:
    priors["hybrid_lodes_d80"] = hybrid_lodes

PRIOR_SPECS = [
    dict(key="adaptive_pwccf", label="Adaptive PWCCF", kind="adaptive"),
    dict(key="identity_fixed", label="Identity (fixed)", kind="identity"),
    dict(key="distance_d80", label="Distance d80", kind="static"),
    dict(key="adjacency_d80", label="Adjacency d80", kind="static"),
    dict(key="hybrid_d80", label="Geographic hybrid d80", kind="static"),
    dict(key="lodes_work_d80", label="LODES work d80", kind="static"),
    dict(key="lodes_home_d80", label="LODES home d80", kind="static"),
    dict(key="lodes_sym_d80", label="LODES symmetric d80", kind="static"),
    dict(key="hybrid_lodes_d80", label="Hybrid LODES d80", kind="static"),
    dict(key="wrong_d80", label="Anti-distance d80", kind="static"),
    dict(
        key="lodes_home_permuted_d80",
        label="Permuted LODES d80",
        kind="static",
    ),
    dict(key="no_prior", label="No prior", kind="no_prior"),
]
available_specs = []
for spec in PRIOR_SPECS:
    if spec["kind"] in {"adaptive", "identity", "no_prior"} or spec["key"] in priors:
        available_specs.append(spec)
    else:
        print(f"Skipping unavailable prior: {spec['key']}")
PRIOR_SPECS = available_specs
PRIOR_LABELS = {spec["key"]: spec["label"] for spec in PRIOR_SPECS}


def prior_for_split(spec: dict, split: str) -> np.ndarray:
    if spec["kind"] == "adaptive":
        return adaptive_priors[split]
    if spec["kind"] == "identity":
        return priors["identity"]
    return priors[spec["key"]]


def train_prior_variant(
    data: FitData,
    C0: np.ndarray,
    seed: int,
    mixed_weight: float,
    mask_weight: float,
    epochs: int,
    kind: str,
):
    train_end = int(data.train_mask.sum().item())
    model = GeoIDDiagnosticPINN(
        P=len(selected_fips),
        T=len(cases_df),
        C0=C0,
        hidden=CFG["hidden"],
        theta_init=THETA,
        theta_fixed=THETA,
        external_mode="outside",
        use_parish_reporting_intercept=False,
        seed=seed,
    ).to(DEVICE)

    fixed_identity = kind == "identity"
    no_prior = kind == "no_prior"
    if fixed_identity:
        model.C_logits.requires_grad_(False)
    elif no_prior:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1729)
        with torch.no_grad():
            model.C_logits.copy_(torch.randn(model.C_logits.shape, generator=generator) * 0.05)

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-3,
    )
    warmup = min(CFG["warmup_epochs"], max(2, epochs // 3))
    for epoch in range(epochs):
        if not fixed_identity:
            model.C_logits.requires_grad_(epoch >= warmup)
        optimizer.zero_grad()
        losses = compute_losses(model, data)
        ordinary_mixed = mixed_direct_loss(losses["out"], data)
        masked = masked_origin_loss(losses["out"], data, train_end)
        objective = losses["total"]
        if no_prior:
            objective = objective - LOSS_WEIGHTS["prior"] * losses["prior"]
        objective = objective + mixed_weight * ordinary_mixed + mask_weight * masked
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        losses = compute_losses(model, data)
        ordinary_mixed = mixed_direct_loss(losses["out"], data)
        masked = masked_origin_loss(losses["out"], data, train_end)
    return model, losses, ordinary_mixed, masked


def matrix_diagnostics(C_hat: np.ndarray, C0: np.ndarray, kind: str) -> dict:
    offdiag = C_hat.copy()
    np.fill_diagonal(offdiag, 0.0)
    effective_sources = 1.0 / np.maximum(np.sum(C_hat**2, axis=1), EPS)
    result = dict(
        c_diag_mean=float(np.mean(np.diag(C_hat))),
        c_max_offdiagonal=float(np.max(offdiag)),
        c_effective_sources_mean=float(np.mean(effective_sources)),
        c_fro_to_c0=float(np.linalg.norm(C_hat - C0)),
    )
    if kind == "no_prior":
        result["c_kl_to_c0"] = np.nan
    else:
        result["c_kl_to_c0"] = float(
            np.sum(C_hat * (np.log(C_hat + EPS) - np.log(C0 + EPS))) / C_hat.shape[0]
        )
    return result



# -----------------------------------------------------------------------------
# Parish-window diagnostic: pure PINN versus adjacency Geo-PINN
# -----------------------------------------------------------------------------
# Identity and adjacency use the same observed outside source, theta, direct-NN
# prediction, mixed loss, masked-origin loss, epochs, splits, and seeds. Only C0
# and whether cross-parish coupling is permitted differ.

WINDOW_SPECS = [
    dict(key="identity_fixed", label="Pure PINN (identity C)", kind="identity"),
    dict(key="adjacency_d80", label="Geo-PINN (adjacency)", kind="static"),
]

prediction_rows = []
run_rows = []
for split in CFG["splits"]:
    base = make_fit_data("cases_tests_nb", split)
    train_end = int(base.train_mask.sum().item())
    test_indices = np.flatnonzero(base.test_mask.cpu().numpy())
    test_end = int(test_indices[-1] + 1)
    mixed_weight, mask_weight = MAIN_WEIGHTS[split]
    data = make_direct_data(split, train_end, test_end, "observed")
    for spec in WINDOW_SPECS:
        C0 = priors["identity"] if spec["kind"] == "identity" else priors[spec["key"]]
        for seed in CFG["seeds"]:
            print(
                f"WINDOW FIT {split} seed={seed} {spec['key']} "
                f"mixed={mixed_weight:g} mask={mask_weight:g}"
            )
            model, losses, ordinary_mixed, masked = train_prior_variant(
                data=data,
                C0=C0,
                seed=seed,
                mixed_weight=mixed_weight,
                mask_weight=mask_weight,
                epochs=CFG["epochs"],
                kind=spec["kind"],
            )
            metrics = evaluate_direct(model, losses, data)
            run_rows.append(
                dict(
                    model=spec["key"], model_label=spec["label"],
                    split=split, seed=seed, mixed_weight=mixed_weight,
                    mask_weight=mask_weight, **metrics,
                )
            )
            mu = losses["out"]["pred_cases"].cpu().numpy()
            y = data.cases.cpu().numpy()
            elem_nll = losses["elem_nll"].cpu().numpy()
            for time_index in test_indices:
                for parish_index, (fips, parish) in enumerate(zip(selected_fips, selected_names)):
                    prediction_rows.append(
                        dict(
                            model=spec["key"], model_label=spec["label"],
                            split=split, seed=seed,
                            week=pd.Timestamp(cases_df.index[time_index]),
                            parish_fips=fips, parish=parish,
                            actual=float(y[time_index, parish_index]),
                            predicted=float(mu[time_index, parish_index]),
                            nll=float(elem_nll[time_index, parish_index]),
                        )
                    )

predictions = pd.DataFrame(prediction_rows)
runs = pd.DataFrame(run_rows)
predictions.to_csv(TABLE_DIR / "pure_vs_geo_predictions_by_seed.csv", index=False)
runs.to_csv(TABLE_DIR / "pure_vs_geo_metrics_by_run.csv", index=False)

run_summary = (
    runs.groupby(["model", "model_label"], as_index=False)
    .agg(
        n=("seed", "count"), test_nll_mean=("test_nll", "mean"),
        test_nll_sd=("test_nll", "std"), mae_mean=("mae", "mean"),
        mse_mean=("mse", "mean"), rmse_mean=("rmse", "mean"),
        coverage90_mean=("coverage90", "mean"),
    )
)
run_summary.to_csv(TABLE_DIR / "pure_vs_geo_overall_comparison.csv", index=False)

# A window is one parish in one six-week outer forecast origin. Metrics are
# computed within each seed and then averaged, preserving the paired design.
window_seed = (
    predictions.assign(
        absolute_error=lambda frame: np.abs(frame.predicted - frame.actual),
        squared_error=lambda frame: (frame.predicted - frame.actual) ** 2,
    )
    .groupby(
        ["model", "model_label", "split", "seed", "parish_fips", "parish"],
        as_index=False,
    )
    .agg(
        mae=("absolute_error", "mean"), mse=("squared_error", "mean"),
        nll=("nll", "mean"), mean_actual=("actual", "mean"),
        total_actual=("actual", "sum"),
    )
)
window_seed.to_csv(TABLE_DIR / "parish_window_metrics_by_seed.csv", index=False)

window_model = (
    window_seed.groupby(
        ["model", "model_label", "split", "parish_fips", "parish"],
        as_index=False,
    )
    .agg(
        mae=("mae", "mean"), mse=("mse", "mean"), nll=("nll", "mean"),
        mean_actual=("mean_actual", "mean"), total_actual=("total_actual", "mean"),
    )
)
pure = window_model[window_model.model.eq("identity_fixed")].drop(
    columns=["model", "model_label"]
)
geo = window_model[window_model.model.eq("adjacency_d80")].drop(
    columns=["model", "model_label"]
)
window_comparison = pure.merge(
    geo,
    on=["split", "parish_fips", "parish"],
    suffixes=("_pure", "_geo"),
    validate="one_to_one",
)
for metric in ["mae", "mse", "nll"]:
    window_comparison[f"{metric}_reduction"] = (
        window_comparison[f"{metric}_pure"] - window_comparison[f"{metric}_geo"]
    )
    window_comparison[f"{metric}_relative_reduction"] = (
        window_comparison[f"{metric}_reduction"]
        / np.maximum(np.abs(window_comparison[f"{metric}_pure"]), 1e-8)
    )
window_comparison["mae_improved"] = window_comparison.mae_reduction > 0
window_comparison["mse_improved"] = window_comparison.mse_reduction > 0
window_comparison["nll_improved"] = window_comparison.nll_reduction > 0
window_comparison["mae_and_mse_improved"] = (
    window_comparison.mae_improved & window_comparison.mse_improved
)
window_comparison = window_comparison.sort_values("mse_reduction", ascending=False)
window_comparison.to_csv(TABLE_DIR / "all_60_parish_window_comparisons.csv", index=False)

eligible = window_comparison[window_comparison.mae_and_mse_improved]
if len(eligible) < 10:
    raise RuntimeError(f"Only {len(eligible)} windows improve both MAE and MSE")
top10 = eligible.head(10).copy()
top10.insert(0, "improvement_rank", np.arange(1, 11))
top10.to_csv(TABLE_DIR / "top10_geo_improved_windows.csv", index=False)

ensemble = (
    predictions.groupby(
        ["model", "model_label", "split", "week", "parish_fips", "parish"],
        as_index=False,
    )
    .agg(
        actual=("actual", "first"), prediction_mean=("predicted", "mean"),
        prediction_min=("predicted", "min"), prediction_max=("predicted", "max"),
        prediction_sd=("predicted", "std"),
    )
)
ensemble.to_csv(TABLE_DIR / "pure_vs_geo_ensemble_predictions.csv", index=False)

pure_color = "#6f7782"
geo_color = "#b53a32"
actual_color = "#111111"
fig, axes = plt.subplots(5, 2, figsize=(14, 19), sharex=False)
for ax, (_, selected) in zip(axes.flat, top10.iterrows()):
    subset = ensemble[
        ensemble.split.eq(selected.split)
        & ensemble.parish_fips.eq(selected.parish_fips)
    ]
    for model, color, label in [
        ("identity_fixed", pure_color, "Pure PINN"),
        ("adjacency_d80", geo_color, "Geo-PINN"),
    ]:
        line = subset[subset.model.eq(model)].sort_values("week")
        x = pd.to_datetime(line.week).to_numpy()
        ax.plot(x, line.prediction_mean, color=color, linewidth=2.1, marker="o", label=label)
        ax.fill_between(
            x, line.prediction_min.to_numpy(), line.prediction_max.to_numpy(),
            color=color, alpha=0.12,
        )
    actual = subset[subset.model.eq("identity_fixed")].sort_values("week")
    ax.plot(
        pd.to_datetime(actual.week), actual.actual,
        color=actual_color, linewidth=2.4, marker="s", label="Observed",
    )
    ax.set_title(
        f"#{int(selected.improvement_rank)} {selected.parish} | "
        f"{selected.split.replace('roll_', '')}\n"
        f"MAE {selected.mae_pure:.1f} -> {selected.mae_geo:.1f}; "
        f"MSE {selected.mse_pure:,.0f} -> {selected.mse_geo:,.0f}",
        fontsize=10,
    )
    ax.grid(alpha=0.22)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Weekly cases")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.suptitle(
    "Ten parish-windows with the largest adjacency Geo-PINN MSE reduction",
    y=0.995, fontsize=15,
)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(FIG_DIR / "top10_geo_improved_prediction_windows.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Transparent overview of all 60 windows, with the selected ten highlighted.
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
top_keys = set(zip(top10.split, top10.parish_fips))
highlight = np.array([
    (split, fips) in top_keys
    for split, fips in zip(window_comparison.split, window_comparison.parish_fips)
])
for ax, metric, title in [
    (axes[0], "mae", "Parish-window MAE"),
    (axes[1], "mse", "Parish-window MSE"),
]:
    x = window_comparison[f"{metric}_pure"]
    y = window_comparison[f"{metric}_geo"]
    limit = float(max(x.max(), y.max()) * 1.03)
    ax.scatter(x[~highlight], y[~highlight], color="#76838a", alpha=0.65, s=35)
    ax.scatter(x[highlight], y[highlight], color=geo_color, s=55, label="Selected top 10")
    ax.plot([0, limit], [0, limit], linestyle="--", color="#222222", linewidth=1)
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_xlabel("Pure PINN")
    ax.set_ylabel("Adjacency Geo-PINN")
    ax.set_title(title)
    ax.grid(alpha=0.2)
axes[1].legend(frameon=False)
fig.suptitle("All 60 parish-windows; points below the diagonal favor Geo-PINN")
fig.tight_layout()
fig.savefig(FIG_DIR / "all_windows_pure_vs_geo_scatter.png", dpi=220, bbox_inches="tight")
plt.close(fig)

identity_row = run_summary.set_index("model").loc["identity_fixed"]
geo_row = run_summary.set_index("model").loc["adjacency_d80"]
overall_effect = {
    metric: dict(
        pure=float(identity_row[f"{metric}_mean"]),
        geo=float(geo_row[f"{metric}_mean"]),
        absolute_reduction=float(identity_row[f"{metric}_mean"] - geo_row[f"{metric}_mean"]),
        relative_reduction=float(
            (identity_row[f"{metric}_mean"] - geo_row[f"{metric}_mean"])
            / identity_row[f"{metric}_mean"]
        ),
    )
    for metric in ["test_nll", "mae", "mse"]
}
diagnostic_summary = dict(
    design="post_hoc_parish_window_geo_improvement_diagnostic",
    window_definition="one parish x one six-week outer forecast origin, averaged over three seeds",
    n_windows=int(len(window_comparison)),
    n_mae_improved=int(window_comparison.mae_improved.sum()),
    n_mse_improved=int(window_comparison.mse_improved.sum()),
    n_nll_improved=int(window_comparison.nll_improved.sum()),
    n_mae_and_mse_improved=int(window_comparison.mae_and_mse_improved.sum()),
    selection_rule="MAE and MSE both improve; rank by absolute MSE reduction",
    comparison="identity-fixed pure PINN versus adjacency d80 Geo-PINN",
    controlled_factors=[
        "same observed outside source", "same Combined + masked-origin training",
        "same theta=2", "same mixed/masked weights", "same splits and seeds",
    ],
    overall_effect=overall_effect,
    post_hoc_selection_warning=(
        "The top ten are selected on held-out outcomes for diagnosis and visualization; "
        "they are not an unbiased estimate of prospective geo-information benefit."
    ),
)
with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as handle:
    json.dump(diagnostic_summary, handle, indent=2)

print("\nOverall pure PINN versus adjacency Geo-PINN:")
print(run_summary.to_string(index=False))
print("\nWindow improvement counts:")
print(json.dumps({
    "total": diagnostic_summary["n_windows"],
    "MAE": diagnostic_summary["n_mae_improved"],
    "MSE": diagnostic_summary["n_mse_improved"],
    "NLL": diagnostic_summary["n_nll_improved"],
    "MAE_and_MSE": diagnostic_summary["n_mae_and_mse_improved"],
}, indent=2))
print("\nTop ten geo-improved parish-windows:")
print(top10[[
    "improvement_rank", "parish", "split", "mae_pure", "mae_geo",
    "mse_pure", "mse_geo", "mae_relative_reduction", "mse_relative_reduction",
]].to_string(index=False))
print("\nSaved outputs to", OUTPUT_DIR)
