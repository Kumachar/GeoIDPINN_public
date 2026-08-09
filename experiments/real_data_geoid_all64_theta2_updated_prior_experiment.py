from __future__ import annotations

import os
import re
import sys
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
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

OUTPUT_DIR = RESULTS_ROOT / "real_data_geoid_all64_theta2_updated_prior_outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MATRIX_DIR = OUTPUT_DIR / "learned_matrices"
PROFILE_DIR = OUTPUT_DIR / "profile_loss"
BASELINE_DIR = OUTPUT_DIR / "baselines"
DATA_PRIOR_DIR = OUTPUT_DIR / "data_derived_priors"
for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, MATRIX_DIR, PROFILE_DIR, BASELINE_DIR, DATA_PRIOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# All-64 county experiment.
# Scientific goal: test whether a closed Louisiana system reduces the need for an outside reservoir.
# Main model: cases_tests_nb + time-varying alpha + full 64-county C matrix.
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
        top_n=64, seeds=[0, 1, 2], epochs=450, warmup_epochs=120, hidden=64,
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
print("Weekly county rows:", weekly_df.shape)
print("Counties:", weekly_df["parish_fips"].nunique(), "Weeks:", weekly_df["week"].nunique())
weekly_df.head()

def build_case_test_matrices(weekly: pd.DataFrame, fips_order: List[str], start: pd.Timestamp, end: pd.Timestamp):
    sub = weekly[(weekly["week"] >= start) & (weekly["week"] <= end)].copy()
    all_weeks = pd.Index(sorted(sub["week"].unique()), name="week")
    cases = (sub.pivot_table(index="week", columns="parish_fips", values="cases", aggfunc="sum")
               .reindex(index=all_weeks, columns=fips_order).fillna(0.0))
    tests = (sub.pivot_table(index="week", columns="parish_fips", values="tests", aggfunc="sum")
               .reindex(index=all_weeks, columns=fips_order).fillna(0.0))
    return cases, tests

# Select counties and prior directory.
# For top_n=64, use the full 64-county prior matrices under outputs/.
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
print("Selected counties:", len(selected_fips))
selected_display = selected_meta.rename(
    columns={"parish_fips": "county_fips", "parish": "county_name"}
)
display(
    selected_display[
        ["county_fips", "county_name", "total_cases", "total_tests", "cenpop2020_population"]
    ].head(20)
)
cases_df, tests_df = build_case_test_matrices(weekly_df, selected_fips, START_DATE, END_DATE)
pop_s = selected_meta.set_index("parish_fips").loc[selected_fips, "cenpop2020_population"].astype(float)

# Observed epidemic pressure from Louisiana's three land-border states.
CDC_OUTSIDE_FILE = PROJECT_ROOT / "data" / "external" / "cdc_neighbor_states_weekly_cases_2020.csv"
if not CDC_OUTSIDE_FILE.exists():
    raise FileNotFoundError(
        f"Missing {CDC_OUTSIDE_FILE}. Run the documented CDC Socrata download first."
    )

outside_states = ["TX", "AR", "MS"]
outside_raw = pd.read_csv(CDC_OUTSIDE_FILE)
outside_raw["date"] = pd.to_datetime(outside_raw["date"]) + pd.Timedelta(days=1)
outside_raw["new_cases"] = (
    pd.to_numeric(outside_raw["new_cases"], errors="coerce").fillna(0).clip(lower=0)
)
outside_weekly = outside_raw.groupby("date")["new_cases"].sum().sort_index()
outside_weekly = outside_weekly.reindex(cases_df.index).interpolate().ffill().bfill()
outside_smoothed = outside_weekly.rolling(3, min_periods=1).mean()
outside_signal = outside_smoothed / max(float(outside_smoothed.mean()), 1e-8)
outside_signal = outside_signal.clip(lower=0, upper=5).to_numpy(dtype=np.float32)
outside_fips = outside_states
outside_pop = float("nan")

outside_signal_df = pd.DataFrame({
    "week": cases_df.index,
    "neighbor_new_cases": outside_weekly.to_numpy(dtype=float),
    "neighbor_new_cases_smoothed": outside_smoothed.to_numpy(dtype=float),
    "outside_signal": outside_signal,
})
outside_signal_df.to_csv(DATA_PRIOR_DIR / "cdc_neighbor_state_outside_signal.csv", index=False)
print("Case matrix:", cases_df.shape, cases_df.index.min(), cases_df.index.max())
print("Outside states:", outside_states)
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
ax.set_title("Total weekly cases in selected Louisiana counties")
ax.set_ylabel("Cases")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "selected_total_cases.png", dpi=180, bbox_inches="tight")
plt.close(fig)

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


ADAPTIVE_KEY = "circular_fdr_adaptive_diag"
ADAPTIVE_SOURCE_DIR = (
    RESULTS_ROOT / "real_data_geoid_all64_pwccf_significance_outputs"
    / "data_derived_priors"
)
adaptive_priors = {}
for split in CFG["splits"]:
    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "_", split)
    prior_path = ADAPTIVE_SOURCE_DIR / f"C0_{ADAPTIVE_KEY}_{safe_split}.csv"
    if not prior_path.exists():
        raise FileNotFoundError(f"Missing adaptive prior: {prior_path}")
    prior_frame = normalize_fips_index(pd.read_csv(prior_path, index_col=0))
    adaptive_priors[split] = row_normalize(
        prior_frame.loc[selected_fips, selected_fips].to_numpy(dtype=np.float64)
    )
    pd.DataFrame(
        adaptive_priors[split], index=selected_fips, columns=selected_fips
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
# Updated all-64 county network comparison.
# Everything except C0 is held fixed at the completed Combined + masked-origin
# design. Mixed and masked-origin weights are reused, not retuned by network.
# -----------------------------------------------------------------------------

MATRIX_DIR = OUTPUT_DIR / "learned_matrices"
MATRIX_DIR.mkdir(parents=True, exist_ok=True)

mixed_weights_path = (
    RESULTS_ROOT / "real_data_geoid_all64_theta2_direct_nn_observed_combined_outputs"
    / "combined_interpretation.json"
)
masked_weights_path = (
    RESULTS_ROOT / "real_data_geoid_all64_theta2_combined_masked_origin_outputs"
    / "masked_origin_interpretation.json"
)
with mixed_weights_path.open(encoding="utf-8") as handle:
    mixed_weights = {
        key: float(value) for key, value in json.load(handle)["selected_weights"].items()
    }
with masked_weights_path.open(encoding="utf-8") as handle:
    masked_weights = {
        key: float(value)
        for key, value in json.load(handle)["selected_mask_weights"].items()
    }
MAIN_WEIGHTS = {
    split: (mixed_weights[split], masked_weights[split]) for split in CFG["splits"]
}

PRIOR_SPECS = [
    dict(key="adaptive_pwccf", label="Data-Inferred Network", kind="adaptive"),
    dict(key="adjacency_d80", label="Neighbor Network", kind="static"),
    dict(key="identity_fixed", label="Identity (fixed)", kind="identity"),
    dict(key="distance_d80", label="Distance", kind="static"),
    dict(key="hybrid_d80", label="Geographic Hybrid", kind="static"),
    dict(key="lodes_work_d80", label="LODES Work", kind="static"),
    dict(key="lodes_home_d80", label="LODES Home", kind="static"),
    dict(key="lodes_sym_d80", label="LODES Symmetric", kind="static"),
    dict(key="wrong_d80", label="Anti-Distance", kind="static"),
    dict(
        key="lodes_home_permuted_d80",
        label="Permuted LODES",
        kind="static",
    ),
    dict(key="no_prior", label="No Prior", kind="no_prior"),
]
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


print("Prior experiment variants:", [spec["key"] for spec in PRIOR_SPECS])
print("Frozen Combined + masked-origin weights:", MAIN_WEIGHTS)

outer_rows = []
for split in CFG["splits"]:
    base = make_fit_data("cases_tests_nb", split)
    train_end = int(base.train_mask.sum().item())
    test_end = int(np.flatnonzero(base.test_mask.cpu().numpy())[-1] + 1)
    mixed_weight, mask_weight = MAIN_WEIGHTS[split]
    data = make_direct_data(split, train_end, test_end, "observed")
    for spec in PRIOR_SPECS:
        C0 = prior_for_split(spec, split)
        for seed in CFG["seeds"]:
            print(
                f"PRIOR OUTER {split} seed={seed} {spec['key']} "
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
            C_hat = losses["out"]["C"].cpu().numpy()
            diagnostics = matrix_diagnostics(C_hat, C0, spec["kind"])
            safe_split = split.replace("roll_", "").replace("-", "")
            pd.DataFrame(C_hat, index=selected_fips, columns=selected_fips).to_csv(
                MATRIX_DIR / f"C_hat_{spec['key']}_{safe_split}_seed{seed}.csv"
            )
            outer_rows.append(
                dict(
                    prior=spec["key"],
                    prior_label=spec["label"],
                    prior_kind=spec["kind"],
                    split=split,
                    seed=seed,
                    mixed_weight=mixed_weight,
                    mask_weight=mask_weight,
                    theta=THETA,
                    source_policy="observed",
                    external_mode="outside",
                    prediction_method="direct_neural_network_output",
                    c_trainable=spec["kind"] != "identity",
                    prior_penalty_active=spec["kind"] not in {"identity", "no_prior"},
                    train_nll=float(losses["elem_nll"][data.train_mask].mean().cpu()),
                    internal_mixed_loss=float(ordinary_mixed.cpu()),
                    internal_masked_loss=float(masked.cpu()),
                    physics_loss=float(losses["physics"].cpu()),
                    **diagnostics,
                    **metrics,
                )
            )

results_df = pd.DataFrame(outer_rows)
results_df.to_csv(TABLE_DIR / "all64_updated_prior_metrics_by_run.csv", index=False)

summary_df = (
    results_df.groupby(["prior", "prior_label"], as_index=False)
    .agg(
        n=("seed", "count"),
        test_nll_mean=("test_nll", "mean"),
        test_nll_sd=("test_nll", "std"),
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
        mse_mean=("mse", "mean"),
        mse_sd=("mse", "std"),
        rmse_mean=("rmse", "mean"),
        coverage90_mean=("coverage90", "mean"),
        c_diag_mean=("c_diag_mean", "mean"),
        c_effective_sources_mean=("c_effective_sources_mean", "mean"),
        c_fro_to_c0_mean=("c_fro_to_c0", "mean"),
    )
    .sort_values("test_nll_mean")
)
summary_df.to_csv(TABLE_DIR / "all64_updated_prior_comparison.csv", index=False)

split_summary = (
    results_df.groupby(["prior", "prior_label", "split"], as_index=False)
    .agg(
        n=("seed", "count"),
        test_nll=("test_nll", "mean"),
        mae=("mae", "mean"),
        mse=("mse", "mean"),
        c_diag_mean=("c_diag_mean", "mean"),
    )
)
split_summary.to_csv(TABLE_DIR / "all64_updated_prior_by_split.csv", index=False)

neighbor = results_df[results_df.prior.eq("adjacency_d80")]
data_inferred = results_df[results_df.prior.eq("adaptive_pwccf")]
paired = neighbor.merge(
    data_inferred,
    on=["split", "seed"],
    suffixes=("_neighbor", "_data_inferred"),
    validate="one_to_one",
)
for metric in ["test_nll", "mae", "mse"]:
    paired[f"delta_{metric}_data_inferred_minus_neighbor"] = (
        paired[f"{metric}_data_inferred"] - paired[f"{metric}_neighbor"]
    )
paired.to_csv(TABLE_DIR / "all64_neighbor_vs_data_inferred_paired_by_run.csv", index=False)

run_summary = dict(
    design="all64_updated_prior_sensitivity",
    unit_name="County",
    top_n=CFG["top_n"],
    outside_definition="Texas, Arkansas, and Mississippi; observed during evaluation",
    theta_fixed=THETA,
    prediction_method="direct neural network output; no mechanistic rollout",
    training="Combined + masked-origin",
    mixed_masked_weights={key: list(value) for key, value in MAIN_WEIGHTS.items()},
    weights_reused_for_all_priors=True,
    splits=CFG["splits"],
    seeds=CFG["seeds"],
    priors={spec["key"]: spec["label"] for spec in PRIOR_SPECS},
    network_prior_is_only_model_factor=True,
)
with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as handle:
    json.dump(run_summary, handle, indent=2)

print("\nAll-64 county prior comparison:")
print(
    summary_df[[
        "prior_label", "test_nll_mean", "mae_mean", "mse_mean",
        "c_diag_mean", "c_effective_sources_mean",
    ]].to_string(index=False)
)
print("\nSaved outputs to", OUTPUT_DIR)
