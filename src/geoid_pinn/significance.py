"""Significance-aware PWCCF priors with circular-shift null tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .priors import (
    EPS,
    PWCCFPriorResult,
    _estimate_ar1,
    _make_prior,
    _safe_corr,
    build_nb_residual_pwccf_prior,
    permute_prior_offdiagonal,
)


@dataclass
class PWCCFSignificanceResult:
    baseline_C0: np.ndarray
    circular_C0: np.ndarray
    fdr_C0: np.ndarray
    adaptive_C0: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    significant_unadjusted: np.ndarray
    significant_fdr: np.ndarray
    edge_table: pd.DataFrame
    diagnostics: Dict[str, float]
    base_result: PWCCFPriorResult


def _column_correlations(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Correlate every matrix column with one vector."""
    matrix = np.asarray(matrix, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    matrix_centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    vector_centered = vector - np.mean(vector)
    numerator = matrix_centered.T @ vector_centered
    denominator = np.sqrt(
        np.sum(matrix_centered**2, axis=0) * np.sum(vector_centered**2)
    )
    result = np.zeros(matrix.shape[1], dtype=np.float64)
    valid = denominator > EPS
    result[valid] = numerator[valid] / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def circular_shift_pvalues(
    residuals: np.ndarray,
    observed_scores: np.ndarray,
    lags: Sequence[int],
    repetitions: int = 500,
    candidate_mask: Optional[np.ndarray] = None,
    seed: int = 2026,
) -> np.ndarray:
    """Edgewise max-lag p-values under a source circular-shift null.

    Each null draw circularly shifts the prewhitened source series while the
    recipient stays fixed. This preserves the source's serial structure and
    applies the same maximum-over-lags selection used for the observed score.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    observed_scores = np.asarray(observed_scores, dtype=np.float64)
    n_time, n_parish = residuals.shape
    if observed_scores.shape != (n_parish, n_parish):
        raise ValueError("observed_scores has the wrong shape")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    if candidate_mask is None:
        candidate_mask = np.ones((n_parish, n_parish), dtype=bool)
        np.fill_diagonal(candidate_mask, False)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool).copy()
        np.fill_diagonal(candidate_mask, False)

    lags = tuple(sorted({int(lag) for lag in lags if int(lag) >= 0}))
    p_value = np.ones((n_parish, n_parish), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for source in range(n_parish):
        phi = _estimate_ar1(residuals[:, source])
        source_white = residuals[1:, source] - phi * residuals[:-1, source]
        recipient_white = residuals[1:] - phi * residuals[:-1]
        eligible = candidate_mask[:, source] & (observed_scores[:, source] > 0)
        if not np.any(eligible):
            continue

        exceedances = np.zeros(n_parish, dtype=np.int64)
        max_lag = max(lags)
        min_shift = min(max_lag + 2, max(len(source_white) // 3, 1))
        possible_shifts = np.arange(1, len(source_white), dtype=np.int64)
        if len(source_white) > 2 * min_shift:
            possible_shifts = possible_shifts[
                (possible_shifts >= min_shift)
                & (possible_shifts <= len(source_white) - min_shift)
            ]

        for _ in range(repetitions):
            shifted = np.roll(source_white, int(rng.choice(possible_shifts)))
            null_max = np.full(n_parish, -1.0, dtype=np.float64)
            for lag in lags:
                if lag == 0:
                    correlations = _column_correlations(recipient_white, shifted)
                elif lag < len(shifted) - 5:
                    correlations = _column_correlations(
                        recipient_white[lag:], shifted[:-lag]
                    )
                else:
                    correlations = np.zeros(n_parish, dtype=np.float64)
                null_max = np.maximum(null_max, correlations)
            exceedances += eligible & (
                null_max >= observed_scores[:, source] - 1e-12
            )
        p_value[eligible, source] = (
            1.0 + exceedances[eligible]
        ) / float(repetitions + 1)

    return p_value


def benjamini_hochberg(
    p_value: np.ndarray,
    eligible_mask: np.ndarray,
) -> np.ndarray:
    """Return BH-adjusted q-values over all eligible directed edges."""
    p_value = np.asarray(p_value, dtype=np.float64)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    flat_p = p_value[eligible_mask]
    q_value = np.ones_like(p_value, dtype=np.float64)
    if flat_p.size == 0:
        return q_value

    order = np.argsort(flat_p)
    ranked = flat_p[order]
    m = float(len(ranked))
    adjusted = ranked * m / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    unsorted = np.empty_like(adjusted)
    unsorted[order] = adjusted
    q_value[eligible_mask] = unsorted
    return q_value


def make_adaptive_diagonal_prior(
    weighted_scores: np.ndarray,
    significance_mask: np.ndarray,
    base_diagonal_mass: float = 0.80,
    top_k: int = 8,
) -> np.ndarray:
    """Increase diagonal mass when a recipient has little edge evidence."""
    weighted_scores = np.asarray(weighted_scores, dtype=np.float64)
    significance_mask = np.asarray(significance_mask, dtype=bool)
    n_parish = weighted_scores.shape[0]
    result = np.zeros_like(weighted_scores)

    for recipient in range(n_parish):
        row = np.maximum(weighted_scores[recipient], 0.0).copy()
        row[~significance_mask[recipient]] = 0.0
        row[recipient] = 0.0
        positive = np.flatnonzero(row > 0)
        if len(positive) == 0:
            result[recipient, recipient] = 1.0
            continue
        if len(positive) > top_k:
            keep = positive[np.argsort(row[positive])[-top_k:]]
            mask = np.zeros(n_parish, dtype=bool)
            mask[keep] = True
            row[~mask] = 0.0
            positive = keep

        evidence_fraction = min(len(positive) / float(top_k), 1.0)
        diagonal_mass = 1.0 - (1.0 - base_diagonal_mass) * evidence_fraction
        result[recipient, recipient] = diagonal_mass
        result[recipient] += (1.0 - diagonal_mass) * row / row.sum()
    return result


def build_significance_ladder(
    cases: np.ndarray,
    tests: np.ndarray,
    population: np.ndarray,
    train_mask: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    lags: Iterable[int] = (0, 1, 2),
    bootstrap_repetitions: int = 200,
    circular_repetitions: int = 500,
    stability_threshold: float = 0.60,
    alpha: float = 0.05,
    diagonal_mass: float = 0.80,
    top_k: int = 8,
    seed: int = 2026,
) -> PWCCFSignificanceResult:
    """Build sequential one-change-at-a-time PWCCF prior variants."""
    lags = tuple(lags)
    base = build_nb_residual_pwccf_prior(
        cases=cases,
        tests=tests,
        population=population,
        train_mask=train_mask,
        diagonal_mass=diagonal_mass,
        lags=lags,
        bootstrap_repetitions=bootstrap_repetitions,
        block_length=3,
        stability_threshold=stability_threshold,
        top_k=top_k,
        labels=labels,
        seed=seed,
    )
    n_parish = base.C0.shape[0]
    eligible = base.raw_score > 0
    np.fill_diagonal(eligible, False)
    p_value = circular_shift_pvalues(
        residuals=base.pearson_residuals,
        observed_scores=base.raw_score,
        lags=lags,
        repetitions=circular_repetitions,
        candidate_mask=eligible,
        seed=seed + 10000,
    )
    q_value = benjamini_hochberg(p_value, eligible)
    significant_unadjusted = eligible & (p_value <= alpha)
    significant_fdr = eligible & (q_value <= alpha)

    weighted = base.raw_score * base.stability
    circular_scores = weighted * significant_unadjusted
    fdr_scores = weighted * significant_fdr
    circular_C0 = _make_prior(circular_scores, diagonal_mass, top_k)
    fdr_C0 = _make_prior(fdr_scores, diagonal_mass, top_k)
    adaptive_C0 = make_adaptive_diagonal_prior(
        weighted, significant_fdr, diagonal_mass, top_k
    )

    labels = list(labels) if labels is not None else [str(i) for i in range(n_parish)]
    edge_rows = []
    for recipient in range(n_parish):
        for source in range(n_parish):
            if recipient == source or not eligible[recipient, source]:
                continue
            edge_rows.append(dict(
                recipient_index=recipient,
                recipient=labels[recipient],
                source_index=source,
                source=labels[source],
                raw_correlation=base.raw_score[recipient, source],
                best_lag_weeks=int(base.best_lag[recipient, source]),
                bootstrap_stability=base.stability[recipient, source],
                circular_p_value=p_value[recipient, source],
                fdr_q_value=q_value[recipient, source],
                significant_unadjusted=bool(significant_unadjusted[recipient, source]),
                significant_fdr=bool(significant_fdr[recipient, source]),
                C0_circular=float(circular_C0[recipient, source]),
                C0_fdr=float(fdr_C0[recipient, source]),
                C0_adaptive=float(adaptive_C0[recipient, source]),
            ))
    edge_table = pd.DataFrame(edge_rows)

    diagnostics = dict(
        train_weeks=float(np.sum(train_mask)),
        bootstrap_repetitions=float(bootstrap_repetitions),
        circular_repetitions=float(circular_repetitions),
        positive_candidate_edges=float(np.sum(eligible)),
        unadjusted_significant_edges=float(np.sum(significant_unadjusted)),
        fdr_significant_edges=float(np.sum(significant_fdr)),
        unadjusted_rows_with_edges=float(np.sum(np.any(significant_unadjusted, axis=1))),
        fdr_rows_with_edges=float(np.sum(np.any(significant_fdr, axis=1))),
        baseline_mean_diagonal=float(np.mean(np.diag(base.C0))),
        circular_mean_diagonal=float(np.mean(np.diag(circular_C0))),
        fdr_mean_diagonal=float(np.mean(np.diag(fdr_C0))),
        adaptive_mean_diagonal=float(np.mean(np.diag(adaptive_C0))),
        adaptive_identity_rows=float(np.sum(np.diag(adaptive_C0) >= 1.0 - EPS)),
    )
    return PWCCFSignificanceResult(
        baseline_C0=base.C0,
        circular_C0=circular_C0,
        fdr_C0=fdr_C0,
        adaptive_C0=adaptive_C0,
        p_value=p_value,
        q_value=q_value,
        significant_unadjusted=significant_unadjusted,
        significant_fdr=significant_fdr,
        edge_table=edge_table,
        diagnostics=diagnostics,
        base_result=base,
    )


__all__ = [
    "PWCCFSignificanceResult",
    "benjamini_hochberg",
    "build_significance_ladder",
    "circular_shift_pvalues",
    "make_adaptive_diagonal_prior",
    "permute_prior_offdiagonal",
]
