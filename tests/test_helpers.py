from __future__ import annotations

import numpy as np

from geoid_pinn.priors import permute_prior_offdiagonal
from geoid_pinn.significance import benjamini_hochberg, make_adaptive_diagonal_prior


def test_permutation_preserves_diagonal_and_row_mass() -> None:
    prior = np.array(
        [
            [0.8, 0.15, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8],
        ]
    )
    permuted = permute_prior_offdiagonal(prior, seed=7)
    np.testing.assert_allclose(np.diag(permuted), np.diag(prior))
    np.testing.assert_allclose(permuted.sum(axis=1), 1.0)


def test_adaptive_prior_adds_diagonal_mass_without_edges() -> None:
    scores = np.array(
        [
            [0.0, 0.7, 0.0],
            [0.2, 0.0, 0.4],
            [0.0, 0.0, 0.0],
        ]
    )
    significant = scores > 0
    prior = make_adaptive_diagonal_prior(
        scores, significant, base_diagonal_mass=0.8, top_k=2
    )
    np.testing.assert_allclose(prior.sum(axis=1), 1.0)
    assert prior[2, 2] == 1.0
    assert prior[0, 0] > 0.8


def test_bh_adjustment_is_bounded() -> None:
    p_value = np.array([[1.0, 0.001], [0.03, 1.0]])
    eligible = np.array([[False, True], [True, False]])
    q_value = benjamini_hochberg(p_value, eligible)
    assert np.all((q_value >= 0) & (q_value <= 1))
    assert q_value[0, 1] <= q_value[1, 0]
