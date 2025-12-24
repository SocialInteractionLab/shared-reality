#!/usr/bin/env python3
"""
Tests for the Bayesian Factor Model.

Mostly behavior tests, with a few sanity checks on model setup.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bayesian_factor_model import (
    BayesianFactorModel,
    load_factor_loadings,
    load_question_means,
)


class TestModelSetup:
    """Sanity checks that model is configured correctly."""

    def test_k_stored(self):
        """Model should store k."""
        assert BayesianFactorModel(k=4).k == 4
        assert BayesianFactorModel(k=0).k == 0

    def test_epsilon_clipped(self):
        """Epsilon should be clipped to [0, 1]."""
        assert BayesianFactorModel(epsilon=1.5).epsilon == 1.0
        assert BayesianFactorModel(epsilon=-0.5).epsilon == 0.0

    def test_repr(self):
        """Repr should show key parameters."""
        assert 'k=4' in repr(BayesianFactorModel(k=4))
        assert 'infer_λ=True' in repr(BayesianFactorModel(infer_lambda=True))
        assert 'λ=0.5' in repr(BayesianFactorModel(infer_lambda=False, lam=0.5))


class TestPredictions:
    """Test prediction behavior."""

    def test_output_shape(self):
        """Should return 35 predictions."""
        model = BayesianFactorModel(k=4)
        preds = model.predict(0, 3.0, np.random.uniform(1, 5, 35))
        assert preds.shape == (35,)

    def test_output_range(self):
        """Predictions should be probabilities in [0, 1]."""
        model = BayesianFactorModel(k=4)
        preds = model.predict(0, 3.0, np.random.uniform(1, 5, 35))
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_deterministic(self):
        """Same inputs should give same outputs."""
        model = BayesianFactorModel(k=4)
        r_self = np.random.uniform(1, 5, 35)
        preds1 = model.predict(0, 3.0, r_self)
        preds2 = model.predict(0, 3.0, r_self)
        assert np.allclose(preds1, preds2)

    def test_epsilon_compresses_to_half(self):
        """Higher epsilon should push predictions toward 0.5."""
        r_self = np.random.uniform(1, 5, 35)
        preds_low = BayesianFactorModel(k=4, epsilon=0.0).predict(0, 3.0, r_self)
        preds_high = BayesianFactorModel(k=4, epsilon=0.8).predict(0, 3.0, r_self)

        dist_low = np.abs(preds_low - 0.5).mean()
        dist_high = np.abs(preds_high - 0.5).mean()
        assert dist_high < dist_low


class TestModelBehavior:
    """Test that different configurations behave differently."""

    def test_lambda_affects_predictions(self):
        """λ=0 vs λ=1 should give different predictions."""
        r_self = np.array([1.0] * 17 + [5.0] * 18)  # Polarized responses
        preds_0 = BayesianFactorModel(k=0, infer_lambda=False, lam=0.0).predict(0, 3.0, r_self)
        preds_1 = BayesianFactorModel(k=0, infer_lambda=False, lam=1.0).predict(0, 3.0, r_self)
        assert not np.allclose(preds_0, preds_1)

    def test_k4_creates_gradient(self):
        """With k>0, predictions should vary across questions."""
        model = BayesianFactorModel(k=4, infer_lambda=False, lam=0.0, epsilon=0.0)
        r_self = np.random.uniform(1, 5, 35)
        preds = model.predict(0, r_self[0], r_self)
        assert preds.std() > 0.01  # Non-trivial variation

    def test_works_for_all_k(self):
        """Model should work for k=0,1,4,35."""
        r_self = np.random.uniform(1, 5, 35)
        for k in [0, 1, 4, 35]:
            preds = BayesianFactorModel(k=k).predict(0, 3.0, r_self)
            assert np.all(np.isfinite(preds))

    def test_works_for_all_questions(self):
        """Should work when observing any question."""
        model = BayesianFactorModel(k=4)
        r_self = np.random.uniform(1, 5, 35)
        for obs_q in [0, 17, 34]:
            preds = model.predict(obs_q, 3.0, r_self)
            assert np.all(np.isfinite(preds))


class TestEdgeCases:
    """Test boundary conditions."""

    def test_extreme_responses(self):
        """Should handle all-1s or all-5s responses."""
        model = BayesianFactorModel(k=4)
        assert np.all(np.isfinite(model.predict(0, 5.0, np.ones(35))))
        assert np.all(np.isfinite(model.predict(0, 1.0, np.ones(35) * 5)))

    def test_identical_responses(self):
        """Should handle identical self-responses."""
        model = BayesianFactorModel(k=4)
        preds = model.predict(0, 3.0, np.ones(35) * 3.0)
        assert np.all(np.isfinite(preds))


class TestDataLoading:
    """Test data loading functions."""

    def test_loadings_shape(self):
        assert load_factor_loadings(k=4).shape == (35, 4)
        assert load_factor_loadings().shape == (35, 35)

    def test_means_shape_and_range(self):
        means = load_question_means()
        assert means.shape == (35,)
        assert np.all((means >= 1) & (means <= 5))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
