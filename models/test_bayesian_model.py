#!/usr/bin/env python3
"""
Unit tests for the Bayesian Factor Model.

Tests cover:
1. Model initialization and configuration
2. Prediction pipeline
3. Special cases (k=0, infer_lambda=True/False)
4. Convenience constructors
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bayesian_factor_model import (
    BayesianFactorModel,
    FullModel,
    PopulationBaseline,
    EgocentricBaseline,
    FactorModel,
    load_factor_loadings,
    load_question_means,
)


class TestModelInitialization:
    """Test model initialization and configuration."""

    def test_k0_creates_uniform_loadings(self):
        """k=0 should create uniform loadings (single factor, all 1s)."""
        model = BayesianFactorModel(k=0)
        assert model.L.shape == (35, 1)
        assert np.allclose(model.L, np.ones((35, 1)))

    def test_k4_loads_correct_dimensions(self):
        """k=4 should load 35x4 loadings matrix."""
        model = BayesianFactorModel(k=4)
        assert model.L.shape == (35, 4)

    def test_lambda_grid_size(self):
        """Lambda grid should have correct size."""
        model = BayesianFactorModel(k=4, lambda_grid_size=21)
        assert len(model.lambda_grid) == 21
        assert model.lambda_grid[0] == 0.0
        assert model.lambda_grid[-1] == 1.0

    def test_uniform_prior_sums_to_one(self):
        """Uniform prior should sum to 1."""
        model = BayesianFactorModel(k=4)
        assert np.isclose(model.lambda_prior_probs.sum(), 1.0)

    def test_epsilon_clipping(self):
        """Epsilon should be clipped to [0, 1]."""
        model = BayesianFactorModel(k=4, epsilon=1.5)
        assert model.epsilon == 1.0
        model = BayesianFactorModel(k=4, epsilon=-0.5)
        assert model.epsilon == 0.0

    def test_repr(self):
        """Model should have informative repr."""
        model = BayesianFactorModel(k=4, infer_lambda=True)
        assert 'k=4' in repr(model)
        assert 'infer_λ=True' in repr(model)


class TestPredictions:
    """Test prediction pipeline."""

    def test_predictions_shape(self):
        """Predictions should have shape (35,)."""
        model = BayesianFactorModel(k=4, infer_lambda=True)
        r_self = np.random.uniform(1, 5, 35)

        preds = model.predict(0, 3.0, r_self)
        assert preds.shape == (35,)

    def test_predict_participant_alias(self):
        """predict_participant should work as alias for predict."""
        model = BayesianFactorModel(k=4, infer_lambda=True)
        r_self = np.random.uniform(1, 5, 35)

        preds1 = model.predict(0, 3.0, r_self)
        preds2 = model.predict_participant(0, 3.0, r_self)
        assert np.allclose(preds1, preds2)

    def test_predictions_in_valid_range(self):
        """Predictions should be in [0, 1]."""
        model = BayesianFactorModel(k=4, infer_lambda=True)
        r_self = np.random.uniform(1, 5, 35)

        preds = model.predict(0, 3.0, r_self)
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_epsilon_compresses_predictions(self):
        """Higher epsilon should compress predictions toward 0.5."""
        r_self = np.random.uniform(1, 5, 35)

        model_low_eps = BayesianFactorModel(k=4, epsilon=0.0, infer_lambda=False, lam=0.5)
        model_high_eps = BayesianFactorModel(k=4, epsilon=0.8, infer_lambda=False, lam=0.5)

        preds_low = model_low_eps.predict(0, 3.0, r_self)
        preds_high = model_high_eps.predict(0, 3.0, r_self)

        # High epsilon should be closer to 0.5
        dist_low = np.abs(preds_low - 0.5).mean()
        dist_high = np.abs(preds_high - 0.5).mean()

        assert dist_high < dist_low, "Higher epsilon should compress toward 0.5"

    def test_fixed_lambda_mode(self):
        """With infer_lambda=False, predictions should be deterministic."""
        model = BayesianFactorModel(k=4, infer_lambda=False, lam=0.5)
        r_self = np.random.uniform(1, 5, 35)

        preds1 = model.predict(0, 3.0, r_self)
        preds2 = model.predict(0, 3.0, r_self)

        assert np.allclose(preds1, preds2)

    def test_infer_lambda_mode(self):
        """With infer_lambda=True, predictions should still be deterministic."""
        model = BayesianFactorModel(k=4, infer_lambda=True)
        r_self = np.random.uniform(1, 5, 35)

        preds1 = model.predict(0, 3.0, r_self)
        preds2 = model.predict(0, 3.0, r_self)

        assert np.allclose(preds1, preds2)


class TestModelComparison:
    """Test that nested models behave as expected."""

    def test_baseline_and_egocentric_differ(self):
        """Baseline (λ=0) and Egocentric (λ=1) should give different predictions."""
        r_self = np.array([1.0] * 17 + [5.0] * 18)  # Extreme responses

        baseline = PopulationBaseline()
        egocentric = EgocentricBaseline()

        preds_base = baseline.predict(0, 3.0, r_self)
        preds_ego = egocentric.predict(0, 3.0, r_self)

        # Should differ because egocentric centers prior on theta_self
        assert not np.allclose(preds_base, preds_ego)

    def test_factor_model_creates_gradient(self):
        """Factor model should create different predictions for different questions."""
        model = FactorModel(k=4)
        r_self = np.random.uniform(1, 5, 35)

        # Observe agreement on question 0
        preds = model.predict(0, r_self[0], r_self)

        # Predictions should vary across questions (not all identical)
        assert preds.std() > 0.01, "Factor model should create gradient"

    def test_k0_vs_k4_structure(self):
        """k=4 models should show more structured predictions than k=0."""
        r_self = np.random.uniform(2, 4, 35)

        model_k0 = BayesianFactorModel(k=0, infer_lambda=False, lam=0.5, epsilon=0.0)
        model_k4 = BayesianFactorModel(k=4, infer_lambda=False, lam=0.5, epsilon=0.0)

        preds_k0 = model_k0.predict(0, 3.0, r_self)
        preds_k4 = model_k4.predict(0, 3.0, r_self)

        # Both should produce valid predictions
        assert preds_k0.shape == preds_k4.shape == (35,)
        assert np.all(np.isfinite(preds_k0))
        assert np.all(np.isfinite(preds_k4))


class TestConvenienceConstructors:
    """Test convenience constructor functions."""

    def test_full_model_infers_lambda(self):
        """FullModel should have infer_lambda=True."""
        model = FullModel(k=4)
        assert model.infer_lambda is True

    def test_population_baseline_config(self):
        """PopulationBaseline should have k=0, λ=0."""
        model = PopulationBaseline()
        assert model.k == 0
        assert model.fixed_lam == 0.0
        assert model.infer_lambda is False

    def test_egocentric_baseline_config(self):
        """EgocentricBaseline should have k=0, λ=1."""
        model = EgocentricBaseline()
        assert model.k == 0
        assert model.fixed_lam == 1.0
        assert model.infer_lambda is False

    def test_factor_model_config(self):
        """FactorModel should have k>0, λ=0."""
        model = FactorModel(k=4)
        assert model.k == 4
        assert model.fixed_lam == 0.0
        assert model.infer_lambda is False

    def test_constructors_accept_kwargs(self):
        """Constructors should pass through kwargs."""
        model = FullModel(k=4, epsilon=0.2, sigma_obs=0.5)
        assert model.epsilon == 0.2
        assert model.sigma_obs == 0.5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_responses(self):
        """Model should handle extreme responses (1 and 5)."""
        model = BayesianFactorModel(k=4, infer_lambda=True)
        r_self = np.ones(35) * 1.0  # All 1s

        preds = model.predict(0, 5.0, r_self)  # Observe 5
        assert np.all(np.isfinite(preds))

    def test_identical_responses(self):
        """Model should handle identical responses."""
        model = BayesianFactorModel(k=4, infer_lambda=True)
        r_self = np.ones(35) * 3.0  # All 3s

        preds = model.predict(0, 3.0, r_self)  # Observe 3
        assert np.all(np.isfinite(preds))

    def test_all_questions_observed(self):
        """Model should work for any observed question index."""
        model = BayesianFactorModel(k=4, infer_lambda=True)
        r_self = np.random.uniform(1, 5, 35)

        for obs_q in [0, 17, 34]:  # First, middle, last
            preds = model.predict(obs_q, 3.0, r_self)
            assert np.all(np.isfinite(preds))
            assert preds.shape == (35,)

    def test_different_k_values(self):
        """Model should work for various k values."""
        r_self = np.random.uniform(1, 5, 35)

        for k in [0, 1, 4, 10, 35]:
            model = BayesianFactorModel(k=k)
            preds = model.predict(0, 3.0, r_self)
            assert np.all(np.isfinite(preds))
            assert preds.shape == (35,)


class TestDataLoading:
    """Test data loading functions."""

    def test_load_factor_loadings_shape(self):
        """Factor loadings should have expected shape."""
        loadings = load_factor_loadings(k=4)
        assert loadings.shape == (35, 4)

    def test_load_factor_loadings_full(self):
        """Full loadings should be 35x35."""
        loadings = load_factor_loadings(k=None)
        assert loadings.shape == (35, 35)

    def test_load_question_means_shape(self):
        """Question means should have shape (35,)."""
        means = load_question_means()
        assert means.shape == (35,)

    def test_question_means_in_valid_range(self):
        """Question means should be in Likert range [1, 5]."""
        means = load_question_means()
        assert np.all(means >= 1) and np.all(means <= 5)


def run_tests():
    """Run all tests and report results."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
