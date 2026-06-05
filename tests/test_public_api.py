from __future__ import annotations

import importlib

import pytest

import gmm_divergence
from gmm_divergence import distributions, divergence, fitting


def test_top_level_exports_are_curated() -> None:
    assert set(gmm_divergence.__all__) == {
        "BidirectionalKL",
        "ClosedForm",
        "CombinedGaussianMixture",
        "DivergenceResult",
        "ForwardKL",
        "Gaussian",
        "GaussianApproximation",
        "GaussianMixture",
        "KLFitResult",
        "MixtureMapping",
        "MomentMatching",
        "MonteCarlo",
        "ReverseKL",
        "SimplexSLSQP",
        "SoftmaxLBFGSB",
        "Unscented",
        "combine_gaussians",
        "fit_mixture_weights",
        "kl_divergence",
    }


def test_domain_exports_are_curated() -> None:
    assert set(distributions.__all__) == {
        "CombinedGaussianMixture",
        "Gaussian",
        "GaussianMixture",
        "MixtureMapping",
        "combine_gaussians",
    }
    assert set(divergence.__all__) == {
        "ClosedForm",
        "GaussianApproximation",
        "KLMethod",
        "MonteCarlo",
        "Unscented",
        "kl_divergence",
    }
    assert set(fitting.__all__) == {
        "BidirectionalKL",
        "FitObjective",
        "FitParameterization",
        "ForwardKL",
        "MomentMatching",
        "ReverseKL",
        "SimplexSLSQP",
        "SoftmaxLBFGSB",
        "WeightFitMethod",
        "WeightFitObjective",
        "fit_mixture_weights",
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "gmm_divergence.distribution",
        "gmm_divergence.estimators",
        "gmm_divergence.fit",
        "gmm_divergence.typing",
        "gmm_divergence.utils",
    ],
)
def test_removed_legacy_modules_are_not_importable(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        _ = importlib.import_module(module_name)
