---
icon: lucide/chart-no-axes-combined
---

# Fitting API

Top-level fitting helpers are documented under [Top-level API](root.md). This
page documents prepared fits, optimizer and objective configurations, and
candidate-selection helpers from `gmm_divergence.fitting`.


::: gmm_divergence.fitting
    options:
        members: false
        show_root_full_path: true

## Objective Configuration

::: gmm_divergence.fitting.ForwardKL

::: gmm_divergence.fitting.ReverseKL

::: gmm_divergence.fitting.BidirectionalKL

::: gmm_divergence.fitting.JensenShannon

::: gmm_divergence.fitting.MomentMatching

## Optimizers

::: gmm_divergence.fitting.SoftmaxLBFGSB

::: gmm_divergence.fitting.SimplexSLSQP

## Prepared Fits

::: gmm_divergence.fitting.prepare_mixture_weight_fit

::: gmm_divergence.fitting.PreparedFit

::: gmm_divergence.fitting.FitSolution

## Candidate Selectors

::: gmm_divergence.fitting.score_candidates

::: gmm_divergence.fitting.rank_candidates

::: gmm_divergence.fitting.TopKSelector

::: gmm_divergence.fitting.ThresholdSelector

::: gmm_divergence.fitting.ToleranceSelector

::: gmm_divergence.fitting.QuantileSelector
