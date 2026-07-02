---
icon: lucide/chart-no-axes-combined
---

# Fitting API

Top-level fitting helpers are documented under [Top-level API](root.md). This
page documents objective, optimizer, and candidate-selection configuration
classes from `gmm_divergence.fitting`.


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

## Candidate Selectors

::: gmm_divergence.fitting.TopKSelector

::: gmm_divergence.fitting.ThresholdSelector

::: gmm_divergence.fitting.ToleranceSelector

::: gmm_divergence.fitting.QuantileSelector
