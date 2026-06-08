---
icon: lucide/house
---

# Introduction

`gmm-divergence` provides utilities for working with Gaussian mixture models
and estimating divergences between them.

The package currently includes:

- typed Gaussian and Gaussian mixture representations
- density, log-density, and sampling utilities
- KL divergence estimators for Gaussian mixtures
- mixture-weight fitting against a target mixture

!!! note "Active development"
    APIs, estimators, and numerical behavior may change between early releases.


## Next steps

<div class="grid cards" markdown>

- [:material-rocket-launch:{ .lg .middle } __Getting started__](getting_started.md)

    ---

    A quick introduction to the package, its features, and how to get started.

-  [:lucide-equal-approximately:{ .lg .middle } __Kl estimation__](kl_estimation.md)

    ---

    Estimate the KL divergence between two Gaussian mixtures using various numerical methods.

-   [:lucide-chart-no-axes-combined:{ .lg .middle }  __Mixture weight fitting__](fit_weights.md)

    ---

    Fit the mixture weights of a Gaussian mixture to minimize KL divergence against a target mixture.


-   [:material-api:{ .lg .middle } __API reference__](reference/index.md)

    ---

    Jump into the generated Python API docs when you already know the area you want to inspect in code.

</div>
