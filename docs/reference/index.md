---
icon: lucide/book-marked
---

# API reference

The API reference is organized by import namespace. The top-level
`gmm_divergence` module is a curated convenience layer for common objects and
primary functions. Configuration classes live in the namespace pages.

| Page | Import namespace | Contents |
| --- | --- | --- |
| [Top-level API](root.md) | `gmm_divergence` | Root exports: common classes, helper functions, and results |
| [Distributions](distributions.md) | `gmm_divergence.distributions` | Distribution metadata not exported at the root |
| [Divergence](divergence.md) | `gmm_divergence.divergence` | Estimator configuration |
| [Fitting](fitting.md) | `gmm_divergence.fitting` | Objectives, optimizers, and selectors |
| [Sampling](sampling.md) | `gmm_divergence.sampling` | Sample specifications used by Monte Carlo estimators |
| [Covariance](covariance.md) | `gmm_divergence.covariance` | Covariance regularization and epsilon heuristics |

For day-to-day use, import the package once:

```python
import gmm_divergence as gd
```

Then access configuration objects through namespaces such as
`gd.divergence.MonteCarlo`, `gd.sampling.Draw`, and `gd.fitting.ForwardKL`.
