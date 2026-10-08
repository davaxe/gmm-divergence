---
icon: lucide/equal-approximately
---

# Divergence API

Top-level divergence helpers are documented under [Top-level API](root.md). This
page documents estimator configuration classes and mode-occupancy diagnostics
from `gmm_divergence.divergence`.

::: gmm_divergence.divergence
    options:
        members: false
        show_root_full_path: true

::: gmm_divergence.divergence.MonteCarlo

::: gmm_divergence.divergence.Unscented

::: gmm_divergence.divergence.MomentMatchedGaussian

::: gmm_divergence.divergence.ClosedForm

::: gmm_divergence.divergence.Variational

## Mode Occupancy

`mode_occupancy_kl` compares probability vectors over shared categories. It is
separate from `kl_divergence`: it measures categorical occupancy differences,
not divergence between mixture densities. Corresponding indices must refer to
the same category; independently fitted GMM components are not automatically
aligned, and components need not represent distinct modes.

```python
import gmm_divergence as gd

result = gd.divergence.mode_occupancy_kl([0.8, 0.2], [0.5, 0.5])
assert result.value > 0

# Smoothing changes the probability vectors and makes this comparison finite.
smoothed = gd.divergence.mode_occupancy_kl([1.0, 0.0], [0.0, 1.0], epsilon=1e-8)
assert smoothed.value > 0
```

::: gmm_divergence.divergence.mode_occupancy_kl
