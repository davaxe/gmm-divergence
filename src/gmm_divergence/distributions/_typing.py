"""The supported Gaussian-family distribution union."""

from typing import TypeAlias

from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.distributions._mixture import GaussianMixture

GaussianDistribution: TypeAlias = Gaussian | GaussianMixture
