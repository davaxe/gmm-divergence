from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

import gmm_divergence as gd

DATA_PATH = Path(__file__).parent / "data32dim.npz"


class Args(NamedTuple):
    target: str
    data_path: Path = DATA_PATH


def parse_args() -> Args:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("target", type=str, help="The target distribution.")
    _ = parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help=f"Path to the data file (default: {DATA_PATH}).",
    )
    args = parser.parse_args()
    return Args(target=args.target)


def main() -> None:
    """Fit source distribution weights for a named target distribution."""
    args = parse_args()
    data = np.load(args.data_path)
    labels = data["labels"]
    if args.target not in labels:
        msg = f"Target distribution '{args.target}' not found in data. Available labels: {labels}"
        raise ValueError(msg)
    target_index = np.where(labels == args.target)[0][0]
    target = gd.Gaussian.from_arrays(
        mean=data["means"][target_index], covariance=data["covs"][target_index]
    )
    components: list[gd.Gaussian] = [
        gd.Gaussian.from_arrays(mean=data["means"][i], covariance=data["covs"][i])
        for i in range(len(labels))
        if i != target_index
    ]
    component_labels = [str(label) for i, label in enumerate(labels) if i != target_index]
    res = gd.fit_mixture_weights(
        target,
        components,
        method="softmax-lbfgsb",
        objective=gd.ForwardKL(rng=0),
        candidate_selector=gd.fitting.KLToleranceSelector(delta=15, mode="relative"),
    )
    _ = sys.stdout.write(f"{res.display(source_labels=component_labels)}\n")


if __name__ == "__main__":
    main()
