from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DivergenceResult:
    value: float
    method: str | None = None
    num_samples: int | None = None
