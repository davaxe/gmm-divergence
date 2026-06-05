from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, final

if TYPE_CHECKING:
    from collections.abc import Iterable

OptionsT_co = TypeVar("OptionsT_co", covariant=True)


@dataclass(frozen=True, slots=True)
class MethodSpec(Generic[OptionsT_co]):
    """Registered dispatcher method."""

    name: str
    option_type: type[OptionsT_co]
    default: OptionsT_co


@final
class Registry:
    """Resolve public string methods and typed option objects."""

    def __init__(self, *, label: str, specs: Iterable[MethodSpec[object]]) -> None:
        self._label: str = label
        self._by_name: dict[str, MethodSpec[object]] = {}
        self._by_type: dict[type[object], MethodSpec[object]] = {}
        for spec in specs:
            if spec.name in self._by_name:
                msg = f"Duplicate {label} method name {spec.name!r}."
                raise ValueError(msg)
            if spec.option_type in self._by_type:
                msg = f"Duplicate {label} option type {spec.option_type.__name__}."
                raise ValueError(msg)
            self._by_name[spec.name] = spec
            self._by_type[spec.option_type] = spec

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def resolve(self, method: str | object) -> tuple[MethodSpec[object], object]:
        if isinstance(method, str):
            try:
                spec = self._by_name[method]
            except KeyError:
                supported = ", ".join(self.names)
                msg = (
                    f"Unknown {self._label} method {method!r}. Supported methods are: {supported}."
                )
                raise ValueError(msg) from None
            return spec, spec.default

        spec = self._by_type.get(type(method))
        if spec is None:
            supported = ", ".join(self.names)
            msg = (
                f"Unknown {self._label} method option {type(method).__name__}. "
                f"Supported methods are: {supported}."
            )
            raise TypeError(msg)
        return spec, method
