"""Explicit, sparse HSL-only witnesses; never infer a photo's Adobe Look selection.

These numeric defaults were verified against installed top-level Look metadata
and saved ARM64 readers. They are not copies of the complete Adobe profiles:
LUTs, curves, local corrections, camera predicates and other settings are absent.
"""

from __future__ import annotations


ACR_HSL_FRAGMENT_VERSION = "acr-hsl-fragments-v1"
SENTINEL = -999999


def _sparse(lanes: dict[int, int]) -> tuple[int, ...]:
    values = [SENTINEL] * 24
    for lane, value in lanes.items():
        values[lane] = value
    return tuple(values)


# H8, S8, L8; each octet is Red/Orange/Yellow/Green/Aqua/Blue/Purple/Magenta.
_FRAGMENTS = {
    "modern03_hsl": _sparse({9: -14, 17: -7}),
    "modern04_hsl": _sparse({9: -10}),
    "vintage07_hsl": _sparse({13: -29}),
}


def hsl_fragment_defaults(fragment: str) -> tuple[int, ...] | None:
    """Return immutable prepared defaults for an explicit selection, not a DCP."""
    if fragment == "none":
        return None
    try:
        return _FRAGMENTS[fragment]
    except (KeyError, TypeError) as exc:
        raise ValueError("unknown ACR HSL-only fragment") from exc
