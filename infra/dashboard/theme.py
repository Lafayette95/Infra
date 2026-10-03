"""Design tokens (validated dataviz palette) for light and dark chart surfaces.

Direction is encoded blue (up) / orange (down) - categorical slots 1 and 2 of the
reference palette, separable under colour-vision deficiency - never red/green.
"""
from __future__ import annotations

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "up": "#2a78d6", "down": "#eb6834", "volume": "#898781",
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "up": "#3987e5", "down": "#d95926", "volume": "#898781",
    },
}


def tokens(theme: str) -> dict[str, str]:
    """Token dict for ``theme`` (falls back to light)."""
    return THEMES.get(theme, THEMES["light"])


# Categorical series colours, fixed order (never cycled): the dataviz skill's reference
# palette, validated by the skill in both modes; slots 1-2 are the up/down pair above.
SERIES: dict[str, list[str]] = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}


def series_colors(theme: str) -> list[str]:
    return SERIES.get(theme, SERIES["light"])
