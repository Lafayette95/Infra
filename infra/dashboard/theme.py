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
