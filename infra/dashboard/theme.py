"""Design tokens for light and dark chart surfaces - the Gemini Capital metallic scheme
(2026-10-07): dark = metallic black page, carbon-grey surfaces, silver ink; light = silver
/ platinum page, near-white chart surface, carbon ink. Chrome only: the DATA colours below
(up/down, categorical series) are the validated dataviz palette, kept as is.

Direction is encoded blue (up) / orange (down) - categorical slots 1 and 2 of the
reference palette, separable under colour-vision deficiency - never red/green.

Surfaces were checked against the series palette (WCAG contrast): the dark carbon surface
#17181b RAISES every series' contrast slightly vs the old #1a1a19; the light CHART surface
stays near-white (#fbfbfc) while the page around it is silver, because a fully silver
chart surface pushed the orange 'down' series under 3:1 (3.12 -> 2.98).
"""
from __future__ import annotations

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fbfbfc", "page": "#e9ebee", "ink": "#16171a", "ink2": "#4a4d52",
        "muted": "#80848a", "grid": "#dde0e4", "axis": "#b9bdc3",
        "up": "#2a78d6", "down": "#eb6834", "volume": "#80848a",
    },
    "dark": {
        "surface": "#17181b", "page": "#08090a", "ink": "#e6e8eb", "ink2": "#b4b8be",
        "muted": "#858a91", "grid": "#2a2c30", "axis": "#3a3d42",
        "up": "#3987e5", "down": "#d95926", "volume": "#858a91",
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
