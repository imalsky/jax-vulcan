"""VULCAN-master's published plotting palette."""

from __future__ import annotations

# VULCAN-master's Tableau 20 palette, 0-1 normalized. Published in the .vul
# `parameter` dict (`para.tableau20`), which upstream plot_py/ scripts read,
# so the values are a compatibility contract.
_TABLEAU20_RGB255 = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (174, 199, 232),
    (255, 187, 120),
    (152, 223, 138),
    (255, 152, 150),
    (197, 176, 213),
    (196, 156, 148),
    (247, 182, 210),
    (199, 199, 199),
    (219, 219, 141),
    (158, 218, 229),
]


def master_tableau20() -> list[tuple[float, float, float]]:
    """VULCAN-master's normalized Tableau 20 plotting palette."""
    return [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in _TABLEAU20_RGB255]

