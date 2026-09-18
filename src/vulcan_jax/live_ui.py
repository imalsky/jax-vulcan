"""Matplotlib backend policy plus VULCAN-master's published plotting palette."""

from __future__ import annotations

import os

# VULCAN-master's Tableau 20 palette, 0-1 normalized. THE one copy: it is
# published in the .vul `parameter` dict (`para.tableau20`), which upstream
# plot_py/ scripts read, so the values are a compatibility contract.
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


def import_plt():
    """Lazy-import pyplot, selecting a backend by ONE merged policy.

    Both the post-run plotters and the live UI resolve the backend here, so
    `VULCAN_HEADLESS_PLOT` works for both. The order, most explicit first:

    1. `MPLBACKEND` set  -> matplotlib already honours it; touch nothing.
    2. `VULCAN_HEADLESS_PLOT` set -> Agg (the project's explicit override).
    3. No `DISPLAY`, and not macOS -> Agg (there is no display to draw on;
       macOS is excluded because its native backend needs no DISPLAY).
    4. Otherwise leave the default backend alone.

    Import is lazy so a run that never plots does not pay for matplotlib.
    """
    import matplotlib

    if not os.environ.get("MPLBACKEND"):
        if os.environ.get("VULCAN_HEADLESS_PLOT"):
            matplotlib.use("Agg")
        elif not os.environ.get("DISPLAY") and "darwin" not in os.uname().sysname.lower():
            matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
