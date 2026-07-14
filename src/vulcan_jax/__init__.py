"""VULCAN-JAX: JAX-accelerated photochemical kinetics for exoplanet atmospheres.

Float64 is enabled at import time (non-negotiable for rate-constant
arithmetic spanning ~50 orders of magnitude).
"""

from __future__ import annotations

from typing import Any

from ._version import __version__

# Unguarded on purpose (loud-errors rule): a jax that cannot enable x64 must
# fail the import, not run float32 chemistry (rate constants span ~50 dex).
from jax import config as _config

_config.update("jax_enable_x64", True)

from .config import Config, default_config, load_config
from .state import RunState

__all__ = [
    "__version__",
    "RunState",
    "Config",
    "load_config",
    "default_config",
    "make_config",
]


def make_config(**overrides: Any) -> Config:
    """Load the default config with user overrides (alias for ``load_config``).

    Example::

        cfg = vulcan_jax.make_config(nz=100, use_photo=False)
        rs = vulcan_jax.RunState.with_pre_loop_setup(cfg)
    """
    return load_config("default", **overrides)
