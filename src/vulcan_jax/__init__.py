"""VULCAN-JAX: JAX-accelerated photochemical kinetics for exoplanet atmospheres.

Float64 is enabled at import time (non-negotiable for rate-constant
arithmetic spanning ~50 orders of magnitude).
"""

from __future__ import annotations

import copy as _copy
import types as _types
from typing import Any

from ._version import __version__

try:
    from jax import config as _config

    _config.update("jax_enable_x64", True)
except Exception:
    pass

from .state import RunState
from . import vulcan_cfg

__all__ = [
    "__version__",
    "RunState",
    "vulcan_cfg",
    "make_config",
]


def make_config(**overrides: Any) -> _types.SimpleNamespace:
    """Create a config namespace from shipped defaults with user overrides.

    Example::

        cfg = vulcan_jax.make_config(nz=100, use_photo=False)
        rs = vulcan_jax.RunState.with_pre_loop_setup(cfg)
    """
    cfg = _types.SimpleNamespace(
        **{
            k: _copy.deepcopy(v)
            for k, v in vars(vulcan_cfg).items()
            if not k.startswith("_")
            and not callable(v)
            and not hasattr(v, "__loader__")
        }
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
