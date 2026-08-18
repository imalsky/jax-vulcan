"""Upfront-validation coverage for two master-parity failure modes.

1. `const_mix` keys that are not network species. VULCAN-master fails on
   the same configuration (`build_atm.ini_y` calls `species.index(sp)`
   unconditionally — its shipped Earth example crashes on 'Ar' at
   `build_atm.py:200`), so inert/background gases are NOT live master
   physics to port. `validate_runtime_config` rejects the config upfront
   with an explanation rather than letting a bare `ValueError` surface deep
   inside setup.

2. `condense_sp` entries outside the supported condensate set. Master's
   `op.conden` has explicit branches for {H2O, NH3, H2SO4, S2, S4, S8, C}
   and silently leaves any other condensate's rate at zero; VULCAN-JAX
   raises. The validator lists both tiers (kinetics vs sat-data-only H2S)
   before any setup work runs. H2S alone stays master-legal (saturation
   capping without conden kinetics) and must validate cleanly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def test_const_mix_inert_key_rejected():
    from vulcan_jax import make_config
    from vulcan_jax.runtime_validation import validate_runtime_config

    cfg = make_config(ini_mix="const_mix", const_mix={"H2": 0.9, "Ar": 0.1})
    with pytest.raises(RuntimeError, match=r"const_mix key 'Ar'"):
        validate_runtime_config(cfg)


def test_const_mix_network_keys_accepted():
    from vulcan_jax import make_config
    from vulcan_jax.runtime_validation import validate_runtime_config

    cfg = make_config(ini_mix="const_mix", const_mix={"H2": 0.9, "He": 0.1})
    validate_runtime_config(cfg)  # must not raise


def test_condense_sp_unknown_rejected():
    from vulcan_jax import make_config
    from vulcan_jax.runtime_validation import validate_runtime_config

    cfg = make_config(use_condense=True, condense_sp=["Fe"])
    with pytest.raises(RuntimeError, match=r"condense_sp entry 'Fe'"):
        validate_runtime_config(cfg)


def test_condense_sp_h2s_sat_only_accepted():
    from vulcan_jax import make_config
    from vulcan_jax.runtime_validation import validate_runtime_config

    cfg = make_config(use_condense=True, condense_sp=["H2S"])
    validate_runtime_config(cfg)  # sat-only tier is master-legal; must not raise


def test_kinetics_set_matches_sat_set():
    """The kinetics tier must stay a subset of the saturation tier, and the
    only sat-only species must be H2S (drift guard between the two constants)."""
    from vulcan_jax.atm_setup import _SUPPORTED_CONDENSABLES
    from vulcan_jax.conden import SUPPORTED_CONDEN_KINETICS

    assert set(SUPPORTED_CONDEN_KINETICS) <= set(_SUPPORTED_CONDENSABLES)
    assert set(_SUPPORTED_CONDENSABLES) - set(SUPPORTED_CONDEN_KINETICS) == {"H2S"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
