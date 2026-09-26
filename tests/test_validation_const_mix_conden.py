"""Up-front refusal of two configs master cannot run.

1. A const_mix key that is not a network species: master's ini_y calls
   species.index(sp) and fails (build_atm.py:200).
2. A condense_sp entry outside the supported set: master's op.conden
   leaves its rate at zero.

H2S alone (saturation capping, no kinetics) is master-legal and must pass.
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
