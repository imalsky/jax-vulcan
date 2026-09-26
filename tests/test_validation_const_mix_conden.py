"""Up-front refusal of two configs master cannot run.

1. A const_mix key that is not a network species: master's ini_y calls
   species.index(sp) and fails (build_atm.py:200).
2. A condense_sp entry outside the supported set: master's op.conden
   leaves its rate at zero.

H2S alone (saturation capping, no kinetics) is master-legal and must pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


@pytest.mark.parametrize(
    "overrides, error",
    [
        (dict(ini_mix="const_mix", const_mix={"H2": 0.9, "Ar": 0.1}), r"const_mix key 'Ar'"),
        (dict(ini_mix="const_mix", const_mix={"H2": 0.9, "He": 0.1}), None),
        (dict(use_condense=True, condense_sp=["Fe"]), r"condense_sp entry 'Fe'"),
        (dict(use_condense=True, condense_sp=["H2S"]), None),  # sat-only tier
    ],
    ids=["inert_const_mix_key", "network_const_mix_keys", "unknown_condensate", "h2s_sat_only"],
)
def test_up_front_validation(overrides, error):
    from vulcan_jax import make_config
    from vulcan_jax.runtime_validation import validate_runtime_config

    cfg = make_config(**overrides)
    if error is None:
        validate_runtime_config(cfg)
    else:
        with pytest.raises(RuntimeError, match=error):
            validate_runtime_config(cfg)


def test_kinetics_set_matches_sat_set():
    """The kinetics tier must stay a subset of the saturation tier, and the
    only sat-only species must be H2S (drift guard between the two constants)."""
    from vulcan_jax.atm_setup import _SUPPORTED_CONDENSABLES
    from vulcan_jax.conden import SUPPORTED_CONDEN_KINETICS

    assert set(SUPPORTED_CONDEN_KINETICS) <= set(_SUPPORTED_CONDENSABLES)
    assert set(_SUPPORTED_CONDENSABLES) - set(SUPPORTED_CONDEN_KINETICS) == {"H2S"}
