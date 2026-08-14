"""Guards that keep condensation honest under differentiation.

Condensation is a forward-model capability only. The completed, pinned
condensation state is not differentiable-through (transient snapshot, discrete
phase-boundary switches; pinned-species jvp vs FD ~0.91 rel). These tests pin
the guards that enforce that contract:

* forward-config hardening (`_validate_condensation`), lifted into the core so a
  bare VULCAN-JAX run is guarded like the retrieval wrapper;
* the input-sensitivity refusal (`steady_state_input_sensitivity` raises on a
  condensation-active state, both in-window and post-pin).

Full scope: ../README.md (Differentiability).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from vulcan_jax.runtime_validation import _validate_condensation
from vulcan_jax.steady_state_grad import BodyTerms, steady_state_input_sensitivity


def _cfg(**overrides):
    base = dict(
        use_condense=True,
        use_moldiff=True,
        condense_sp=["H2O"],
        start_conden_time=0.0,
        stop_conden_time=1.0e5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- forward-config hardening (F3) ------------------------------------------


def test_conden_off_is_a_noop():
    assert _validate_condensation(SimpleNamespace(use_condense=False)) == []


def test_valid_conden_config_passes():
    assert _validate_condensation(_cfg()) == []


def test_conden_requires_moldiff():
    errs = _validate_condensation(_cfg(use_moldiff=False))
    assert any("use_moldiff" in e for e in errs)


def test_conden_rejects_empty_condense_sp():
    errs = _validate_condensation(_cfg(condense_sp=[]))
    assert any("empty condense_sp" in e for e in errs)


def test_conden_rejects_inverted_window():
    errs = _validate_condensation(_cfg(start_conden_time=100.0, stop_conden_time=10.0))
    assert any("would never open" in e for e in errs)


def test_conden_rejects_unsupported_species():
    errs = _validate_condensation(_cfg(condense_sp=["NOPE"]))
    assert any("not a supported condensate" in e for e in errs)


# --- input-sensitivity refusal (F1) -----------------------------------------


def _dummy_input_sensitivity(body_terms, **kw):
    # The condensation refusal fires before any solver work, so the remaining
    # arguments are never touched; dummies are fine.
    return steady_state_input_sensitivity(
        lambda y: y.sum(),
        jnp.ones((2, 3)),
        jnp.ones((5, 2)),
        None,
        None,
        0.0,
        lambda p: (None, None),
        compo_array=jnp.ones((3, 2)),
        dz=jnp.ones(2),
        body_terms=body_terms,
        **kw,
    )


def test_input_sensitivity_refuses_pinned_condensation():
    bt = BodyTerms(fix_mask=jnp.ones((2, 3), dtype=bool), fix_y=jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="reliably differentiable"):
        _dummy_input_sensitivity(bt)


def test_input_sensitivity_refuses_inwindow_condensation():
    # A non-None conden_static (any object) marks the in-window regime.
    bt = BodyTerms(conden_static=object())
    with pytest.raises(ValueError, match="reliably differentiable"):
        _dummy_input_sensitivity(bt)


def test_input_sensitivity_no_condensation_passes_the_guard():
    # Without condensation the refusal does not fire; the call proceeds past the
    # guard and fails later on the dummy arguments (NOT with the conden message).
    with pytest.raises(Exception) as exc:
        _dummy_input_sensitivity(BodyTerms())
    assert "reliably differentiable" not in str(exc.value)
