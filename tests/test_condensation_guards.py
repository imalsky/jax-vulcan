"""Guards that keep condensation honest under differentiation.

Condensation is a forward-model capability only. The completed, pinned
condensation state is not differentiable-through (transient snapshot, discrete
phase-boundary switches; pinned-species jvp vs FD ~0.91 rel). These tests pin
the guards that enforce that contract:

* forward-config hardening (`_validate_condensation`), lifted into the core so a
  bare VULCAN-JAX run is guarded like the retrieval wrapper;
* the input-sensitivity refusal (`steady_state_input_sensitivity` raises on a
  condensation-active state, both in-window and post-pin).

Full scope: notes.md (Differentiability).
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


@pytest.mark.parametrize("overrides,expect", [
    (dict(_off=True), None),                       # conden off -> no errors
    (dict(), None),                                # valid config -> no errors
    (dict(use_moldiff=False), "use_moldiff"),
    (dict(condense_sp=[]), "empty condense_sp"),
    (dict(start_conden_time=100.0, stop_conden_time=10.0), "would never open"),
    (dict(condense_sp=["NOPE"]), "not a supported condensate"),
], ids=["off_noop", "valid", "needs_moldiff", "empty_sp", "inverted_window",
        "unsupported_species"])
def test_validate_condensation(overrides, expect):
    if overrides.pop("_off", False):
        errs = _validate_condensation(SimpleNamespace(use_condense=False))
    else:
        errs = _validate_condensation(_cfg(**overrides))
    if expect is None:
        assert errs == []
    else:
        assert any(expect in e for e in errs), errs


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


def test_input_sensitivity_refuses_condensation_active_states():
    """Refused both post-pin (fix_mask set) and in-window (conden_static);
    without condensation the guard passes and failure comes later from the
    dummy arguments, never with the conden message."""
    bt = BodyTerms(fix_mask=jnp.ones((2, 3), dtype=bool),
                   fix_y=jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="reliably differentiable"):
        _dummy_input_sensitivity(bt)
    with pytest.raises(ValueError, match="reliably differentiable"):
        _dummy_input_sensitivity(BodyTerms(conden_static=object()))
    with pytest.raises(Exception) as exc:
        _dummy_input_sensitivity(BodyTerms())
    assert "reliably differentiable" not in str(exc.value)
