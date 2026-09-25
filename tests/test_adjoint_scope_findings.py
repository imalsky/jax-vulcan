"""Scope checks for the adjoint body map (`_adjoint_scope_findings`).

Pure host logic -- no arrays, no converged column, no Krylov -- so the whole
matrix is one parametrized case list. It is worth pinning because
`vulcan-jwst-tool` gates a release on `audit_adjoint_scope`, which builds its
verdict from these findings: a dropped or mis-severitied code there silently
downgrades a wrong gradient to a passing run.

The contract under test is the severity ladder, not the prose: a process that
is active and NOT carried by `body_terms` must be "error"; the same process
once `body_terms` carries it must drop to "info".
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from vulcan_jax.steady_state_grad import BodyTerms, _adjoint_scope_findings

SEVERITIES = {"error", "warning", "info"}

# The refresh feedback is unconditional, so every case carries it.
ALWAYS = ("atm_refresh_feedback", "info")

_MASK = jnp.ones((2, 3), dtype=bool)
_BOT = jnp.asarray([0], dtype=jnp.int32)

# (id, cfg, final_state, body_terms, photo_recompute_k, expected (code, severity))
CASES = [
    ("baseline_all_off", {}, None, None, None, ()),
    (
        "ion",
        {"use_ion": True},
        None,
        None,
        None,
        (("ion_charge_balance", "error"),),
    ),
    (
        "conden_uncarried",
        {"use_condense": True},
        None,
        None,
        None,
        (("condensation", "error"),),
    ),
    (
        "conden_uncarried_with_fix_species",
        {"use_condense": True, "fix_species": ["H2O"]},
        None,
        None,
        None,
        (("condensation", "error"), ("fix_species_pins", "error")),
    ),
    (
        "conden_carried_by_conden_static",
        {"use_condense": True},
        None,
        BodyTerms(conden_static=object()),
        None,
        (("condensation", "info"),),
    ),
    (
        "conden_carried_by_fix_mask_after_window",
        {"use_condense": True},
        {"fix_species_started": True},
        BodyTerms(fix_mask=_MASK),
        None,
        (("condensation", "info"),),
    ),
    (
        "conden_fix_started_but_uncarried",
        {"use_condense": True},
        {"fix_species_started": True},
        None,
        None,
        (("condensation", "error"), ("fix_species_pins", "error")),
    ),
    (
        "bottom_pins_uncarried",
        {"use_fix_all_bot": True},
        None,
        None,
        None,
        (
            ("bottom_boundary_pins", "error"),
            ("open_boundary_deflation", "info"),
        ),
    ),
    (
        "bottom_pins_carried",
        {"use_fix_all_bot": True},
        None,
        BodyTerms(bot_idx=_BOT),
        None,
        (
            ("bottom_boundary_pins", "info"),
            ("open_boundary_deflation", "info"),
        ),
    ),
    (
        "bottom_pins_via_sp_bot_dict",
        {"use_fix_sp_bot": {"H2O": 1.0e-4}},
        None,
        None,
        None,
        (
            ("bottom_boundary_pins", "error"),
            ("open_boundary_deflation", "info"),
        ),
    ),
    (
        "hycean_state_unknown",
        {"use_fix_H2He": True},
        None,
        None,
        None,
        (("hycean_h2he_pin", "error"),),
    ),
    (
        "hycean_not_yet_tripped",
        {"use_fix_H2He": True},
        {"h2he_pinned": False},
        None,
        None,
        (("hycean_h2he_pin", "warning"),),
    ),
    (
        "hycean_tripped_and_carried",
        {"use_fix_H2He": True},
        {"h2he_pinned": True},
        BodyTerms(bot_idx=_BOT),
        None,
        (("hycean_h2he_pin", "info"),),
    ),
    (
        "photo_frozen_J",
        {"use_photo": True},
        None,
        None,
        None,
        (("photolysis_feedback", "error"),),
    ),
    (
        "photo_with_recompute",
        {"use_photo": True},
        None,
        None,
        lambda y: y,
        (("photo_dflux_recursion", "info"),),
    ),
    (
        "diff_esc",
        {"diff_esc": ["H2"]},
        None,
        None,
        None,
        (
            ("escape_flux_feedback", "warning"),
            ("open_boundary_deflation", "info"),
        ),
    ),
    (
        "vm_mol_upwind",
        {"use_vm_mol": True},
        None,
        None,
        None,
        (("vm_mol_feedback", "warning"),),
    ),
    (
        "vm_mol_hybrid",
        {"use_vm_mol": True, "use_hybrid_vm_mol": True},
        None,
        None,
        None,
        (("vm_mol_hybrid", "info"),),
    ),
]


def _run(cfg_kw, state_kw, terms, photo_recompute_k):
    return _adjoint_scope_findings(
        SimpleNamespace(**cfg_kw),
        final_state=None if state_kw is None else SimpleNamespace(**state_kw),
        photo_recompute_k=photo_recompute_k,
        body_terms=terms,
    )


@pytest.mark.parametrize(
    "cfg_kw,state_kw,terms,photo_k,expected",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_scope_findings_match_expected_severities(
    cfg_kw, state_kw, terms, photo_k, expected
):
    """Each configuration emits exactly the expected codes at the expected
    severity -- no more, no fewer. Exactness is the point: an extra "error"
    blocks a valid gradient and a missing one ships a wrong gradient."""
    got = {(f["code"], f["severity"]) for f in _run(cfg_kw, state_kw, terms, photo_k)}
    assert got == {ALWAYS, *expected}


def test_every_finding_is_well_formed_and_unique():
    """Structural invariant over the union of every case: the three keys are
    always present, severity is from the ladder, the message is real prose,
    and no configuration emits the same code twice (a duplicate code would
    make a downstream `dict(findings)` silently drop one)."""
    seen_codes = set()
    for case_id, cfg_kw, state_kw, terms, photo_k, _ in CASES:
        findings = _run(cfg_kw, state_kw, terms, photo_k)
        codes = [f["code"] for f in findings]
        assert len(codes) == len(set(codes)), f"{case_id}: duplicate code in {codes}"
        for f in findings:
            assert set(f) == {"code", "severity", "message"}, f"{case_id}: {f.keys()}"
            assert f["severity"] in SEVERITIES, f"{case_id}: {f['severity']}"
            assert len(f["message"]) > 40, f"{case_id}/{f['code']}: stub message"
        seen_codes.update(codes)

    # 12 distinct codes across the emitter's 17 `add(...)` sites (several codes
    # are raised at more than one severity). If a code is added to the emitter
    # without a case here, this is what notices.
    assert len(seen_codes) == 12, sorted(seen_codes)
