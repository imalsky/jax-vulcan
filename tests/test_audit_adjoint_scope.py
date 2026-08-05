"""Units for the adjoint scope machinery: `audit_adjoint_scope` static
findings, `_guard_unmodeled_processes` fingerprints, and `BodyTerms` plumbing.

The adjoint body map covers only ros2_step (+ renorm) (+ photo) (+ body_terms).
These tests pin the severity class of every dropped runner process (error =
active and wrong, warning = leading-order-only, info = second-order); a
body-map change that starts covering one must update the scan in the same pass.
"""

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest

from vulcan_jax.steady_state_grad import (
    BodyTerms,
    PHOTO_RECOMPUTE_AUTO,
    _adjoint_scope_findings,
    _guard_unmodeled_processes,
    _clip_dead_mask,
    _make_body_map,
    _resolve_photo_recompute_k,
)


def _cfg(**overrides):
    base = dict(
        use_photo=False,
        use_ion=False,
        use_condense=False,
        use_topflux=False,
        use_botflux=False,
        use_fix_sp_bot={},
        use_fix_all_bot=False,
        use_fix_H2He=False,
        use_vm_mol=False,
        diff_esc=[],
        fix_species=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _codes(findings, severity=None):
    return {
        f["code"] for f in findings if severity is None or f["severity"] == severity
    }


def test_default_photo_off_column_has_no_errors():
    findings = _adjoint_scope_findings(_cfg())
    assert _codes(findings, "error") == set()
    # Atm-refresh feedback is always on in the runner, never in the body map,
    # so it must always be reported.
    assert "atm_refresh_feedback" in _codes(findings, "info")


def test_photo_without_recompute_warns_and_with_recompute_downgrades():
    frozen = _adjoint_scope_findings(_cfg(use_photo=True))
    assert "photolysis_feedback" in _codes(frozen, "warning")

    fed = _adjoint_scope_findings(_cfg(use_photo=True), photo_recompute_k=lambda y: y)
    assert "photolysis_feedback" not in _codes(fed)
    assert "photo_dflux_recursion" in _codes(fed, "info")


def test_ion_column_is_an_error():
    findings = _adjoint_scope_findings(_cfg(use_ion=True))
    assert "ion_charge_balance" in _codes(findings, "error")


def test_condensation_is_an_error_and_state_refines_fix_species():
    plain = _adjoint_scope_findings(_cfg(use_condense=True))
    assert "condensation" in _codes(plain, "error")

    # No final state but fix_species configured: flagged conservatively.
    configured = _adjoint_scope_findings(_cfg(use_condense=True, fix_species=["H2O"]))
    assert "fix_species_pins" in _codes(configured, "error")

    # Pin tripped: active error.
    tripped = _adjoint_scope_findings(
        _cfg(use_condense=True, fix_species=["H2O"]),
        final_state=SimpleNamespace(fix_species_started=np.bool_(True)),
    )
    assert "fix_species_pins" in _codes(tripped, "error")

    # Pin not tripped: pin finding dropped, condensation error kept.
    untripped = _adjoint_scope_findings(
        _cfg(use_condense=True, fix_species=["H2O"]),
        final_state=SimpleNamespace(fix_species_started=np.bool_(False)),
    )
    assert "fix_species_pins" not in _codes(untripped)
    assert "condensation" in _codes(untripped, "error")


def test_bottom_pins_error_and_open_boundary_info():
    for cfg in (
        _cfg(use_fix_all_bot=True),
        _cfg(use_fix_sp_bot={"H2O": 1e-3}),
    ):
        findings = _adjoint_scope_findings(cfg)
        assert "bottom_boundary_pins" in _codes(findings, "error")
        assert "open_boundary_deflation" in _codes(findings, "info")


def test_hycean_pin_severity_tracks_trip_state():
    armed = _adjoint_scope_findings(_cfg(use_fix_H2He=True))
    assert "hycean_h2he_pin" in _codes(armed, "error")

    pre_trip = _adjoint_scope_findings(
        _cfg(use_fix_H2He=True),
        final_state=SimpleNamespace(h2he_pinned=np.bool_(False)),
    )
    assert "hycean_h2he_pin" in _codes(pre_trip, "warning")
    assert "hycean_h2he_pin" not in _codes(pre_trip, "error")


def test_escape_vm_mol_and_boundary_flux_classification():
    findings = _adjoint_scope_findings(
        _cfg(diff_esc=["H"], use_vm_mol=True, use_topflux=True)
    )
    assert "escape_flux_feedback" in _codes(findings, "warning")
    assert "vm_mol_feedback" in _codes(findings, "warning")
    assert "open_boundary_deflation" in _codes(findings, "info")
    assert _codes(findings, "error") == set()


# --- body_terms downgrades the findings the terms actually cover ---


def test_body_terms_downgrade_conden_and_pins():
    conden_terms = BodyTerms(conden_static=object(), hydro_partial=True)
    f = _adjoint_scope_findings(_cfg(use_condense=True), body_terms=conden_terms)
    assert "condensation" in _codes(f, "info")
    assert "condensation" not in _codes(f, "error")

    pin_terms = BodyTerms(bot_idx=jnp.asarray([0], dtype=jnp.int32))
    f = _adjoint_scope_findings(_cfg(use_fix_all_bot=True), body_terms=pin_terms)
    assert "bottom_boundary_pins" in _codes(f, "info")
    assert "bottom_boundary_pins" not in _codes(f, "error")

    fix_terms = BodyTerms(fix_mask=object(), fix_y=object())
    f = _adjoint_scope_findings(
        _cfg(use_condense=True, fix_species=["H2O"]),
        final_state=SimpleNamespace(fix_species_started=np.bool_(True)),
        body_terms=fix_terms,
    )
    assert "fix_species_pins" not in _codes(f)
    assert "condensation" in _codes(f, "info")


# --- fingerprint guard: raise/warn instead of silently-wrong gradients ---


def _fake_network(nr, ion_rows=(), conden_rows=(), photo_rows=()):
    def mask(rows):
        m = np.zeros(nr + 1, dtype=bool)
        for r in rows:
            m[r] = True
        return m

    return SimpleNamespace(
        nr=nr,
        is_ion=mask(ion_rows),
        is_conden=mask(conden_rows),
        is_photo=mask(photo_rows),
    )


def test_guard_raises_on_active_conden_rows_without_terms():
    nz, ni, nr = 3, 4, 6
    y = np.ones((nz, ni))
    k = np.zeros((nr + 1, nz))
    k[5] = 1.0  # active conden row
    net = SimpleNamespace(nr=nr)
    fake = _fake_network(nr, conden_rows=(5,))
    with pytest.raises(ValueError, match="condensation active"):
        _guard_unmodeled_processes(y, k, net, None, None, network=fake, species=[])
    # conden terms (or fix_species pins) clear it
    _guard_unmodeled_processes(
        y, k, net, BodyTerms(conden_static=object()), None, network=fake, species=[]
    )
    _guard_unmodeled_processes(
        y, k, net, BodyTerms(fix_mask=object()), None, network=fake, species=[]
    )


def test_guard_raises_on_populated_condensate_species():
    nz, ni, nr = 3, 3, 2
    y = np.zeros((nz, ni))
    y[:, :2] = 1.0
    k = np.zeros((nr + 1, nz))
    net = SimpleNamespace(nr=nr)
    fake = _fake_network(nr)
    species = ["H2", "H2O", "H2O_l_s"]
    # dry condensate column: fine
    _guard_unmodeled_processes(y, k, net, None, None, network=fake, species=species)
    y[10 % nz, 2] = 1e3  # populated condensate -> conden was active
    with pytest.raises(ValueError, match="condensation active"):
        _guard_unmodeled_processes(y, k, net, None, None, network=fake, species=species)


def test_guard_raises_on_active_ion_rows():
    nz, ni, nr = 2, 3, 4
    y = np.ones((nz, ni))
    k = np.zeros((nr + 1, nz))
    k[2] = 1.0
    net = SimpleNamespace(nr=nr)
    fake = _fake_network(nr, ion_rows=(2,))
    with pytest.raises(NotImplementedError, match="ion"):
        _guard_unmodeled_processes(y, k, net, None, None, network=fake, species=[])


def test_guard_warns_on_frozen_photolysis():
    nz, ni, nr = 2, 3, 4
    y = np.ones((nz, ni))
    k = np.zeros((nr + 1, nz))
    k[1] = 1.0
    net = SimpleNamespace(nr=nr)
    fake = _fake_network(nr, photo_rows=(1,))
    with pytest.warns(UserWarning, match="photolysis"):
        _guard_unmodeled_processes(y, k, net, None, None, network=fake, species=[])
    # passing photo_recompute_k silences it
    _guard_unmodeled_processes(y, k, net, None, lambda yy: yy, network=fake, species=[])


def test_auto_photo_recompute_requires_context_on_default_map():
    nz, nr = 2, 4
    k = np.zeros((nr + 1, nz))
    k[1] = 1.0
    net = SimpleNamespace(nr=nr)
    fake = _fake_network(nr, photo_rows=(1,))

    with pytest.raises(ValueError, match="photo_recompute_k='auto'"):
        _resolve_photo_recompute_k(
            PHOTO_RECOMPUTE_AUTO,
            k,
            net,
            "renorm",
            network=fake,
        )

    assert (
        _resolve_photo_recompute_k(
            PHOTO_RECOMPUTE_AUTO,
            k,
            net,
            "bare",
            network=fake,
        )
        is None
    )
    assert _resolve_photo_recompute_k(None, k, net, "renorm", network=fake) is None


def test_auto_photo_recompute_is_noop_without_active_photo_rows():
    nz, nr = 2, 4
    k = np.zeros((nr + 1, nz))
    net = SimpleNamespace(nr=nr)
    fake = _fake_network(nr, photo_rows=(1,))

    assert (
        _resolve_photo_recompute_k(
            PHOTO_RECOMPUTE_AUTO,
            k,
            net,
            "renorm",
            network=fake,
        )
        is None
    )


def test_body_terms_require_renorm_map():
    terms = BodyTerms(gas_mask=jnp.ones(3, dtype=bool))
    atm = SimpleNamespace(M=jnp.ones(2))
    with pytest.raises(ValueError, match="renorm"):
        _make_body_map(None, None, atm, None, 1e7, "bare", None, terms)


# --- zero-clip dead cells ---------------------------------------------------
# The per-step clip is deliberately outside the body map. Where it fires the
# runner zeroes the cell while the map keeps the raw step, so the cell has no
# fixed point and its defect measures the clip, not physics. These pin that
# the mask mirrors outer_loop._make_clip_fn term for term.


def _clip_cfg(**kw):
    base = dict(pos_cut=0.0, nega_cut=-1.0, mtol=1e-22)
    base.update(kw)
    return SimpleNamespace(**base)


def test_clip_dead_mask_matches_the_runner_clip_terms():
    cfg = _clip_cfg()
    ymix = np.full((1, 6), 1e-10)
    ymix[0, 5] = 1e-30  # below mtol -> trace-negative cut
    G = np.array([[1e10, -0.5, -5.0, 0.0, 1e-3, -1e12]])
    dead = _clip_dead_mask(G, ymix, cfg)[0]
    assert not dead[0]  # healthy positive density: clip is identity
    assert dead[1]  # in [nega_cut, pos_cut) -> runner zeroes it
    assert not dead[2]  # below nega_cut: the runner LEAVES it (not clipped)
    assert not dead[3]  # exactly 0 -> clipping is already the identity
    assert not dead[4]  # small but positive, above pos_cut=0
    assert dead[5]  # ymix_old < mtol and negative -> trace cut


def test_clip_dead_mask_is_absolute_not_a_mixing_ratio():
    """Pins that the clip window is absolute (cm^-3), not a mixing ratio:
    the same ymix (1e-16) is clip-dead at the cold top and healthy deep down,
    so min_ymix cannot do this job."""
    cfg = _clip_cfg()
    ymix = np.array([[1e-16, 1e-16]])
    G = np.array([[-7.5e-4, -1e3]])  # same ymix, different densities
    dead = _clip_dead_mask(G, ymix, cfg)[0]
    assert dead[0]  # cold top: inside the |nega_cut| = 1 window
    assert not dead[1]  # deep: far outside it


def test_clip_dead_mask_is_inert_without_cfg():
    # no cfg -> nothing to mirror -> exclude nothing (cells stay in the scan)
    G = np.array([[-0.5, -0.5]])
    ymix = np.full((1, 2), 1e-10)
    assert not _clip_dead_mask(G, ymix, None).any()


def test_clip_dead_mask_honours_a_positive_pos_cut():
    # pos_cut > 0 also zeroes small POSITIVE values; the mask must follow it
    cfg = _clip_cfg(pos_cut=10.0)
    G = np.array([[5.0, 50.0]])
    ymix = np.full((1, 2), 1e-10)
    dead = _clip_dead_mask(G, ymix, cfg)[0]
    assert dead[0] and not dead[1]
