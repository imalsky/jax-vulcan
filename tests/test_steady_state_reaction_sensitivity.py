"""Tests for the production reverse-mode steady-state reaction sensitivity.

Two layers:

* Fast unit tests of the load-bearing algebra — the log-abundance zero mask,
  the conserved-mass null-space deflation, the projector, and the
  `dL/d(ln k)` chain-rule assembly. These need no converged column and run in
  milliseconds (they are where a wrong `dz`/`compo` slice or a sign error would
  show up).
* A slow HD189 fixture regression that runs the full adjoint on a real, closed
  column and reproduces the documented finite-difference anchors. It pays the
  step-VJP cold compile (~10-20 min), so it is skipped unless
  `VULCAN_JAX_RUN_SLOW=1`.

End-to-end correctness on production columns (HD189, WASP-39b SO2) is what the
slow test and `jax_paper/scripts/adj_solvermap_gmres.py` cover; the fast tests
cannot, because `jax_step.jax_ros2_step` uses import-baked atom-projection
tables sized for the active network, not an arbitrary synthetic one.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from vulcan_jax.steady_state_grad import (
    steady_state_reaction_sensitivity,
    _safe_inv_y,
    _conserved_null_basis,
    _BODY_MAP_DT_MAX,
)

DATA = Path(__file__).resolve().parent / "data"
HD189_FIXTURE = DATA / "adj_state_hd189.npz"
_RUN_SLOW = os.environ.get("VULCAN_JAX_RUN_SLOW") == "1"

# Centered-finite-difference truth for the HD189 photo-off CH4 loss, copied from
# jax_paper/scripts/adj_solvermap_gmres.py (the validated reference run). The
# 13/14 and 115/116 pairs are a forward/reverse rate pair each.
HD189_FD_ANCHORS = {13: -5.651e-01, 14: +5.651e-01, 115: -1.919e-05, 116: +2.712e-05}


# --------------------------------------------------------------------------- #
# Fast unit tests (no converged column, no Krylov)                            #
# --------------------------------------------------------------------------- #


def test_safe_inv_y_masks_exact_zeros():
    """1/y* maps exact zeros to 0, not inf (the closed-column NaN guard)."""
    y = jnp.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0]])
    inv = _safe_inv_y(y)
    assert bool(jnp.all(jnp.isfinite(inv)))
    # nonzero entries are reciprocated, zeros stay zero
    assert np.allclose(np.asarray(inv), np.array([[1.0, 0.0, 0.25], [0.0, 0.5, 0.0]]))


def test_conserved_null_basis_annihilates_atom_vectors():
    """The deflation projector kills the analytic atom-count vectors.

    This is the single most load-bearing correctness property: a wrong `dz`,
    a mis-sliced `compo`, or a wrong log-space scaling all break it.
    """
    rng = np.random.default_rng(0)
    nz, ni, n_atoms = 6, 4, 3
    y_star = jnp.asarray(rng.uniform(1.0, 10.0, size=(nz, ni)))
    compo = jnp.asarray(rng.integers(0, 3, size=(ni, n_atoms)).astype(float))
    dz = jnp.asarray(rng.uniform(0.5, 2.0, size=nz))

    Q = _conserved_null_basis(y_star, compo, dz)

    # Q is orthonormal and spans exactly the populated atom columns.
    n_pop = int((np.asarray(compo).sum(axis=0) > 0).sum())
    assert Q.shape == (nz * ni, n_pop)
    assert float(jnp.linalg.norm(Q.T @ Q - jnp.eye(Q.shape[1]))) < 1e-10

    # proj annihilates every log-space atom-count vector c_e.
    Q_np = np.asarray(Q)
    compo_np = np.asarray(compo)
    for e in np.where(compo_np.sum(axis=0) > 0)[0]:
        c_e = (
            np.asarray(y_star) * (compo_np[:, e][None, :] * np.asarray(dz)[:, None])
        ).ravel()
        proj_c = c_e - Q_np @ (Q_np.T @ c_e)
        assert np.linalg.norm(proj_c) / np.linalg.norm(c_e) < 1e-12


def test_conserved_null_basis_requires_populated_columns():
    """An all-zero compo (no atoms) is a clear error, not a silent empty solve."""
    y_star = jnp.ones((3, 2))
    compo = jnp.zeros((2, 4))
    dz = jnp.ones((3,))
    with pytest.raises(ValueError, match="no populated atom columns"):
        _conserved_null_basis(y_star, compo, dz)


def test_conserved_null_basis_rejects_rank_deficiency():
    """Linearly dependent atom columns fail fast instead of silently deflating
    an arbitrary (non-null) direction from the unpivoted QR."""
    rng = np.random.default_rng(3)
    nz, ni = 4, 3
    y_star = jnp.asarray(rng.uniform(1.0, 5.0, size=(nz, ni)))
    dz = jnp.asarray(rng.uniform(0.5, 2.0, size=nz))
    # Two identical atom columns -> the c_e stack is rank 1.
    col = np.array([1.0, 2.0, 0.0])
    compo = jnp.asarray(np.stack([col, col], axis=1))
    with pytest.raises(ValueError, match="rank-deficient"):
        _conserved_null_basis(y_star, compo, dz)


def test_conserved_null_basis_rejects_zero_column():
    """An atom whose every carrier has y* == 0 is degenerate -> hard error."""
    y_star = jnp.asarray(np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]]))
    dz = jnp.ones((3,))
    compo = jnp.asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))  # atom 1 only in sp 1
    with pytest.raises(ValueError, match="all-zero"):
        _conserved_null_basis(y_star, compo, dz)


def test_deflation_projector_idempotent():
    """proj(proj(z)) == proj(z) and proj removes the spanned directions."""
    rng = np.random.default_rng(1)
    nz, ni = 5, 3
    y_star = jnp.asarray(rng.uniform(1.0, 5.0, size=(nz, ni)))
    compo = jnp.asarray(np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    dz = jnp.asarray(rng.uniform(0.5, 2.0, size=nz))
    Q = _conserved_null_basis(y_star, compo, dz)

    def proj(z):
        zf = z.reshape(-1)
        return (zf - Q @ (Q.T @ zf)).reshape(nz, ni)

    z = jnp.asarray(rng.normal(size=(nz, ni)))
    pz = proj(z)
    assert float(jnp.max(jnp.abs(proj(pz) - pz))) < 1e-12
    # a vector already in the null space projects to ~0
    null_vec = jnp.asarray(np.asarray(Q)[:, 0].reshape(nz, ni))
    assert float(jnp.linalg.norm(proj(null_vec))) < 1e-10


def test_reaction_cotangent_chain_rule_identity():
    """Validate the assembly `dL/d(ln k_r) = (k .* G_k^T lambda)_r`.

    Uses a cheap surrogate body map so the identity (vjp contraction + the
    d/d(ln k) = k * d/dk chain rule, summed over layers) is checked in
    isolation, independent of `jax_ros2_step`.
    """
    rng = np.random.default_rng(2)
    nrp1, nz, ni = 5, 3, 4
    k = jnp.asarray(rng.uniform(0.1, 2.0, size=(nrp1, nz)))
    lam = jnp.asarray(rng.normal(size=(nz, ni)))
    W = jnp.asarray(rng.normal(size=(nrp1, ni)))

    def surrogate(kk):  # (nrp1, nz) -> (nz, ni)
        return jnp.sin(kk).T @ W

    # The function's assembly:
    _, vjp_k = jax.vjp(surrogate, k)
    (cot_k,) = vjp_k(lam)
    assembled = (k * cot_k).sum(axis=1)

    # Independent reference: grad wrt ln k of <lam, surrogate(exp(ln k))>, summed
    # over layers (the global per-reaction scaling FD perturbs).
    ref = jax.grad(lambda lnk: jnp.vdot(lam, surrogate(jnp.exp(lnk))))(jnp.log(k)).sum(
        axis=1
    )
    assert float(jnp.max(jnp.abs(assembled - ref))) < 1e-9


def test_poor_convergence_warns():
    """Default-on diagnostics: a poor residual or a loose fixed point warns even
    when the caller never inspects `info`; a good solve is silent."""
    import warnings as _w

    from vulcan_jax.steady_state_grad import _warn_poor_convergence

    with pytest.warns(UserWarning, match="stagnation"):
        _warn_poor_convergence(0.5, 0.0)
    with pytest.warns(UserWarning, match="fixed point"):
        _warn_poor_convergence(0.0, 0.1)
    with pytest.warns(UserWarning, match="ensemble spread"):
        _warn_poor_convergence(0.0, 0.0, 0.3)
    with _w.catch_warnings():
        _w.simplefilter("error")  # a well-converged solve must not warn
        _warn_poor_convergence(1e-3, 1e-4, 0.05)


def test_body_dt_danger_zone_rejected():
    """A body_dt in the near-singular danger zone is refused up front."""
    dummy = jnp.ones((2, 2))
    with pytest.raises(ValueError, match="danger zone"):
        steady_state_reaction_sensitivity(
            lambda y: y.sum(),
            dummy,
            jnp.ones((3, 2)),
            None,
            None,
            compo_array=jnp.ones((2, 1)),
            dz=jnp.ones((2,)),
            body_dt=10 * _BODY_MAP_DT_MAX,
        )


# --------------------------------------------------------------------------- #
# Slow end-to-end regression on a real HD189 column                           #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.skipif(not HD189_FIXTURE.exists(), reason="HD189 adjoint fixture missing")
@pytest.mark.skipif(
    not _RUN_SLOW,
    reason="slow reverse-mode HD189 regression; set VULCAN_JAX_RUN_SLOW=1 to run",
)
def test_hd189_reaction_sensitivity_regression():
    """Reproduce the documented HD189 reaction-ranking finite-difference anchors.

    Runs the full solver-map adjoint (defaults: body_dt=1e7, n_solves=3
    ensemble) on the saved converged HD189 (photo-off) state and checks the
    ensemble-mean CH4 sensitivity against the centered finite-difference truth
    from `adj_solvermap_gmres.py`. At body_dt=1e7 the solve reaches a low
    residual (0.04-0.15 measured) and independent twins land 0.3-6% from FD
    (mean 3.5%; docs/notes.md 2026-07-01 campaign), so the anchor tolerance is
    12% on the ensemble mean, with resid/spread guards. At the old body_dt=1e8
    default the solve stagnated and single samples bounced ±25% (band
    -0.43..-0.63 vs FD -0.565) — the dt map lives in `BODY_MAP_DT`'s comment.
    Sign and top-6 ranking remain the strongest assertions.
    """
    import vulcan_jax.chem_funs as chem_funs
    from vulcan_jax.jax_step import AtmStatic

    d = np.load(HD189_FIXTURE, allow_pickle=True)
    nz = int(d["nz"])
    y_star = jnp.asarray(d["y_star"])
    k_arr = jnp.asarray(d["k_arr"])
    dz = jnp.asarray(d["dz"])
    compo = jnp.asarray(d["compo"])
    atm = AtmStatic(
        **{k[5:]: jnp.asarray(d[k]) for k in d.files if k.startswith("atm__")},
        **{k[9:]: bool(d[k]) for k in d.files if k.startswith("atmbool__")},
    )
    net = chem_funs._NET_JAX
    ch4 = chem_funs._NETWORK.species_idx["CH4"]
    L0 = nz // 2

    def loss(y):  # log10 CH4 VMR at mid-column (matches the reference run)
        ymix = y / jnp.sum(y, axis=1, keepdims=True)
        return jnp.log10(ymix[L0, ch4])

    dLdlnk, info = steady_state_reaction_sensitivity(
        loss,
        y_star,
        k_arr,
        atm,
        net,
        compo_array=compo,
        dz=dz,
        lgmres_inner_m=250,
        lgmres_cycles=8,
        return_info=True,
    )
    dLdlnk = np.asarray(dLdlnk)

    assert info["fp_err"] < 1e-1, (
        f"y* is not a body-map fixed point: {info['fp_err']:.2e}"
    )
    # null_quality is max_e ||A_eta^T q_e|| relative to the operator's action
    # on a random unit direction. Measured ~3e-5 at body_dt=1e8 (2026-07-01);
    # a genuinely non-null deflated direction (broken conservation) reads O(1).
    assert info["null_quality"] < 1e-3
    # Solver-regime guards, calibrated 2026-07-01 (measured per-twin residuals
    # {0.29, 0.05, 0.10}, spread 0.047): the ensemble MEDIAN residual must stay
    # out of the stagnation regime (one wandering twin is tolerated — the
    # best-iterate safeguard bounds it and the mean absorbs it), and the twins
    # must agree on the top-10 reactions.
    assert float(np.median(info["resids"])) < 0.2, (
        f"median resid {np.median(info['resids']):.2e} — stagnation regime"
    )
    assert info["resid"] < 0.5, f"max twin resid {info['resid']:.2e}"
    assert info["ensemble_spread"] < 0.15, (
        f"ensemble spread {info['ensemble_spread']:.2e} — twins disagree"
    )
    assert np.all(np.isfinite(dLdlnk))

    # Sign and ranking are the most robust assertions.
    assert dLdlnk[13] < 0 and dLdlnk[14] > 0
    top = np.argsort(np.abs(dLdlnk[: net.nr + 1]))[::-1][:6]
    assert 13 in top or 14 in top, f"dominant CH4 reaction missing from top-6: {top}"

    # Finite-difference agreement of the ensemble mean on the dominant pair
    # (measured 3.5% mean over twins at body_dt=1e7; 12% gives headroom).
    for r in (13, 14):
        rel = abs(dLdlnk[r] - HD189_FD_ANCHORS[r]) / abs(HD189_FD_ANCHORS[r])
        assert rel < 0.12, (
            f"r{r}: {dLdlnk[r]:+.3e} vs FD {HD189_FD_ANCHORS[r]:+.3e} (rel {rel:.2f})"
        )


def main() -> int:
    """Script-style runner for the always-on fast unit tests."""
    test_safe_inv_y_masks_exact_zeros()
    test_conserved_null_basis_annihilates_atom_vectors()
    test_conserved_null_basis_requires_populated_columns()
    test_conserved_null_basis_rejects_rank_deficiency()
    test_conserved_null_basis_rejects_zero_column()
    test_deflation_projector_idempotent()
    test_reaction_cotangent_chain_rule_identity()
    return 0


def test_main():
    """Pytest entry consistency with the suite's `test_main()` convention."""
    assert main() == 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
