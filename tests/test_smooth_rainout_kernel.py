"""Unit tests for the smooth-rainout sink kernel (Route B B0-1).

Network-independent (pure array kernel): FD closure through ALL inputs
(n, n_sat, C), the pointwise w->0 legacy-branch limit, exact zero at and
below saturation, safe-arithmetic guards at n_sat extremes, C1 continuity
across both hinge breakpoints, and jit/vmap consistency.
"""

import jax
import jax.numpy as jnp
import numpy as np

from vulcan_jax.conden import N_SAT_FLOOR, smooth_rainout_loss

W = 0.1


def _mk(nz=7):
    n_sat = jnp.asarray(np.geomspace(1e2, 1e14, nz))
    # saturation ratios from deep-subsaturated through the quadratic hinge
    # region into the linear branch
    s_ratio = jnp.asarray([0.0, 0.5, 1.0, 1.0 + W / 3, 1.0 + W, 2.0, 50.0])
    n = n_sat * s_ratio
    C = jnp.asarray(np.geomspace(1e-9, 1e-5, nz))
    return n, C, n_sat


def test_zero_at_and_below_saturation():
    n, C, n_sat = _mk()
    L, dL = smooth_rainout_loss(n, C, n_sat, W)
    assert float(jnp.max(jnp.abs(L[:3]))) == 0.0
    assert float(jnp.max(jnp.abs(dL[:3]))) == 0.0
    assert bool(jnp.all(L[3:] > 0.0))


def test_w_to_zero_pointwise_limit_is_legacy_branch():
    n, C, n_sat = _mk()
    L0, _ = smooth_rainout_loss(n, C, n_sat, 0.0)
    legacy = C * n * jnp.maximum(n - n_sat, 0.0)
    np.testing.assert_allclose(np.asarray(L0), np.asarray(legacy), rtol=0.0)
    # and the finite-w kernel converges to it as w shrinks; the hinge
    # offset makes the relative error O(w/s), so the bound scales with w
    Lw, _ = smooth_rainout_loss(n, C, n_sat, 1e-10)
    live = np.asarray(legacy) > 0
    np.testing.assert_allclose(
        np.asarray(Lw)[live], np.asarray(legacy)[live], rtol=1e-8
    )


def test_analytic_dL_dn_matches_ad():
    n, C, n_sat = _mk()
    _, dL = smooth_rainout_loss(n, C, n_sat, W)
    ad = jnp.diag(jax.jacfwd(lambda x: smooth_rainout_loss(x, C, n_sat, W)[0])(n))
    np.testing.assert_allclose(np.asarray(ad), np.asarray(dL), rtol=1e-12)


def test_fd_through_all_inputs():
    """Centered FD vs jvp through n, n_sat, and C at finite w.

    Nodes sit strictly inside the hinge branches: centered FD straddling a
    breakpoint measures the (C1) kink rather than the derivative — the
    exact-saturation point's one-sided derivative 0 is covered by
    test_zero_at_and_below_saturation.
    """
    nz = 6
    n_sat = jnp.asarray(np.geomspace(1e3, 1e13, nz))
    s_ratio = jnp.asarray([0.3, 0.9, 1.0 + W / 3, 1.0 + 2 * W / 3, 2.0, 40.0])
    n = n_sat * s_ratio
    C = jnp.asarray(np.geomspace(1e-9, 1e-5, nz))
    rng = np.random.default_rng(7)

    for which in range(3):
        args = [n, C, n_sat]
        tang = jnp.asarray(rng.uniform(0.5, 1.0, nz))

        def f(x, wi=which, a=args):
            a2 = list(a)
            a2[wi] = x
            return smooth_rainout_loss(a2[0], a2[1], a2[2], W)[0]

        x0 = args[which]
        _, jv = jax.jvp(f, (x0,), (tang * x0,))  # relative direction
        h = 1e-6
        fd = (f(x0 * (1 + h * tang)) - f(x0 * (1 - h * tang))) / (2 * h)
        live = np.abs(np.asarray(fd)) > 0
        np.testing.assert_allclose(
            np.asarray(jv)[live], np.asarray(fd)[live], rtol=5e-6
        )


def test_c1_continuity_at_hinge_breakpoints():
    """dL/dn continuous across s=0 and s=w (the two hinge breakpoints)."""
    n_sat = jnp.asarray([1e10])
    C = jnp.asarray([1e-7])
    for s_break in (0.0, W):
        eps_rel = 1e-9
        lo = n_sat * (1.0 + s_break - eps_rel)
        hi = n_sat * (1.0 + s_break + eps_rel)
        _, d_lo = smooth_rainout_loss(lo, C, n_sat, W)
        _, d_hi = smooth_rainout_loss(hi, C, n_sat, W)
        # C1: derivative jump vanishes with the offset (scale: C*n ~ 1e3)
        assert abs(float(d_hi[0] - d_lo[0])) < 1e-5 * float(C[0] * n_sat[0])


def test_safe_arithmetic_at_n_sat_extremes():
    C = jnp.asarray([1e-7, 1e-7, 1e-7])
    n = jnp.asarray([1e10, 1e10, 1e-25])
    n_sat = jnp.asarray([0.0, 1e-300, 1e20])  # underflow, sub-floor, huge
    L, dL = smooth_rainout_loss(n, C, n_sat, W)
    assert bool(jnp.all(jnp.isfinite(L))) and bool(jnp.all(jnp.isfinite(dL)))
    # underflowed n_sat: the hinge width floors at w*N_SAT_FLOOR and the
    # supersaturated branch reduces to the legacy loss to high accuracy
    legacy = float(C[0] * n[0] * (n[0] - 0.0))
    assert abs(float(L[0]) - legacy) / legacy < 1e-12
    assert N_SAT_FLOOR > 0.0
    # deeply subsaturated cell: exactly zero
    assert float(L[2]) == 0.0
    # tangents stay finite through the guard region
    g = jax.jacfwd(lambda ns: smooth_rainout_loss(n, C, ns, W)[0])(n_sat)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_jit_vmap_consistency():
    n, C, n_sat = _mk()
    L, dL = smooth_rainout_loss(n, C, n_sat, W)
    Lj, dLj = jax.jit(smooth_rainout_loss, static_argnums=3)(n, C, n_sat, W)
    # XLA fusion may reassociate; float64-precision agreement, not bitwise
    np.testing.assert_allclose(np.asarray(L), np.asarray(Lj), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(dL), np.asarray(dLj), rtol=1e-14)
    batch = jax.vmap(lambda k: smooth_rainout_loss(n * k, C, n_sat, W)[0])(
        jnp.asarray([0.5, 1.0, 2.0])
    )
    solo = smooth_rainout_loss(n * 2.0, C, n_sat, W)[0]
    np.testing.assert_allclose(np.asarray(batch[2]), np.asarray(solo), rtol=1e-12)
