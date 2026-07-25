"""Forward-mode AD through a physical *transport* knob (eddy diffusion Kzz).

The rate path has its own parity tests; this guards that a physical input (Kzz)
is correctly on the AD graph through one Ros2 step. It is the forward-mode /
physical-knob complement to the reverse-mode reaction-sensitivity tests, and
stays fast by differentiating a single step (no convergence).

Why not a tight jvp-vs-FD assertion here: the single-step diffusion signal is
~1e-6 of the abundances, so a finite difference of the ~1e19 scalar loss hits
the float64 cancellation floor (verified: FD swings sign as eps shrinks). The
robust correctness check is forward-vs-reverse-mode agreement, which is immune to
that cancellation; a coarse FD is kept only as an independent sign/magnitude
sanity. Tight FD validation of Kzz is intrinsically *end-to-end* (the cumulative
effect on a converged column is O(1) relative) and lives in
`jax_paper/scripts/fig_kzz_jvp_validate.py` (<0.1% vs re-converged FD).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _synthetic_atm(net, nz):
    from vulcan_jax.jax_step import AtmStatic

    ni = net.ni
    return AtmStatic(
        Kzz=jnp.full((nz - 1,), 1e9),
        Dzz=jnp.full((nz - 1, ni), 1e2),
        dzi=jnp.full((nz - 1,), 1e5),
        vz=jnp.zeros(nz - 1),
        Hpi=jnp.full((nz - 1,), 1e6),
        Ti=jnp.full((nz - 1,), 1500.0),
        Tco=jnp.full((nz,), 1500.0),
        g=jnp.full((nz,), 2140.0),
        ms=jnp.full((ni,), 1e-23),
        alpha=jnp.zeros(ni),
        M=jnp.full((nz,), 1e15),
        vm=jnp.zeros((nz - 1, ni)),
        vs=jnp.zeros((nz - 1, ni)),
        top_flux=jnp.zeros(ni),
        bot_flux=jnp.zeros(ni),
        bot_vdep=jnp.zeros(ni),
        gas_indx_mask=jnp.ones(ni, dtype=jnp.bool_),
        diff_esc_mask=jnp.zeros(ni, dtype=jnp.bool_),
        use_vm_mol=jnp.bool_(False),
        use_settling=jnp.bool_(False),
        use_topflux=jnp.bool_(False),
        use_botflux=jnp.bool_(False),
    )


def test_per_step_kzz_forward_mode():
    """Kzz is correctly differentiable through one Ros2 step (jvp==vjp; FD sanity)."""
    from vulcan_jax import network as net_mod, chem as chem_mod
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    from vulcan_jax.jax_step import jax_ros2_step

    net = chem_mod.to_jax(net_mod.parse_network(vulcan_cfg.network))
    nz = 8
    atm = _synthetic_atm(net, nz)
    prof = jnp.linspace(1.0, 3.0, nz)[:, None]  # vertical gradient -> diffusion acts
    y = jnp.full((nz, net.ni), 1e10) * prof
    k_arr = jnp.full((net.nr + 1, nz), 1e-15)  # near-zero chem -> isolate transport
    dt = jnp.float64(5e-1)
    ch = nz // 2

    def loss(ln_kzz):
        a = atm._replace(Kzz=jnp.exp(ln_kzz))
        sol, _ = jax_ros2_step(y, k_arr, dt, a, net)
        return jnp.sum(sol[:ch])  # upper-half column mass; Kzz redistributes it

    ln_kzz = jnp.log(atm.Kzz)
    ones = jnp.ones_like(ln_kzz)
    g_jvp = float(jax.jvp(loss, (ln_kzz,), (ones,))[1])
    g_vjp = float(jnp.sum(jax.grad(loss)(ln_kzz)))

    # Kzz is on the graph (finite, nonzero tangent).
    assert np.isfinite(g_jvp) and abs(g_jvp) > 0.0, "Kzz tangent is zero/non-finite"
    # Forward and reverse mode agree on the transport gradient (the robust check).
    assert abs(g_jvp - g_vjp) / abs(g_jvp) < 1e-3, (
        f"jvp vs vjp disagree on Kzz: {g_jvp:.3e} vs {g_vjp:.3e}"
    )
    # Coarse FD cross-check (cancellation-limited per step, so loose tolerance).
    eps = 1e-2
    fd = (float(loss(ln_kzz + eps)) - float(loss(ln_kzz - eps))) / (2 * eps)
    assert np.sign(fd) == np.sign(g_jvp) and abs(fd - g_jvp) / abs(g_jvp) < 0.3, (
        f"coarse FD cross-check failed: fd={fd:.3e} jvp={g_jvp:.3e}"
    )


def test_main():
    test_per_step_kzz_forward_mode()
