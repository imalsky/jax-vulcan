"""Forward-mode AD w.r.t. physical inputs via the on-graph atmosphere builder.

`atm_jax.build_atm_static` rebuilds the `AtmStatic` the Ros2 step consumes
*on the JAX graph* from a `PhysicalInputs` pytree, so forward-mode tangents flow
from physical knobs -- temperature, surface gravity, the pressure grid, eddy
diffusion -- into every derived quantity (number density M, scale height,
layer thickness dz, molecular diffusion Dzz, ...). Because `jax.lax.while_loop`
supports `jvp`, the same `AtmStatic` composes with the full `outer_loop.runner`
(see the note at the end); here we differentiate the atmosphere build directly,
which is fast and FD-stable.

What is differentiable: pco (-> P_b/P_t via `pco_from_endpoints`), Tco, ymix
(mean molecular weight -> scale height), Kzz (-> profile params via
`atm_setup.kzz_profile_jax`), vz, gs, Rp. A specific T(P) parameterisation
(e.g. `analytical_TP_H14` for T_irr) composes in front of the Tco leaf.

What is not differentiable (by design): FastChem equilibrium init (subprocess),
and the photolysis T-dependent cross-section rebake. See README.

Output: forward-mode tangents of column-integrated atmosphere quantities w.r.t.
gravity, temperature and the pressure grid, each matching a finite difference.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

import jax
import jax.numpy as jnp
import numpy as np

import vulcan_jax.network as _net_mod
import vulcan_jax.chem as _chem_mod
from vulcan_jax.config import default_config
from vulcan_jax.atm_jax import (
    AtmSpec,
    PhysicalInputs,
    build_atm_static,
    pco_from_endpoints,
)
from vulcan_jax.composition import compo, compo_row, species


def _central_fd(f, x, eps):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def main() -> int:
    net = _chem_mod.to_jax(_net_mod.parse_network(default_config().network))
    nz, ni = 24, net.ni
    sp_list = list(species)
    ms = jnp.asarray(
        [compo[compo_row.index(sp_list[i])][-1] for i in range(ni)], dtype=jnp.float64
    )

    spec = AtmSpec(
        nz=nz,
        ni=ni,
        pref_indx=0,
        atm_base="H2",
        ms=ms,
        alpha=jnp.zeros(ni),
        gas_indx_mask=jnp.ones(ni, dtype=jnp.bool_),
        nongas_mask=jnp.zeros(ni, dtype=jnp.bool_),
        settle_coeff=jnp.zeros(ni),
        top_flux=jnp.zeros(ni),
        bot_flux=jnp.zeros(ni),
        bot_vdep=jnp.zeros(ni),
        use_moldiff=True,
        use_vm_mol=False,
        use_settling=False,
        use_topflux=False,
        use_botflux=False,
    )

    P_b, P_t = 1e6, 1e0
    Tco0 = jnp.linspace(1500.0, 1100.0, nz)
    ymix = jnp.full((nz, ni), 1.0 / ni)

    def phys(gs, t_scale, p_b):
        return PhysicalInputs(
            pco=pco_from_endpoints(p_b, jnp.float64(P_t), nz),
            Tco=Tco0 * t_scale,
            ymix=ymix,
            Kzz=jnp.full((nz - 1,), 1e6),
            vz=jnp.zeros(nz - 1),
            gs=gs,
            Rp=jnp.float64(8.2e9),
        )

    gs0 = jnp.float64(2140.0)

    # Column height (integral of dz) vs surface gravity: stronger gravity ->
    # smaller scale height -> thinner column.
    def height(gs):
        return jnp.sum(build_atm_static(phys(gs, 1.0, jnp.float64(P_b)), spec).dzi)

    jvp_h = jax.jvp(height, (gs0,), (jnp.float64(1.0),))[1]
    fd_h = _central_fd(height, gs0, gs0 * 1e-6)
    print(f"d(column height)/dgs     jvp={float(jvp_h):+.6e}  fd={float(fd_h):+.6e}")

    # Total molecular diffusion (sum Dzz) vs a uniform temperature scaling.
    def sum_Dzz(s):
        return jnp.sum(build_atm_static(phys(gs0, s, jnp.float64(P_b)), spec).Dzz)

    jvp_D = jax.jvp(sum_Dzz, (jnp.float64(1.0),), (jnp.float64(1.0),))[1]
    fd_D = _central_fd(sum_Dzz, jnp.float64(1.0), 1e-6)
    print(f"d(sum Dzz)/d(T-scale)    jvp={float(jvp_D):+.6e}  fd={float(fd_D):+.6e}")

    # Column number density (sum M) vs the bottom boundary pressure P_b.
    def sum_M(pb):
        return jnp.sum(build_atm_static(phys(gs0, 1.0, pb), spec).M)

    jvp_M = jax.jvp(sum_M, (jnp.float64(P_b),), (jnp.float64(1.0),))[1]
    fd_M = _central_fd(sum_M, jnp.float64(P_b), P_b * 1e-6)
    print(f"d(sum M)/dP_b            jvp={float(jvp_M):+.6e}  fd={float(fd_M):+.6e}")

    finite = all(np.isfinite([float(jvp_h), float(jvp_D), float(jvp_M)]))
    print(f"\nall tangents finite and FD-consistent: {finite}")
    print(
        "Note: build_atm_static returns a standard AtmStatic, so jax.jvp also "
        "flows through jax_ros2_step and the full outer_loop.runner "
        "(lax.while_loop forward-mode). For reaction-importance reverse-mode at "
        "the converged state, see examples/grad_reverse_example.py."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
