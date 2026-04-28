"""Photochemistry helpers wrapping VULCAN-JAX's pure-JAX `photo.py` kernels.

`Ros2JAX` no longer subclasses `op.Ros2`. It exists purely as the photo
adapter: when `vulcan_jax.py`'s pre-loop setup or `outer_loop.OuterLoop`'s
between-step orchestration needs `compute_tau` / `compute_flux` / `compute_J`,
those calls dispatch through this class's instance methods so the JAX kernels
get the lazy-initialised `PhotoData` / `PhotoJData` snapshots.

Phase 10.1 cleanup: removed all step-control overrides (`clip`, `loss`,
`step_ok`, `step_reject`, `reset_y`, `step_size`), the `solver` / `solver_fix_all_bot`
hot path, and the `op.Ros2` inheritance. The numerics those methods carried
all live in `outer_loop.py` now as pure-JAX equivalents.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import vulcan_cfg
from chem_funs import ni as _ni, nr as _nr  # noqa: F401  (re-exported for back-compat)

import network as _net_mod
import photo as _photo_mod
import phy_const as _phy_const

jax.config.update("jax_enable_x64", True)


# Parse the network once.
_NETWORK = _net_mod.parse_network(vulcan_cfg.network)


class Ros2JAX:
    """Photo wrapper. Holds lazily-built `PhotoData` / `PhotoJData` caches
    and forwards `compute_tau` / `compute_flux` / `compute_J` to JAX kernels.
    """

    def __init__(self, photo_static=None):
        # `photo_static` is the dense `PhotoStaticInputs` pytree from
        # `photo_setup.populate_photo` / `build_photo_static`. When set,
        # the per-call lazy builders below derive `PhotoData` /
        # `PhotoJData` from it. When None, the lazy builders construct
        # the static themselves from `(var, atm)` on first call so test
        # call sites that do `Ros2JAX()` then `compute_tau(var, atm)`
        # keep working post-Phase-22e (the legacy dict surface is gone).
        self._photo_static = photo_static
        self._photo_data = None
        self._photo_J_data = None
        self._photo_ion_data = None

    def _ensure_photo_static(self, var, atm):
        if self._photo_static is None:
            import photo_setup as _photo_setup
            self._photo_static = _photo_setup._build_photo_static_dense(var, atm)
        if int(self._photo_static.din12_indx) < 0 and hasattr(var, "sflux_din12_indx"):
            self._photo_static = self._photo_static.with_din12_indx(
                int(var.sflux_din12_indx)
            )
        return self._photo_static

    # ------------------------------------------------------------------
    # Photochemistry: pure-JAX kernels with NumPy boundaries.
    # ------------------------------------------------------------------
    def compute_tau(self, var, atm):
        """Optical depth via JAX. Mirrors op.compute_tau (op.py:2583-2603)."""
        if self._photo_data is None:
            static = self._ensure_photo_static(var, atm)
            self._photo_data = _photo_mod.photo_data_from_static(
                static, list(_NETWORK.species)
            )
        tau_j = _photo_mod.compute_tau_jax(
            jnp.asarray(var.y),
            jnp.asarray(atm.dz),
            self._photo_data,
        )
        var.tau = np.asarray(tau_j, dtype=np.float64)

    def compute_flux(self, var, atm):
        """Two-stream Eddington RT via JAX. Mirrors op.compute_flux
        (op.py:2606-2742). Bit-equivalent to the upstream NumPy impl when
        `var.dflux_u` carries the prior call's value (matches op.py:2694).
        """
        static = self._ensure_photo_static(var, atm)
        ag0 = float(_phy_const.ag0)
        aflux_j, sflux_j, dfd_j, dfu_j = _photo_mod.compute_flux_jax(
            jnp.asarray(var.tau),
            jnp.asarray(var.sflux_top),
            jnp.asarray(var.ymix),
            self._photo_data,
            jnp.asarray(static.bins, dtype=jnp.float64),
            float(np.cos(vulcan_cfg.sl_angle)),
            float(vulcan_cfg.edd),
            ag0,
            float(_phy_const.hc),
            jnp.asarray(var.dflux_u),
            ag0_is_zero=(ag0 == 0.0),
        )
        var.prev_aflux = np.copy(var.aflux)
        var.aflux = np.asarray(aflux_j, dtype=np.float64)
        var.sflux = np.asarray(sflux_j, dtype=np.float64)
        var.dflux_d = np.asarray(dfd_j, dtype=np.float64)
        var.dflux_u = np.asarray(dfu_j, dtype=np.float64)
        # aflux_change matches op.py:2740 (suppress NaN when all aflux==0).
        mask = var.aflux > vulcan_cfg.flux_atol
        if np.any(mask):
            with np.errstate(invalid="ignore"):
                var.aflux_change = float(np.nanmax(
                    np.abs(var.aflux - var.prev_aflux)[mask] / var.aflux[mask]
                ))
        else:
            var.aflux_change = 0.0

    def compute_J(self, var, atm):
        """Wavelength-integrated photolysis rates per (species, branch) via JAX.

        Mirrors op.compute_J (op.py:2754-2790): populates var.J_sp dict and
        writes per-(species, branch) J-rates into var.k_arr at the appropriate
        reaction indices. Phase 22d: writes to dense `var.k_arr` directly;
        the legacy dict mirror is rebuilt by `rates.apply_photo_remove` at
        the end of the photo block.
        """
        if self._photo_J_data is None:
            static = self._ensure_photo_static(var, atm)
            self._photo_J_data = _photo_mod.photo_J_data_from_static(static)

        J_sp_jax = _photo_mod.compute_J_jax(
            jnp.asarray(var.aflux), self._photo_J_data
        )
        nz_ = var.aflux.shape[0]
        var.J_sp = {
            (sp, bn): np.zeros(nz_)
            for sp in var.photo_sp
            for bn in range(var.n_branch[sp] + 1)
        }
        for (sp, nbr), Jrow in J_sp_jax.items():
            var.J_sp[(sp, nbr)] = np.asarray(Jrow, dtype=np.float64)
            var.J_sp[(sp, 0)] = var.J_sp[(sp, 0)] + var.J_sp[(sp, nbr)]
            ridx = var.pho_rate_index.get((sp, nbr))
            if ridx is not None and ridx not in vulcan_cfg.remove_list:
                var.k_arr[ridx, :] = var.J_sp[(sp, nbr)] * vulcan_cfg.f_diurnal

    def compute_Jion(self, var, atm):
        """Photo-ionisation rate via JAX. Mirrors op.compute_Jion."""
        if self._photo_ion_data is None:
            static = self._ensure_photo_static(var, atm)
            self._photo_ion_data = _photo_mod.photo_ion_data_from_static(static)

        Jion_sp_jax = _photo_mod.compute_Jion_jax(
            jnp.asarray(var.aflux), self._photo_ion_data
        )
        nz_ = var.aflux.shape[0]
        var.Jion_sp = {
            (sp, bn): np.zeros(nz_)
            for sp in var.ion_sp
            for bn in range(var.ion_branch[sp] + 1)
        }
        for (sp, nbr), Jrow in Jion_sp_jax.items():
            var.Jion_sp[(sp, nbr)] = np.asarray(Jrow, dtype=np.float64)
            var.Jion_sp[(sp, 0)] = var.Jion_sp[(sp, 0)] + var.Jion_sp[(sp, nbr)]
            ridx = var.ion_rate_index.get((sp, nbr))
            if ridx is not None and ridx not in vulcan_cfg.remove_list:
                var.k_arr[ridx, :] = var.Jion_sp[(sp, nbr)] * vulcan_cfg.f_diurnal

    def naming_solver(self, para):
        """Compatibility shim. VULCAN-JAX targets only Ros2; the solver
        dispatch happens at compile time inside `outer_loop.OuterLoop`, so
        this is a print-only courtesy. Phase 10.6 added in-body support
        for `use_fix_all_bot` and ion charge balance, so all the
        remaining op.Ros2 BC variants are covered.
        """
        if vulcan_cfg.use_moldiff:
            print("Include molecular diffusion.")
        else:
            print("No molecular diffusion.")
        if getattr(vulcan_cfg, "use_fix_all_bot", False):
            print("Use fixed bottom BC.")
        para.solver_str = "solver"
