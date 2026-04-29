"""Photochemistry adapter: lazy `PhotoStaticInputs` cache + JAX kernel dispatch."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import vulcan_cfg
from chem_funs import ni as _ni, nr as _nr  # noqa: F401

import network as _net_mod
import photo as _photo_mod
import phy_const as _phy_const

jax.config.update("jax_enable_x64", True)


_NETWORK = _net_mod.parse_network(vulcan_cfg.network)


class Ros2JAX:
    """Photo wrapper holding lazy PhotoData/PhotoJData caches."""

    def __init__(self, photo_static=None):
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

    def compute_tau(self, var, atm):
        """Compute optical depth and write to `var.tau`."""
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
        """Two-stream Eddington RT. Reads var.dflux_u from prior call."""
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
        mask = var.aflux > vulcan_cfg.flux_atol
        if np.any(mask):
            with np.errstate(invalid="ignore"):
                var.aflux_change = float(np.nanmax(
                    np.abs(var.aflux - var.prev_aflux)[mask] / var.aflux[mask]
                ))
        else:
            var.aflux_change = 0.0

    def compute_J(self, var, atm):
        """Photolysis rates per (species, branch). Writes to var.J_sp + var.k_arr."""
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
        """Photo-ionisation rates per (species, branch). Writes to var.Jion_sp + var.k_arr."""
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
        if vulcan_cfg.use_moldiff:
            print("Include molecular diffusion.")
        else:
            print("No molecular diffusion.")
        if getattr(vulcan_cfg, "use_fix_all_bot", False):
            print("Use fixed bottom BC.")
        para.solver_str = "solver"
