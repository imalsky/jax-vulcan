"""Vmap-compatible JAX Ros2 step.

`jax_ros2_step(y, k_arr, dt, atm, net)` operates on a static atmosphere
snapshot (Kzz, Dzz, Hpi, ... plus the third-body density `atm.M`); ysum(y) is
handled inside the step. For batch parallelism over multiple atmospheres, vmap
over the leading axis of (y, k_arr, atm).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .chem import chem_jac_analytical, NetworkArrays
from .chem_funs import chem_rhs_codegen as _chem_rhs, spec_list as _SPEC_LIST
from . import composition as _composition
from .phy_const import kb, Navo
from .solver import (
    factor_block_thomas_diag_offdiag,
    solve_block_thomas_diag_offdiag,
)
from .config import default_config

jax.config.update("jax_enable_x64", True)

_CFG = default_config()
# Pure numerical floor for ysum denominators; not a tuning knob.
_UNDERFLOW_DENOM = 1e-300

# (atom, reservoir) pairs for the atom-conservation projection: each atom's
# per-layer production residual is distributed onto its abundant reservoir
# carrier. Only pairs the active network supports are used (C-H-N-O networks
# conserve H/O/C/N; SNCHO adds S via H2S). One reservoir per atom keeps the
# reservoir count matrix square, so one linear solve zeros every atom residual
# at once, including the H that H2S also carries.
_ATOM_RESERVOIRS = (
    ("H", "H2"),
    ("O", "H2O"),
    ("C", "CO"),
    ("N", "N2"),
    ("S", "H2S"),
)


def _build_chem_projection_tables() -> tuple[
    bool, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Build static reservoir-projection tables for chemistry conservation.

    Conserves every atom in ``cfg.atom_list`` that has a composition column and
    a tracked reservoir species, so SNCHO conserves S while the C-H-N-O
    networks stay bit-for-bit unchanged. ``reservoir_counts`` is invertible for
    every atom subset: the C, N and S columns each have a single nonzero entry,
    so the solve always reduces to the invertible H/O block.
    """
    compo_names = _composition.compo.dtype.names
    cfg_atoms = getattr(_CFG, "atom_list", ())
    pairs = tuple(
        (atom, reservoir)
        for atom, reservoir in _ATOM_RESERVOIRS
        if atom in cfg_atoms and atom in compo_names and reservoir in _SPEC_LIST
    )
    if not pairs:
        return (
            False,
            jnp.zeros((len(_SPEC_LIST), 0), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0, 0), dtype=jnp.float64),
        )

    project_atoms = tuple(atom for atom, _ in pairs)
    reservoir_species = tuple(reservoir for _, reservoir in pairs)

    row_by_species = {str(row["species"]): row for row in _composition.compo}
    atom_counts = np.asarray(
        [
            [float(row_by_species[sp][atom]) for atom in project_atoms]
            for sp in _SPEC_LIST
        ],
        dtype=np.float64,
    )
    reservoir_idx = np.asarray(
        [_SPEC_LIST.index(sp) for sp in reservoir_species],
        dtype=np.int32,
    )
    reservoir_counts = atom_counts[reservoir_idx]
    inv_reservoir_counts = np.linalg.inv(reservoir_counts)
    return (
        True,
        jnp.asarray(atom_counts, dtype=jnp.float64),
        jnp.asarray(reservoir_idx, dtype=jnp.int32),
        jnp.asarray(inv_reservoir_counts, dtype=jnp.float64),
    )


(
    _CHEM_PROJECTION_ENABLED,
    _CHEM_ATOM_COUNTS,
    _CHEM_RESERVOIR_IDX,
    _CHEM_INV_RESERVOIR_COUNTS,
) = _build_chem_projection_tables()

# atom_list baked into the projection tables above; import-frozen (the tables
# are module constants). state._assert_atom_list_matches_import fails fast if
# a later make_config passes a different atom_list.
IMPORT_ATOM_LIST = tuple(getattr(_CFG, "atom_list", ()))


def _project_chem_rhs(rhs: jnp.ndarray) -> jnp.ndarray:
    """Project chemistry RHS onto per-atom conservation via abundant reservoirs.

    Atoms and reservoirs are whichever the active network supports
    (H/O/C/N for C--H--N--O networks; H/O/C/N/S via H2S for SNCHO).
    """
    if not _CHEM_PROJECTION_ENABLED:
        return rhs
    residual = rhs @ _CHEM_ATOM_COUNTS  # shape: (nz, n_atoms)
    correction = -residual @ _CHEM_INV_RESERVOIR_COUNTS
    # correction shape: (nz, n_reservoir)
    return rhs.at[:, _CHEM_RESERVOIR_IDX].add(correction)


def _project_chem_jac(chem_jac: jnp.ndarray) -> jnp.ndarray:
    """Apply the same reservoir projection to chemistry Jacobian rows."""
    if not _CHEM_PROJECTION_ENABLED:
        return chem_jac
    residual = jnp.einsum("ia,zij->zaj", _CHEM_ATOM_COUNTS, chem_jac)
    # residual shape: (nz, n_atoms, ni)
    correction = -jnp.einsum("ar,zaj->zrj", _CHEM_INV_RESERVOIR_COUNTS, residual)
    # correction shape: (nz, n_reservoir, ni)
    return chem_jac.at[:, _CHEM_RESERVOIR_IDX, :].add(correction)


def _projected_chem_rhs(
    y: jnp.ndarray, M: jnp.ndarray, k_arr: jnp.ndarray
) -> jnp.ndarray:
    """Evaluate and conservatively project the generated chemistry RHS."""
    return _project_chem_rhs(_chem_rhs(y, M, k_arr))


class AtmStatic(NamedTuple):
    """Atmosphere parameters held constant within a Ros2 step. nz from
    Tco's leading axis, ni from Dzz's trailing axis."""

    Kzz: jnp.ndarray  # (nz-1,)
    Dzz: jnp.ndarray  # (nz-1, ni)
    dzi: jnp.ndarray  # (nz-1,)
    vz: jnp.ndarray  # (nz-1,)
    Hpi: jnp.ndarray  # (nz-1,)
    Ti: jnp.ndarray  # (nz-1,)
    Tco: jnp.ndarray  # (nz,)
    g: jnp.ndarray  # (nz,)
    ms: jnp.ndarray  # (ni,)
    alpha: jnp.ndarray  # (ni,)
    M: jnp.ndarray  # (nz,)
    vm: jnp.ndarray  # (nz-1, ni) interface molecular-diffusion drift velocity
    vs: jnp.ndarray  # (nz-1, ni)
    top_flux: jnp.ndarray  # (ni,)
    bot_flux: jnp.ndarray  # (ni,)
    bot_vdep: jnp.ndarray  # (ni,)
    gas_indx_mask: jnp.ndarray  # (ni,) bool
    diff_esc_mask: jnp.ndarray  # (ni,) bool, species in cfg.diff_esc
    # Read as a FLOAT blend weight, not a plain bool: 1.0 = upwind molecular
    # diffusion, 0.0 = central difference. `make_atm_static` seeds a Python
    # bool, but the hybrid runner splices the carry's `hybrid_use_vm` (float64)
    # in here mid-run, so anything consuming it must accept either.
    use_vm_mol: bool | float | jnp.ndarray
    use_settling: bool
    use_topflux: bool
    use_botflux: bool


class DiffGrav(NamedTuple):
    """Pre-baked y-independent transport contributions to the mol-diff blocks.

    Computed once per Ros2 step (atm refresh recomputes); reused for both
    stages of the Rosenbrock step. `dz_ave = 0.5*(dzi[j-1] + dzi[j])` is
    also consumed by the ysum-dependent eddy/mol terms.
    """

    A_grav_int: jnp.ndarray
    B_grav_int: jnp.ndarray
    C_grav_int: jnp.ndarray
    bdry0_grav: jnp.ndarray
    bdry_top_grav: jnp.ndarray
    dz_ave: jnp.ndarray


def compute_diff_grav(atm: AtmStatic) -> DiffGrav:
    """y-independent transport piece of the molecular-diffusion blocks."""
    dzi, Hpi, Ti, Tco, g, ms, alpha, Dzz = (
        atm.dzi,
        atm.Hpi,
        atm.Ti,
        atm.Tco,
        atm.g,
        atm.ms,
        atm.alpha,
        atm.Dzz,
    )
    vm = atm.vm
    vs = atm.vs
    nz = atm.Tco.shape[0]
    j_int = jnp.arange(1, nz - 1)
    dz_ave = 0.5 * (dzi[j_int - 1] + dzi[j_int])  # (nz-2,)
    use_vm = jnp.asarray(atm.use_vm_mol, dtype=jnp.float64)
    use_set = jnp.asarray(atm.use_settling, dtype=jnp.float64)
    mol_active = jnp.any(Dzz != 0.0)
    Hpi = jnp.where(mol_active, Hpi, jnp.ones_like(Hpi))

    grav_j = (
        -1.0 / Hpi[j_int][:, None]
        + ms[None, :] * g[j_int][:, None] / (Navo * kb * Ti[j_int][:, None])
        + alpha[None, :]
        / Ti[j_int][:, None]
        * (Tco[j_int + 1][:, None] - Tco[j_int][:, None])
        / dzi[j_int][:, None]
    )
    grav_jm = (
        -1.0 / Hpi[j_int - 1][:, None]
        + ms[None, :] * g[j_int][:, None] / (Navo * kb * Ti[j_int - 1][:, None])
        + alpha[None, :]
        / Ti[j_int - 1][:, None]
        * (Tco[j_int][:, None] - Tco[j_int - 1][:, None])
        / dzi[j_int - 1][:, None]
    )
    grav_jp_b = (
        -1.0 / Hpi[j_int][:, None]
        + ms[None, :] * g[j_int + 1][:, None] / (Navo * kb * Ti[j_int][:, None])
        + alpha[None, :]
        / Ti[j_int][:, None]
        * (Tco[j_int + 1][:, None] - Tco[j_int][:, None])
        / dzi[j_int][:, None]
    )
    grav_jm_c = (
        -1.0 / Hpi[j_int - 1][:, None]
        + ms[None, :] * g[j_int - 1][:, None] / (Navo * kb * Ti[j_int - 1][:, None])
        + alpha[None, :]
        / Ti[j_int - 1][:, None]
        * (Tco[j_int][:, None] - Tco[j_int - 1][:, None])
        / dzi[j_int - 1][:, None]
    )

    inv_2dz_ave = 1.0 / (2.0 * dz_ave[:, None])
    A_grav_int = inv_2dz_ave * (Dzz[j_int] * grav_j - Dzz[j_int - 1] * grav_jm)
    B_grav_int = inv_2dz_ave * Dzz[j_int] * grav_jp_b
    C_grav_int = -inv_2dz_ave * Dzz[j_int - 1] * grav_jm_c

    bdry0_grav = (
        1.0
        / dzi[0]
        * Dzz[0]
        / 2.0
        * (
            -1.0 / Hpi[0]
            + ms * g[0] / (Navo * kb * Ti[0])
            + alpha / Ti[0] * (Tco[1] - Tco[0]) / dzi[0]
        )
    )
    bdry_top_grav = (
        -1.0
        / dzi[-1]
        * Dzz[-1]
        / 2.0
        * (
            -1.0 / Hpi[-1]
            + ms * g[-1] / (Navo * kb * Ti[-1])
            + alpha / Ti[-1] * (Tco[-1] - Tco[-2]) / dzi[-1]
        )
    )

    # Upwind molecular-diffusion advection variant (`use_vm_mol`). `vm` is the
    # interface drift velocity (nz-1, ni): vm[k] acts on the interface between
    # cells k and k+1, so vm[-1] is the top interface (matches op.diffdf_vm).
    A_vm_int = (
        -((vm[j_int] > 0) * vm[j_int] - (vm[j_int - 1] < 0) * vm[j_int - 1])
        / dz_ave[:, None]
    )
    B_vm_int = -((vm[j_int] < 0) * vm[j_int]) / dz_ave[:, None]
    C_vm_int = ((vm[j_int - 1] > 0) * vm[j_int - 1]) / dz_ave[:, None]
    bdry0_vm = -((vm[0] > 0) * vm[0]) / dzi[0]
    bdry0_vm_B = -((vm[0] < 0) * vm[0]) / dzi[0]
    bdry_top_vm = ((vm[-1] < 0) * vm[-1]) / dzi[-1]
    bdry_top_vm_C = ((vm[-1] > 0) * vm[-1]) / dzi[-1]

    # Settling velocity is additive to either gravity-mode or vm-mode mol-diff.
    A_vs_int = (
        -((vs[j_int] > 0) * vs[j_int] - (vs[j_int - 1] < 0) * vs[j_int - 1])
        / dz_ave[:, None]
    )
    B_vs_int = -((vs[j_int] < 0) * vs[j_int]) / dz_ave[:, None]
    C_vs_int = ((vs[j_int - 1] > 0) * vs[j_int - 1]) / dz_ave[:, None]
    bdry0_vs = -((vs[0] > 0) * vs[0]) / dzi[0]
    bdry0_vs_B = -((vs[0] < 0) * vs[0]) / dzi[0]
    bdry_top_vs = ((vs[-1] < 0) * vs[-1]) / dzi[-1]
    bdry_top_vs_C = ((vs[-1] > 0) * vs[-1]) / dzi[-1]

    A_base = (1.0 - use_vm) * A_grav_int + use_vm * A_vm_int
    B_base = (1.0 - use_vm) * B_grav_int + use_vm * B_vm_int
    C_base = (1.0 - use_vm) * C_grav_int + use_vm * C_vm_int

    bdry0_A_base = (1.0 - use_vm) * bdry0_grav + use_vm * bdry0_vm
    bdry0_B_base = (1.0 - use_vm) * bdry0_grav + use_vm * bdry0_vm_B
    bdry_top_A_base = (1.0 - use_vm) * bdry_top_grav + use_vm * bdry_top_vm
    bdry_top_C_base = (1.0 - use_vm) * bdry_top_grav + use_vm * bdry_top_vm_C

    A_extra = A_base + use_set * A_vs_int
    B_extra = B_base + use_set * B_vs_int
    C_extra = C_base + use_set * C_vs_int
    bdry0_A = bdry0_A_base + use_set * bdry0_vs
    bdry0_B = bdry0_B_base + use_set * bdry0_vs_B
    bdry_top_A = bdry_top_A_base + use_set * bdry_top_vs
    bdry_top_C = bdry_top_C_base + use_set * bdry_top_vs_C

    return DiffGrav(
        A_grav_int=A_extra,
        B_grav_int=B_extra,
        C_grav_int=C_extra,
        bdry0_grav=jnp.stack([bdry0_A, bdry0_B], axis=0),
        bdry_top_grav=jnp.stack([bdry_top_A, bdry_top_C], axis=0),
        dz_ave=dz_ave,
    )


def _build_diff_coeffs_jax(y, atm: AtmStatic, grav: DiffGrav):
    """Diffusion (eddy + molecular) coefficient blocks for one Ros2 stage.

    `grav` carries the y-independent gravity contribution; this function
    layers in the y-dependent part. Returns
    (A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, ysum).
    """
    Kzz, Dzz, dzi, vz = atm.Kzz, atm.Dzz, atm.dzi, atm.vz
    ni = atm.ms.shape[0]
    nz = atm.Tco.shape[0]

    ysum = jnp.sum(jnp.where(atm.gas_indx_mask[None, :], y, 0.0), axis=1)
    ysum = jnp.maximum(ysum, _UNDERFLOW_DENOM)

    # Build full nz arrays of interior values, then overwrite the boundaries.
    j_int = jnp.arange(1, nz - 1)
    dz_ave = grav.dz_ave  # (nz-2,)

    A_eddy_int = (
        -1.0
        / dz_ave
        * (
            Kzz[j_int] / dzi[j_int] * (ysum[j_int + 1] + ysum[j_int]) / 2.0
            + Kzz[j_int - 1] / dzi[j_int - 1] * (ysum[j_int] + ysum[j_int - 1]) / 2.0
        )
        / ysum[j_int]
    )
    B_eddy_int = (
        1.0
        / dz_ave
        * Kzz[j_int]
        / dzi[j_int]
        * (ysum[j_int + 1] + ysum[j_int])
        / 2.0
        / ysum[j_int + 1]
    )
    C_eddy_int = (
        1.0
        / dz_ave
        * Kzz[j_int - 1]
        / dzi[j_int - 1]
        * (ysum[j_int] + ysum[j_int - 1])
        / 2.0
        / ysum[j_int - 1]
    )

    # Vertical advection (bool*value gives the upwind switch).
    A_eddy_int = (
        A_eddy_int
        - ((vz[j_int] > 0) * vz[j_int] - (vz[j_int - 1] < 0) * vz[j_int - 1]) / dz_ave
    )
    B_eddy_int = B_eddy_int - ((vz[j_int] < 0) * vz[j_int]) / dz_ave
    C_eddy_int = C_eddy_int + ((vz[j_int - 1] > 0) * vz[j_int - 1]) / dz_ave

    A_eddy_0 = -1.0 / dzi[0] * (Kzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[0]
    A_eddy_0 = A_eddy_0 - ((vz[0] > 0) * vz[0]) / dzi[0]
    B_eddy_0 = 1.0 / dzi[0] * (Kzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[1]
    B_eddy_0 = B_eddy_0 - ((vz[0] < 0) * vz[0]) / dzi[0]
    C_eddy_0 = 0.0

    A_eddy_top = (
        -1.0 / dzi[-1] * (Kzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-1]
    )
    A_eddy_top = A_eddy_top + ((vz[-1] < 0) * vz[-1]) / dzi[-1]
    C_eddy_top = (
        1.0 / dzi[-1] * (Kzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-2]
    )
    C_eddy_top = C_eddy_top + ((vz[-1] > 0) * vz[-1]) / dzi[-1]
    B_eddy_top = 0.0

    A_eddy = jnp.concatenate(
        [jnp.array([A_eddy_0]), A_eddy_int, jnp.array([A_eddy_top])]
    )
    B_eddy = jnp.concatenate(
        [jnp.array([B_eddy_0]), B_eddy_int, jnp.array([B_eddy_top])]
    )
    C_eddy = jnp.concatenate(
        [jnp.array([C_eddy_0]), C_eddy_int, jnp.array([C_eddy_top])]
    )

    # Molecular-diffusion per-species blocks: y-dependent part + pre-baked grav.
    Ai_int = (
        -1.0
        / dz_ave[:, None]
        * (
            Dzz[j_int]
            / dzi[j_int][:, None]
            * (ysum[j_int + 1][:, None] + ysum[j_int][:, None])
            / 2.0
            + Dzz[j_int - 1]
            / dzi[j_int - 1][:, None]
            * (ysum[j_int][:, None] + ysum[j_int - 1][:, None])
            / 2.0
        )
        / ysum[j_int][:, None]
    )
    Ai_int = Ai_int + grav.A_grav_int

    Bi_int = (
        1.0
        / dz_ave[:, None]
        * Dzz[j_int]
        / dzi[j_int][:, None]
        * (ysum[j_int + 1][:, None] + ysum[j_int][:, None])
        / 2.0
        / ysum[j_int + 1][:, None]
    )
    Bi_int = Bi_int + grav.B_grav_int

    Ci_int = (
        1.0
        / dz_ave[:, None]
        * Dzz[j_int - 1]
        / dzi[j_int - 1][:, None]
        * (ysum[j_int][:, None] + ysum[j_int - 1][:, None])
        / 2.0
        / ysum[j_int - 1][:, None]
    )
    Ci_int = Ci_int + grav.C_grav_int

    Ai_0 = (
        -1.0 / dzi[0] * (Dzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[0]
        + grav.bdry0_grav[0]
    )
    Bi_0 = (
        1.0 / dzi[0] * (Dzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[1]
        + grav.bdry0_grav[1]
    )
    Ci_0 = jnp.zeros(ni)

    Ai_top = (
        -1.0 / dzi[-1] * (Dzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-1]
        + grav.bdry_top_grav[0]
    )
    Ci_top = (
        1.0 / dzi[-1] * (Dzz[-1] / dzi[-1]) * (ysum[-1] + ysum[-2]) / 2.0 / ysum[-2]
        + grav.bdry_top_grav[1]
    )
    Bi_top = jnp.zeros(ni)

    A_mol = jnp.concatenate([Ai_0[None], Ai_int, Ai_top[None]], axis=0)
    B_mol = jnp.concatenate([Bi_0[None], Bi_int, Bi_top[None]], axis=0)
    C_mol = jnp.concatenate([Ci_0[None], Ci_int, Ci_top[None]], axis=0)

    return A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, ysum


def _apply_diffusion_jax(
    y, A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, atm: AtmStatic
):
    """diff[j] = (A+Ai)*y[j] + (B+Bi)*y[j+1] + (C+Ci)*y[j-1], plus BC fluxes."""
    A_total = A_eddy[:, None] + A_mol
    B_total = B_eddy[:, None] + B_mol
    C_total = C_eddy[:, None] + C_mol
    diff_0 = A_total[0] * y[0] + B_total[0] * y[1]
    diff_top = A_total[-1] * y[-1] + C_total[-1] * y[-2]
    diff_int = A_total[1:-1] * y[1:-1] + B_total[1:-1] * y[2:] + C_total[1:-1] * y[:-2]
    diff = jnp.concatenate([diff_0[None], diff_int, diff_top[None]], axis=0)
    diff = diff.at[-1].add(
        jnp.where(
            atm.use_topflux, atm.top_flux / atm.dzi[-1], jnp.zeros_like(atm.top_flux)
        )
    )
    diff = diff.at[0].add(
        jnp.where(
            atm.use_botflux,
            (atm.bot_flux - y[0] * atm.bot_vdep) / atm.dzi[0],
            jnp.zeros_like(atm.bot_flux),
        )
    )
    return diff


# Ros2 free parameter gamma = 1 + 1/sqrt(2) (Verwer et al. 1997). The stage-2
# factor 2/(gamma*dt) and solution weights 3/(2*gamma), 1/(2*gamma) below
# derive from it; matches VULCAN-master op.py.
#
# gamma is the L-stable choice and second order holds, but L-stability is a
# property of the EXACT-Jacobian Rosenbrock. The Jacobian actually supplied
# below is exact for chemistry and models transport with a diagonal-in-species
# tridiagonal that omits the ysum coupling, so this is a second-order W-method:
# second order for an arbitrary approximate Jacobian, with contractivity on
# these columns coming from the post-step hydrostatic rebalance rather than
# from the linear solve alone. The same omission is in both pinned upstreams.
_ROS2_GAMMA = 1.0 + 2.0**-0.5


@jax.jit
def jax_ros2_step(y, k_arr, dt, atm: AtmStatic, net: NetworkArrays, fix_mask=None):
    """One 2nd-order Rosenbrock step.

    Returns (sol, delta_arr), both (nz, ni). `fix_mask` (nz, ni) optionally
    pins selected (layer, species) entries by zeroing the corresponding
    rows/cols of the LHS and RHS.
    """
    r = _ROS2_GAMMA
    c0 = 1.0 / (r * dt)
    ni = atm.ms.shape[0]
    M = atm.M

    # y-independent gravity terms; reused for the y and yk2 evaluations.
    grav = compute_diff_grav(atm)

    A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, _ = _build_diff_coeffs_jax(
        y, atm, grav
    )

    diff_at_y = _apply_diffusion_jax(
        y, A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, atm
    )
    rhs_y = _projected_chem_rhs(y, M, k_arr) + diff_at_y
    # Analytical Jacobian: <=1e-13 vs the AD (jacrev) path, and 16x faster
    # measured on HD189 (92.1 -> 5.66 ms; notes.md dev log).
    # It skips the structurally-zero entries AD would materialize.
    chem_J = _project_chem_jac(chem_jac_analytical(y, M, k_arr, net))

    # Diffusion blocks are diagonal-in-species: pass off-diagonals as (nz-1, ni)
    # vectors to skip the O(ni^3) C @ invA_B matmul in forward elimination.
    diag_d = A_eddy[:, None] + A_mol
    sup_d = B_eddy[:-1, None] + B_mol[:-1]
    sub_d = C_eddy[1:, None] + C_mol[1:]
    bot_vdep_term = jnp.where(
        atm.use_botflux,
        -atm.bot_vdep / atm.dzi[0],
        jnp.zeros_like(atm.bot_vdep),
    )
    diag_d = diag_d.at[0].add(bot_vdep_term)
    # Diffusion-limited escape at TOA (`top_flux / y[-1]` on the top-layer
    # diagonal). Upstream carries this term ONLY in the upwind variants
    # `lhs_jac_tot_vm` and `lhs_jac_settling_vm` (exoclime@80f75b9 op.py:2044-
    # 2121, 2366+; vm_branch@84d010d op.py:2123-2200, 2445+); `lhs_jac_tot`,
    # `lhs_jac_settling` and `lhs_jac_no_mol` have no escape term, and the
    # `lhs_jac_fix_all_bot` copy is dead (its solver is never selected,
    # op.py:3080-3083). So the gate is `use_vm_mol`, not "diff_esc non-empty";
    # the RHS flux itself is gated on `use_topflux` above, as upstream.
    # The inner `where` mirrors upstream's `y > 0` guard and keeps the division
    # and its derivative finite at y = 0.
    #
    # DELIBERATE DIVERGENCE from the analytic derivative, inherited from both
    # pinned upstreams (vulcan2_ncho op.py:2106-2107, vm_branch op.py:2185-2186):
    # the RHS adds top_flux/dzi[-1] (a flux DIVERGENCE) while this entry is
    # d(top_flux)/dy, so it is larger than the true derivative by dzi[-1].
    # top_flux is exactly linear in y[-1] (atm_refresh.py:175), so the correct
    # entry would be top_flux/(y*dzi[-1]). Not corrected: it is on the LHS only
    # -- a W-method tolerates an approximate Jacobian without moving the fixed
    # point -- and correcting it would break bit-parity with both oracles.
    y_top_pos = y[-1] > 0.0
    diff_lim = jnp.where(
        atm.diff_esc_mask & y_top_pos,
        atm.top_flux / jnp.where(y_top_pos, y[-1], 1.0),
        0.0,
    )
    diag_d = diag_d.at[-1].add(
        diff_lim * jnp.asarray(atm.use_vm_mol, dtype=jnp.float64)
    )

    eye = jnp.eye(ni)
    di = jnp.arange(ni)
    diag = c0 * eye[None] - chem_J
    diag = diag.at[:, di, di].add(-diag_d)
    sup_neg = -sup_d  # (nz-1, ni)
    sub_neg = -sub_d  # (nz-1, ni)

    if fix_mask is not None:
        diag = jnp.where(fix_mask[:, :, None], 0.0, diag)
        diag_diag = diag[:, di, di]
        diag = diag.at[:, di, di].set(jnp.where(fix_mask, c0, diag_diag))
        rhs_y = jnp.where(fix_mask, 0.0, rhs_y)
        sup_neg = jnp.where(fix_mask[:-1], 0.0, sup_neg)
        sub_neg = jnp.where(fix_mask[1:], 0.0, sub_neg)

    factors = factor_block_thomas_diag_offdiag(diag, sup_neg, sub_neg)
    k1 = solve_block_thomas_diag_offdiag(factors, rhs_y)

    yk2 = y + k1 / r
    A_eddy2, B_eddy2, C_eddy2, A_mol2, B_mol2, C_mol2, _ = _build_diff_coeffs_jax(
        yk2, atm, grav
    )
    diff_at_yk2 = _apply_diffusion_jax(
        yk2, A_eddy2, B_eddy2, C_eddy2, A_mol2, B_mol2, C_mol2, atm
    )
    rhs_yk2 = _projected_chem_rhs(yk2, M, k_arr) + diff_at_yk2
    if fix_mask is not None:
        rhs_yk2 = jnp.where(fix_mask, 0.0, rhs_yk2)

    rhs2 = rhs_yk2 - (2.0 / (r * dt)) * k1
    k2 = solve_block_thomas_diag_offdiag(factors, rhs2)

    sol = y + (3.0 / (2.0 * r)) * k1 + (1.0 / (2.0 * r)) * k2
    delta_arr = jnp.abs(sol - yk2)
    return sol, delta_arr


def make_atm_static(atm, ni: int, nz: int, cfg=None) -> AtmStatic:
    """Build an AtmStatic from a legacy AtmData container.

    `cfg` defaults to the process default; OuterLoop passes its own cfg so the
    transport toggles honor a load_config() cfg (this runs at integration
    time, after state._cfg_overlay has restored the default).
    """
    if cfg is None:
        cfg = default_config()
    use_vm = bool(
        getattr(cfg, "use_vm_mol", False) and getattr(cfg, "use_moldiff", True)
    )
    use_set = bool(
        getattr(cfg, "use_settling", False) and getattr(cfg, "use_moldiff", True)
    )
    use_topflux = bool(getattr(cfg, "use_topflux", False))
    use_botflux = bool(getattr(cfg, "use_botflux", False))
    gas_mask = jnp.zeros((ni,), dtype=jnp.bool_)
    gas_mask = gas_mask.at[jnp.asarray(atm.gas_indx, dtype=jnp.int32)].set(True)
    # Independent of the toggles above (see the diff_esc note at the Jacobian
    # assembly). Species not in the network are caught by runtime_validation,
    # so the index lookup is safe here.
    diff_esc_np = np.zeros((ni,), dtype=bool)
    for _sp in getattr(cfg, "diff_esc", []) or []:
        diff_esc_np[_SPEC_LIST.index(_sp)] = True
    vm = atm.vm if use_vm else jnp.zeros((nz - 1, ni), dtype=jnp.float64)
    vs = atm.vs if use_set else jnp.zeros((nz - 1, ni), dtype=jnp.float64)
    Dzz = (
        atm.Dzz
        if getattr(cfg, "use_moldiff", True)
        else jnp.zeros((nz - 1, ni), dtype=jnp.float64)
    )
    return AtmStatic(
        Kzz=jnp.asarray(atm.Kzz),
        Dzz=jnp.asarray(Dzz),
        dzi=jnp.asarray(atm.dzi),
        vz=jnp.asarray(atm.vz),
        Hpi=jnp.asarray(atm.Hpi),
        Ti=jnp.asarray(atm.Ti),
        Tco=jnp.asarray(atm.Tco),
        g=jnp.asarray(atm.g),
        ms=jnp.asarray(atm.ms),
        alpha=jnp.asarray(atm.alpha),
        M=jnp.asarray(atm.M),
        vm=jnp.asarray(vm),
        vs=jnp.asarray(vs),
        top_flux=jnp.asarray(atm.top_flux),
        bot_flux=jnp.asarray(atm.bot_flux),
        bot_vdep=jnp.asarray(atm.bot_vdep),
        gas_indx_mask=gas_mask,
        diff_esc_mask=jnp.asarray(diff_esc_np),
        use_vm_mol=use_vm,
        use_settling=use_set,
        use_topflux=use_topflux,
        use_botflux=use_botflux,
    )
