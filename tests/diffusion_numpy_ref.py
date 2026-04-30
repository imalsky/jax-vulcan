"""NumPy reference diffusion operator (test-only).

This file lives in `tests/` because it is no longer on the VULCAN-JAX hot
path; the production diffusion is the JAX implementation inside
`jax_step._build_diff_coeffs_jax` / `_apply_diffusion_jax`. Tests import this
module to validate the JAX path (and op.diffdf) against an explicit NumPy
reference.

Mirrors `op.py:diffdf` (lines 1499-1600) and `op.py:lhs_jac_tot` (1976-2045).

The discretized diffusion operator at layer j has the form
    diff[j, :] = A_eff[j]*y[j] + B_eff[j]*y[j+1] + C_eff[j]*y[j-1]
where A_eff, B_eff, C_eff combine eddy (Kzz, scalar per layer) and molecular
(Dzz, per-species per-layer) diffusion plus thermal-diffusion / gravitational
sedimentation terms.

The Jacobian d(diff)/d(y) is block-tridiagonal: per-layer diagonal block has
chem_jac (dense [ni, ni]) + diag(diff_diag) (the ni-vector eddy+mol coeff at
layer j); off-diagonal blocks are diagonal in species (no inter-species
coupling from diffusion alone), so we represent them as ni-vectors that get
broadcast onto the dense blocks consumed by block_thomas.

VULCAN's lhs_jac_tot treats `ysum` as constant when differentiating (it shows
up only via mole-fraction conversion, and the cross-species coupling is
small for stiff chemistry). VULCAN-JAX matches this convention exactly.

Optional features supported (gated by vulcan_cfg flags):
    use_moldiff      molecular-diffusion contribution (Ai, Bi, Ci)
    use_topflux     fixed flux at TOA (top_flux array)
    use_botflux     deposition velocity at surface (bot_vdep, bot_flux)
    use_vz          vertical advection (with upwind switching on sign of vz)
    non_gas_sp      exclude condensed species from ysum (uses gas_indx)

Variants supported via the `mode` argument to build_diffusion_coeffs:
    'gravity'       diffdf:           central scheme using Hpi/ms*g/alpha terms
    'vm'            diffdf_vm:        upwind scheme using atm.vm
    'settling'      diffdf_settling:  diffdf + atm.vs settling velocity (upwind)
    'settling_vm'   diffdf_settling_vm: diffdf_vm + atm.vs settling
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phy_const import Navo, kb


@dataclass
class DiffusionCoeffs:
    """Per-layer eddy (scalar) + molecular (per-species) diffusion coefficients
    for the operator form  diff[j] = A*y[j] + B*y[j+1] + C*y[j-1].

    Boundary handling:  C[0] = 0,  B[nz-1] = 0   (zero-flux unless overridden
    by use_topflux / use_botflux).

    Eddy (scalar per layer):
        A_eddy: shape (nz,)
        B_eddy: shape (nz,)
        C_eddy: shape (nz,)
    Molecular (per species per layer):
        A_mol: shape (nz, ni)
        B_mol: shape (nz, ni)
        C_mol: shape (nz, ni)
    Net per-layer coefficient applied to y[j, i] is A_eddy[j] + A_mol[j, i].
    """

    A_eddy: np.ndarray
    B_eddy: np.ndarray
    C_eddy: np.ndarray
    A_mol: np.ndarray
    B_mol: np.ndarray
    C_mol: np.ndarray
    # Boundary-condition arrays (BC contribution to RHS, separate from coeffs).
    use_topflux: bool
    top_flux: np.ndarray            # shape (ni,)
    use_botflux: bool
    bot_flux: np.ndarray
    bot_vdep: np.ndarray            # shape (ni,)
    dzi: np.ndarray


def build_diffusion_coeffs(
    y: np.ndarray, atm, cfg, mode: str = "auto"
) -> DiffusionCoeffs:
    """Build A, B, C coefficients given the current state and atmosphere.

    Args:
        y: number densities, shape (nz, ni). Used to compute `ysum` per layer.
        atm: VULCAN-style AtmData object with .Kzz, .Dzz, .vz, .alpha, .Tco,
             .ms, .Hp, .Hpi, .Ti, .g, .dzi, .top_flux, .bot_flux, .bot_vdep,
             .gas_indx attributes. For non-default modes also needs .vm and .vs.
        cfg: vulcan_cfg module (or any namespace with use_moldiff, use_vz,
             use_topflux, use_botflux, use_vm_mol, use_settling, non_gas_sp,
             use_condense flags).
        mode: one of 'auto', 'gravity', 'vm', 'settling', 'settling_vm'.
              'auto' picks based on cfg.use_vm_mol and cfg.use_settling flags.

    Returns:
        DiffusionCoeffs ready for `apply_diffusion` and `diffusion_block_diags`.
    """
    # Resolve mode from cfg if 'auto'
    if mode == "auto":
        use_vm = bool(getattr(cfg, "use_vm_mol", False))
        use_set = bool(getattr(cfg, "use_settling", False))
        if use_vm and use_set:
            mode = "settling_vm"
        elif use_vm:
            mode = "vm"
        elif use_set:
            mode = "settling"
        else:
            mode = "gravity"
    if mode not in ("gravity", "vm", "settling", "settling_vm"):
        raise ValueError(f"Unknown diffusion mode: {mode!r}")
    nz, ni = y.shape

    # ysum: layer-wise total number density (excluding non-gaseous species
    # if condensation is on).
    if cfg.non_gas_sp:
        ysum = np.sum(y[:, atm.gas_indx], axis=1)
    else:
        ysum = np.sum(y, axis=1)

    dzi = atm.dzi
    Kzz = atm.Kzz
    Dzz = atm.Dzz
    alpha = atm.alpha       # (ni,)
    ms = atm.ms             # (ni,)
    Tco = atm.Tco           # (nz,)
    Hpi = atm.Hpi           # (nz-1,)
    Ti = atm.Ti             # (nz-1,)
    g = atm.g               # (nz,)
    vz = atm.vz             # (nz-1,)

    A = np.zeros(nz, dtype=np.float64)
    B = np.zeros(nz, dtype=np.float64)
    C = np.zeros(nz, dtype=np.float64)
    Ai = np.zeros((nz, ni), dtype=np.float64)
    Bi = np.zeros((nz, ni), dtype=np.float64)
    Ci = np.zeros((nz, ni), dtype=np.float64)

    use_mol = bool(cfg.use_moldiff)
    use_vm = mode in ("vm", "settling_vm")
    use_set = mode in ("settling", "settling_vm")
    vm = atm.vm if use_vm else np.zeros((y.shape[0], y.shape[1]), dtype=np.float64)
    vs = atm.vs if use_set else np.zeros((y.shape[0] - 1, y.shape[1]), dtype=np.float64)

    # --- Surface (j = 0) ---
    A[0] = -1.0 / dzi[0] * (Kzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[0]
    B[0] = 1.0 / dzi[0] * (Kzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[1]
    # vertical advection at j=0
    A[0] += -((vz[0] > 0) * vz[0]) / dzi[0]
    B[0] += -((vz[0] < 0) * vz[0]) / dzi[0]

    if use_mol:
        if use_vm:
            # vm-mode: use upwind advection with vm[0] in place of gravity term
            Ai[0] = -1.0 / dzi[0] * (Dzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[0]
            Ai[0] -= ((vm[0] > 0) * vm[0]) / dzi[0]
            Bi[0] = 1.0 / dzi[0] * (Dzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[1]
            Bi[0] -= ((vm[0] < 0) * vm[0]) / dzi[0]
        else:
            bdry0_mol_term = (
                1.0 / dzi[0] * Dzz[0] / 2.0
                * (-1.0 / Hpi[0] + ms * g[0] / (Navo * kb * Ti[0])
                   + alpha / Ti[0] * (Tco[1] - Tco[0]) / dzi[0])
            )
            Ai[0] = -1.0 / dzi[0] * (Dzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[0] + bdry0_mol_term
            Bi[0] = 1.0 / dzi[0] * (Dzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[1] + bdry0_mol_term
        if use_set:
            # Add settling velocity (upwind) at j=0
            Ai[0] -= ((vs[0] > 0) * vs[0]) / dzi[0]
            Bi[0] -= ((vs[0] < 0) * vs[0]) / dzi[0]

    # --- Top (j = nz - 1) ---
    A[nz - 1] = -1.0 / dzi[nz - 2] * (Kzz[nz - 2] / dzi[nz - 2]) * (ysum[nz - 1] + ysum[nz - 2]) / 2.0 / ysum[nz - 1]
    C[nz - 1] = 1.0 / dzi[nz - 2] * (Kzz[nz - 2] / dzi[nz - 2]) * (ysum[nz - 1] + ysum[nz - 2]) / 2.0 / ysum[nz - 2]
    A[nz - 1] += ((vz[-1] < 0) * vz[-1]) / dzi[-1]
    C[nz - 1] += ((vz[-1] > 0) * vz[-1]) / dzi[-1]

    if use_mol:
        if use_vm:
            Ai[nz - 1] = -1.0 / dzi[-1] * (Dzz[nz - 2] / dzi[-1]) * (ysum[nz - 1] + ysum[nz - 2]) / 2.0 / ysum[nz - 1]
            Ai[nz - 1] += ((vm[-1] < 0) * vm[-1]) / dzi[-1]
            Ci[nz - 1] = 1.0 / dzi[-1] * (Dzz[nz - 2] / dzi[-1]) * (ysum[nz - 1] + ysum[nz - 2]) / 2.0 / ysum[nz - 2]
            Ci[nz - 1] += ((vm[-1] > 0) * vm[-1]) / dzi[-1]
        else:
            bdry_top_mol_term = (
                -1.0 / dzi[-1] * Dzz[-1] / 2.0
                * (-1.0 / Hpi[-1] + ms * g[-1] / (Navo * kb * Ti[-1])
                   + alpha / Ti[-1] * (Tco[-1] - Tco[-2]) / dzi[-1])
            )
            Ai[nz - 1] = -1.0 / dzi[-1] * (Dzz[nz - 2] / dzi[-1]) * (ysum[nz - 1] + ysum[nz - 2]) / 2.0 / ysum[nz - 1] + bdry_top_mol_term
            Ci[nz - 1] = 1.0 / dzi[-1] * (Dzz[nz - 2] / dzi[-1]) * (ysum[nz - 1] + ysum[nz - 2]) / 2.0 / ysum[nz - 2] + bdry_top_mol_term
        if use_set:
            Ai[nz - 1] += ((vs[-1] < 0) * vs[-1]) / dzi[-1]
            Ci[nz - 1] += ((vs[-1] > 0) * vs[-1]) / dzi[-1]

    # --- Interior (1 <= j <= nz - 2) ---
    for j in range(1, nz - 1):
        dz_ave = 0.5 * (dzi[j - 1] + dzi[j])
        A[j] = -1.0 / dz_ave * (
            Kzz[j] / dzi[j] * (ysum[j + 1] + ysum[j]) / 2.0
            + Kzz[j - 1] / dzi[j - 1] * (ysum[j] + ysum[j - 1]) / 2.0
        ) / ysum[j]
        B[j] = 1.0 / dz_ave * Kzz[j] / dzi[j] * (ysum[j + 1] + ysum[j]) / 2.0 / ysum[j + 1]
        C[j] = 1.0 / dz_ave * Kzz[j - 1] / dzi[j - 1] * (ysum[j] + ysum[j - 1]) / 2.0 / ysum[j - 1]

        # Vertical advection
        A[j] += -((vz[j] > 0) * vz[j] - (vz[j - 1] < 0) * vz[j - 1]) / dz_ave
        B[j] += -((vz[j] < 0) * vz[j]) / dz_ave
        C[j] += ((vz[j - 1] > 0) * vz[j - 1]) / dz_ave

        if use_mol:
            # Diffusion (central) part of Ai/Bi/Ci -- common to all variants
            Ai[j] = -1.0 / dz_ave * (
                Dzz[j] / dzi[j] * (ysum[j + 1] + ysum[j]) / 2.0
                + Dzz[j - 1] / dzi[j - 1] * (ysum[j] + ysum[j - 1]) / 2.0
            ) / ysum[j]
            Bi[j] = 1.0 / dz_ave * Dzz[j] / dzi[j] * (ysum[j + 1] + ysum[j]) / 2.0 / ysum[j + 1]
            Ci[j] = 1.0 / dz_ave * Dzz[j - 1] / dzi[j - 1] * (ysum[j] + ysum[j - 1]) / 2.0 / ysum[j - 1]

            if use_vm:
                # vm-mode: upwind advection in place of gravity terms
                Ai[j] -= ((vm[j] > 0) * vm[j] - (vm[j - 1] < 0) * vm[j - 1]) / dz_ave
                Bi[j] -= ((vm[j] < 0) * vm[j]) / dz_ave
                Ci[j] += ((vm[j - 1] > 0) * vm[j - 1]) / dz_ave
            else:
                # Gravity-style central terms
                grav_j = -1.0 / Hpi[j] + ms * g[j] / (Navo * kb * Ti[j]) + alpha / Ti[j] * (Tco[j + 1] - Tco[j]) / dzi[j]
                grav_jm = -1.0 / Hpi[j - 1] + ms * g[j] / (Navo * kb * Ti[j - 1]) + alpha / Ti[j - 1] * (Tco[j] - Tco[j - 1]) / dzi[j - 1]
                Ai[j] += 1.0 / (2.0 * dz_ave) * (Dzz[j] * grav_j - Dzz[j - 1] * grav_jm)

                grav_jp_b = -1.0 / Hpi[j] + ms * g[j + 1] / (Navo * kb * Ti[j]) + alpha / Ti[j] * (Tco[j + 1] - Tco[j]) / dzi[j]
                Bi[j] += 1.0 / (2.0 * dz_ave) * Dzz[j] * grav_jp_b

                grav_jm_c = -1.0 / Hpi[j - 1] + ms * g[j - 1] / (Navo * kb * Ti[j - 1]) + alpha / Ti[j - 1] * (Tco[j] - Tco[j - 1]) / dzi[j - 1]
                Ci[j] += -1.0 / (2.0 * dz_ave) * Dzz[j - 1] * grav_jm_c

            if use_set:
                # Settling velocity (upwind), additional to whatever variant above
                Ai[j] -= ((vs[j] > 0) * vs[j] - (vs[j - 1] < 0) * vs[j - 1]) / dz_ave
                Bi[j] -= ((vs[j] < 0) * vs[j]) / dz_ave
                Ci[j] += ((vs[j - 1] > 0) * vs[j - 1]) / dz_ave

    return DiffusionCoeffs(
        A_eddy=A, B_eddy=B, C_eddy=C,
        A_mol=Ai, B_mol=Bi, C_mol=Ci,
        use_topflux=bool(cfg.use_topflux),
        top_flux=np.asarray(atm.top_flux, dtype=np.float64),
        use_botflux=bool(cfg.use_botflux),
        bot_flux=np.asarray(atm.bot_flux, dtype=np.float64),
        bot_vdep=np.asarray(atm.bot_vdep, dtype=np.float64),
        dzi=np.asarray(dzi, dtype=np.float64),
    )


def apply_diffusion(y: np.ndarray, coeffs: DiffusionCoeffs) -> np.ndarray:
    """Apply the diffusion operator: returns diff[nz, ni] = A*y[j] + B*y[j+1] + C*y[j-1].

    Includes boundary-condition contributions if use_topflux / use_botflux.
    """
    nz, ni = y.shape
    A_total = coeffs.A_eddy[:, None] + coeffs.A_mol      # (nz, ni)
    B_total = coeffs.B_eddy[:, None] + coeffs.B_mol
    C_total = coeffs.C_eddy[:, None] + coeffs.C_mol

    diff = np.zeros_like(y)
    # j = 0
    diff[0] = A_total[0] * y[0] + B_total[0] * y[1]
    # j = nz - 1
    diff[-1] = A_total[-1] * y[-1] + C_total[-1] * y[-2]
    # interior
    diff[1:-1] = (
        A_total[1:-1] * y[1:-1]
        + B_total[1:-1] * y[2:]
        + C_total[1:-1] * y[:-2]
    )

    # BC: top flux
    if coeffs.use_topflux:
        diff[-1] += coeffs.top_flux / coeffs.dzi[-1]
    # BC: bottom flux + deposition velocity (the deposition velocity term DOES
    # contribute to the Jacobian; it's added in `diffusion_block_diags`).
    if coeffs.use_botflux:
        diff[0] += (coeffs.bot_flux - y[0] * coeffs.bot_vdep) / coeffs.dzi[0]

    return diff


def diffusion_block_diags(coeffs: DiffusionCoeffs, ni: int):
    """Return the per-layer block-Jacobian DIAGONALS for the diffusion operator.

    The diffusion contribution to the Jacobian d(diff[j])/d(y[j', i]) is
    DIAGONAL in species (no inter-species coupling from diffusion alone), so
    each "block" can be represented as an ni-vector instead of an ni x ni
    dense matrix. Callers convert these into dense diagonal blocks when
    assembling the full LHS for block_thomas.

    Returns three arrays:
        diag_d  : shape (nz, ni)   -- diagonal entries of self-block
        sup_d   : shape (nz-1, ni) -- diagonal entries of super-diagonal block (layer j -> j+1)
        sub_d   : shape (nz-1, ni) -- diagonal entries of sub-diagonal block (layer j -> j-1)

    Sign convention: these match VULCAN's `lhs_jac_tot`. To build the LHS of
    the Rosenbrock implicit step `[c0*I - dF/dy]`, subtract these from
    chem_jac and add c0 to the diagonal:
        LHS_diag[j]  = c0*I  -  chem_jac[j]  -  diag(diag_d[j])
        LHS_sup[j]   =                       -  diag(sup_d[j])
        LHS_sub[j]   =                       -  diag(sub_d[j])
    """
    A = coeffs.A_eddy
    B = coeffs.B_eddy
    C = coeffs.C_eddy
    Ai = coeffs.A_mol
    Bi = coeffs.B_mol
    Ci = coeffs.C_mol

    # Diagonal of the j-block: A[j] + Ai[j, :]
    diag_d = A[:, None] + Ai                      # (nz, ni)
    # Super-diagonal (j -> j+1): B[j] + Bi[j, :], but only for j in [0, nz-2]
    sup_d = B[:-1, None] + Bi[:-1]                # (nz-1, ni)
    # Sub-diagonal (j -> j-1): C[j] + Ci[j, :], for j in [1, nz-1]
    sub_d = C[1:, None] + Ci[1:]                  # (nz-1, ni)

    # Add deposition velocity to the surface diagonal
    if coeffs.use_botflux:
        diag_d = diag_d.copy()
        diag_d[0] -= coeffs.bot_vdep / coeffs.dzi[0]

    return diag_d, sup_d, sub_d
