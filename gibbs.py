"""NASA-9 polynomial Gibbs energy and reverse-rate computation for VULCAN-JAX.

Mirrors `chem_funs.gibbs_sp` and `chem_funs.Gibbs` from VULCAN's auto-generated
output (lines 4023-4090 of chem_funs.py).

NASA-9 file format (`thermo/NASA9/<species>.txt`):
    Two temperature ranges (low T<1000 K, high T>=1000 K), 10 coefficients each.
    Indices 0..6 are the temperature polynomial coefficients, 7 is unused, 8
    and 9 are integration constants:
        h_RT(T,a) = -a[0]/T^2 + a[1]*ln(T)/T + a[2] + a[3]*T/2 + a[4]*T^2/3
                  + a[5]*T^3/4 + a[6]*T^4/5 + a[8]/T
        s_R (T,a) = -a[0]/(2 T^2) - a[1]/T + a[2]*ln(T) + a[3]*T + a[4]*T^2/2
                  + a[5]*T^3/3 + a[6]*T^4/4 + a[9]
        g_RT(T,a) = h_RT(T,a) - s_R(T,a)        (= G/(RT), dimensionless)

Equilibrium constant (concentration units) for forward reaction `i`:
    K_eq(i, T) = exp(-(stoich_prod * g_RT_prod  -  stoich_reac * g_RT_reac))
                 * (corr * T)^(n_reac - n_prod)
    corr = k_B / P0,   P0 = 1e6 dyne/cm^2

Reverse rate:  k[i+1] = k[i] / K_eq(i, T)  for i in [1, 3, ..., stop_rev_indx-2]

Implementation: vectorized NumPy.  We pre-stack NASA-9 coefficients into one
[ni, 2, 10] array indexed by species and (low/high) regime; species missing a
NASA-9 file have all-zero coefficients (caller must check that no reverse-rate
calc actually needs them).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from network import Network
from phy_const import kb

# P0 = 1e6 dyne/cm^2 (1 atm in cgs, conventional reference pressure)
_P0 = 1.0e6
CORR = kb / _P0  # = k_B / P0


def load_nasa9(species: tuple[str, ...], thermo_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NASA-9 polynomial coefficients for each species.

    Args:
        species: ordered species tuple (length ni).
        thermo_dir: path to `thermo/` directory containing `NASA9/<sp>.txt`.

    Returns:
        coeffs: float64 array of shape (ni, 2, 10).
            coeffs[i, 0] = low-T coefficients (T < 1000 K)
            coeffs[i, 1] = high-T coefficients (T >= 1000 K)
        present:  bool array (ni,). True iff a NASA-9 file was found.

    Species without a NASA-9 file (e.g. condensed species H2O_l_s, S8_l_s) are
    given all-zero coefficients. Callers that depend on the Gibbs energy of
    such species must guard against this.
    """
    thermo_dir = Path(thermo_dir)
    ni = len(species)
    coeffs = np.zeros((ni, 2, 10), dtype=np.float64)
    present = np.zeros(ni, dtype=bool)
    for j, sp in enumerate(species):
        fp = thermo_dir / "NASA9" / f"{sp}.txt"
        if not fp.exists():
            continue
        flat = np.loadtxt(fp).flatten()
        # Some species (e.g. HO2) have 3 temperature ranges = 30 numbers; some
        # have just 20. VULCAN's chem_funs.py uses [0:10] for low-T and [10:20]
        # for high-T regardless. Truncate to 20 to match exactly.
        if flat.size < 20:
            raise ValueError(
                f"NASA-9 file {fp} has only {flat.size} entries; need at least 20"
            )
        coeffs[j, 0] = flat[0:10]
        coeffs[j, 1] = flat[10:20]
        present[j] = True
    return coeffs, present


def _h_RT(T: np.ndarray, a: np.ndarray) -> np.ndarray:
    """h(T)/(R T) from NASA-9 polynomial. T shape arbitrary; a shape (..., 10) broadcastable."""
    return (
        -a[..., 0] / T**2
        + a[..., 1] * np.log(T) / T
        + a[..., 2]
        + a[..., 3] * T / 2.0
        + a[..., 4] * T**2 / 3.0
        + a[..., 5] * T**3 / 4.0
        + a[..., 6] * T**4 / 5.0
        + a[..., 8] / T
    )


def _s_R(T: np.ndarray, a: np.ndarray) -> np.ndarray:
    """s(T)/R from NASA-9 polynomial."""
    return (
        -a[..., 0] / T**2 / 2.0
        - a[..., 1] / T
        + a[..., 2] * np.log(T)
        + a[..., 3] * T
        + a[..., 4] * T**2 / 2.0
        + a[..., 5] * T**3 / 3.0
        + a[..., 6] * T**4 / 4.0
        + a[..., 9]
    )


def gibbs_sp_vector(coeffs: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Vectorized g(T)/(RT) for all species.

    Args:
        coeffs: shape (ni, 2, 10) -- NASA-9 coefficients per species.
        T:      shape (nz,)       -- temperatures.

    Returns:
        gi: shape (ni, nz) -- g_sp / (R T) per species per layer.
    """
    T = np.asarray(T, dtype=np.float64)

    a_low_2d = coeffs[:, 0, :]               # (ni, 10)
    a_high_2d = coeffs[:, 1, :]              # (ni, 10)
    T_2d = T[None, :]                        # (1, nz)

    def _g_branch(T_grid: np.ndarray, a: np.ndarray) -> np.ndarray:
        # T_grid: (1, nz); a: (ni, 10); output: (ni, nz)
        return (
            -a[:, 0:1] / T_grid**2
            + a[:, 1:2] * np.log(T_grid) / T_grid
            + a[:, 2:3]
            + a[:, 3:4] * T_grid / 2.0
            + a[:, 4:5] * T_grid**2 / 3.0
            + a[:, 5:6] * T_grid**3 / 4.0
            + a[:, 6:7] * T_grid**4 / 5.0
            + a[:, 8:9] / T_grid
        ) - (
            -a[:, 0:1] / T_grid**2 / 2.0
            - a[:, 1:2] / T_grid
            + a[:, 2:3] * np.log(T_grid)
            + a[:, 3:4] * T_grid
            + a[:, 4:5] * T_grid**2 / 2.0
            + a[:, 5:6] * T_grid**3 / 3.0
            + a[:, 6:7] * T_grid**4 / 4.0
            + a[:, 9:10]
        )

    g_low = _g_branch(T_2d, a_low_2d)        # (ni, nz)
    g_high = _g_branch(T_2d, a_high_2d)      # (ni, nz)

    mask_low = (T < 1000.0)[None, :]          # (1, nz)
    g = np.where(mask_low, g_low, g_high)    # (ni, nz)
    return g


def K_eq_array(net: Network, gibbs_sp: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Equilibrium constants K_eq(forward_i, T) for each forward reaction.

    Args:
        net: parsed Network.
        gibbs_sp: (ni, nz) array of g_sp/(RT) values.
        T: (nz,) temperatures.

    Returns:
        K_eq: (nr+1, nz). Only the forward slots [1, 3, ..., stop_rev_indx-2]
        contain valid values; everything else is 1 (so dividing by 1 gives 0
        when k_forward is 0 -- safe sentinel).

    The formula is:  K_eq = exp( sum_reac stoich*g_sp - sum_prod stoich*g_sp )
                          * (corr*T)^(n_reac - n_prod)
    Sign convention matches `make_chem_funs.make_Gibbs`:
        gibbs = exp( -(- Σ_reac stoich*g + Σ_prod stoich*g) ) * (corr*T)^Δn
              = exp( Σ_reac stoich*g - Σ_prod stoich*g ) * (corr*T)^Δn
        with Δn = n_reac - n_prod.
    """
    T = np.asarray(T, dtype=np.float64)
    nz = T.shape[0]
    nr = net.nr

    K = np.ones((nr + 1, nz), dtype=np.float64)

    # For each forward reaction: compute the linear combination of g_sp and Δn.
    r_idx = net.reactant_idx
    p_idx = net.product_idx
    r_st = net.reactant_stoich
    p_st = net.product_stoich

    # Broadcast: gibbs_sp_padded[ni] = 0 (so padding contributes nothing)
    g_padded = np.concatenate([gibbs_sp, np.zeros((1, nz), dtype=np.float64)], axis=0)
    # Now g_padded[reactant_idx[i, k], :] -> the per-layer Gibbs for that slot

    for i in range(1, nr + 1, 2):
        if not net.is_forward[i]:
            continue
        # Compute K only for reactions with reverses (i.e. before stop_rev_indx)
        if i + 1 >= net.stop_rev_indx:
            continue
        # Skip photo, conden, ion, radiative (they have no thermal reverse anyway)
        if (
            net.is_photo[i] or net.is_ion[i]
            or net.is_conden[i] or net.is_radiative[i]
        ):
            continue

        # Σ_reac stoich*g - Σ_prod stoich*g
        reac_sum = np.zeros(nz, dtype=np.float64)
        for k_slot in range(r_idx.shape[1]):
            sp = r_idx[i, k_slot]
            st = r_st[i, k_slot]
            if st == 0.0:
                continue
            reac_sum += st * g_padded[sp]
        prod_sum = np.zeros(nz, dtype=np.float64)
        for k_slot in range(p_idx.shape[1]):
            sp = p_idx[i, k_slot]
            st = p_st[i, k_slot]
            if st == 0.0:
                continue
            prod_sum += st * g_padded[sp]

        n_reac = float(r_st[i].sum())
        n_prod = float(p_st[i].sum())
        delta_n = n_reac - n_prod

        K[i] = np.exp(reac_sum - prod_sum)
        if delta_n != 0.0:
            K[i] *= (CORR * T) ** delta_n

    return K


def fill_reverse_k(
    net: Network, k: np.ndarray, K_eq: np.ndarray, remove_list: list[int] | None = None
) -> np.ndarray:
    """Fill in reverse rate constants in-place based on forward k and K_eq.

    Args:
        net: parsed Network.
        k:   (nr+1, nz) array of rates; forward slots populated, reverses zero.
        K_eq: (nr+1, nz) equilibrium constants from `K_eq_array`.
        remove_list: list of reaction indices to forcibly zero out (from
            vulcan_cfg.remove_list). Both forward and reverse slots get zeroed.

    Returns:
        k with reverse slots filled. Slot i (even) gets `k[i-1] / K_eq[i-1]`
        for i in [2, 4, ..., stop_rev_indx-2]. Beyond stop_rev_indx, all
        reverse slots are forced to zero.
    """
    k = np.asarray(k, dtype=np.float64).copy()
    remove = set(remove_list or [])

    # Compute reverses for the well-defined range
    for i in range(2, net.stop_rev_indx, 2):
        i_fwd = i - 1
        if i in remove or i_fwd in remove:
            k[i] = 0.0
            k[i_fwd] = 0.0
            continue
        # Avoid divide-by-zero (K_eq=1 by default sentinel; division yields k_fwd)
        K = K_eq[i_fwd]
        # Where K==0 set reverse to 0 to avoid inf
        with np.errstate(divide="ignore", invalid="ignore"):
            k[i] = np.where(K > 0, k[i_fwd] / K, 0.0)

    # Zero out reverses beyond stop_rev_indx (photo, conden have no reverses)
    for i in range(net.stop_rev_indx + 1, net.nr + 1, 2):
        # Even slot beyond stop_rev_indx
        k[i] = 0.0

    return k


def compute_all_k(
    net: Network,
    T: np.ndarray,
    M: np.ndarray,
    nasa9_coeffs: np.ndarray,
    remove_list: list[int] | None = None,
) -> np.ndarray:
    """Convenience: compute_forward_k + reverse fill in one call."""
    from rates import compute_forward_k

    k_fwd = compute_forward_k(net, T, M)
    g_sp = gibbs_sp_vector(nasa9_coeffs, T)
    K_eq = K_eq_array(net, g_sp, T)
    k_full = fill_reverse_k(net, k_fwd, K_eq, remove_list=remove_list)
    return k_full
