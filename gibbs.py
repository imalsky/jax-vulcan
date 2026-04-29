"""NASA-9 polynomial Gibbs energy and reverse-rate computation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from network import Network
from phy_const import kb

_P0 = 1.0e6
CORR = kb / _P0


def load_nasa9(species: tuple[str, ...], thermo_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NASA-9 polynomial coefficients per species.

    Returns (coeffs[ni, 2, 10], present[ni]). Index 0 is low-T (T<1000 K),
    index 1 is high-T. Species without a NASA-9 file get zeros.
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
        if flat.size < 20:
            raise ValueError(
                f"NASA-9 file {fp} has only {flat.size} entries; need at least 20"
            )
        coeffs[j, 0] = flat[0:10]
        coeffs[j, 1] = flat[10:20]
        present[j] = True
    return coeffs, present


def gibbs_sp_vector(coeffs: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Compute g_sp/(RT) per species per layer. coeffs (ni, 2, 10), T (nz,) → (ni, nz)."""
    T = np.asarray(T, dtype=np.float64)

    a_low_2d = coeffs[:, 0, :]
    a_high_2d = coeffs[:, 1, :]
    T_2d = T[None, :]

    def _g_branch(T_grid: np.ndarray, a: np.ndarray) -> np.ndarray:
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

    g_low = _g_branch(T_2d, a_low_2d)
    g_high = _g_branch(T_2d, a_high_2d)

    mask_low = (T < 1000.0)[None, :]
    return np.where(mask_low, g_low, g_high)


def K_eq_array(net: Network, gibbs_sp: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Equilibrium constants for each forward reaction. (nr+1, nz)

    Slots without a thermal reverse (photo/ion/conden/radiative, beyond
    stop_rev_indx) are 1 — a sentinel so the reverse-fill divide is safe.
    """
    T = np.asarray(T, dtype=np.float64)
    nz = T.shape[0]
    nr = net.nr

    K = np.ones((nr + 1, nz), dtype=np.float64)

    r_idx = net.reactant_idx
    p_idx = net.product_idx
    r_st = net.reactant_stoich
    p_st = net.product_stoich

    g_padded = np.concatenate([gibbs_sp, np.zeros((1, nz), dtype=np.float64)], axis=0)

    for i in range(1, nr + 1, 2):
        if not net.is_forward[i]:
            continue
        if i + 1 >= net.stop_rev_indx:
            continue
        if (
            net.is_photo[i] or net.is_ion[i]
            or net.is_conden[i] or net.is_radiative[i]
        ):
            continue

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

        delta_n = float(r_st[i].sum()) - float(p_st[i].sum())

        K[i] = np.exp(reac_sum - prod_sum)
        if delta_n != 0.0:
            K[i] *= (CORR * T) ** delta_n

    return K


def fill_reverse_k(
    net: Network, k: np.ndarray, K_eq: np.ndarray, remove_list: list[int] | None = None
) -> np.ndarray:
    """Fill reverse rate slots from forward rates and K_eq.

    Slots beyond stop_rev_indx are zeroed (photo/conden have no reverse).
    """
    k = np.asarray(k, dtype=np.float64).copy()
    remove = set(remove_list or [])

    for i in range(2, net.stop_rev_indx, 2):
        i_fwd = i - 1
        if i in remove or i_fwd in remove:
            k[i] = 0.0
            k[i_fwd] = 0.0
            continue
        K = K_eq[i_fwd]
        with np.errstate(divide="ignore", invalid="ignore"):
            k[i] = np.where(K > 0, k[i_fwd] / K, 0.0)

    for i in range(net.stop_rev_indx + 1, net.nr + 1, 2):
        k[i] = 0.0

    return k


def compute_all_k(
    net: Network,
    T: np.ndarray,
    M: np.ndarray,
    nasa9_coeffs: np.ndarray,
    remove_list: list[int] | None = None,
) -> np.ndarray:
    """Forward rates from rate constants + reverse fill from NASA-9 Gibbs."""
    from rates import compute_forward_k

    k_fwd = compute_forward_k(net, T, M)
    g_sp = gibbs_sp_vector(nasa9_coeffs, T)
    K_eq = K_eq_array(net, g_sp, T)
    return fill_reverse_k(net, k_fwd, K_eq, remove_list=remove_list)
