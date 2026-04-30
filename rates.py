"""Forward-rate-constant evaluation for VULCAN-JAX.

Supported forms: modified Arrhenius, Lindemann falloff (with k_inf), bare
3-body (M-multiplied at RHS time), and a hardcoded Troe expression for
`OH + CH3 + M -> CH3OH + M`. Photo / conden / radiative / ion slots are
zero here and filled at runtime.

Output is `k[nr+1, nz]` with 1-based reaction indexing. The 3-body [M]
factor is applied in the chemistry RHS (depends on time-evolving sum(y)),
not here. Implementation is pure NumPy: setup runs once, and keeping it
out of JAX gives bit-exact agreement with VULCAN-master's `var.k`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from network import Network


_RATES_ROOT = Path(__file__).resolve().parent


_SPECIAL_OH_CH3 = "OH + CH3 + M -> CH3OH + M"

# Three hardcoded low-T rate caps (Moses et al. 2005). Each is applied to
# the post-Lindemann rate below the T threshold.
_LOWT_CAP_RXN_CH3 = "H + CH3 + M -> CH4 + M"     # T <= 277.5 K, Lindemann form
_LOWT_CAP_RXN_C2H4 = "H + C2H4 + M -> C2H5 + M"  # T <= 300 K, constant 3.7e-30
_LOWT_CAP_RXN_C2H5 = "H + C2H5 + M -> C2H6 + M"  # T <= 200 K, constant 2.49e-27


def _arrhenius(a: float, n: float, E: float, T: np.ndarray) -> np.ndarray:
    """k(T) = a * T^n * exp(-E/T)."""
    return a * T**n * np.exp(-E / T)


def _troe_OH_CH3(T: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Hardcoded Troe form for `OH + CH3 + M -> CH3OH + M` (Jasper 2017).

    Returns k of shape [nz] in cm^3/s.
    """
    k0 = (
        1.932e3 * T**-9.88 * np.exp(-7544.0 / T)
        + 5.109e-11 * T**-6.25 * np.exp(-1433.0 / T)
    )
    kinf = 1.031e-10 * T**-0.018 * np.exp(16.74 / T)
    Fc = (
        0.1855 * np.exp(-T / 155.8)
        + 0.8145 * np.exp(-T / 1675.0)
        + np.exp(-4531.0 / T)
    )
    nn = 0.75 - 1.27 * np.log(Fc)
    ff = np.exp(np.log(Fc) / (1.0 + (np.log(k0 * M / kinf) / nn) ** 2))
    return k0 / (1.0 + k0 * M / kinf) * ff


def compute_forward_k(net: Network, T: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Compute forward rate coefficients on the atmospheric grid.

    Args:
        net: Parsed Network.
        T:   Temperature [K], shape (nz,).
        M:   Total number density of third bodies [cm^-3], shape (nz,).

    Returns:
        k: array of shape (nr+1, nz). `k[0, :]` is unused (1-based indexing).
        Reverse-rate slots (even indices) are zeroed -- filled by gibbs module.
        Photo, ion, condensation, radiative-recombination slots are zero --
        filled by the photo update / condensation logic at runtime.
    """
    T = np.asarray(T, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    nz = T.shape[0]
    if M.shape != (nz,):
        raise ValueError(f"T and M must have the same shape; got {T.shape} and {M.shape}")

    nr = net.nr
    k = np.zeros((nr + 1, nz), dtype=np.float64)

    # Mask flags as 1-D arrays (length nr+1); we iterate odd indices (forward rxns).
    # `is_three_body` is implicit in `has_kinf` here -- 3-body w/ k_inf goes
    # through the Lindemann branch, 3-body w/o k_inf falls through to the
    # plain Arrhenius branch (the [M] factor lives in the chemistry RHS).
    is_forward = net.is_forward
    has_kinf = net.has_kinf
    is_special = net.is_special
    is_conden = net.is_conden
    is_photo = net.is_photo
    is_ion = net.is_ion
    is_radiative = net.is_radiative
    a = net.a
    n = net.n
    E = net.E
    a_inf = net.a_inf
    n_inf = net.n_inf
    E_inf = net.E_inf

    for i in range(1, nr + 1, 2):
        if not is_forward[i]:
            continue
        if is_photo[i] or is_ion[i] or is_conden[i] or is_radiative[i]:
            continue

        if is_special[i]:
            if net.Rf.get(i, "").strip() == _SPECIAL_OH_CH3:
                k[i] = _troe_OH_CH3(T, M)
            elif a[i] != 0.0:
                k[i] = _arrhenius(a[i], n[i], E[i], T)
            continue

        if has_kinf[i]:
            k0 = _arrhenius(a[i], n[i], E[i], T)
            kinf = _arrhenius(a_inf[i], n_inf[i], E_inf[i], T)
            k[i] = k0 / (1.0 + k0 * M / kinf)
        else:
            k[i] = _arrhenius(a[i], n[i], E[i], T)

    return k


def k_dict_from_array(net: Network, k_arr: np.ndarray) -> dict:
    """Convert (nr+1, nz) into a `{i: array(nz)}` dict; skips index 0."""
    out: dict = {}
    for i in range(1, net.nr + 1):
        out[i] = np.asarray(k_arr[i]).copy()
    return out


def k_array_from_dict(net: Network, k_dict: dict, nz: int) -> np.ndarray:
    """Pack a `{i: array(nz)}` dict into a (nr+1, nz) ndarray."""
    out = np.zeros((net.nr + 1, nz), dtype=np.float64)
    for i, vec in k_dict.items():
        if 1 <= i <= net.nr:
            out[i] = np.asarray(vec, dtype=np.float64)
    return out


def apply_lowT_caps(
    net: Network, k: np.ndarray, T: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Apply three Moses+2005 low-T rate caps. Caller gates on
    `cfg.use_lowT_limit_rates` (only the Jupiter cfg sets it)."""
    k = np.asarray(k, dtype=np.float64).copy()
    T = np.asarray(T, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    for i in range(1, net.nr + 1, 2):
        if not net.is_forward[i]:
            continue
        rf = net.Rf.get(i, "")
        if rf == _LOWT_CAP_RXN_CH3:
            # Moses+2005 cap: k0=6e-29, kinf=2.06e-10*T^-0.4 (Lindemann form).
            T_mask = T <= 277.5
            k0 = 6.0e-29
            kinf = 2.06e-10 * T**-0.4
            cap = k0 / (1.0 + k0 * M / kinf)
            k[i] = np.where(T_mask, cap, k[i])
        elif rf == _LOWT_CAP_RXN_C2H4:
            k[i] = np.where(T <= 300.0, 3.7e-30, k[i])
        elif rf == _LOWT_CAP_RXN_C2H5:
            k[i] = np.where(T <= 200.0, 2.49e-27, k[i])

    return k


def apply_remove_list(
    net: Network, k: np.ndarray, remove_list: Iterable[int] | None
) -> np.ndarray:
    """Zero the rows in `remove_list`. No auto-pairing: passing a lone
    forward leaves its reverse intact."""
    k = np.asarray(k, dtype=np.float64).copy()
    if not remove_list:
        return k
    for i in remove_list:
        idx = int(i)
        if 0 <= idx <= net.nr:
            k[idx] = 0.0
    return k


def build_rate_array(cfg, net: Network, atm, nasa9_coeffs: np.ndarray) -> np.ndarray:
    """End-to-end rate build: forward → (lowT caps) → reverse via Gibbs → remove.

    Index 0 is unused (1-based reactions); reverse slots beyond
    `net.stop_rev_indx` are zero.
    """
    # Late import: gibbs.compute_all_k → rates.compute_forward_k cycle.
    from gibbs import K_eq_array, fill_reverse_k, gibbs_sp_vector

    T = np.asarray(atm.Tco, dtype=np.float64)
    M = np.asarray(atm.M, dtype=np.float64)

    k = compute_forward_k(net, T, M)

    if bool(getattr(cfg, "use_lowT_limit_rates", False)):
        k = apply_lowT_caps(net, k, T, M)

    # Pass `remove_list=None` here; we apply remove_list in its own pass
    # to match legacy semantics (no auto fwd/rev pairing).
    g_sp = gibbs_sp_vector(nasa9_coeffs, T)
    K_eq = K_eq_array(net, g_sp, T)
    k = fill_reverse_k(net, k, K_eq, remove_list=None)

    return apply_remove_list(net, k, getattr(cfg, "remove_list", None))


def setup_var_k(cfg, var, atm) -> Network:
    """Parse network, load NASA-9 coeffs, write `var.k_arr`. Returns the Network."""
    from gibbs import load_nasa9
    from network import parse_network

    network = parse_network(cfg.network)
    thermo_dir = Path(cfg.network).parent
    if not (thermo_dir / "NASA9").exists():
        thermo_dir = _RATES_ROOT / "thermo"
    nasa9_coeffs, _ = load_nasa9(network.species, thermo_dir)
    var.k_arr = build_rate_array(cfg, network, atm, nasa9_coeffs)
    return network


def apply_photo_remove(cfg, var, network: Network, atm) -> None:
    """Re-apply `cfg.remove_list` after `compute_J`/`compute_Jion` has
    overwritten the photolysis rows of `var.k_arr`."""
    del atm
    var.k_arr = apply_remove_list(network, var.k_arr, cfg.remove_list)
