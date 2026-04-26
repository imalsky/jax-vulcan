"""Rate-coefficient evaluation for VULCAN-JAX.

Given a parsed `Network` and an atmospheric (T, M) pair, compute the forward
rate constants `k[nr+1, nz]` (1-based reaction indexing). Reverse-rate slots
are left at zero -- they're filled in by `gibbs.compute_reverse_k` once the
Gibbs free-energy data is loaded.

Supported functional forms (mirrors `op.py:ReadRate.read_rate`):
  - Modified Arrhenius:        k = A * T^n * exp(-E/T)
  - Lindemann falloff (3-body w/ k_inf):
        k_0   = A   * T^n   * exp(-E/T)
        k_inf = A_∞ * T^n_∞ * exp(-E_∞/T)
        k     = k_0 / (1 + k_0 * M / k_inf)
  - 3-body w/o k_inf:           k = A * T^n * exp(-E/T)  (multiplied by [M] in rate eqn)
  - Hardcoded Troe form for `OH + CH3 + M -> CH3OH + M` (from Jasper 2017)
  - Photo / condensation / radiative / ion: k = 0 here; filled by photo update
                                             or condensation logic at runtime.

The output `k[i, :]` for 3-body reactions does NOT include the [M] factor --
that multiplication happens in the chemistry RHS (`rates.compute_chem_rate`)
because it depends on the time-evolving total-density M(z) = sum(y).

Implementation note: NumPy (not jax.numpy). Forward-rate evaluation runs once
at startup; the inner chemistry RHS that runs every step uses k as a fixed
input. Keeping this NumPy means we don't recompile when T changes (and gives
us bit-exact agreement with VULCAN's `var.k`).
"""

from __future__ import annotations

import numpy as np

from network import Network


_SPECIAL_OH_CH3 = "OH + CH3 + M -> CH3OH + M"


def _arrhenius(a: float, n: float, E: float, T: np.ndarray) -> np.ndarray:
    """k(T) = a * T^n * exp(-E/T)."""
    return a * T**n * np.exp(-E / T)


def _troe_OH_CH3(T: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Hardcoded Troe form for the OH + CH3 + M -> CH3OH + M reaction.

    Mirrors `op.py:200-207` exactly. Returns k array shape [nz] in cm^3/s.
    """
    # k_0: sum of two modified-Arrhenius terms
    k0 = (
        1.932e3 * T**-9.88 * np.exp(-7544.0 / T)
        + 5.109e-11 * T**-6.25 * np.exp(-1433.0 / T)
    )
    # k_inf: single modified-Arrhenius
    kinf = 1.031e-10 * T**-0.018 * np.exp(16.74 / T)
    # Troe broadening factor
    Fc = (
        0.1855 * np.exp(-T / 155.8)
        + 0.8145 * np.exp(-T / 1675.0)
        + np.exp(-4531.0 / T)
    )
    nn = 0.75 - 1.27 * np.log(Fc)
    # Pressure-dependent broadening (Jasper 2017)
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

    # Mask flags as 1-D arrays (length nr+1); we iterate odd indices (forward rxns)
    is_forward = net.is_forward
    has_kinf = net.has_kinf
    is_three_body = net.is_three_body
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
        # Photo / ion / condensation / radiative-recomb: rate filled at runtime.
        if is_photo[i] or is_ion[i] or is_conden[i] or is_radiative[i]:
            continue

        if is_special[i]:
            if net.Rf.get(i, "").strip() == _SPECIAL_OH_CH3:
                k[i] = _troe_OH_CH3(T, M)
            else:
                # Unknown special reaction: fall back to Arrhenius if A>0 else zero.
                if a[i] != 0.0:
                    k[i] = _arrhenius(a[i], n[i], E[i], T)
            continue

        if has_kinf[i]:
            # Lindemann-Hinshelwood falloff
            k0 = _arrhenius(a[i], n[i], E[i], T)
            kinf = _arrhenius(a_inf[i], n_inf[i], E_inf[i], T)
            k[i] = k0 / (1.0 + k0 * M / kinf)
        else:
            # Two-body or 3-body w/o k_inf
            k[i] = _arrhenius(a[i], n[i], E[i], T)

    return k


def k_dict_from_array(net: Network, k_arr: np.ndarray) -> dict:
    """Convert k array shape (nr+1, nz) into VULCAN-style dict `{i: array(nz)}`.

    Useful for comparing against VULCAN's `var.k` dict. Skips index 0 (unused).
    """
    out: dict = {}
    for i in range(1, net.nr + 1):
        out[i] = np.asarray(k_arr[i]).copy()
    return out


def k_array_from_dict(net: Network, k_dict: dict, nz: int) -> np.ndarray:
    """Inverse of `k_dict_from_array` -- pack a VULCAN dict into a (nr+1, nz) array."""
    out = np.zeros((net.nr + 1, nz), dtype=np.float64)
    for i, vec in k_dict.items():
        if 1 <= i <= net.nr:
            out[i] = np.asarray(vec, dtype=np.float64)
    return out
