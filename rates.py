"""Rate-coefficient evaluation for VULCAN-JAX.

Given a parsed `Network` and an atmospheric (T, M) pair, compute the forward
rate constants `k[nr+1, nz]` (1-based reaction indexing). Reverse-rate slots
are filled in by `gibbs.fill_reverse_k` (or, end-to-end, by `build_rate_array`
below).

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

Phase 22a entry points (consolidate `legacy_io.ReadRate.{read_rate, rev_rate,
remove_rate, lim_lowT_rates}` into pure-NumPy functions over `Network`):
  - `compute_forward_k(net, T, M)`  -- forward (Arrhenius / Lindemann / Troe)
  - `apply_lowT_caps(net, k, T, M)` -- mirrors `lim_lowT_rates` (3 hardcoded
    reactions; caller gates on `cfg.use_lowT_limit_rates`).
  - `apply_remove_list(net, k, remove_list)` -- mirrors `remove_rate` (literal
    zero of the indices in `remove_list`, nothing more).
  - `build_rate_array(cfg, net, atm, nasa9_coeffs)` -- end-to-end glue:
    forward -> (lowT caps if cfg flag) -> reverse via Gibbs -> remove pass.
    Returns dense `(nr+1, nz)` `np.ndarray`. Bit-exact (<=1e-13) to the legacy
    `legacy_io.ReadRate` pipeline.

Implementation note: NumPy (not jax.numpy). Forward-rate evaluation runs once
at startup; the inner chemistry RHS that runs every step uses k as a fixed
input. Keeping this NumPy means we don't recompile when T changes (and gives
us bit-exact agreement with VULCAN's `var.k`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from network import Network


_RATES_ROOT = Path(__file__).resolve().parent


_SPECIAL_OH_CH3 = "OH + CH3 + M -> CH3OH + M"

# Low-T rate caps (mirrors `legacy_io.ReadRate.lim_lowT_rates`, lines 292-314,
# itself transcribed from upstream `op.py:320-342`). Three hardcoded
# reactions; the cap is applied to the post-Lindemann k_i below the
# corresponding T threshold. Values are from Moses et al. 2005.
_LOWT_CAP_RXN_CH3 = "H + CH3 + M -> CH4 + M"     # T <= 277.5 K, Lindemann form
_LOWT_CAP_RXN_C2H4 = "H + C2H4 + M -> C2H5 + M"  # T <= 300 K, constant 3.7e-30
_LOWT_CAP_RXN_C2H5 = "H + C2H5 + M -> C2H6 + M"  # T <= 200 K, constant 2.49e-27


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


# ---------------------------------------------------------------------------
# Phase 22a: low-T caps + remove_list + end-to-end build
# ---------------------------------------------------------------------------

def apply_lowT_caps(
    net: Network, k: np.ndarray, T: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Cap the post-Lindemann rate of three hardcoded reactions below their T
    thresholds. Mirrors `legacy_io.ReadRate.lim_lowT_rates` (which itself
    mirrors upstream `op.py:320-342`).

    Args:
        net: parsed Network.
        k:   (nr+1, nz) rate array; only forward slots are read/written here.
        T:   (nz,) temperatures.
        M:   (nz,) total third-body number density (`atm.M`).

    Returns:
        New (nr+1, nz) array with caps applied where T is below the cap
        threshold. Always operates -- the caller is responsible for gating
        on `cfg.use_lowT_limit_rates` (the default cfg has it False; only
        the Jupiter cfg switches it on).
    """
    k = np.asarray(k, dtype=np.float64).copy()
    T = np.asarray(T, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    for i in range(1, net.nr + 1, 2):
        if not net.is_forward[i]:
            continue
        rf = net.Rf.get(i, "")
        if rf == _LOWT_CAP_RXN_CH3:
            # Moses+2005 low-T Lindemann cap: k0=6e-29, kinf=2.06e-10*T^-0.4
            T_mask = T <= 277.5
            k0 = 6.0e-29
            kinf = 2.06e-10 * T**-0.4
            cap = k0 / (1.0 + k0 * M / kinf)
            k[i] = np.where(T_mask, cap, k[i])
        elif rf == _LOWT_CAP_RXN_C2H4:
            T_mask = T <= 300.0
            k[i] = np.where(T_mask, 3.7e-30, k[i])
        elif rf == _LOWT_CAP_RXN_C2H5:
            T_mask = T <= 200.0
            k[i] = np.where(T_mask, 2.49e-27, k[i])

    return k


def apply_remove_list(
    net: Network, k: np.ndarray, remove_list: Iterable[int] | None
) -> np.ndarray:
    """Zero rows in `remove_list`. Mirrors `legacy_io.ReadRate.remove_rate`.

    Important: this zeros literally what is in `remove_list`. It does not
    auto-zero a paired forward/reverse partner. Upstream's typical usage is
    to pass pairs (e.g. `[1, 2]`) so both ends are zeroed; passing a lone
    forward leaves the reverse intact (matching upstream).
    """
    k = np.asarray(k, dtype=np.float64).copy()
    if not remove_list:
        return k
    for i in remove_list:
        idx = int(i)
        if 0 <= idx <= net.nr:
            k[idx] = 0.0
    return k


def build_rate_array(cfg, net: Network, atm, nasa9_coeffs: np.ndarray) -> np.ndarray:
    """End-to-end rate-array build: forward -> (lowT caps) -> reverse -> remove.

    Args:
        cfg: vulcan_cfg-style module/object exposing `use_lowT_limit_rates`
            and `remove_list`.
        net: parsed Network.
        atm: atmosphere container exposing `Tco` (nz,) and `M` (nz,).
        nasa9_coeffs: (ni, 2, 10) NASA-9 coefficient table from
            `gibbs.load_nasa9`.

    Returns:
        (nr+1, nz) `np.ndarray`. Index 0 is unused (1-based reaction
        indexing). Reverse slots beyond `net.stop_rev_indx` are zero.

    Bit-exact (<=1e-13) replacement for the legacy
    `read_rate -> lim_lowT_rates -> rev_rate -> remove_rate` chain in
    `vulcan_jax.py:97-104`.
    """
    # Late import to avoid a top-level cycle (gibbs.compute_all_k imports
    # rates.compute_forward_k).
    from gibbs import K_eq_array, fill_reverse_k, gibbs_sp_vector

    T = np.asarray(atm.Tco, dtype=np.float64)
    M = np.asarray(atm.M, dtype=np.float64)

    # 1. Forward rates (Arrhenius / Lindemann / Troe).
    k = compute_forward_k(net, T, M)

    # 2. Optional low-T caps (Jupiter cfg sets the flag on; HD189 does not).
    if bool(getattr(cfg, "use_lowT_limit_rates", False)):
        k = apply_lowT_caps(net, k, T, M)

    # 3. Reverse rates from Gibbs. Pass `remove_list=None` so this routine
    #    does NOT auto-zero forward/reverse pairs; we apply remove_list in
    #    its own pass below to match legacy semantics exactly.
    g_sp = gibbs_sp_vector(nasa9_coeffs, T)
    K_eq = K_eq_array(net, g_sp, T)
    k = fill_reverse_k(net, k, K_eq, remove_list=None)

    # 4. Literal remove pass: zero only the entries in `cfg.remove_list`.
    k = apply_remove_list(net, k, getattr(cfg, "remove_list", None))

    return k


# ---------------------------------------------------------------------------
# Phase 22c convenience: one-shot rate setup helpers for tests + production.
# ---------------------------------------------------------------------------

def setup_var_k(cfg, var, atm) -> Network:
    """Parse network, load NASA-9 coeffs, build dense `var.k_arr`. Returns
    the parsed Network for callers that need it.

    Replaces the legacy `read_rate -> rev_rate -> remove_rate
    [-> lim_lowT_rates]` chain. The legacy `read_rate` is still required
    upstream of this call for metadata population (`var.Rf`,
    `var.pho_rate_index`, `var.n_branch`, `var.photo_sp`, ...).
    """
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
    """After `compute_J` (and optionally `compute_Jion`) has overwritten
    `var.k_arr` rows for photodissociation slots, apply `cfg.remove_list`
    on the dense array.
    """
    del atm  # unused after Phase 22d (no shape inference needed)
    var.k_arr = apply_remove_list(network, var.k_arr, cfg.remove_list)
