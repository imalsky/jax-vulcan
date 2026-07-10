"""Differentiable (JAX) rate-coefficient build: `T -> k_arr` on the AD graph.

Vectorized port of the NumPy `rates.compute_forward_k` (Arrhenius / Lindemann /
the one hardcoded Troe form) and the Gibbs reverse-rate path
(`gibbs.gibbs_sp_vector` / `K_eq_array` / `fill_reverse_k`). The NumPy versions
run host-side at setup and are bit-exact vs VULCAN-master; this module is the
*differentiable* counterpart so a temperature perturbation flows through the
rate constants (the dominant chemical T-dependence), which the frozen
host-side `k_arr` does not. It also exposes the Arrhenius rate coefficients
(`compute_forward_k(..., a=, n=, E=, ...)` / `build_rate_array(..., rate_coeffs=)`)
and the NASA-9 thermo table (`nasa9_coeffs`, already on the graph via the Gibbs
reverse path) as differentiable inputs, for rate-coefficient / thermochemistry
uncertainty gradients. Validate with `tests`/scripts against
`rates.build_rate_array` before trusting it.

Low-T caps (`rates.apply_lowT_caps`, Moses+2005) ARE ported (`apply_lowT_caps`
below, applied when `build_rate_array(..., use_lowT_caps=True)`), so dL/dT is
correct on cool networks too. They only fire below 277.5/300/200 K and are off
by default (the hot benchmarks never trigger them). Photo / ion / conden /
radiative slots are zero here (filled at runtime), exactly as in the NumPy path.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .network import Network
from .phy_const import kb

_P0 = 1.0e6
CORR = kb / _P0
_SPECIAL_OH_CH3 = "OH + CH3 + M -> CH3OH + M"

# Upper bound on the Gibbs exponent (reac - prod) before exp() in K_eq_array.
# float64 exp overflows at ~709; on cold columns (upper atmospheres of cool
# planets, ~100-160 K) a strongly forward-favored reaction's exponent can exceed
# it, sending K_eq -> +inf. The primal is unaffected (k_rev = k_fwd / inf -> 0,
# which is physically correct for an effectively-irreversible cold reaction), but
# the forward-mode jvp of k_fwd/K then evaluates finite*(inf/inf) = NaN, silently
# poisoning d(k)/dT for the whole column. Clipping the *upper* side only keeps K
# huge-but-finite (so k_rev stays ~0 to float precision, matching the unclamped
# primal) while making the tangent finite. The lower side needs no clip: exp
# underflow gives K -> 0, and fill_reverse_k's `where(K > 0, ...)` already returns
# a tangent-safe 0 there.
_EXP_ARG_MAX = 500.0

# Three Moses+2005 low-T rate caps (mirror rates.py), applied to the forward
# slot below a temperature threshold on cool-atmosphere networks.
_LOWT_CAP_RXN_CH3 = "H + CH3 + M -> CH4 + M"  # T <= 277.5 K, Lindemann cap
_LOWT_CAP_RXN_C2H4 = "H + C2H4 + M -> C2H5 + M"  # T <= 300 K, constant 3.7e-30
_LOWT_CAP_RXN_C2H5 = "H + C2H5 + M -> C2H6 + M"  # T <= 200 K, constant 2.49e-27


def _arrhenius(a, n, E, T):
    return a * T**n * jnp.exp(-E / T)


def _troe_OH_CH3(T, M):
    """Hardcoded Troe form for OH + CH3 + M -> CH3OH + M (Jasper 2017)."""
    k0 = 1.932e3 * T**-9.88 * jnp.exp(-7544.0 / T) + 5.109e-11 * T**-6.25 * jnp.exp(
        -1433.0 / T
    )
    kinf = 1.031e-10 * T**-0.018 * jnp.exp(16.74 / T)
    Fc = (
        0.1855 * jnp.exp(-T / 155.8)
        + 0.8145 * jnp.exp(-T / 1675.0)
        + jnp.exp(-4531.0 / T)
    )
    nn = 0.75 - 1.27 * jnp.log(Fc)
    ff = jnp.exp(jnp.log(Fc) / (1.0 + (jnp.log(k0 * M / kinf) / nn) ** 2))
    return k0 / (1.0 + k0 * M / kinf) * ff


def _col(x):
    """(nr+1,) network array -> (nr+1, 1) jnp float column for broadcasting."""
    return jnp.asarray(np.asarray(x, dtype=np.float64))[:, None]


def compute_forward_k(
    net: Network,
    T: jnp.ndarray,
    M: jnp.ndarray,
    *,
    a=None,
    n=None,
    E=None,
    a_inf=None,
    n_inf=None,
    E_inf=None,
) -> jnp.ndarray:
    """Vectorized JAX forward rates. T, M are (nz,) JAX arrays -> (nr+1, nz).

    The six Arrhenius/Lindemann coefficient arrays default to the network's
    static values; pass any as a (nr+1,) JAX array to differentiate the rates
    w.r.t. rate-coefficient uncertainty (`jvp`/`grad`). The one hardcoded Troe
    row (OH+CH3) is not overridable (its constants stay fixed).
    """
    T = jnp.asarray(T)[None, :]  # (1, nz)
    M = jnp.asarray(M)[None, :]

    def _ovr(val, default):
        return _col(default) if val is None else jnp.asarray(val)[:, None]

    a, n, E = _ovr(a, net.a), _ovr(n, net.n), _ovr(E, net.E)
    a_inf, n_inf, E_inf = (
        _ovr(a_inf, net.a_inf),
        _ovr(n_inf, net.n_inf),
        _ovr(E_inf, net.E_inf),
    )

    arr = _arrhenius(a, n, E, T)  # plain Arrhenius (a==0 -> 0)
    kinf = _arrhenius(a_inf, n_inf, E_inf, T)
    kinf_safe = jnp.where(kinf > 0, kinf, 1.0)  # guard unused branch
    lindemann = arr / (1.0 + arr * M / kinf_safe)
    has_kinf = jnp.asarray(np.asarray(net.has_kinf, dtype=bool))[:, None]
    general = jnp.where(has_kinf, lindemann, arr)

    # Special handling: the one OH+CH3 Troe row; other "special" rows fall back to
    # plain Arrhenius (a==0 specials give 0 automatically).
    is_special = jnp.asarray(np.asarray(net.is_special, dtype=bool))[:, None]
    oh = np.zeros(net.nr + 1, dtype=bool)
    for i in range(1, net.nr + 1):
        if net.is_special[i] and net.Rf.get(i, "").strip() == _SPECIAL_OH_CH3:
            oh[i] = True
    oh_mask = jnp.asarray(oh)[:, None]
    troe = _troe_OH_CH3(T, M)
    special_val = jnp.where(oh_mask, troe, arr)
    k = jnp.where(is_special, special_val, general)

    # Keep only thermal forward slots; zero photo/ion/conden/radiative/non-forward.
    thermal_fwd = (
        net.is_forward
        & ~net.is_photo
        & ~net.is_ion
        & ~net.is_conden
        & ~net.is_radiative
    )
    keep = jnp.asarray(np.asarray(thermal_fwd, dtype=bool))[:, None]
    return jnp.where(keep, k, 0.0)


def apply_lowT_caps(
    net: Network, k_fwd: jnp.ndarray, T: jnp.ndarray, M: jnp.ndarray
) -> jnp.ndarray:
    """JAX port of `rates.apply_lowT_caps` (Moses+2005 low-T recombination caps).

    Caps the three forward rows when present in `net`, gated by a temperature
    threshold via `jnp.where`. The cap branch is a T-independent constant for
    C2H4/C2H5 (so dk/dT = 0 below threshold) and a Lindemann form for CH3; the
    tangent is non-smooth only exactly at the threshold (a measure-zero kink),
    smooth on either side. Mirrors the NumPy path so dL/dT is correct on cool
    networks. `k_fwd` (nr+1, nz); `T`, `M` (nz,).
    """
    Tz = jnp.asarray(T)
    Mz = jnp.asarray(M)
    # Locate the cap rows on the host (Rf is static); first forward slot each.
    rows: dict[str, int] = {}
    for i in range(1, net.nr + 1, 2):
        if net.is_forward[i]:
            rows.setdefault(net.Rf.get(i, ""), i)

    k = k_fwd
    i = rows.get(_LOWT_CAP_RXN_CH3)
    if i is not None:
        kinf = 2.06e-10 * Tz**-0.4
        cap = 6.0e-29 / (1.0 + 6.0e-29 * Mz / kinf)
        k = k.at[i].set(jnp.where(Tz <= 277.5, cap, k[i]))
    i = rows.get(_LOWT_CAP_RXN_C2H4)
    if i is not None:
        k = k.at[i].set(jnp.where(Tz <= 300.0, 3.7e-30, k[i]))
    i = rows.get(_LOWT_CAP_RXN_C2H5)
    if i is not None:
        k = k.at[i].set(jnp.where(Tz <= 200.0, 2.49e-27, k[i]))
    return k


def gibbs_sp_vector(coeffs, T: jnp.ndarray) -> jnp.ndarray:
    """g_sp/(RT) per species per layer. coeffs (ni,2,10), T (nz,) -> (ni,nz).

    `coeffs` may be a NumPy array (the static thermo) or a JAX array/tracer:
    `jnp.asarray` (not `np.asarray`) keeps it on the graph so reverse rates are
    differentiable w.r.t. the NASA-9 coefficients.
    """
    c = jnp.asarray(coeffs, dtype=jnp.float64)
    a_low, a_high = c[:, 0, :], c[:, 1, :]
    Tg = jnp.asarray(T)[None, :]

    def _g(a):
        return (
            -a[:, 0:1] / Tg**2
            + a[:, 1:2] * jnp.log(Tg) / Tg
            + a[:, 2:3]
            + a[:, 3:4] * Tg / 2.0
            + a[:, 4:5] * Tg**2 / 3.0
            + a[:, 5:6] * Tg**3 / 4.0
            + a[:, 6:7] * Tg**4 / 5.0
            + a[:, 8:9] / Tg
        ) - (
            -a[:, 0:1] / Tg**2 / 2.0
            - a[:, 1:2] / Tg
            + a[:, 2:3] * jnp.log(Tg)
            + a[:, 3:4] * Tg
            + a[:, 4:5] * Tg**2 / 2.0
            + a[:, 5:6] * Tg**3 / 3.0
            + a[:, 6:7] * Tg**4 / 4.0
            + a[:, 9:10]
        )

    mask_low = (jnp.asarray(T) < 1000.0)[None, :]
    return jnp.where(mask_low, _g(a_low), _g(a_high))


def K_eq_array(net: Network, gibbs_sp: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
    """Equilibrium constants per forward reaction (nr+1, nz). Vectorized."""
    nz = gibbs_sp.shape[1]
    g_pad = jnp.concatenate([gibbs_sp, jnp.zeros((1, nz))], axis=0)  # (ni+1, nz)
    r_idx = jnp.asarray(np.asarray(net.reactant_idx))  # (nr+1, S)
    p_idx = jnp.asarray(np.asarray(net.product_idx))
    r_st = jnp.asarray(np.asarray(net.reactant_stoich, dtype=np.float64))
    p_st = jnp.asarray(np.asarray(net.product_stoich, dtype=np.float64))

    reac = jnp.einsum("rs,rsz->rz", r_st, g_pad[r_idx])  # (nr+1, nz)
    prod = jnp.einsum("rs,rsz->rz", p_st, g_pad[p_idx])
    delta_n = (r_st.sum(axis=1) - p_st.sum(axis=1))[:, None]  # (nr+1, 1)
    Tg = jnp.asarray(T)[None, :]
    # Clip the upper side of the exponent so exp() cannot overflow to +inf on
    # cold columns (would give a NaN forward-mode tangent in the reverse divide;
    # see _EXP_ARG_MAX). Primal is unchanged to float precision.
    K_raw = jnp.exp(jnp.minimum(reac - prod, _EXP_ARG_MAX)) * (CORR * Tg) ** delta_n

    # Valid forward slots get a real K; everything else stays 1 (sentinel).
    nr = net.nr
    idx = np.arange(nr + 1)
    valid = (
        net.is_forward
        & (idx + 1 < net.stop_rev_indx)
        & ~net.is_photo
        & ~net.is_ion
        & ~net.is_conden
        & ~net.is_radiative
    )
    valid_m = jnp.asarray(np.asarray(valid, dtype=bool))[:, None]
    return jnp.where(valid_m, K_raw, 1.0)


def fill_reverse_k(net: Network, k_fwd: jnp.ndarray, K_eq: jnp.ndarray) -> jnp.ndarray:
    """Fill even (reverse) slots from forward rates / K_eq. (nr+1, nz)."""
    nr = net.nr
    idx = np.arange(nr + 1)
    even_rev = (idx % 2 == 0) & (idx >= 2) & (idx < net.stop_rev_indx)
    partner = np.where(even_rev, idx - 1, 0)  # forward partner index
    even_rev_m = jnp.asarray(even_rev)[:, None]
    partner_j = jnp.asarray(partner)

    k_partner = k_fwd[partner_j]  # (nr+1, nz)
    K_partner = K_eq[partner_j]
    k_rev = jnp.where(
        K_partner > 0, k_partner / jnp.where(K_partner > 0, K_partner, 1.0), 0.0
    )
    return jnp.where(even_rev_m, k_rev, k_fwd)


def build_rate_array(
    net: Network,
    T: jnp.ndarray,
    M: jnp.ndarray,
    nasa9_coeffs: np.ndarray,
    remove_list=None,
    use_lowT_caps: bool = False,
    rate_coeffs: dict | None = None,
) -> jnp.ndarray:
    """Differentiable end-to-end: forward(T) -> (lowT caps) -> reverse via Gibbs(T) -> remove.

    Mirrors `rates.build_rate_array`. `T`, `M` are JAX arrays; `nasa9_coeffs` is
    the static (ni,2,10) table (already differentiable — it flows through the
    Gibbs reverse path); `remove_list` zeros those rows. Pass `use_lowT_caps=True`
    to match `cfg.use_lowT_limit_rates` on cool networks (off by default — the hot
    benchmarks never trigger the caps). Pass `rate_coeffs` (a dict of any of
    `a/n/E/a_inf/n_inf/E_inf` as (nr+1,) JAX arrays) to differentiate w.r.t.
    Arrhenius rate-coefficient uncertainty; defaults to the network's values.
    """
    k_fwd = compute_forward_k(net, T, M, **(rate_coeffs or {}))
    if use_lowT_caps:
        k_fwd = apply_lowT_caps(net, k_fwd, T, M)
    g_sp = gibbs_sp_vector(nasa9_coeffs, T)
    K_eq = K_eq_array(net, g_sp, T)
    k = fill_reverse_k(net, k_fwd, K_eq)
    if remove_list:
        nr = net.nr
        rm = np.zeros(nr + 1, dtype=bool)
        for i in remove_list:
            if 0 <= int(i) <= nr:
                rm[int(i)] = True
        k = jnp.where(jnp.asarray(rm)[:, None], 0.0, k)
    return k
