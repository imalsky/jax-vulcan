"""JAX-native atmosphere setup (Phase 20).

Replaces the NumPy/SciPy `Atm` class that lived in `build_atm.py`. Each
public function is a pure JAX kernel that takes cfg + arrays and returns
JAX arrays — no `data_atm` mutation. The thin `Atm` facade at the bottom
preserves the legacy `make_atm.f_pico(data_atm)` / `load_TPK(...)` / ...
call sites used by existing tests and the runner so this rewrite is
behavior-preserving.

Per the Phase-19 schema in `state.py`, the per-method outputs feed the
`AtmInputs` pytree at the boundary; the legacy `data_atm` container is
populated via the facade for callers that still rely on its attribute
surface (`atm_refresh`, `outer_loop._build_refresh_static`, the rate
parser, the `.vul` writer).

Numerical contract (vs VULCAN-master `build_atm.Atm`):
  - bit-exact for `f_pico`, `BC_flux`
  - ≤ 1e-13 for `load_TPK` (jnp.interp matches scipy.interp1d when xp
    is sorted ascending; we flip descending atm files before the call)
  - ≤ 1e-13 for `mol_diff`, `f_mu_dz` (pure arithmetic; the sequential
    upward/downward `f_mu_dz` loops use `jax.lax.scan`)
  - ≤ 1e-13 for `sp_sat` (per-species explicit formulae)
  - ≤ 1e-12 for `read_sflux` (linear interp + trapezoidal sum
    book-keeping is identical)
"""
from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np

import vulcan_cfg
from phy_const import Navo, au, kb, r_sun

# x64 is required for the rate-constant dynamic range; mirrors the same
# enable in every other JAX module (chem.py, jax_step.py, ...).
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# 1. Pressure interface grid (f_pico)
# ---------------------------------------------------------------------------

def compute_pico(pco: jnp.ndarray) -> jnp.ndarray:
    """Compute the pressure grid at cell interfaces from cell centres.

    Geometric mean between adjacent cells; outer boundaries by
    log-extrapolation of the nearest pair. Returns a `(nz+1,)` array.
    Equivalent to `Atm.f_pico` in `build_atm.py:314`.
    """
    pco = jnp.asarray(pco)
    pco_up1 = jnp.roll(pco, 1)
    interior = jnp.sqrt(pco * pco_up1)
    bottom = pco[0] ** 1.5 * pco[1] ** -0.5
    top = pco[-1] ** 1.5 * pco[-2] ** -0.5
    return jnp.concatenate([
        jnp.array([bottom]),
        interior[1:],
        jnp.array([top]),
    ])


# ---------------------------------------------------------------------------
# 2. Analytical T(P) profile (TP_H14)
# ---------------------------------------------------------------------------

def analytical_TP_H14(pco: jnp.ndarray, params, *, gs: float, Pb: float) -> jnp.ndarray:
    """Heng et al. 2014 analytical T(P) profile, JAX-native.

    Mirrors `Atm.TP_H14` in `build_atm.py:493`. Vectorised over `pco`;
    the original function vectorised by being called once per cell
    inside an `interp1d` lambda. `params` is the 6-tuple `(T_int,
    T_irr, ka_0, ka_s, beta_s, beta_l)`.
    """
    T_int, T_irr, ka_0, ka_s, beta_s, beta_l = params
    pco = jnp.asarray(pco)

    albedo = (1.0 - beta_s) / (1.0 + beta_s)
    T_irr_eff = T_irr * (1.0 - albedo) ** 0.25
    eps_L = 3.0 / 8.0
    eps_L3 = 1.0 / 3.0
    ka_CIA = 0.0
    m = pco / gs
    m_0 = Pb / gs
    ka_l = ka_0 + ka_CIA * m / m_0
    term1 = T_int**4 / 4.0 * (
        1.0 / eps_L + m / (eps_L3 * beta_l**2) * (ka_0 + ka_CIA * m / (2.0 * m_0))
    )
    arg = ka_s * m / beta_s
    expn2 = jsp_special.expn(2, arg)
    expn4 = jsp_special.expn(4, arg)
    term2 = (
        1.0 / (2.0 * eps_L)
        + expn2 * (
            ka_s / (ka_l * beta_s)
            - ka_CIA * m * beta_s / (eps_L3 * ka_s * m_0 * beta_l**2)
        )
    )
    term3 = ka_0 * beta_s / (eps_L3 * ka_s * beta_l**2) * (1.0 / 3.0 - expn4)
    term4 = 0.0
    T = (term1 + T_irr_eff**4 / 8.0 * (term2 + term3 + term4)) ** 0.25
    return T


# ---------------------------------------------------------------------------
# 3. Load TPK (T, Kzz, vz from cfg / file)  — host-side I/O
# ---------------------------------------------------------------------------

def _interp_descending_or_ascending(
    x_query: np.ndarray,
    xp_raw: np.ndarray,
    fp_raw: np.ndarray,
    fill_left: float,
    fill_right: float,
) -> np.ndarray:
    """`scipy.interp1d(..., bounds_error=False, fill_value=(left, right))`
    behaviour, in JAX.

    `interp1d` accepts unsorted `xp`; `jnp.interp` requires ascending
    `xp`. We sort once on the host, mapping `(left, right)` to the
    sorted-min / sorted-max edge values consistently.
    """
    xp_raw = np.asarray(xp_raw, dtype=np.float64)
    fp_raw = np.asarray(fp_raw, dtype=np.float64)
    order = np.argsort(xp_raw)
    xp_sorted = xp_raw[order]
    fp_sorted = fp_raw[order]
    return np.asarray(
        jnp.interp(
            jnp.asarray(x_query, dtype=jnp.float64),
            jnp.asarray(xp_sorted),
            jnp.asarray(fp_sorted),
            left=float(fill_left),
            right=float(fill_right),
        )
    )


def _read_atm_table(atm_file: str) -> dict[str, np.ndarray]:
    """Host-side read of an `atm/atm_*.txt` file.

    Header is one line of comments + one line of column names; the
    parser uses `skip_header=1, names=True`. Returns a dict with
    `Pressure` / `Temp` / (optional) `Kzz` / (optional) `vz`.
    """
    table = np.genfromtxt(atm_file, names=True, dtype=None, skip_header=1)
    out: dict[str, np.ndarray] = {
        "Pressure": np.asarray(table["Pressure"], dtype=np.float64),
        "Temp": np.asarray(table["Temp"], dtype=np.float64),
    }
    if "Kzz" in table.dtype.names:
        out["Kzz"] = np.asarray(table["Kzz"], dtype=np.float64)
    if "vz" in table.dtype.names:
        out["vz"] = np.asarray(table["vz"], dtype=np.float64)
    return out


def load_TPK(cfg, pco: np.ndarray, *, pico: np.ndarray) -> dict[str, jnp.ndarray]:
    """Build (Tco, Kzz, vz, M, n_0) for the configured atm/Kzz/vz modes.

    Mirrors the dispatch in `Atm.load_TPK` (build_atm.py:332). Returns a
    dict of JAX arrays. `pco` and `pico` are inputs because some modes
    (`vulcan_ini`, `table`) overwrite `pco` from a saved file.
    """
    nz = int(pco.shape[0])
    atm_type = cfg.atm_type
    Kzz_prof = cfg.Kzz_prof
    vz_prof = cfg.vz_prof
    use_Kzz = bool(cfg.use_Kzz)
    use_vz = bool(cfg.use_vz)

    out: dict[str, np.ndarray] = {}
    table_for_vz: dict[str, np.ndarray] | None = None  # holds the file table for vz_prof='file'

    # -------- Tco (and, for atm_type='file', the table for Kzz/vz reads) --------
    if atm_type == "isothermal":
        Tco = np.full(nz, float(cfg.Tiso), dtype=np.float64)
    elif atm_type == "analytical":
        Tco = np.asarray(
            analytical_TP_H14(
                jnp.asarray(pco),
                cfg.para_anaTP,
                gs=float(cfg.gs),
                Pb=float(cfg.P_b),
            )
        )
    elif atm_type == "file":
        table = _read_atm_table(cfg.atm_file)
        table_for_vz = table
        p_file = table["Pressure"]
        T_file = table["Temp"]
        if max(p_file) < pco[0] or min(p_file) > pco[-1]:
            print(
                "Warning: P_b and P_t assgined in vulcan.cfg are out of range "
                "of the input.\nConstant extension is used."
            )
        Tco = _interp_descending_or_ascending(
            pco, p_file, T_file,
            fill_left=float(T_file[np.argmin(p_file)]),
            fill_right=float(T_file[np.argmax(p_file)]),
        )
        if use_Kzz and Kzz_prof == "file":
            Kzz_file = table["Kzz"]
            out["Kzz"] = _interp_descending_or_ascending(
                pico[1:-1], p_file, Kzz_file,
                fill_left=float(Kzz_file[np.argmin(p_file)]),
                fill_right=float(Kzz_file[np.argmax(p_file)]),
            )
    elif atm_type == "vulcan_ini":
        import pickle
        print(f"Initializing PT from the prvious run {cfg.vul_ini}")
        with open(cfg.vul_ini, "rb") as handle:
            vul_data = pickle.load(handle)
        Tco = np.asarray(vul_data["atm"]["Tco"], dtype=np.float64)
    elif atm_type == "table":
        print(f"Initializing PT from the prvious run {cfg.vul_ini}")
        table = np.genfromtxt(cfg.vul_ini, names=True, dtype=None, skip_header=1)
        if not nz == len(table["Pressure"]):
            raise IOError(
                "Initial profile has different layers than the current setting"
            )
        out["pco"] = np.asarray(table["Pressure"], dtype=np.float64)
        Tco = np.asarray(table["Temp"], dtype=np.float64)
    else:
        raise IOError(
            f'\n"atm_type"={atm_type!r} cannot be recongized.\n'
            f"Please reassign it in vulcan_cfg."
        )
    out["Tco"] = Tco

    # -------- Kzz (independent of atm_type for non-'file' Kzz_prof) --------
    if Kzz_prof == "const":
        out["Kzz"] = np.full(nz - 1, float(cfg.const_Kzz), dtype=np.float64)
    elif Kzz_prof == "JM16":
        Kzz = 1e5 * (300.0 / (pico[1:-1] * 1e-3)) ** 0.5
        out["Kzz"] = np.maximum(float(cfg.K_deep), Kzz)
    elif Kzz_prof == "Pfunc":
        Kzz = float(cfg.K_max) * (float(cfg.K_p_lev) * 1e6 / pico[1:-1]) ** 0.4
        out["Kzz"] = np.maximum(float(cfg.K_max), Kzz)
    elif Kzz_prof == "file":
        if "Kzz" not in out:
            raise IOError(
                'Kzz_prof="file" requires atm_type="file" with a Kzz column.'
            )
    else:
        raise IOError(
            f'\n"Kzz_prof"={Kzz_prof!r} cannot be recongized.\n'
            f'Assign it as "file", "const", "JM16" or "Pfunc" in vulcan_cfg.'
        )

    # -------- vz --------
    if vz_prof == "const":
        out["vz"] = np.full(nz - 1, float(cfg.const_vz), dtype=np.float64)
    elif vz_prof == "file":
        if table_for_vz is None or "vz" not in table_for_vz:
            raise IOError(
                'vz_prof="file" requires atm_type="file" with a vz column.'
            )
        p_file = table_for_vz["Pressure"]
        vz_file = table_for_vz["vz"]
        out["vz"] = _interp_descending_or_ascending(
            pico[1:-1], p_file, vz_file,
            fill_left=0.0, fill_right=0.0,
        )
    else:
        raise IOError(
            f'\n"vz_prof"={vz_prof!r} cannot be recongized.\n'
            f'Assign it as "file" or "const" in vulcan_cfg.'
        )

    # Force-zero when the corresponding `use_*` switch is off (master overrides
    # the cfg-derived array unconditionally; preserve that).
    if not use_Kzz:
        out["Kzz"] = np.zeros(nz - 1, dtype=np.float64)
    if not use_vz:
        out["vz"] = np.zeros(nz - 1, dtype=np.float64)

    # M and n_0 follow directly from (pco, Tco). Done in JAX so downstream
    # call sites see jnp arrays.
    pco_for_M = out.get("pco", pco)
    M = jnp.asarray(pco_for_M, dtype=jnp.float64) / (
        kb * jnp.asarray(out["Tco"], dtype=jnp.float64)
    )
    out["M"] = np.asarray(M)
    out["n_0"] = np.asarray(M)

    # Promote everything to plain numpy for the legacy facade; the runner
    # converts to JAX at the pytree boundary.
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 4. mean_mass + f_mu_dz
# ---------------------------------------------------------------------------

def compute_mean_mass(ymix: jnp.ndarray, ms_arr: jnp.ndarray) -> jnp.ndarray:
    """Mean molecular weight (g/mol) per layer.

    `ymix` is `(nz, ni)`, `ms_arr` is `(ni,)`. Vectorised replacement
    for the per-species accumulator in `Atm.mean_mass`.
    """
    return jnp.einsum("zi,i->z", jnp.asarray(ymix), jnp.asarray(ms_arr))


def _scan_up_mu_dz_g(
    pref_indx: int, gs: float, Rp: float,
    Tco: jnp.ndarray, mu: jnp.ndarray, pico: jnp.ndarray,
    nz: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sequential upward fill from `pref_indx` (1 bar / surface) to top."""
    Tco_up = Tco[pref_indx:nz]
    mu_up = mu[pref_indx:nz]
    pico_up_lo = pico[pref_indx:nz]
    pico_up_hi = pico[pref_indx + 1:nz + 1]

    def body(carry, scan_in):
        z_prev, _gz_prev, _Hp_prev = carry
        i, T_i, mu_i, p_lo, p_hi = scan_in
        is_first = i == 0
        gz_i = jnp.where(is_first, gs, gs * (Rp / (Rp + z_prev)) ** 2)
        Hp_i = kb * T_i / (mu_i / Navo * gz_i)
        dz_i = Hp_i * jnp.log(p_lo / p_hi)
        z_next = z_prev + dz_i
        return (z_next, gz_i, Hp_i), (gz_i, Hp_i, dz_i, z_next)

    init = (jnp.float64(0.0), jnp.float64(gs), jnp.float64(0.0))
    n_up = nz - pref_indx
    idx = jnp.arange(n_up)
    _, (gz_seq, Hp_seq, dz_seq, z_after_seq) = jax.lax.scan(
        body, init, (idx, Tco_up, mu_up, pico_up_lo, pico_up_hi)
    )
    return gz_seq, Hp_seq, dz_seq, z_after_seq


def _scan_down_mu_dz_g(
    pref_indx: int, gs: float, Rp: float,
    Tco: jnp.ndarray, mu: jnp.ndarray, pico: jnp.ndarray,
    z_at_pref: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sequential downward fill from `pref_indx-1` to 0, given `zco[pref_indx]`.

    Returns arrays in the natural (low→high index) order; pref_indx must
    be ≥ 1 to call this.
    """
    Tco_dn_rev = Tco[:pref_indx][::-1]
    mu_dn_rev = mu[:pref_indx][::-1]
    pico_lo_rev = pico[:pref_indx][::-1]
    pico_hi_rev = pico[1:pref_indx + 1][::-1]

    def body(carry, scan_in):
        z_above = carry
        T_i, mu_i, p_lo, p_hi = scan_in
        gz_i = gs * (Rp / (Rp + z_above)) ** 2
        Hp_i = kb * T_i / (mu_i / Navo * gz_i)
        dz_i = Hp_i * jnp.log(p_lo / p_hi)
        z_here = z_above - dz_i
        return z_here, (gz_i, Hp_i, dz_i, z_here)

    init = jnp.float64(z_at_pref)
    _, (gz_rev, Hp_rev, dz_rev, z_rev) = jax.lax.scan(
        body, init, (Tco_dn_rev, mu_dn_rev, pico_lo_rev, pico_hi_rev)
    )
    return gz_rev[::-1], Hp_rev[::-1], dz_rev[::-1], z_rev[::-1]


def compute_mu_dz_g(
    cfg, ymix: np.ndarray, ms_arr: np.ndarray,
    pico: np.ndarray, Tco: np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Build (mu, g, gs, Hp, dz, dzi, zco, zmco, pref_indx, [Ti, Hpi]).

    Mirrors `Atm.f_mu_dz` (build_atm.py:526). The two sequential
    upward/downward height-integration loops are JAX `lax.scan`s for
    autodiff-friendliness; the rest is direct array math.
    """
    nz = int(Tco.shape[0])
    pico_j = jnp.asarray(pico, dtype=jnp.float64)
    Tco_j = jnp.asarray(Tco, dtype=jnp.float64)
    mu_j = compute_mean_mass(jnp.asarray(ymix, dtype=jnp.float64),
                             jnp.asarray(ms_arr, dtype=jnp.float64))

    gs = float(cfg.gs)
    Rp = float(cfg.Rp)
    rocky = bool(cfg.rocky)
    Pb = float(cfg.P_b)

    # `pref_indx` is the layer where g(z)=gs anchors. For gas giants
    # (rocky=False) with P_b ≥ 1 bar the anchor sits at the layer
    # closest to log10(pico)=6 (1 bar in dyne/cm^2). Done on the host
    # since it's a pure scalar lookup over a static array.
    pico_host = np.asarray(pico_j)
    if (not rocky) and Pb >= 1e6:
        pref_indx = int(min(range(nz + 1), key=lambda i: abs(np.log10(pico_host[i]) - 6.0)))
    else:
        pref_indx = 0

    # Upward integration (pref_indx → top).
    gz_up, Hp_up, dz_up, z_after_up = _scan_up_mu_dz_g(
        pref_indx, gs, Rp, Tco_j, mu_j, pico_j, nz,
    )
    gz = jnp.zeros(nz, dtype=jnp.float64).at[pref_indx:nz].set(gz_up)
    Hp = jnp.zeros(nz, dtype=jnp.float64).at[pref_indx:nz].set(Hp_up)
    dz = jnp.zeros(nz, dtype=jnp.float64).at[pref_indx:nz].set(dz_up)
    # zco is (nz+1,) and zco[pref_indx] = 0 by definition.
    zco = jnp.zeros(nz + 1, dtype=jnp.float64).at[pref_indx + 1:nz + 1].set(z_after_up)

    if pref_indx > 0:
        gz_dn, Hp_dn, dz_dn, z_dn = _scan_down_mu_dz_g(
            pref_indx, gs, Rp, Tco_j, mu_j, pico_j, z_at_pref=0.0,
        )
        gz = gz.at[:pref_indx].set(gz_dn)
        Hp = Hp.at[:pref_indx].set(Hp_dn)
        dz = dz.at[:pref_indx].set(dz_dn)
        zco = zco.at[:pref_indx].set(z_dn)

    zmco = 0.5 * (zco[:-1] + jnp.roll(zco, -1)[:-1])
    dzi = 0.5 * (dz[1:] + jnp.roll(dz, 1)[1:])

    out: dict[str, np.ndarray | int] = {
        "mu": np.asarray(mu_j),
        "g": np.asarray(gz),
        "gs": gs,
        "Hp": np.asarray(Hp),
        "dz": np.asarray(dz),
        "dzi": np.asarray(dzi),
        "zco": np.asarray(zco),
        "zmco": np.asarray(zmco),
        "pref_indx": pref_indx,
    }
    if bool(cfg.use_moldiff):
        Ti = 0.5 * (Tco_j[:-1] + jnp.roll(Tco_j, -1)[:-1])
        Hpi = 0.5 * (Hp[:-1] + jnp.roll(Hp, -1)[:-1])
        out["Ti"] = np.asarray(Ti)
        out["Hpi"] = np.asarray(Hpi)
    return out


# ---------------------------------------------------------------------------
# 5. Settling velocity (use_settling branch of f_mu_dz)
# ---------------------------------------------------------------------------

# (na, a, b) per atm_base — Cloutman dynamic-viscosity polynomial.
_VISCOSITY_TABLE: Mapping[str, tuple[float, float, float]] = {
    "N2":  (1.52, 1.186e-5, 86.54),
    "H2":  (1.67, 1.936e-6,  2.187),
    "CO2": (1.52, 1.186e-5, 86.54),  # CO2 not tabulated; use N2 fallback
    "H2O": (1.5,  1.6e-5,    0.0),
    "O2":  (1.46, 2.294e-5, 164.4),
}


def compute_settling_velocity(
    cfg, Tco: np.ndarray, g: np.ndarray, species_list: list[str],
    rho_p: Mapping[str, float], r_p: Mapping[str, float],
) -> np.ndarray:
    """Per-species terminal-fall velocity (Stokes regime).

    Returns `(nz-1, ni)`; only `non_gas_sp` entries are non-zero. Mirrors
    the `if self.use_settling:` block at `build_atm.py:585`.
    """
    nz = int(Tco.shape[0])
    ni = len(species_list)
    vs = np.zeros((nz - 1, ni), dtype=np.float64)
    if not bool(cfg.use_settling):
        return vs
    atm_base = cfg.atm_base
    if atm_base == "CO2":
        print("NO CO2 viscosity yet! (using N2 instead)")
    if atm_base not in _VISCOSITY_TABLE:
        raise IOError(f"No viscosity polynomial for atm_base={atm_base!r}")
    na, a, b = _VISCOSITY_TABLE[atm_base]
    Tco_np = np.asarray(Tco, dtype=np.float64)
    g_np = np.asarray(g, dtype=np.float64)
    dmu = a * Tco_np**na / (b + Tco_np)  # (nz,) g cm^-1 s^-1
    gi = 0.5 * (g_np[:-1] + g_np[1:])

    for sp in cfg.non_gas_sp:
        if sp not in species_list:
            continue
        if sp not in rho_p or sp not in r_p:
            raise IOError(f"{sp} has not been prescribed size and density!")
        idx = species_list.index(sp)
        vs[:, idx] = -1.0 * (
            2.0 / 9.0 * float(rho_p[sp]) * float(r_p[sp]) ** 2 * gi / dmu[1:]
        )
    return vs


# ---------------------------------------------------------------------------
# 6. mol_diff (Dzz, alpha, ms, vm)
# ---------------------------------------------------------------------------

def _Dzz_gen_for_base(atm_base: str):
    """Return a JAX-compatible lambda `(T, n_tot, mi) -> Dzz` for an atm_base."""
    if atm_base == "H2":
        # Moses 2000a: 2.2965e17 * T**0.765 / n_tot * (16.04/mi*(mi+2.016)/18.059)**0.5
        return lambda T, n_tot, mi: (
            2.2965e17 * T**0.765 / n_tot
            * (16.04 / mi * (mi + 2.016) / 18.059) ** 0.5
        )
    if atm_base == "N2":
        return lambda T, n_tot, mi: (
            7.34e16 * T**0.75 / n_tot
            * (16.04 / mi * (mi + 28.014) / 44.054) ** 0.5
        )
    if atm_base == "O2":
        return lambda T, n_tot, mi: (
            7.51e16 * T**0.759 / n_tot
            * (16.04 / mi * (mi + 32.0) / 48.04) ** 0.5
        )
    if atm_base == "CO2":
        return lambda T, n_tot, mi: (
            2.15e17 * T**0.750 / n_tot
            * (2.016 / mi * (mi + 44.001) / 46.017) ** 0.5
        )
    raise IOError(f"\n Unknow atm_base={atm_base!r}!")


def _alpha_array_for_base(atm_base: str, species_list: list[str], mol_mass) -> np.ndarray:
    """Thermal-diffusion factor per species — atm_base-dependent overrides
    on top of the implicit-zero default. For H2 base, `alpha[heavy]=0.25`
    is applied to every species heavier than 4 amu.
    """
    ni = len(species_list)
    alpha = np.zeros(ni, dtype=np.float64)
    if atm_base == "H2":
        if "H" in species_list:
            alpha[species_list.index("H")] = -0.1
        if "He" in species_list:
            alpha[species_list.index("He")] = 0.145
        for sp in species_list:
            if mol_mass(sp) > 4.0:
                alpha[species_list.index(sp)] = 0.25
    elif atm_base in ("N2", "O2", "CO2"):
        for sp in ("H", "H2", "He"):
            if sp in species_list:
                alpha[species_list.index(sp)] = -0.25
        if "Ar" in species_list:
            alpha[species_list.index("Ar")] = 0.17
    else:
        raise IOError(f"\n Unknow atm_base={atm_base!r}!")
    return alpha


def compute_mol_diff(
    cfg, Tco: np.ndarray, n_0: np.ndarray, g: np.ndarray, Hp: np.ndarray,
    dz: np.ndarray, ms_arr: np.ndarray, alpha_arr: np.ndarray,
    species_list: list[str],
) -> dict[str, np.ndarray]:
    """Build (Dzz, Dzz_cen, vm) for the cfg's `atm_base`.

    Mirrors `Atm.mol_diff` (build_atm.py:664). Returns vm as the
    advective component of molecular diffusion (zero unless
    `use_vm_mol=True`). When `use_moldiff=False`, Dzz/Dzz_cen/vm are
    all zero — only `ms` and `alpha` (already passed in) populate.
    """
    nz = int(Tco.shape[0])
    ni = len(species_list)
    Tco_j = jnp.asarray(Tco, dtype=jnp.float64)
    n0_j = jnp.asarray(n_0, dtype=jnp.float64)
    ms_j = jnp.asarray(ms_arr, dtype=jnp.float64)

    out: dict[str, np.ndarray] = {
        "Dzz": np.zeros((nz - 1, ni), dtype=np.float64),
        "Dzz_cen": np.zeros((nz, ni), dtype=np.float64),
        "vm": np.zeros((nz, ni), dtype=np.float64),
    }
    if not bool(cfg.use_moldiff):
        return out

    atm_base = cfg.atm_base
    Dzz_gen = _Dzz_gen_for_base(atm_base)

    # Use values defined on the interface for the (nz-1, ni) Dzz array
    # — matches `op.diffdf` consumption pattern exactly.
    Tco_i = (Tco_j[1:] + Tco_j[:-1]) * 0.5
    n0_i = (n0_j[1:] + n0_j[:-1]) * 0.5

    # vmap over species: Dzz[:, i] = Dzz_gen(T, n, ms[i])
    Dzz = jax.vmap(lambda mi: Dzz_gen(Tco_i, n0_i, mi), out_axes=1)(ms_j)
    Dzz_cen = jax.vmap(lambda mi: Dzz_gen(Tco_j, n0_j, mi), out_axes=1)(ms_j)

    # Zero out non-gaseous species (H2O_l_s etc.) on the interface array.
    for sp in cfg.non_gas_sp:
        if sp in species_list:
            Dzz = Dzz.at[:, species_list.index(sp)].set(0.0)

    out["Dzz"] = np.asarray(Dzz)
    out["Dzz_cen"] = np.asarray(Dzz_cen)

    if bool(getattr(cfg, "use_vm_mol", False)):
        Hp_j = jnp.asarray(Hp, dtype=jnp.float64)
        g_j = jnp.asarray(g, dtype=jnp.float64)
        dz_j = jnp.asarray(dz, dtype=jnp.float64)
        alpha_j = jnp.asarray(alpha_arr, dtype=jnp.float64)
        # delta_T[i] = T[i+1] - T[i]; the master code aliases delta_T[0] to
        # delta_T[1] (extrapolation) and the `np.insert` line is a no-op
        # (returned array is discarded). Reproduce both behaviours.
        delta_T = jnp.roll(Tco_j, -1) - Tco_j
        delta_T = delta_T.at[0].set(delta_T[1])
        gravity_term = ms_j[None, :] * g_j[:, None] / (Navo * kb * Tco_j[:, None])
        scale_term = 1.0 / Hp_j[:, None]
        thermal_term = alpha_j[None, :] / Tco_j[:, None] * delta_T[:, None] / dz_j[:, None]
        vm = -Dzz_cen * (gravity_term - scale_term + thermal_term)
        if bool(cfg.use_condense):
            for sp in cfg.non_gas_sp:
                if sp in species_list:
                    vm = vm.at[:, species_list.index(sp)].set(0.0)
        out["vm"] = np.asarray(vm)
    return out


# ---------------------------------------------------------------------------
# 7. read_sflux — interpolate stellar flux onto uniform photo bins
# ---------------------------------------------------------------------------

def read_sflux_binned(
    cfg, bins: np.ndarray, sflux_raw: np.ndarray | None = None,
) -> dict[str, Any]:
    """Interpolate stellar flux onto VULCAN's uniform `var.bins` grid.

    Mirrors `Atm.read_sflux` (build_atm.py:621). Writes the bin-domain
    flux `sflux_top`, the dbin1→dbin2 transition index
    `sflux_din12_indx`, and the raw read for downstream diagnostics.

    `sflux_raw` may be passed in (test path) — otherwise we read it
    from `cfg.sflux_file` here.
    """
    if sflux_raw is None:
        sflux_raw = np.genfromtxt(
            cfg.sflux_file, dtype=float, skip_header=1,
            names=["lambda", "flux"],
        )
    bins_np = np.asarray(bins, dtype=np.float64)
    dbin1 = float(cfg.dbin1)
    dbin2 = float(cfg.dbin2)
    dbin_12 = float(cfg.dbin_12trans)
    raw_lambda = np.asarray(sflux_raw["lambda"], dtype=np.float64)
    # Stellar flux at the planet: scale by (R_star / r_orbit)^2.
    geom = (float(cfg.r_star) * r_sun / (au * float(cfg.orbit_radius))) ** 2
    raw_flux = np.asarray(sflux_raw["flux"], dtype=np.float64) * geom

    # Linear interp; outside [raw_min, raw_max] -> 0.
    sflux_top = np.asarray(jnp.interp(
        jnp.asarray(bins_np), jnp.asarray(raw_lambda), jnp.asarray(raw_flux),
        left=0.0, right=0.0,
    ), dtype=np.float64)

    # The dbin1->dbin2 transition index in the bins grid.
    sflux_din12_indx = -1
    for n, ld in enumerate(bins_np):
        if ld == dbin_12:
            sflux_din12_indx = n
            break

    # Energy conservation diagnostic — raw integral vs. bin integral.
    raw_left_indx = int(np.searchsorted(raw_lambda, bins_np[0], side="right"))
    raw_right_indx = int(np.searchsorted(raw_lambda, bins_np[-1], side="right")) - 1
    sum_orgin = 0.0
    for n in range(raw_left_indx, raw_right_indx):
        sum_orgin += 0.5 * (raw_flux[n] + raw_flux[n + 1]) * (
            raw_lambda[n + 1] - raw_lambda[n]
        )
    sum_orgin += 0.5 * (
        float(np.interp(bins_np[0], raw_lambda, raw_flux)) + raw_flux[raw_left_indx]
    ) * (raw_lambda[raw_left_indx] - bins_np[0])
    sum_orgin += 0.5 * (
        float(np.interp(bins_np[-1], raw_lambda, raw_flux)) + raw_flux[raw_right_indx]
    ) * (bins_np[-1] - raw_lambda[raw_right_indx])

    sum_bin = dbin1 * np.sum(sflux_top[:sflux_din12_indx])
    sum_bin -= dbin1 * 0.5 * (
        sflux_top[0] + sflux_top[sflux_din12_indx - 1]
    )
    sum_bin += dbin2 * np.sum(sflux_top[sflux_din12_indx:])
    sum_bin -= dbin2 * 0.5 * (sflux_top[sflux_din12_indx] + sflux_top[-1])

    print(
        "The stellar flux is interpolated onto uniform grid of "
        f"{dbin1} (<{dbin_12} nm) and {dbin2} (>={dbin_12} nm)"
        f" and conserving {100 * sum_bin / sum_orgin:.2f} % energy."
    )
    return {
        "sflux_top": sflux_top,
        "sflux_din12_indx": sflux_din12_indx,
        "sflux_raw": sflux_raw,
    }


# ---------------------------------------------------------------------------
# 8. BC flux reader
# ---------------------------------------------------------------------------

def _parse_bc_file(path: str) -> list[list[str]]:
    """Read a BC text file into rows of token strings; skip comments / blanks."""
    rows: list[list[str]] = []
    with open(path) as f:
        for line in f.readlines():
            if not line.startswith("#") and line.strip():
                rows.append(line.split())
    return rows


def read_bc_flux(cfg, species_list: list[str]) -> dict[str, np.ndarray]:
    """Build (top_flux, bot_flux, bot_vdep, bot_fix_sp).

    Mirrors `Atm.BC_flux` (build_atm.py:747). Three optional flags drive
    populate-or-skip — defaults are zero arrays of length `ni`.
    """
    ni = len(species_list)
    out = {
        "top_flux": np.zeros(ni, dtype=np.float64),
        "bot_flux": np.zeros(ni, dtype=np.float64),
        "bot_vdep": np.zeros(ni, dtype=np.float64),
        "bot_fix_sp": np.zeros(ni, dtype=np.float64),
    }
    if bool(cfg.use_topflux):
        print("Using the prescribed constant top flux.")
        for tokens in _parse_bc_file(cfg.top_BC_flux_file):
            sp = tokens[0]
            if sp in species_list:
                out["top_flux"][species_list.index(sp)] = float(tokens[1])
    if bool(cfg.use_botflux):
        print("Using the prescribed constant bottom flux.")
        for tokens in _parse_bc_file(cfg.bot_BC_flux_file):
            sp = tokens[0]
            if sp in species_list:
                out["bot_flux"][species_list.index(sp)] = float(tokens[1])
                out["bot_vdep"][species_list.index(sp)] = float(tokens[2])
    # Master treats `use_fix_sp_bot` as a truthy-dict in `Atm.BC_flux`. The
    # only call that hits `==True` is when the cfg sets it to the literal
    # `True` (not a dict) — we preserve that path verbatim. Production
    # callers feed a dict and the OuterLoop pin handles those entries.
    if getattr(cfg, "use_fix_sp_bot", False) is True:
        print("Using the prescribed fixed bottom mixing ratios.")
        for tokens in _parse_bc_file(cfg.bot_BC_flux_file):
            sp = tokens[0]
            if sp in species_list and len(tokens) >= 4:
                out["bot_fix_sp"][species_list.index(sp)] = float(tokens[3])
    return out


# ---------------------------------------------------------------------------
# 9. Saturation pressure (sp_sat — per-condensable explicit formulae)
# ---------------------------------------------------------------------------

_SUPPORTED_CONDENSABLES: tuple[str, ...] = (
    "H2O", "NH3", "H2SO4", "S2", "S4", "S8", "C", "H2S",
)


def compute_sat_p(condense_sp: list[str], Tco: np.ndarray) -> dict[str, np.ndarray]:
    """Saturation vapour pressure (dyne/cm^2) per condensable species.

    Mirrors `Atm.sp_sat` (build_atm.py:780). Each species has a hand-
    coded explicit formula; unsupported species raise.
    """
    Tco_np = np.asarray(Tco, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for sp in condense_sp:
        if sp not in _SUPPORTED_CONDENSABLES:
            raise IOError(
                f"No saturation vapor data for {sp}. "
                f"Check `compute_sat_p` in atm_setup.py"
            )
        T = Tco_np.copy()
        if sp == "H2O":
            T_C = T - 273.0
            c0, c1, c2, c3 = 6111.5, 23.036, -333.7, 279.82  # ice
            w0, w1, w2, w3 = 6112.1, 18.729, -227.3, 257.87  # water
            sat_p = (T_C < 0) * (c0 * np.exp((c1 * T_C + T_C**2 / c2) / (T_C + c3)))
            sat_p += (T_C > 0) * (w0 * np.exp((w1 * T_C + T_C**2 / w2) / (T_C + w3)))
            out[sp] = sat_p
        elif sp == "NH3":
            c0, c1, c2 = 10.53, -2161.0, -86596.0
            out[sp] = np.exp(c0 + c1 / T + c2 / T**2) * 1e6
        elif sp == "H2SO4":
            p_atm = np.exp(-10156.0 / T + 16.259)
            out[sp] = p_atm * 1.01325 * 1e6
        elif sp == "S2":
            sat_p = np.zeros_like(T)
            mask_lo = T < 413
            sat_p[mask_lo] = np.exp(27.0 - 18500.0 / T[mask_lo]) * 1e6
            sat_p[~mask_lo] = np.exp(16.1 - 14000.0 / T[~mask_lo]) * 1e6
            out[sp] = sat_p
        elif sp == "S4":
            out[sp] = 10 ** (6.0028 - 6047.5 / T) * 1.01325e6
        elif sp == "S8":
            sat_p = np.zeros_like(T)
            mask_lo = T < 413
            sat_p[mask_lo] = np.exp(20.0 - 11800.0 / T[mask_lo]) * 1e6
            sat_p[~mask_lo] = np.exp(9.6 - 7510.0 / T[~mask_lo]) * 1e6
            out[sp] = sat_p
        elif sp == "C":
            a, b, c = 3.27860e1, -8.65139e4, 4.80395e-1
            out[sp] = np.exp(a + b / (T + c))
        elif sp == "H2S":
            mask_ice = T <= 187.6
            ice_log10 = -1329.0 / T + 9.28588 - 0.0051263 * T
            l_log10 = -1145.0 / T + 7.94746 - 0.00322 * T
            sat_p = 10 ** (mask_ice * ice_log10 + (~mask_ice) * l_log10)
            out[sp] = sat_p * 0.001333 * 1e6
    return out


# ---------------------------------------------------------------------------
# 10. Compatibility facade (`Atm` class)
# ---------------------------------------------------------------------------

class Atm:
    """Thin facade that routes legacy `make_atm.X(data_atm)` calls into
    the JAX-native pure functions above and writes the results back into
    the legacy `data_atm` / `data_var` containers.

    The facade is the path that pre-existing tests (and `vulcan_jax.py`)
    use today. Phases 21+ and the runner can call the JAX functions
    directly to skip the legacy container entirely.
    """

    def __init__(self):
        self.gs = vulcan_cfg.gs
        self.P_b = vulcan_cfg.P_b
        self.P_t = vulcan_cfg.P_t
        self.type = vulcan_cfg.atm_type
        self.use_Kzz = vulcan_cfg.use_Kzz
        self.Kzz_prof = vulcan_cfg.Kzz_prof
        self.const_Kzz = vulcan_cfg.const_Kzz
        self.use_vz = vulcan_cfg.use_vz
        self.vz_prof = vulcan_cfg.vz_prof
        self.const_vz = vulcan_cfg.const_vz
        self.use_settling = vulcan_cfg.use_settling
        self.non_gas_sp = vulcan_cfg.non_gas_sp

    # ---- pre-loop callable surface (matches build_atm.Atm) ----
    def f_pico(self, data_atm):
        data_atm.pico = np.asarray(compute_pico(jnp.asarray(data_atm.pco)))
        return data_atm

    def load_TPK(self, data_atm):
        out = load_TPK(vulcan_cfg, np.asarray(data_atm.pco), pico=np.asarray(data_atm.pico))
        # `table` mode rewrites pco; honour that.
        if "pco" in out:
            data_atm.pco = out["pco"]
        for k in ("Tco", "Kzz", "vz", "M", "n_0"):
            setattr(data_atm, k, out[k])
        return data_atm

    def TP_H14(self, pco, *args_analytical):
        return np.asarray(
            analytical_TP_H14(
                jnp.asarray(pco), args_analytical,
                gs=float(vulcan_cfg.gs), Pb=float(vulcan_cfg.P_b),
            )
        )

    def mol_mass(self, sp):
        from composition import compo, compo_row
        return compo["mass"][compo_row.index(sp)]

    def mean_mass(self, var, atm, ni):
        from composition import species
        ms_arr = np.array(
            [self.mol_mass(species[i]) for i in range(ni)],
            dtype=np.float64,
        )
        atm.mu = np.asarray(compute_mean_mass(
            jnp.asarray(var.ymix), jnp.asarray(ms_arr),
        ))
        return atm

    def f_mu_dz(self, data_var, data_atm, output):
        from composition import species
        ni = len(species)
        ms_arr = np.array(
            [self.mol_mass(species[i]) for i in range(ni)],
            dtype=np.float64,
        )
        out = compute_mu_dz_g(
            vulcan_cfg, np.asarray(data_var.ymix), ms_arr,
            np.asarray(data_atm.pico), np.asarray(data_atm.Tco),
        )
        data_atm.mu = out["mu"]
        data_atm.g = out["g"]
        data_atm.gs = out["gs"]
        data_atm.Hp = out["Hp"]
        data_atm.dz = out["dz"]
        data_atm.dzi = out["dzi"]
        data_atm.zco = out["zco"]
        data_atm.zmco = out["zmco"]
        data_atm.pref_indx = out["pref_indx"]
        if "Ti" in out:
            data_atm.Ti = out["Ti"]
            data_atm.Hpi = out["Hpi"]
        if vulcan_cfg.use_settling:
            data_atm.vs = compute_settling_velocity(
                vulcan_cfg, data_atm.Tco, data_atm.g, list(species),
                rho_p=getattr(data_atm, "rho_p", {}),
                r_p=getattr(data_atm, "r_p", {}),
            )
        if vulcan_cfg.plot_TP:
            output.plot_TP(data_atm)
        if np.any(np.logical_or(data_atm.Tco < 200, data_atm.Tco > 6000)):
            print("Temperatures exceed the valid range of Gibbs free energy.\n")
        return data_atm

    def mol_diff(self, atm):
        from composition import compo, compo_row, species
        ni = len(species)
        ms_arr = np.array(
            [compo[compo_row.index(species[i])][-1] for i in range(ni)],
            dtype=np.float64,
        )
        atm.ms = ms_arr
        alpha_arr = _alpha_array_for_base(
            vulcan_cfg.atm_base, list(species), self.mol_mass,
        )
        atm.alpha = alpha_arr
        out = compute_mol_diff(
            vulcan_cfg, np.asarray(atm.Tco), np.asarray(atm.n_0),
            np.asarray(atm.g), np.asarray(atm.Hp), np.asarray(atm.dz),
            ms_arr, alpha_arr, list(species),
        )
        atm.Dzz = out["Dzz"]
        atm.Dzz_cen = out["Dzz_cen"]
        atm.vm = out["vm"]

    def BC_flux(self, atm):
        from composition import species
        out = read_bc_flux(vulcan_cfg, list(species))
        atm.top_flux = out["top_flux"]
        atm.bot_flux = out["bot_flux"]
        atm.bot_vdep = out["bot_vdep"]
        atm.bot_fix_sp = out["bot_fix_sp"]

    def sp_sat(self, atm):
        out = compute_sat_p(list(vulcan_cfg.condense_sp), np.asarray(atm.Tco))
        for sp, sp_arr in out.items():
            atm.sat_p[sp] = sp_arr

    def read_sflux(self, var, atm):
        out = read_sflux_binned(vulcan_cfg, np.asarray(var.bins))
        atm.sflux_raw = out["sflux_raw"]
        var.sflux_top = out["sflux_top"]
        var.sflux_din12_indx = out["sflux_din12_indx"]
