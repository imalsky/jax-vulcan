"""JAX-native photo cross-section preprocessing (Phase 22b → 22e).

Replaces `legacy_io.ReadRate.make_bins_read_cross`. Builds the wavelength
bin grid + interpolates per-species absorption / dissociation /
ionization / Rayleigh cross sections + branch ratios onto that grid.

Phase 22e: the canonical entry point is `populate_photo(var, atm) ->
PhotoStaticInputs`. The dict-keyed view (`var.cross[sp]`,
`var.cross_J[(sp, i)]`, ...) is no longer written; the `.vul` writer
synthesizes it from the dense pytree at pickle time.

Numerical contract: bit-exact (<= 1e-13; in practice 0.0 abs/rel err)
vs the legacy `make_bins_read_cross` on the supported config matrix
(HD189 / Earth / Jupiter / HD209 cfg_examples). `scipy.interpolate.
interp1d(kind='linear', bounds_error=False, fill_value=...)` and
`numpy.interp(..., left=..., right=...)` compute the same linear
formula and clip the same way; we use `np.interp` throughout.

Design:
  - Pure NumPy. No JAX or scipy here. Setup runs once at startup; JIT
    overhead is pure cost. NumPy gives bit-exact match to scipy's
    `interp1d(kind='linear')`.
  - Host-side CSV I/O at the boundary; the actual interpolation
    kernels are flat array ops the runtime can later vmap or batch.
  - `build_photo_static(cfg, atm)` is the pure builder; it returns a
    `state.PhotoStaticInputs` pytree with every dense cross-section
    array the runtime needs. `populate_photo(var, atm)` is a
    convenience wrapper that also zero-allocates the runtime mutable
    buffers (`var.tau`, `var.sflux`, ...) on `var`.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

import vulcan_cfg
from state import PhotoStaticInputs


_LOWT_SENTINEL = -100.0  # log10 stand-in for log10(0) = -inf; mirrors
# `legacy_io.py:499/500` exactly.


# ---------------------------------------------------------------------------
# Host-side CSV readers
# ---------------------------------------------------------------------------

def _cross_folder() -> str:
    """Return cfg.cross_folder with no trailing-slash gymnastics."""
    return str(vulcan_cfg.cross_folder)


def _load_thresholds(species_in_network) -> dict[str, float]:
    """Read `cross_folder/thresholds.txt` and return the per-species
    photodissociation wavelength threshold (nm).

    Mirrors `legacy_io.py:333-338`: `np.genfromtxt(usecols=0)` for the
    label column, `np.genfromtxt(skip_header=1)[:, 1]` for the value
    column. Filtered to species present in the current network.
    """
    folder = _cross_folder()
    sp_label = np.genfromtxt(folder + "thresholds.txt", dtype=str, usecols=0)
    lmd_data = np.genfromtxt(folder + "thresholds.txt", skip_header=1)[:, 1]
    species_set = set(species_in_network)
    return {
        str(label): float(row)
        for label, row in zip(sp_label, lmd_data)
        if str(label) in species_set
    }


def _load_cross_csv(sp: str, use_ion: bool) -> np.ndarray:
    """Read `cross_folder/{sp}/{sp}_cross.csv`.

    4-column when `use_ion=True` (lambda, cross, disso, ion), else
    3-column (lambda, cross, disso). Mirrors `legacy_io.py:344-352`.
    """
    folder = _cross_folder()
    path = folder + sp + "/" + sp + "_cross.csv"
    names = (["lambda", "cross", "disso", "ion"] if use_ion
             else ["lambda", "cross", "disso"])
    try:
        return np.genfromtxt(
            path, dtype=float, delimiter=",", skip_header=1, names=names,
        )
    except Exception:
        print("\nMissing the cross section from " + sp)
        raise


def _load_branch_csv(sp: str) -> np.ndarray:
    """Read `cross_folder/{sp}/{sp}_branch.csv`. Column names auto-
    detected from header (br_ratio_1, br_ratio_2, ...). Mirrors
    `legacy_io.py:355`."""
    folder = _cross_folder()
    path = folder + sp + "/" + sp + "_branch.csv"
    try:
        return np.genfromtxt(
            path, dtype=float, delimiter=",", skip_header=1, names=True,
        )
    except Exception:
        print("\nMissing the branching ratio from " + sp)
        raise


def _load_ion_branch_csv(sp: str) -> np.ndarray:
    """Read `cross_folder/{sp}/{sp}_ion_branch.csv`. Mirrors
    `legacy_io.py:347`."""
    folder = _cross_folder()
    path = folder + sp + "/" + sp + "_ion_branch.csv"
    try:
        return np.genfromtxt(
            path, dtype=float, delimiter=",", skip_header=1, names=True,
        )
    except Exception:
        print("\nMissing the ion branching ratio from " + sp)
        raise


def _discover_T_cross_files(sp: str) -> list[int]:
    """Scan `thermo/photo_cross/{sp}/` for `{sp}_cross_{T}K.csv` files
    and return the integer T list. Mirrors `legacy_io.py:361-366`.

    Note: legacy hardcodes the path string ``"thermo/photo_cross/" + sp +
    "/"`` (not via cfg.cross_folder). We match that exactly so any user
    who has reconfigured cross_folder still gets the same behavior.
    """
    T_list: list[int] = []
    folder = "thermo/photo_cross/" + sp + "/"
    for temp_file in os.listdir(folder):
        if temp_file.startswith(sp) and temp_file.endswith("K.csv"):
            temp = (
                temp_file
                .replace(sp, "")
                .replace("_cross_", "")
                .replace("K.csv", "")
            )
            T_list.append(int(temp))
    return T_list


def _load_T_cross_csv(sp: str, T: int, use_ion: bool) -> np.ndarray:
    """Read `cross_folder/{sp}/{sp}_cross_{T}K.csv`. Mirrors
    `legacy_io.py:368-370`."""
    folder = _cross_folder()
    path = folder + sp + "/" + sp + "_cross_" + str(T) + "K.csv"
    names = (["lambda", "cross", "disso", "ion"] if use_ion
             else ["lambda", "cross", "disso"])
    return np.genfromtxt(
        path, dtype=float, delimiter=",", skip_header=1, names=names,
    )


def _load_rayleigh_csv(sp: str) -> np.ndarray:
    """Read `cross_folder/rayleigh/{sp}_scat.txt`. Mirrors
    `legacy_io.py:592-593`."""
    folder = _cross_folder()
    path = folder + "rayleigh/" + sp + "_scat.txt"
    return np.genfromtxt(
        path, dtype=float, skip_header=1, names=["lambda", "cross"],
    )


# ---------------------------------------------------------------------------
# Bin grid + per-bin interpolation kernels
# ---------------------------------------------------------------------------

def _make_bins(
    bin_min: float, bin_max: float,
    dbin1: float, dbin2: float, dbin_12trans: float,
) -> np.ndarray:
    """Two-resolution wavelength bin grid (nm). Mirrors
    `legacy_io.py:401-405` exactly.

    If `bin_min <= dbin_12trans <= bin_max`: concat fine-grid arange
    below the transition with coarse-grid arange above. Otherwise a
    single dbin1-spaced arange.
    """
    if dbin_12trans >= bin_min and dbin_12trans <= bin_max:
        return np.concatenate((
            np.arange(bin_min, dbin_12trans, dbin1),
            np.arange(dbin_12trans, bin_max, dbin2),
        ))
    return np.arange(bin_min, bin_max, dbin1)


def _sort_pairs(xp: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort `(xp, fp)` by `xp`. `scipy.interpolate.interp1d` does this
    internally so non-monotonic data files (e.g. CH3SH_branch.csv has a
    typo `354.0` between `253.5` and `309.5`) interpolate correctly;
    `np.interp` does NOT sort, so we have to do it explicitly to match.
    """
    order = np.argsort(xp, kind="stable")
    return xp[order], fp[order]


def _interp_zero_extrap(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation with `fill_value=0` outside [xp[0], xp[-1]].

    Bit-exact equivalent of
    `scipy.interpolate.interp1d(xp, fp, kind='linear',
        bounds_error=False, fill_value=0)(x)`. Sorts (xp, fp) first to
    match scipy's internal sort behavior on non-monotonic inputs.
    """
    xp_s, fp_s = _sort_pairs(xp, fp)
    return np.interp(x, xp_s, fp_s, left=0.0, right=0.0)


def _interp_edge_extrap(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation with `fill_value=(fp[0], fp[-1])` outside
    [xp[0], xp[-1]] -- the asymmetric branch-ratio extrapolation rule.

    Bit-exact equivalent of `scipy.interpolate.interp1d(...,
    fill_value=(fp[0], fp[-1]))(x)`. The fill values are read from the
    *unsorted* fp[0] / fp[-1] to match scipy's exact behavior, while
    the in-bounds interpolation uses the sorted pairs.
    """
    left = float(fp[0])
    right = float(fp[-1])
    xp_s, fp_s = _sort_pairs(xp, fp)
    return np.interp(x, xp_s, fp_s, left=left, right=right)


def _interp_T_log_pair(
    Tlow: float, Thigh: float, low_at_ld: float, high_at_ld: float, Tz: float,
) -> float:
    """Single (lev, bin) T-axis log10-space linear interp between two T
    samples. Mirrors `legacy_io.py:497-504` -- a 1-D interp1d on
    `[Tlow, Thigh] -> [log10_low, log10_high]` evaluated at Tz, with
    `log10(0) -> -100` sentinel and a final `10**` round-trip
    (sentinel collapses to 0 by the `inter_T == -100` check).
    """
    log_low = np.log10(low_at_ld) if low_at_ld > 0 else _LOWT_SENTINEL
    log_high = np.log10(high_at_ld) if high_at_ld > 0 else _LOWT_SENTINEL
    # 1-D linear interp; np.interp matches scipy.interp1d kind='linear'.
    val = np.interp(Tz, [Tlow, Thigh], [log_low, log_high])
    # Mirrors legacy line 503: `if inter_T(Tz) == -100: var.cross_T == 0.`
    # (the legacy uses `==` instead of `=` so this branch is a no-op;
    # cross_T is already 0-initialized, which we mirror by short-circuiting
    # to 0.0 only when val IS the sentinel exactly).
    if val == _LOWT_SENTINEL:
        return 0.0
    return float(10.0 ** val)


# ---------------------------------------------------------------------------
# Per-species cross-section binning helpers
# ---------------------------------------------------------------------------

def _bin_cross_and_branches(
    cross_raw: np.ndarray, ratio_raw: np.ndarray, n_branch: int, bins: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Build `cross[bin]` and `cross_J[branch_i][bin]` for one
    photodissociation species. Mirrors `legacy_io.py:442-458`.
    """
    lam = cross_raw["lambda"]
    cross_at_bins = _interp_zero_extrap(lam, cross_raw["cross"], bins)
    disso_at_bins = _interp_zero_extrap(lam, cross_raw["disso"], bins)

    branches: dict[int, np.ndarray] = {}
    for i in range(1, n_branch + 1):
        br_key = "br_ratio_" + str(i)
        try:
            ratio_at_bins = _interp_edge_extrap(
                ratio_raw["lambda"], ratio_raw[br_key], bins,
            )
        except Exception:
            print(
                "The branches in the network file does not match the "
                "branchong ratio file for "
            )
            raise
        branches[i] = disso_at_bins * ratio_at_bins
    return cross_at_bins, branches


def _bin_T_dependent(
    sp: str, n_branch: int, bins: np.ndarray, Tco: np.ndarray,
    cross_at_bins: np.ndarray, cross_J_at_bins: dict[int, np.ndarray],
    cross_T_raw: dict[tuple[str, int], np.ndarray],
    cross_T_sp_list: list[int],
    inter_ratio_at_bins: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Per-layer T-dependent cross + branch interpolation.

    Mirrors `legacy_io.py:462-567` (the three-case T branching). Returns
    `cross_T[lev, bin]` and per-branch `cross_J_T[i][lev, bin]`. The
    layer/bin loop is preserved as written for byte-for-byte parity --
    vectorising it would require careful handling of the
    `if ld < ld_min or ld > ld_max` mask per-(sp, T) pair, which is
    bookkeeping pain for almost no perf gain (this runs once at startup).
    """
    nz = int(Tco.shape[0])
    nbin = int(bins.shape[0])
    T_list_arr = np.array(cross_T_sp_list)
    max_T_sp = int(T_list_arr.max())
    min_T_sp = int(T_list_arr.min())

    cross_T = np.zeros((nz, nbin), dtype=np.float64)
    cross_J_T = {i: np.zeros((nz, nbin), dtype=np.float64)
                 for i in range(1, n_branch + 1)}

    for lev in range(nz):
        Tz = float(Tco[lev])
        below = T_list_arr[T_list_arr <= Tz]
        above = T_list_arr[T_list_arr > Tz]

        if below.size > 0 and above.size > 0:
            # Tz is between two T samples -> log10-space linear interp
            Tlow = int(below.max())
            Thigh = int(above.min())
            raw_low = cross_T_raw[(sp, Tlow)]
            raw_high = cross_T_raw[(sp, Thigh)]
            ld_min = max(raw_low["lambda"][0], raw_high["lambda"][0])
            ld_max = min(raw_low["lambda"][-1], raw_high["lambda"][-1])

            # Pre-bin the two T samples on the global wavelength grid so
            # the inner (bin, branch) loop is just a couple of numpy ops.
            cross_low_at_bins = _interp_zero_extrap(
                raw_low["lambda"], raw_low["cross"], bins,
            )
            cross_high_at_bins = _interp_zero_extrap(
                raw_high["lambda"], raw_high["cross"], bins,
            )
            disso_low_at_bins = _interp_zero_extrap(
                raw_low["lambda"], raw_low["disso"], bins,
            )
            disso_high_at_bins = _interp_zero_extrap(
                raw_high["lambda"], raw_high["disso"], bins,
            )

            for n in range(nbin):
                ld = float(bins[n])
                if ld < ld_min or ld > ld_max:
                    cross_T[lev, n] = cross_at_bins[n]
                    for i in range(1, n_branch + 1):
                        cross_J_T[i][lev, n] = cross_J_at_bins[i][n]
                else:
                    cross_T[lev, n] = _interp_T_log_pair(
                        Tlow, Thigh,
                        cross_low_at_bins[n], cross_high_at_bins[n], Tz,
                    )
                    for i in range(1, n_branch + 1):
                        val = _interp_T_log_pair(
                            Tlow, Thigh,
                            disso_low_at_bins[n], disso_high_at_bins[n], Tz,
                        )
                        cross_J_T[i][lev, n] = (
                            val * inter_ratio_at_bins[i][n]
                        )

        elif below.size == 0:
            # Tz is below the lowest tabulated T sample. If 300 K is the
            # min, fall back to the room-T cross section (= cross_at_bins);
            # otherwise extrapolate from the lowest tabulated T.
            if min_T_sp == 300:
                cross_T[lev] = cross_at_bins
                for i in range(1, n_branch + 1):
                    cross_J_T[i][lev] = cross_J_at_bins[i]
            else:
                raw_low = cross_T_raw[(sp, min_T_sp)]
                ld_min = float(raw_low["lambda"][0])
                ld_max = float(raw_low["lambda"][-1])
                cross_low_at_bins = _interp_zero_extrap(
                    raw_low["lambda"], raw_low["cross"], bins,
                )
                disso_low_at_bins = _interp_zero_extrap(
                    raw_low["lambda"], raw_low["disso"], bins,
                )
                for n in range(nbin):
                    ld = float(bins[n])
                    if ld < ld_min or ld > ld_max:
                        cross_T[lev, n] = cross_at_bins[n]
                        for i in range(1, n_branch + 1):
                            cross_J_T[i][lev, n] = cross_J_at_bins[i][n]
                    else:
                        cross_T[lev, n] = float(cross_low_at_bins[n])
                        for i in range(1, n_branch + 1):
                            cross_J_T[i][lev, n] = (
                                float(disso_low_at_bins[n])
                                * inter_ratio_at_bins[i][n]
                            )

        else:
            # Tz is above the highest tabulated T sample.
            if max_T_sp == 300:
                cross_T[lev] = cross_at_bins
                for i in range(1, n_branch + 1):
                    cross_J_T[i][lev] = cross_J_at_bins[i]
            else:
                raw_high = cross_T_raw[(sp, max_T_sp)]
                ld_min = float(raw_high["lambda"][0])
                ld_max = float(raw_high["lambda"][-1])
                cross_high_at_bins = _interp_zero_extrap(
                    raw_high["lambda"], raw_high["cross"], bins,
                )
                disso_high_at_bins = _interp_zero_extrap(
                    raw_high["lambda"], raw_high["disso"], bins,
                )
                for n in range(nbin):
                    ld = float(bins[n])
                    if ld < ld_min or ld > ld_max:
                        cross_T[lev, n] = cross_at_bins[n]
                        for i in range(1, n_branch + 1):
                            cross_J_T[i][lev, n] = cross_J_at_bins[i][n]
                    else:
                        cross_T[lev, n] = float(cross_high_at_bins[n])
                        for i in range(1, n_branch + 1):
                            cross_J_T[i][lev, n] = (
                                float(disso_high_at_bins[n])
                                * inter_ratio_at_bins[i][n]
                            )

    return cross_T, cross_J_T


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def populate_photo_arrays(var, atm) -> None:
    """Phase 22e backwards-compat wrapper for `populate_photo`.

    Pre-22e callers expect an in-place mutator that writes the legacy
    `var.cross*` dict surface plus scalars + runtime buffers. After 22e
    the dict surface is gone — only scalars (`var.bins`, `var.nbin`,
    `var.dbin1`, `var.dbin2`, `var.threshold`) and runtime buffers
    (`var.tau`, `var.aflux`, ...) remain on `var`; the dense cross
    sections live in the returned `PhotoStaticInputs` pytree. Callers
    that need the static must switch to `populate_photo` (returns the
    pytree); callers that only need `var.bins` / `var.nbin` / runtime-
    buffer initialization can keep calling this name.
    """
    populate_photo(var, atm)




# ---------------------------------------------------------------------------
# Phase 22e: dense-pytree builder
# ---------------------------------------------------------------------------

def _build_photo_static_dense(var, atm) -> PhotoStaticInputs:
    """Pure builder of the dense `PhotoStaticInputs` pytree.

    Reads the same CSV files as `populate_photo_arrays` and returns the
    same numerical cross sections, but packed as dense `(n, nbin)` /
    `(n, nz, nbin)` arrays instead of Python dicts. `din12_indx`
    initializes to `-1`; the caller must `_replace(din12_indx=...)`
    after `read_sflux` runs.

    `var` is read for the photo-metadata fields populated by
    `legacy_io.read_rate` (`var.photo_sp`, `var.ion_sp`, `var.n_branch`,
    `var.ion_branch`, `var.def_bin_min`, `var.def_bin_max`). No
    mutation of `var` happens here.
    """
    photo_sp = list(var.photo_sp)
    ion_sp = list(var.ion_sp)
    # Match the canonical ordering used by photo.pack_photo_data
    # (sorted union, alphabetical species names) so the dense pytree
    # row order is invariant of network reading order.
    absp_sp_list = sorted(set(photo_sp) | set(ion_sp))
    use_ion = bool(getattr(vulcan_cfg, "use_ion", False))
    T_cross_sp = list(vulcan_cfg.T_cross_sp or [])
    scat_sp_list = list(vulcan_cfg.scat_sp or [])
    nz = int(atm.Tco.shape[0])

    # 1. Threshold table + raw CSV ingestion (locals only).
    import chem_funs
    threshold = _load_thresholds(chem_funs.spec_list)

    cross_raw: dict[str, np.ndarray] = {}
    ratio_raw: dict[str, np.ndarray] = {}
    ion_ratio_raw: dict[str, np.ndarray] = {}
    cross_T_raw: dict[tuple[str, int], np.ndarray] = {}
    cross_T_sp_list_local: dict[str, list[int]] = {}

    bin_min: float | None = None
    bin_max: float | None = None
    diss_max: float | None = None

    for n_idx, sp in enumerate(absp_sp_list):
        cross_raw[sp] = _load_cross_csv(sp, use_ion)
        if use_ion and sp in ion_sp:
            ion_ratio_raw[sp] = _load_ion_branch_csv(sp)
        if sp in photo_sp:
            ratio_raw[sp] = _load_branch_csv(sp)
        if sp in T_cross_sp:
            T_list = _discover_T_cross_files(sp)
            cross_T_sp_list_local[sp] = T_list
            for tt in T_list:
                cross_T_raw[(sp, tt)] = _load_T_cross_csv(sp, tt, use_ion)
            cross_T_raw[(sp, 300)] = cross_raw[sp]
            cross_T_sp_list_local[sp].append(300)

        if cross_raw[sp]["cross"][0] == 0 or cross_raw[sp]["cross"][-1] == 0:
            raise IOError(
                "\n Please remove the zeros in the cross file of " + sp
            )

        sp_min = float(cross_raw[sp]["lambda"][0])
        sp_max = float(cross_raw[sp]["lambda"][-1])
        try:
            sp_diss = float(threshold[sp])
        except KeyError:
            print(sp + " not in threshol.txt")
            raise
        if n_idx == 0:
            bin_min, bin_max, diss_max = sp_min, sp_max, sp_diss
        else:
            if sp_min < bin_min:
                bin_min = sp_min
            if sp_max > bin_max:
                bin_max = sp_max
            if sp_diss > diss_max:
                diss_max = sp_diss

    # 2. Constrain by stellar-flux extents + photolysis threshold.
    bin_min = max(bin_min, var.def_bin_min)
    bin_max = min(bin_max, var.def_bin_max, diss_max)
    print(
        "Input stellar spectrum from "
        + "{:.1f}".format(var.def_bin_min)
        + " to " + "{:.1f}".format(var.def_bin_max)
    )
    print("Photodissociation threshold: " + "{:.1f}".format(diss_max))
    print(
        "Using wavelength bins from "
        + "{:.1f}".format(bin_min) + " to " + str(bin_max)
    )

    # 3. Bin grid.
    dbin1 = float(vulcan_cfg.dbin1)
    dbin2 = float(vulcan_cfg.dbin2)
    bins = _make_bins(bin_min, bin_max, dbin1, dbin2, vulcan_cfg.dbin_12trans)
    nbin = int(bins.shape[0])

    # 4. Per-photo-species absorbing cross + branch cross.
    cross_per_sp: dict[str, np.ndarray] = {
        sp: np.zeros(nbin, dtype=np.float64) for sp in absp_sp_list
    }
    cross_J_per_key: dict[tuple[str, int], np.ndarray] = {}
    cross_T_per_sp: dict[str, np.ndarray] = {
        sp: np.zeros((nz, nbin), dtype=np.float64) for sp in T_cross_sp
    }
    cross_J_T_per_key: dict[tuple[str, int], np.ndarray] = {}

    for sp in photo_sp:
        cross_at_bins, cross_J_at_bins = _bin_cross_and_branches(
            cross_raw[sp], ratio_raw[sp], var.n_branch[sp], bins,
        )
        cross_per_sp[sp][:] = cross_at_bins
        for i in range(1, var.n_branch[sp] + 1):
            cross_J_per_key[(sp, i)] = cross_J_at_bins[i].copy()

        if sp in T_cross_sp:
            inter_ratio_at_bins = {
                i: _interp_edge_extrap(
                    ratio_raw[sp]["lambda"],
                    ratio_raw[sp]["br_ratio_" + str(i)],
                    bins,
                )
                for i in range(1, var.n_branch[sp] + 1)
            }
            cross_T_arr, cross_J_T_per_branch = _bin_T_dependent(
                sp, var.n_branch[sp], bins, atm.Tco,
                cross_at_bins, cross_J_at_bins,
                cross_T_raw, cross_T_sp_list_local[sp],
                inter_ratio_at_bins,
            )
            cross_T_per_sp[sp][:] = cross_T_arr
            for i in range(1, var.n_branch[sp] + 1):
                cross_J_T_per_key[(sp, i)] = cross_J_T_per_branch[i].copy()

    # 5. Ion cross interpolation.
    cross_Jion_per_key: dict[tuple[str, int], np.ndarray] = {}
    if use_ion:
        for sp in ion_sp:
            if sp not in photo_sp:
                cross_per_sp[sp][:] = _interp_zero_extrap(
                    cross_raw[sp]["lambda"], cross_raw[sp]["cross"], bins,
                )
            cross_Jion_at_bins = _interp_zero_extrap(
                cross_raw[sp]["lambda"], cross_raw[sp]["ion"], bins,
            )
            for i in range(1, var.ion_branch[sp] + 1):
                br_key = "br_ratio_" + str(i)
                ratio_at_bins = _interp_edge_extrap(
                    ion_ratio_raw[sp]["lambda"],
                    ion_ratio_raw[sp][br_key],
                    bins,
                )
                cross_Jion_per_key[(sp, i)] = (
                    cross_Jion_at_bins * ratio_at_bins
                )

    # 6. Rayleigh scattering.
    cross_scat_per_sp: dict[str, np.ndarray] = {}
    for sp in scat_sp_list:
        scat_raw = _load_rayleigh_csv(sp)
        cross_scat_per_sp[sp] = _interp_zero_extrap(
            scat_raw["lambda"], scat_raw["cross"], bins,
        )

    # 7. Pack dense arrays. Order is the canonical iteration order
    # consumers (photo runtime kernels + .vul synthesizer) rely on.
    absp_sp_ordered = tuple(absp_sp_list)        # photo_sp + ion_sp
    absp_T_sp_ordered = tuple(sp for sp in absp_sp_list if sp in T_cross_sp)
    scat_sp_ordered = tuple(scat_sp_list)
    branch_keys_ordered = tuple(
        (sp, i)
        for sp in photo_sp
        if sp not in T_cross_sp
        for i in range(1, var.n_branch[sp] + 1)
    )
    branch_T_keys_ordered = tuple(
        (sp, i)
        for sp in photo_sp
        if sp in T_cross_sp
        for i in range(1, var.n_branch[sp] + 1)
    )
    if use_ion:
        ion_branch_keys_ordered = tuple(
            (sp, i)
            for sp in ion_sp
            for i in range(1, var.ion_branch[sp] + 1)
        )
    else:
        ion_branch_keys_ordered = ()

    if absp_sp_ordered:
        absp_cross = np.stack(
            [cross_per_sp[sp] for sp in absp_sp_ordered], axis=0
        )
    else:
        absp_cross = np.zeros((0, nbin), dtype=np.float64)
    if absp_T_sp_ordered:
        absp_T_cross = np.stack(
            [cross_T_per_sp[sp] for sp in absp_T_sp_ordered], axis=0
        )
    else:
        absp_T_cross = np.zeros((0, nz, nbin), dtype=np.float64)
    if scat_sp_ordered:
        scat_cross = np.stack(
            [cross_scat_per_sp[sp] for sp in scat_sp_ordered], axis=0
        )
    else:
        scat_cross = np.zeros((0, nbin), dtype=np.float64)
    if branch_keys_ordered:
        cross_J = np.stack(
            [cross_J_per_key[k] for k in branch_keys_ordered], axis=0
        )
    else:
        cross_J = np.zeros((0, nbin), dtype=np.float64)
    if branch_T_keys_ordered:
        cross_J_T = np.stack(
            [cross_J_T_per_key[k] for k in branch_T_keys_ordered], axis=0
        )
    else:
        cross_J_T = np.zeros((0, nz, nbin), dtype=np.float64)
    if ion_branch_keys_ordered:
        cross_Jion = np.stack(
            [cross_Jion_per_key[k] for k in ion_branch_keys_ordered], axis=0
        )
    else:
        cross_Jion = np.zeros((0, nbin), dtype=np.float64)

    return PhotoStaticInputs(
        bins=jnp.asarray(bins, dtype=jnp.float64),
        nbin=nbin,
        dbin1=dbin1,
        dbin2=dbin2,
        din12_indx=-1,
        absp_sp=absp_sp_ordered,
        absp_T_sp=absp_T_sp_ordered,
        scat_sp=scat_sp_ordered,
        branch_keys=branch_keys_ordered,
        branch_T_keys=branch_T_keys_ordered,
        ion_branch_keys=ion_branch_keys_ordered,
        absp_cross=jnp.asarray(absp_cross, dtype=jnp.float64),
        absp_T_cross=jnp.asarray(absp_T_cross, dtype=jnp.float64),
        scat_cross=jnp.asarray(scat_cross, dtype=jnp.float64),
        cross_J=jnp.asarray(cross_J, dtype=jnp.float64),
        cross_J_T=jnp.asarray(cross_J_T, dtype=jnp.float64),
        cross_Jion=jnp.asarray(cross_Jion, dtype=jnp.float64),
    )


def build_photo_static(cfg, atm, var) -> PhotoStaticInputs:
    """Pure builder: reads CSVs + atm.Tco, returns the dense
    `PhotoStaticInputs` pytree.

    `cfg` is accepted for symmetry with the rest of the JAX-native
    setup API but is not consumed directly -- the per-species CSV
    paths and dbin scalars come from the `vulcan_cfg` module that
    every other host-side reader uses. `var` is read for the
    photo-metadata populated by `legacy_io.read_rate`.

    `din12_indx` is `-1` until the post-photo `read_sflux` write fires;
    use `static.with_din12_indx(int(var.sflux_din12_indx))` to attach
    it.
    """
    del cfg  # match the rest of the setup API; vulcan_cfg is module-global
    return _build_photo_static_dense(var, atm)


def _alloc_runtime_buffers(var, nbin: int, nz: int) -> None:
    """Zero-allocate the host-side runtime mutable buffers on `var`.

    The runner reads these between batches (`var.tau`, `var.sflux`,
    `var.aflux`, ...); they stay on `var` because the chunked-driver
    path and `_initial_photo_carry` initialize the JAX carry from them.
    """
    var.sflux = np.zeros((nz + 1, nbin), dtype=np.float64)
    var.dflux_u = np.zeros((nz + 1, nbin), dtype=np.float64)
    var.dflux_d = np.zeros((nz + 1, nbin), dtype=np.float64)
    var.aflux = np.zeros((nz, nbin), dtype=np.float64)
    var.tau = np.zeros((nz + 1, nbin), dtype=np.float64)
    var.sflux_top = np.zeros(nbin, dtype=np.float64)


def populate_photo(var, atm) -> PhotoStaticInputs:
    """Build the dense `PhotoStaticInputs` pytree, write scalar grid
    metadata + threshold table to `var`, and zero-allocate the runtime
    mutable buffers on `var`.

    Phase 22e replacement for `populate_photo_arrays`. Returns the
    pytree; the caller wires it into `Ros2JAX(...)` and the `.vul`
    writer. After `read_sflux` populates `var.sflux_din12_indx`,
    re-attach via `static = static.with_din12_indx(int(var.sflux_din12_indx))`.

    Scalars (`var.bins` / `var.nbin` / `var.dbin1` / `var.dbin2` /
    `var.threshold`) stay on `var` because `atm_setup.read_sflux` and
    a handful of legacy callers still read them. The cross-section dict
    surface (`var.cross[sp]`, ...) is gone; the .vul writer's synthesizer
    rebuilds the legacy dict view from the pytree at pickle time.
    """
    static = _build_photo_static_dense(var, atm)
    var.bins = np.asarray(static.bins, dtype=np.float64)
    var.nbin = int(static.nbin)
    var.dbin1 = float(static.dbin1)
    var.dbin2 = float(static.dbin2)
    import chem_funs
    var.threshold = _load_thresholds(chem_funs.spec_list)
    _alloc_runtime_buffers(var, int(static.nbin), int(atm.Tco.shape[0]))
    return static
