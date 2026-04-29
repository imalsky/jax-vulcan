"""Photo cross-section preprocessing.

Builds the wavelength bin grid and interpolates per-species absorption /
dissociation / ionization / Rayleigh cross sections + branch ratios onto
that grid. Pure NumPy: setup runs once at startup.

`np.interp` matches `scipy.interpolate.interp1d(kind='linear',
bounds_error=False, fill_value=...)` bit-exactly, so we use np.interp
throughout.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

import vulcan_cfg
from state import PhotoStaticInputs


# log10 stand-in for log10(0) = -inf in T-axis log-space interpolation.
_LOWT_SENTINEL = -100.0


def _cross_folder() -> str:
    return str(vulcan_cfg.cross_folder)


def _load_thresholds(species_in_network) -> dict[str, float]:
    """Read `cross_folder/thresholds.txt`; return per-species photodissociation
    wavelength threshold (nm), filtered to species in the network."""
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
    """Read `cross_folder/{sp}/{sp}_cross.csv`. 4-column when `use_ion`, else 3-column."""
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
    """Read `cross_folder/{sp}/{sp}_branch.csv`. Column names auto-detected
    from header (br_ratio_1, br_ratio_2, ...)."""
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
    """Read `cross_folder/{sp}/{sp}_ion_branch.csv`."""
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
    """Scan `thermo/photo_cross/{sp}/` for `{sp}_cross_{T}K.csv` files;
    return the integer T list. Path is hardcoded (not via cfg.cross_folder)."""
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
    """Read `cross_folder/{sp}/{sp}_cross_{T}K.csv`."""
    folder = _cross_folder()
    path = folder + sp + "/" + sp + "_cross_" + str(T) + "K.csv"
    names = (["lambda", "cross", "disso", "ion"] if use_ion
             else ["lambda", "cross", "disso"])
    return np.genfromtxt(
        path, dtype=float, delimiter=",", skip_header=1, names=names,
    )


def _load_rayleigh_csv(sp: str) -> np.ndarray:
    """Read `cross_folder/rayleigh/{sp}_scat.txt`."""
    folder = _cross_folder()
    path = folder + "rayleigh/" + sp + "_scat.txt"
    return np.genfromtxt(
        path, dtype=float, skip_header=1, names=["lambda", "cross"],
    )


def _make_bins(
    bin_min: float, bin_max: float,
    dbin1: float, dbin2: float, dbin_12trans: float,
) -> np.ndarray:
    """Two-resolution wavelength bin grid (nm).

    If `bin_min <= dbin_12trans <= bin_max`, fine-grid arange below the
    transition is concatenated with coarse-grid arange above; else a single
    dbin1-spaced arange.
    """
    if dbin_12trans >= bin_min and dbin_12trans <= bin_max:
        return np.concatenate((
            np.arange(bin_min, dbin_12trans, dbin1),
            np.arange(dbin_12trans, bin_max, dbin2),
        ))
    return np.arange(bin_min, bin_max, dbin1)


def _sort_pairs(xp: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort `(xp, fp)` by `xp`. scipy.interpolate.interp1d sorts internally;
    np.interp does not, so non-monotonic data files (e.g. CH3SH_branch.csv
    has a typo `354.0` between `253.5` and `309.5`) need an explicit sort."""
    order = np.argsort(xp, kind="stable")
    return xp[order], fp[order]


def _interp_zero_extrap(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation with `fill_value=0` outside [xp[0], xp[-1]]."""
    xp_s, fp_s = _sort_pairs(xp, fp)
    return np.interp(x, xp_s, fp_s, left=0.0, right=0.0)


def _interp_edge_extrap(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation with `fill_value=(fp[0], fp[-1])` outside
    [xp[0], xp[-1]] -- the asymmetric branch-ratio extrapolation rule.

    Fill values are read from the *unsorted* fp[0]/fp[-1] to match scipy's
    exact behavior, while the in-bounds interpolation uses sorted pairs.
    """
    left = float(fp[0])
    right = float(fp[-1])
    xp_s, fp_s = _sort_pairs(xp, fp)
    return np.interp(x, xp_s, fp_s, left=left, right=right)


def _interp_T_log_pair(
    Tlow: float, Thigh: float, low_at_ld: float, high_at_ld: float, Tz: float,
) -> float:
    """Single (lev, bin) T-axis log10-space linear interp between two T samples.

    log10(0) maps to a -100 sentinel; if the interpolated value is exactly the
    sentinel, return 0.0 (no 10**sentinel).
    """
    log_low = np.log10(low_at_ld) if low_at_ld > 0 else _LOWT_SENTINEL
    log_high = np.log10(high_at_ld) if high_at_ld > 0 else _LOWT_SENTINEL
    val = np.interp(Tz, [Tlow, Thigh], [log_low, log_high])
    if val == _LOWT_SENTINEL:
        return 0.0
    return float(10.0 ** val)


def _bin_cross_and_branches(
    cross_raw: np.ndarray, ratio_raw: np.ndarray, n_branch: int, bins: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Build `cross[bin]` and `cross_J[branch_i][bin]` for one photodissociation species."""
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

    Returns `cross_T[lev, bin]` and per-branch `cross_J_T[i][lev, bin]`.
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
            Tlow = int(below.max())
            Thigh = int(above.min())
            raw_low = cross_T_raw[(sp, Tlow)]
            raw_high = cross_T_raw[(sp, Thigh)]
            ld_min = max(raw_low["lambda"][0], raw_high["lambda"][0])
            ld_max = min(raw_low["lambda"][-1], raw_high["lambda"][-1])

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
            # Tz below the lowest tabulated T sample.
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
            # Tz above the highest tabulated T sample.
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


def populate_photo_arrays(var, atm) -> None:
    """In-place mutator wrapper for `populate_photo`."""
    populate_photo(var, atm)


def _build_photo_static_dense(var, atm) -> PhotoStaticInputs:
    """Pure builder of the dense `PhotoStaticInputs` pytree.

    `din12_indx` initializes to `-1`; the caller must `_replace(din12_indx=...)`
    after `read_sflux` runs.
    """
    photo_sp = list(var.photo_sp)
    ion_sp = list(var.ion_sp)
    # Sorted union (alphabetical) so the dense pytree row order is invariant
    # of network reading order.
    absp_sp_list = sorted(set(photo_sp) | set(ion_sp))
    use_ion = bool(getattr(vulcan_cfg, "use_ion", False))
    T_cross_sp = list(vulcan_cfg.T_cross_sp or [])
    scat_sp_list = list(vulcan_cfg.scat_sp or [])
    nz = int(atm.Tco.shape[0])

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

    dbin1 = float(vulcan_cfg.dbin1)
    dbin2 = float(vulcan_cfg.dbin2)
    bins = _make_bins(bin_min, bin_max, dbin1, dbin2, vulcan_cfg.dbin_12trans)
    nbin = int(bins.shape[0])

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

    cross_scat_per_sp: dict[str, np.ndarray] = {}
    for sp in scat_sp_list:
        scat_raw = _load_rayleigh_csv(sp)
        cross_scat_per_sp[sp] = _interp_zero_extrap(
            scat_raw["lambda"], scat_raw["cross"], bins,
        )

    # Canonical iteration order consumers (photo runtime kernels + .vul writer) rely on.
    absp_sp_ordered = tuple(absp_sp_list)
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
    """Pure builder: reads CSVs + atm.Tco, returns the dense `PhotoStaticInputs` pytree.

    `din12_indx` is `-1` until `read_sflux` runs; use
    `static.with_din12_indx(int(var.sflux_din12_indx))` to attach it.
    """
    del cfg
    return _build_photo_static_dense(var, atm)


def _alloc_runtime_buffers(var, nbin: int, nz: int) -> None:
    """Zero-allocate the host-side runtime mutable buffers on `var`."""
    var.sflux = np.zeros((nz + 1, nbin), dtype=np.float64)
    var.dflux_u = np.zeros((nz + 1, nbin), dtype=np.float64)
    var.dflux_d = np.zeros((nz + 1, nbin), dtype=np.float64)
    var.aflux = np.zeros((nz, nbin), dtype=np.float64)
    var.tau = np.zeros((nz + 1, nbin), dtype=np.float64)
    var.sflux_top = np.zeros(nbin, dtype=np.float64)


def populate_photo(var, atm) -> PhotoStaticInputs:
    """Build the dense `PhotoStaticInputs` pytree, write scalar grid metadata
    + threshold table to `var`, and zero-allocate the runtime mutable buffers.

    After `read_sflux` populates `var.sflux_din12_indx`, re-attach via
    `static = static.with_din12_indx(int(var.sflux_din12_indx))`.
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
