"""Legacy I/O surface matching VULCAN-master/op.py's entry points.

`ReadRate` keeps upstream's call signature (`ReadRate().read_rate(var,
atm)`) but is not a parser: `network.parse_network` is the one
network parser, and this copies the host-side metadata the runtime reads
off `var` (`var.Rf`, `var.pho_rate_index`, ...) from it. Rate *values*
come from `rates_jax.build_rate_array`.

`Output` writes the `.vul` pickle with the same public schema upstream
plotting tools read; photo cross-section dicts, the per-reaction `var.k`
dict, and parameter fields are synthesised at pickle time from the typed
`RunState`.
"""

import logging
import numpy as np
import os
import pickle
import time
import warnings

from .config import default_config
from . import chem_funs
from .chem_funs import ni, nr
from .live_ui import master_tableau20
from .state import TERM_NONFINITE, TERM_RUNTIME, TERM_STEP_COUNT

logger = logging.getLogger(__name__)

_CFG = default_config()
species = chem_funs.spec_list


def _warn_stale_reaction_ids(
    network_path: str, stale: list[tuple[int, int, str]]
) -> None:
    """Warn when written photo/ion ids differ from parse positions (the file
    was not renumbered upstream). Rates are indexed by position, but a
    ``cfg.remove_list`` taken from the id column picks the wrong reactions."""
    if not stale:
        return
    shown = ", ".join(
        f"{txt.strip()!r} written {fid} but at {pos}" for pos, fid, txt in stale[:3]
    )
    more = f" (+{len(stale) - 3} more)" if len(stale) > 3 else ""
    warnings.warn(
        f"Network {network_path} has {len(stale)} photo/ion reaction(s) whose "
        f"written id disagrees with its position: {shown}{more}. The file has not "
        "been renumbered by VULCAN's make_chem_funs.py. Rates are indexed by "
        "position, so this run is correct — but any cfg.remove_list entry taken "
        "from this file's id column will select the WRONG reaction. Renumber the "
        "file, or write remove_list using positions (1-based, forward reactions "
        "odd).",
        RuntimeWarning,
        stacklevel=2,
    )


class ReadRate(object):
    """Upstream's rate-setup entry point, backed by `network.parse_network`.

    Upstream's ``op.ReadRate`` re-parsed the network file and built rate
    coefficients; both are done elsewhere here (``network.parse_network``
    owns the parse, ``rates_jax.build_rate_array`` owns ``var.k_arr``). What is
    left is publishing the host-side metadata the runtime reads off the
    legacy ``var``.
    """

    def read_rate(self, var, atm):
        """Copy the parsed network's host-side metadata onto `var`.

        Sets the attributes something reads: `Rf` (reaction text by
        parser position), the photo/ion branch indices, and
        `conden_re_list`. The Arrhenius columns and the section markers are
        read off the `Network` everywhere, never off `var`, so they are not
        republished here.
        """
        del atm  # upstream signature; the metadata is temperature-free
        net = chem_funs._NETWORK
        var.Rf = dict(net.Rf)
        var.pho_rate_index = dict(net.pho_rate_index)
        var.ion_rate_index = dict(net.ion_rate_index)
        var.n_branch = dict(net.n_branch)
        var.ion_branch = dict(net.ion_branch)
        var.photo_sp = set(net.photo_sp)
        if _CFG.use_ion:
            var.ion_sp = set(net.ion_sp)
        var.conden_re_list = [
            int(i) for i in np.flatnonzero(net.is_conden & net.is_forward)
        ]
        _warn_stale_reaction_ids(net.network_path, list(net.stale_ids))
        return var




def _synthesize_cross_dicts(static) -> dict:
    """Build the legacy `var.cross*` dict views from a `PhotoStaticInputs`.

    The .vul writer publishes the same six photo dict keys the upstream
    `plot_py/` scripts index (`d['variable']['cross'][sp]`,
    `d['variable']['cross_J'][(sp,i)]`, etc.); the dicts are rebuilt from
    the dense pytree at pickle time. Every value is wrapped in
    `np.asarray(..., dtype=np.float64)` so downstream consumers see plain
    ndarrays (not jax arrays).
    """
    cross = {
        sp: np.asarray(static.absp_cross[i], dtype=np.float64)
        for i, sp in enumerate(static.absp_sp)
    }
    cross_T = {
        sp: np.asarray(static.absp_T_cross[i], dtype=np.float64)
        for i, sp in enumerate(static.absp_T_sp)
    }
    cross_J = {
        k: np.asarray(static.cross_J[i], dtype=np.float64)
        for i, k in enumerate(static.branch_keys)
    }
    cross_J_T = {
        k: np.asarray(static.cross_J_T[i], dtype=np.float64)
        for i, k in enumerate(static.branch_T_keys)
    }
    cross_scat = {
        sp: np.asarray(static.scat_cross[i], dtype=np.float64)
        for i, sp in enumerate(static.scat_sp)
    }
    cross_Jion = {
        k: np.asarray(static.cross_Jion[i], dtype=np.float64)
        for i, k in enumerate(static.ion_branch_keys)
    }
    return {
        "cross": cross,
        "cross_T": cross_T,
        "cross_J": cross_J,
        "cross_J_T": cross_J_T,
        "cross_scat": cross_scat,
        "cross_Jion": cross_Jion,
    }


def _integrate_J_branch(
    aflux: np.ndarray,
    cross_branch: np.ndarray,
    din12_indx: int,
    dbin1: float,
    dbin2: float,
) -> np.ndarray:
    """Trapezoidal-rule integral of `aflux * cross` over wavelength bins.

    Mirrors `op.compute_J` (op.py:2771-2780): a midpoint sum on each of the
    two-resolution sub-grids minus a half-weight correction at the four
    boundary samples (low-end, split, split-1, high-end). Works for both
    1-D `cross_branch` of shape (nbin,) and 2-D (nz, nbin) — broadcasting
    handles per-layer T-dependent cross sections transparently.
    """
    flo = aflux[:, :din12_indx]
    fhi = aflux[:, din12_indx:]
    if cross_branch.ndim == 1:
        clo = cross_branch[:din12_indx]
        chi = cross_branch[din12_indx:]
        c0 = cross_branch[0]
        c_split_minus = cross_branch[din12_indx - 1]
        c_split_plus = cross_branch[din12_indx]
        c_end = cross_branch[-1]
    else:
        clo = cross_branch[:, :din12_indx]
        chi = cross_branch[:, din12_indx:]
        c0 = cross_branch[:, 0]
        c_split_minus = cross_branch[:, din12_indx - 1]
        c_split_plus = cross_branch[:, din12_indx]
        c_end = cross_branch[:, -1]

    j = np.sum(flo * clo, axis=1) * dbin1
    j -= 0.5 * (aflux[:, 0] * c0 + aflux[:, din12_indx - 1] * c_split_minus) * dbin1
    j += np.sum(fhi * chi, axis=1) * dbin2
    j -= 0.5 * (aflux[:, din12_indx] * c_split_plus + aflux[:, -1] * c_end) * dbin2
    return j


def _synthesize_J_sp_dict(
    runstate,
    branch_count,
    species_iter,
    branch_keys,
    cross_J,
    branch_T_keys,
    cross_J_T,
    T_cross_sp,
):
    """Reconstruct the legacy `{(sp, branch): array(nz)}` J-rate dict
    from cross sections × actinic flux, matching `op.compute_J`.

    Master writes `var.J_sp[(sp, nbr)]` for ALL branches in `n_branch[sp]`
    (op.py:2764) and only skips writing `var.k[idx]` for branches whose
    `pho_rate_index` is in `cfg.remove_list` (op.py:2785). Reading J from
    `runstate.rate.k` therefore loses the J-rate for any removed branch.
    We integrate `aflux * cross` directly so removed branches still report
    their true photolysis rate, as master's writer does.

    Branch 0 is the across-branch sum (op.py:2783).

    Args:
        runstate:       populated RunState; needs `photo_runtime.aflux`
                        and `photo_static.{bins, dbin1, dbin2, din12_indx}`.
        branch_count:   `{sp: n_branch}` mapping (max branch number per sp).
        species_iter:   iterable of species names (photo_sp / ion_sp).
        branch_keys:    tuple of `(sp, br)` for non-T cross_J rows.
        cross_J:        (n_br, nbin) cross sections per branch.
        branch_T_keys:  tuple of `(sp, br)` for T-dep cross_J_T rows.
        cross_J_T:      (n_br_T, nz, nbin) per-layer T-dep cross sections.
        T_cross_sp:     species using cross_J_T (per cfg.T_cross_sp).

    Returns:
        `{(sp, branch): np.ndarray(nz)}` matching master's schema.
    """
    pr = runstate.photo_runtime
    static = runstate.photo_static
    if pr is None or static is None:
        return {}
    aflux = np.asarray(pr.aflux, dtype=np.float64)
    nz = aflux.shape[0]
    din12_indx = int(static.din12_indx)
    dbin1 = float(static.dbin1)
    dbin2 = float(static.dbin2)

    # Lookup tables: (sp, br) -> cross row index.
    nonT_index = {key: i for i, key in enumerate(branch_keys)}
    T_index = {key: i for i, key in enumerate(branch_T_keys)}
    T_cross_set = set(T_cross_sp)

    out: dict = {}
    for sp in species_iter:
        nbr_max = int(branch_count.get(sp, 0))
        for br in range(1, nbr_max + 1):
            if sp in T_cross_set and (sp, br) in T_index:
                cross_row = np.asarray(cross_J_T[T_index[(sp, br)]], dtype=np.float64)
            elif (sp, br) in nonT_index:
                cross_row = np.asarray(cross_J[nonT_index[(sp, br)]], dtype=np.float64)
            else:
                # No cross section table for this branch (e.g. ion-only
                # species in pho_rate_index, or vice versa).
                out[(sp, br)] = np.zeros(nz, dtype=np.float64)
                continue
            out[(sp, br)] = _integrate_J_branch(
                aflux,
                cross_row,
                din12_indx,
                dbin1,
                dbin2,
            )
        # Branch 0 = sum across branches 1..nbr_max (op.py:2783).
        if nbr_max > 0:
            out[(sp, 0)] = sum(out[(sp, br)] for br in range(1, nbr_max + 1))
        else:
            out[(sp, 0)] = np.zeros(nz, dtype=np.float64)
    return out


def _synthesize_save_dicts(runstate, cfg, photo_static=None):
    """Build the three .vul top-level dicts from a `RunState`.

    Returns `(variable_dict, atm_dict, parameter_dict)` matching the
    legacy `(vars(data_var) filtered by var_save, vars(atm), vars(para))`
    public shape so `plot_py/` and downstream consumers see the same keys,
    shapes, and dtypes as upstream.

    `photo_static` defaults to `runstate.photo_static`; pass an explicit
    pytree only when the caller has a different cross-section override.
    """
    use_photo = bool(cfg.use_photo)
    use_ion = bool(cfg.use_ion)
    use_save_evo = bool(cfg.save_evolution)
    T_cross_sp = list(cfg.T_cross_sp)

    # 1. Variable dict — mirrors the legacy var.var_save filter.
    var_save = {"species": species, "nr": nr}

    # Rate dict from the dense (nr+1, nz) array.
    k_arr = np.asarray(runstate.rate.k, dtype=np.float64)
    var_save["k"] = {i: k_arr[i].copy() for i in range(1, k_arr.shape[0])}

    # Step slice.
    if runstate.step is not None:
        var_save["y"] = np.asarray(runstate.step.y, dtype=np.float64)
        var_save["ymix"] = np.asarray(runstate.step.ymix, dtype=np.float64)
        var_save["t"] = float(runstate.step.t)
        var_save["dt"] = float(runstate.step.dt)
        var_save["longdy"] = float(runstate.step.longdy)
        var_save["longdydt"] = float(runstate.step.longdydt)
    # Initial-state snapshot (taken at ini_y; metadata.y_ini holds the
    # canonical reference even after the runner mutates `step.y`).
    if runstate.metadata is not None:
        var_save["y_ini"] = np.asarray(runstate.metadata.y_ini, dtype=np.float64)

    # Atom dicts from the typed atom slot.
    if runstate.atoms is not None:
        a = runstate.atoms
        ai = np.asarray(a.atom_ini)
        al = np.asarray(a.atom_loss)
        as_ = np.asarray(a.atom_sum)
        var_save["atom_ini"] = {sp: float(ai[i]) for i, sp in enumerate(a.atom_order)}
        var_save["atom_sum"] = {sp: float(as_[i]) for i, sp in enumerate(a.atom_order)}
        var_save["atom_loss"] = {sp: float(al[i]) for i, sp in enumerate(a.atom_order)}
        # The runner does not track atom_conden; zeros keep the .vul schema.
        var_save["atom_conden"] = {sp: 0.0 for sp in a.atom_order}

    # Photo runtime (when use_photo).
    if use_photo and runstate.photo_runtime is not None:
        pr = runstate.photo_runtime
        var_save["tau"] = np.asarray(pr.tau, dtype=np.float64)
        var_save["sflux"] = np.asarray(pr.sflux, dtype=np.float64)
        var_save["aflux"] = np.asarray(pr.aflux, dtype=np.float64)
        var_save["aflux_change"] = float(pr.aflux_change)

    # Photo cross-section dicts.
    static = photo_static if photo_static is not None else runstate.photo_static
    if use_photo and static is not None:
        photo_dicts = _synthesize_cross_dicts(static)
        var_save["cross"] = photo_dicts["cross"]
        var_save["cross_scat"] = photo_dicts["cross_scat"]
        var_save["cross_J"] = photo_dicts["cross_J"]
        if T_cross_sp:
            var_save["cross_T"] = photo_dicts["cross_T"]
        if use_ion:
            var_save["cross_Jion"] = photo_dicts["cross_Jion"]
        # Bin grid.
        var_save["nbin"] = int(static.nbin)
        var_save["bins"] = np.asarray(static.bins, dtype=np.float64)
        var_save["dbin1"] = float(static.dbin1)
        var_save["dbin2"] = float(static.dbin2)

    # Metadata (Rf, n_branch, photo_sp, ion_sp, charge_list, ...).
    md = runstate.metadata
    if md is not None:
        var_save["Rf"] = dict(md.Rf)
        if use_photo:
            var_save["n_branch"] = dict(md.n_branch)
            var_save["J_sp"] = _synthesize_J_sp_dict(
                runstate,
                md.n_branch,
                md.photo_sp,
                static.branch_keys,
                static.cross_J,
                static.branch_T_keys,
                static.cross_J_T,
                T_cross_sp,
            )
        if use_ion:
            var_save["charge_list"] = list(md.charge_list)
            var_save["ion_sp"] = set(md.ion_sp)
            var_save["ion_wavelen"] = {}
            var_save["ion_branch"] = dict(md.ion_branch)
            var_save["ion_br_ratio"] = dict(md.ion_br_ratio)
            # Ion branches are non-T-dep; pass empty T tables.
            var_save["Jion_sp"] = _synthesize_J_sp_dict(
                runstate,
                md.ion_branch,
                md.ion_sp,
                static.ion_branch_keys,
                static.cross_Jion,
                tuple(),
                np.zeros((0, 0, 0), dtype=np.float64),
                tuple(),
            )

    # Evolution buffer under the legacy y_time / t_time keys.
    if use_save_evo and runstate.step is not None:
        var_save["y_time"] = np.asarray(runstate.step.y_evo, dtype=np.float64)
        var_save["t_time"] = np.asarray(runstate.step.t_evo, dtype=np.float64)

    # 2. Atm dict — mirrors `vars(data_atm)`.
    atm_save = {}
    a_in = runstate.atm
    for f in a_in._fields:
        atm_save[f] = np.asarray(getattr(a_in, f))
    if md is not None:
        atm_save["Ti"] = np.asarray(md.Ti)
        atm_save["gas_indx"] = list(md.gas_indx)
        atm_save["pref_indx"] = int(md.pref_indx)
        atm_save["gs"] = float(md.gs)
        atm_save["sat_p"] = dict(md.sat_p)
        atm_save["sat_mix"] = dict(md.sat_mix)
        atm_save["r_p"] = dict(md.r_p)
        atm_save["rho_p"] = dict(md.rho_p)
        atm_save["fix_sp_indx"] = dict(md.fix_sp_indx)
    atm_save["conden_min_lev"] = {}

    # 3. Parameter dict — mirrors `vars(data_para)` from state._Parameters.
    para_save = {}
    if runstate.params is not None:
        p = runstate.params
        para_save["count"] = int(p.count)
        para_save["nega_count"] = int(p.nega_count)
        para_save["loss_count"] = int(p.loss_count)
        para_save["delta_count"] = int(p.delta_count)
        para_save["delta"] = float(p.delta)
        para_save["small_y"] = float(p.small_y)
        para_save["nega_y"] = float(p.nega_y)
        para_save["end_case"] = int(getattr(p, "end_case", 0))
        # VULCAN-JAX addition: the runner's termination code, finer than
        # end_case (end_case 5 does not say why the run stopped).
        para_save["termination_reason"] = int(getattr(p, "termination_reason", 0))
        para_save["solver_str"] = "solver"
        para_save["switch_final_photo_frq"] = bool(
            getattr(p, "switch_final_photo_frq", False)
        )
        para_save["where_varies_most"] = np.asarray(
            getattr(
                p,
                "where_varies_most",
                np.zeros_like(np.asarray(runstate.step.y, dtype=np.float64)),
            ),
            dtype=np.float64,
        )
        para_save["pic_count"] = int(getattr(p, "pic_count", 0))
        para_save["fix_species_start"] = bool(p.fix_species_start)
    # Plotting-only master field. Runtime values live in live_ui, but the
    # public .vul schema should still expose the same parameter key.
    para_save["tableau20"] = master_tableau20()
    para_save.setdefault("end_case", 0)
    para_save.setdefault("termination_reason", 0)
    para_save.setdefault("solver_str", "solver")
    para_save.setdefault("switch_final_photo_frq", False)
    if runstate.step is not None:
        para_save.setdefault(
            "where_varies_most",
            np.zeros_like(np.asarray(runstate.step.y, dtype=np.float64)),
        )
    para_save.setdefault("pic_count", 0)
    para_save.setdefault("fix_species_start", False)
    if md is not None:
        para_save["start_time"] = float(md.start_time)

    return var_save, atm_save, para_save


class Output(object):
    """Per-run output: cfg copy, .vul writer, progress prints."""

    def __init__(self, cfg=None):
        """Set up the `.vul` writer for one run: create the output dir and
        warn if the target file already exists.
        """
        # cfg defaults to the process default; load_config() users pass their
        # namespace so output honors the same cfg as setup and the runner.
        # Pair with OuterLoop(cfg=cfg); save_out(..., cfg=...) overrides.
        self._cfg = cfg if cfg is not None else default_config()

        output_dir, out_name = self._cfg.output_dir, self._cfg.out_name
        os.makedirs(output_dir, exist_ok=True)

        if os.path.isfile(output_dir + out_name):
            warnings.warn(
                "The output file: " + str(out_name) + " already exists.", stacklevel=2
            )

    def print_prog(self, var, para):
        """Log one progress block: elapsed model time, step count vs
        `count_max`, longdy / longdy_dt / dt, and the most-varying
        (level, species).
        """
        indx_max = np.nanargmax(para.where_varies_most)
        logger.info(
            "Elapsed time: "
            + "{:.2e}".format(var.t)
            + " || Step number: "
            + str(para.count)
            + "/"
            + str(self._cfg.count_max)
        )
        logger.info(
            "longdy = "
            + "{:.2e}".format(var.longdy)
            + "      || longdy/dt = "
            + "{:.2e}".format(var.longdydt)
            + "  || dt = "
            + "{:.2e}".format(var.dt)
        )
        logger.info("from nz = " + str(int(indx_max / ni)) + " and " + species[indx_max % ni])
        logger.info(
            "------------------------------------------------------------------------"
        )

    def print_end_msg(self, var, para):
        """Log the steady-state success summary: CPU wall time, step count,
        final model time, long dy / dy_dt, per-atom loss (skipping
        `cfg.loss_ex`), and the negative/loss/delta rejection counters.
        """
        logger.info(
            "After ------- %s seconds -------" % (time.time() - para.start_time)
            + " s CPU time"
        )
        logger.info(
            self._cfg.out_name[:-4]
            + " has successfully run to steady-state with "
            + str(para.count)
            + " steps and "
            + str("{:.2e}".format(var.t))
            + " s"
        )
        logger.info(
            "long dy = "
            + f"{var.longdy:.6e}"
            + " and long dy/dt = "
            + f"{var.longdydt:.6e}"
        )

        logger.info("total atom loss:")
        for atom in self._cfg.atom_list:
            if atom not in self._cfg.loss_ex:
                logger.info(atom + ": " + f"{var.atom_loss[atom]:.4e}" + " ")

        logger.info("negative solution counter:")
        logger.info(str(para.nega_count))
        logger.info("loss rejected counter:")
        logger.info(str(para.loss_count))
        logger.info("delta rejected counter:")
        logger.info(str(para.delta_count))
        logger.info("------ Live long and prosper \\V/ ------")

    def print_unconverged_msg(self, var, para, case):
        """Log the non-converged summary for termination `case` (2 = runtime
        budget, 3 = max steps, 5 = stopped without converging and without
        hitting a cap); any other case raises RuntimeError. Also logs
        per-atom loss and rejection counters.
        """
        why = {
            TERM_RUNTIME: f"Maximal allowed runtime exceeded ({self._cfg.runtime:.1e} sec)",
            TERM_STEP_COUNT: f"Maximal allowed steps exceeded ({self._cfg.count_max} steps)",
            TERM_NONFINITE: "Stopped without converging and without hitting a cap "
            f"(termination_reason {para.termination_reason}); "
            "the state may be non-finite",
        }.get(case)
        if why is None:
            raise RuntimeError(f"Unconverged case undefined (case={case})")

        logger.info(
            "After ------- %s seconds -------" % (time.time() - para.start_time)
            + " s CPU time"
        )
        logger.warning(self._cfg.out_name[:-4] + " did not reach steady-state:")
        logger.info("long dy = " + str(var.longdy) + " and long dy/dt = " + str(var.longdydt))
        logger.warning("Integration stopped before converged...\n" + why)

        logger.info("total atom loss:")
        for atom in self._cfg.atom_list:
            if atom not in self._cfg.loss_ex:
                logger.info(atom + ": " + f"{var.atom_loss[atom]:.4e}" + " ")
        logger.info("negative solution counter:")
        logger.info(str(para.nega_count))
        logger.info("loss rejected counter:")
        logger.info(str(para.loss_count))
        logger.info("delta rejected counter:")
        logger.info(str(para.delta_count))

    def save_cfg(self, dname):
        """Write a repr snapshot of the active cfg (including make_config
        overrides) to `cfg_<out_name>.txt` under `dname/output_dir`, skipping
        private, callable, type, and module attributes so it re-reads as Python.
        """
        # Create the directory where the file is written, not under the cwd.
        output_dir, out_name = self._cfg.output_dir, self._cfg.out_name
        target_dir = os.path.join(dname, output_dir)
        os.makedirs(target_dir, exist_ok=True)

        lines = []
        for key in sorted(vars(self._cfg)):
            if key.startswith("_"):
                continue
            val = getattr(self._cfg, key)
            if callable(val) or isinstance(val, type) or hasattr(val, "__loader__"):
                continue
            lines.append(f"{key} = {val!r}")
        out_path = os.path.join(target_dir, "cfg_" + out_name[:-3] + "txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def save_out(self, runstate, dname, photo_static=None, cfg=None):
        """Write the `.vul` pickle output; reads everything from `runstate`."""
        cfg_mod = cfg if cfg is not None else self._cfg
        var_save, atm_save, para_save = _synthesize_save_dicts(
            runstate, cfg_mod, photo_static=photo_static
        )

        output_dir, out_name = cfg_mod.output_dir, cfg_mod.out_name
        target_dir = os.path.join(dname, output_dir)
        os.makedirs(target_dir, exist_ok=True)
        output_file = os.path.join(target_dir, out_name)

        with open(output_file, "wb") as outfile:
            pickle.dump(
                {"variable": var_save, "atm": atm_save, "parameter": para_save},
                outfile,
                protocol=4,
            )
