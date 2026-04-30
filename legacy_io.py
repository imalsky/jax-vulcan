"""Vendored I/O classes from VULCAN-master/op.py.

`ReadRate` is the network parser + rate-coef builder; it populates
host-side metadata dicts (`var.Rf`, `var.pho_rate_index`, ...). Rate
*values* are recomputed by `rates.build_rate_array` after this runs.

`Output` writes the `.vul` pickle with the same public schema upstream
plotting tools read. The photo cross-section dicts, per-reaction `var.k`
dict, and RunState-backed parameter fields are synthesised at pickle time
from typed JAX state.

Most legacy class method bodies retain upstream structure. JAX-specific
schema synthesis helpers are local adapters and should stay explicit.
"""

import numpy as np
import os
import pickle
import time

import vulcan_cfg
import chem_funs
from chem_funs import ni, nr
from vulcan_cfg import nz

species = chem_funs.spec_list


def _master_tableau20() -> list[tuple[float, float, float]]:
    """Return VULCAN-master's normalized Tableau 20 plotting palette."""
    colors = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194),
        (127, 127, 127), (188, 189, 34), (23, 190, 207),
        (174, 199, 232), (255, 187, 120), (152, 223, 138),
        (255, 152, 150), (197, 176, 213), (196, 156, 148),
        (247, 182, 210), (199, 199, 199), (219, 219, 141),
        (158, 218, 229),
    ]
    return [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in colors]


class ReadRate(object):
    """Network parser + rate-coef builder. Populates host-side metadata
    dicts on `var`; rate values are discarded (recomputed downstream)."""

    def __init__(self):

        self.i = 1
        self.re_tri, self.re_tri_k0 = False, False
        self.list_tri = []


    def read_rate(self, var, atm):

        # `k` here is parser scratch; rates.setup_var_k overwrites var.k_arr
        # immediately after this method returns.
        k = {}
        Rf, Rindx, a, n, E, a_inf, n_inf, E_inf, pho_rate_index = (
            var.Rf, var.Rindx, var.a, var.n, var.E,
            var.a_inf, var.n_inf, var.E_inf, var.pho_rate_index,
        )
        ion_rate_index = var.ion_rate_index

        i = self.i
        re_tri, re_tri_k0 = self.re_tri, self.re_tri_k0
        list_tri = self.list_tri

        Tco = atm.Tco.copy()
        M = atm.M.copy()

        special_re = False
        conden_re = False
        recomb_re = False
        photo_re = False
        ion_re = False

        photo_sp = []
        ion_sp = []

        with open(vulcan_cfg.network) as f:
            all_lines = f.readlines()
            for line_indx, line in enumerate(all_lines):

                # switch to 3-body and dissociation reations
                if line.startswith("# 3-body"):
                    re_tri = True

                if line.startswith("# 3-body reactions without high-pressure rates"):
                    re_tri_k0 = True

                elif line.startswith("# special"):
                    re_tri = False
                    re_tri_k0 = False
                    special_re = True # switch to reactions with special forms (hard coded)

                elif line.startswith("# condensation"):
                    re_tri = False
                    re_tri_k0 = False
                    special_re = False
                    conden_re = True
                    var.conden_indx = i

                elif line.startswith("# radiative"):
                    re_tri = False
                    re_tri_k0 = False
                    special_re = False
                    conden_re = False
                    recomb_re = True
                    var.recomb_indx = i

                elif line.startswith("# photo"):
                    re_tri = False
                    re_tri_k0 = False
                    special_re = False # turn off reading in the special form
                    conden_re = False
                    recomb_re = False
                    photo_re = True
                    var.photo_indx = i

                elif line.startswith("# ionisation"):
                    re_tri = False
                    re_tri_k0 = False
                    special_re = False # turn off reading in the special form
                    conden_re = False
                    recomb_re = False
                    photo_re = False
                    ion_re = True
                    var.ion_indx = i

                elif line.startswith("# reverse stops"):
                    var.special_re = False
                    var.stop_rev_indx = i

                # skip common lines and blank lines
                # ========================================================================================
                if not line.startswith("#") and line.strip() and special_re == False and conden_re == False and photo_re == False and ion_re == False: # if not starts

                    Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()
                    li = line.partition(']')[-1].strip()
                    columns = li.split()
                    Rindx[i] = int(line.partition('[')[0].strip())
                    a[i] = float(columns[0])
                    n[i] = float(columns[1])
                    E[i] = float(columns[2])

                    # switching to trimolecular reactions (len(columns) > 3 for those with high-P limit rates)
                    if re_tri == True and re_tri_k0 == False:
                        a_inf[i] = float(columns[3])
                        n_inf[i] = float(columns[4])
                        E_inf[i] = float(columns[5])
                        list_tri.append(i)

                    if columns[-1].strip() == 'He': re_He = i
                    elif columns[-1].strip() == 'ex1': re_CH3OH = i

                    k0 = a[i] * Tco**n[i] * np.exp(-E[i]/Tco)
                    if re_tri == False:
                        k[i] = k0
                    elif re_tri == True and len(columns) >= 6:
                        # 3-body with high-pressure limit (Lindemann)
                        k_inf_val = a_inf[i] * Tco**n_inf[i] * np.exp(-E_inf[i]/Tco)
                        k[i] = k0 / (1 + k0 * M / k_inf_val)
                    else:
                        # 3-body without high-pressure rates
                        k[i] = k0

                    i += 2
                    # end if not
                 # ========================================================================================
                elif special_re == True and line.strip() and not line.startswith("#"):

                    Rindx[i] = int(line.partition('[')[0].strip())
                    Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()

                    if Rf[i] == 'OH + CH3 + M -> CH3OH + M':
                        print ('Using special form for the reaction: ' + Rf[i])

                        k[i] = 1.932E3*Tco**-9.88 *np.exp(-7544./Tco) + 5.109E-11*Tco**-6.25 *np.exp(-1433./Tco)
                        k_inf = 1.031E-10 * Tco**-0.018 *np.exp(16.74/Tco)
                        # the pressure dependence from Jasper 2017
                        Fc = 0.1855*np.exp(-Tco/155.8)+0.8145*np.exp(-Tco/1675.)+np.exp(-4531./Tco)
                        nn = 0.75 - 1.27*np.log(Fc)
                        ff = np.exp( np.log(Fc)/(1.+ (np.log(k[i]*M/k_inf)/nn)**2 ) )

                        k[i] = k[i]/(1 + k[i]*M/k_inf ) *ff

                    i += 2


                # Testing condensation
                elif conden_re == True and line.strip() and not line.startswith("#"):
                    Rindx[i] = int(line.partition('[')[0].strip())
                    Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()

                    var.conden_re_list.append(i)
                    k[i] = np.zeros(nz)
                    k[i+1] = np.zeros(nz)

                    i += 2

                # setting photo dissociation reactions to zeros
                elif photo_re == True and line.strip() and not line.startswith("#"):

                    k[i] = np.zeros(nz)
                    Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()

                    # adding the photo species
                    photo_sp.append(Rf[i].split()[0])

                    li = line.partition(']')[-1].strip()
                    columns = li.split()
                    Rindx[i] = int(line.partition('[')[0].strip())
                    # columns[0]: the species being dissocited; branch index: columns[1]
                    pho_rate_index[(columns[0],int(columns[1]))] = Rindx[i]

                    # store the number of branches
                    var.n_branch[columns[0]] = int(columns[1])

                    i += 2

                # setting photo ionization reactions to zeros
                elif ion_re == True and line.strip() and not line.startswith("#"):

                    k[i] = np.zeros(nz)
                    Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()

                    ion_sp.append(Rf[i].split()[0])

                    li = line.partition(']')[-1].strip()
                    columns = li.split()
                    Rindx[i] = int(line.partition('[')[0].strip())
                    # columns[0]: the species being dissocited; branch index: columns[1]
                    ion_rate_index[(columns[0],int(columns[1]))] = Rindx[i]

                    # store the number of branches
                    var.ion_branch[columns[0]] = int(columns[1])

                    i += 2

        # The local `k` dict is not assigned to `var` — `rates.build_rate_array`
        # writes the canonical dense `var.k_arr` from scratch. The parser is
        # kept only for the metadata side-effects (var.Rf, var.Rindx,
        # var.pho_rate_index, var.n_branch, var.photo_sp, var.ion_sp,
        # var.conden_re_list, var.special_re, var.stop_rev_indx).

        var.photo_sp = set(photo_sp)
        if vulcan_cfg.use_ion == True: var.ion_sp = set(ion_sp)

        return var



    # `make_bins_read_cross` is intentionally not vendored. The dense
    # `PhotoStaticInputs` pytree built by `photo_setup.populate_photo` /
    # `photo_setup._build_photo_static_dense` is the canonical photo-input
    # surface; the .vul writer synthesizes the legacy dict views from it
    # at pickle time (see `_synthesize_cross_dicts` below). Tests that
    # need master's dict view use master's `op.ReadRate().make_bins_read_cross`
    # from sys.path.

def _import_plt():
    """Lazy import of matplotlib so tests that don't plot don't pay the cost."""
    import matplotlib
    matplotlib.use("Agg" if os.environ.get("VULCAN_HEADLESS_PLOT") else
                   matplotlib.get_backend())
    import matplotlib.pyplot as plt
    return plt


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
        'cross': cross,
        'cross_T': cross_T,
        'cross_J': cross_J,
        'cross_J_T': cross_J_T,
        'cross_scat': cross_scat,
        'cross_Jion': cross_Jion,
    }


def _integrate_J_branch(
    aflux: np.ndarray,
    cross_branch: np.ndarray,
    din12_indx: int,
    dbin1: float,
    dbin2: float,
) -> np.ndarray:
    """Trapezoidal-rule integral of `aflux * cross` over wavelength bins.

    Mirrors `op.compute_J` (op.py:2774-2783): a midpoint sum on each of the
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
    j -= 0.5 * (aflux[:, 0] * c0
                + aflux[:, din12_indx - 1] * c_split_minus) * dbin1
    j += np.sum(fhi * chi, axis=1) * dbin2
    j -= 0.5 * (aflux[:, din12_indx] * c_split_plus
                + aflux[:, -1] * c_end) * dbin2
    return j


def _synthesize_J_sp_dict(
    runstate, branch_count, species_iter,
    branch_keys, cross_J, branch_T_keys, cross_J_T, T_cross_sp,
):
    """Reconstruct the legacy `{(sp, branch): array(nz)}` J-rate dict
    from cross sections × actinic flux, matching `op.compute_J`.

    Master writes `var.J_sp[(sp, nbr)]` for ALL branches in `n_branch[sp]`
    (op.py:2767) and only skips writing `var.k[idx]` for branches whose
    `pho_rate_index` is in `cfg.remove_list` (op.py:2788). Reading J from
    `runstate.rate.k` therefore loses the J-rate for any removed branch.
    We integrate `aflux * cross` directly so removed branches still report
    their true photolysis rate, exactly matching master's writer.

    Branch 0 is the across-branch sum (master op.py:2786).

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
                cross_row = np.asarray(cross_J_T[T_index[(sp, br)]],
                                       dtype=np.float64)
            elif (sp, br) in nonT_index:
                cross_row = np.asarray(cross_J[nonT_index[(sp, br)]],
                                       dtype=np.float64)
            else:
                # No cross section table for this branch (e.g. ion-only
                # species in pho_rate_index, or vice versa).
                out[(sp, br)] = np.zeros(nz, dtype=np.float64)
                continue
            out[(sp, br)] = _integrate_J_branch(
                aflux, cross_row, din12_indx, dbin1, dbin2,
            )
        # Branch 0 = sum across branches 1..nbr_max (master op.py:2786).
        if nbr_max > 0:
            out[(sp, 0)] = sum(out[(sp, br)] for br in range(1, nbr_max + 1))
        else:
            out[(sp, 0)] = np.zeros(nz, dtype=np.float64)
    return out


def _is_runstate_arg(obj) -> bool:
    """Return True iff `obj` is a `state.RunState` (avoids a circular import
    at module-load time — `legacy_io` is imported very early in the
    setup pipeline)."""
    cls = type(obj)
    return cls.__name__ == "RunState" and cls.__module__ == "state"


def _synthesize_save_dicts(runstate, cfg, photo_static=None):
    """Build the three .vul top-level dicts from a `RunState`.

    Returns `(variable_dict, atm_dict, parameter_dict)` matching the
    legacy `(vars(data_var) filtered by var_save, vars(atm), vars(para))`
    public shape so `plot_py/` and downstream consumers see the same keys,
    shapes, and dtypes as upstream.

    `photo_static` defaults to `runstate.photo_static`; pass an explicit
    pytree only when the caller has a different cross-section override.
    """
    use_photo = bool(getattr(cfg, "use_photo", False))
    use_ion = bool(getattr(cfg, "use_ion", False))
    use_save_evo = bool(getattr(cfg, "save_evolution", False))
    T_cross_sp = list(getattr(cfg, "T_cross_sp", []) or [])

    # 1. Variable dict — mirrors the legacy var.var_save filter.
    var_save = {'species': species, 'nr': nr}

    # Rate dict from the dense (nr+1, nz) array.
    k_arr = np.asarray(runstate.rate.k, dtype=np.float64)
    var_save['k'] = {i: k_arr[i].copy() for i in range(1, k_arr.shape[0])}

    # Step slice.
    if runstate.step is not None:
        var_save['y'] = np.asarray(runstate.step.y, dtype=np.float64)
        var_save['ymix'] = np.asarray(runstate.step.ymix, dtype=np.float64)
        var_save['t'] = float(runstate.step.t)
        var_save['dt'] = float(runstate.step.dt)
        var_save['longdy'] = float(runstate.step.longdy)
        var_save['longdydt'] = float(runstate.step.longdydt)
    # Initial-state snapshot (taken at ini_y; metadata.y_ini holds the
    # canonical reference even after the runner mutates `step.y`).
    if runstate.metadata is not None:
        var_save['y_ini'] = np.asarray(runstate.metadata.y_ini, dtype=np.float64)

    # Atom dicts from the typed atom slot.
    if runstate.atoms is not None:
        a = runstate.atoms
        ai = np.asarray(a.atom_ini)
        al = np.asarray(a.atom_loss)
        as_ = np.asarray(a.atom_sum)
        var_save['atom_ini'] = {sp: float(ai[i]) for i, sp in enumerate(a.atom_order)}
        var_save['atom_sum'] = {sp: float(as_[i]) for i, sp in enumerate(a.atom_order)}
        var_save['atom_loss'] = {sp: float(al[i]) for i, sp in enumerate(a.atom_order)}
        # `atom_conden` historically tracked condensation losses; until
        # we route conden through the typed schema, publish zeros so the
        # .vul reader's downstream code keeps working.
        var_save['atom_conden'] = {sp: 0.0 for sp in a.atom_order}

    # Photo runtime (when use_photo).
    if use_photo and runstate.photo_runtime is not None:
        pr = runstate.photo_runtime
        var_save['tau'] = np.asarray(pr.tau, dtype=np.float64)
        var_save['sflux'] = np.asarray(pr.sflux, dtype=np.float64)
        var_save['aflux'] = np.asarray(pr.aflux, dtype=np.float64)
        var_save['aflux_change'] = float(pr.aflux_change)

    # Photo cross-section dicts.
    static = photo_static if photo_static is not None else runstate.photo_static
    if use_photo and static is not None:
        photo_dicts = _synthesize_cross_dicts(static)
        var_save['cross'] = photo_dicts['cross']
        var_save['cross_scat'] = photo_dicts['cross_scat']
        var_save['cross_J'] = photo_dicts['cross_J']
        if T_cross_sp:
            var_save['cross_J'] = photo_dicts['cross_J']
            var_save['cross_T'] = photo_dicts['cross_T']
        if use_ion:
            var_save['cross_Jion'] = photo_dicts['cross_Jion']
        # Bin grid.
        var_save['nbin'] = int(static.nbin)
        var_save['bins'] = np.asarray(static.bins, dtype=np.float64)
        var_save['dbin1'] = float(static.dbin1)
        var_save['dbin2'] = float(static.dbin2)

    # Metadata (Rf, n_branch, photo_sp, ion_sp, charge_list, ...).
    md = runstate.metadata
    if md is not None:
        var_save['Rf'] = dict(md.Rf)
        if use_photo:
            var_save['n_branch'] = dict(md.n_branch)
            # Both branches: integrate cross × aflux directly so removed
            # photo/ion reactions still publish their J-rate (master writes
            # J_sp for ALL branches and only skips the var.k assignment).
            var_save['J_sp'] = _synthesize_J_sp_dict(
                runstate, md.n_branch, md.photo_sp,
                static.branch_keys, static.cross_J,
                static.branch_T_keys, static.cross_J_T,
                T_cross_sp,
            )
        if use_ion:
            var_save['charge_list'] = list(md.charge_list)
            var_save['ion_sp'] = set(md.ion_sp)
            var_save['ion_wavelen'] = {}
            var_save['ion_branch'] = dict(md.ion_branch)
            var_save['ion_br_ratio'] = dict(md.ion_br_ratio)
            # Ion branches are non-T-dep; pass empty T tables.
            var_save['Jion_sp'] = _synthesize_J_sp_dict(
                runstate, md.ion_branch, md.ion_sp,
                static.ion_branch_keys, static.cross_Jion,
                tuple(), np.zeros((0, 0, 0), dtype=np.float64),
                tuple(),
            )

    # Evolution buffer — only when save_evolution. The OuterLoop fills
    # `runstate.step.y_evo` / `t_evo` (already sliced to the populated
    # prefix in `_unpack_state_to_runstate`); we expose them under
    # the legacy `var.y_time` / `var.t_time` schema.
    if use_save_evo and runstate.step is not None:
        var_save['y_time'] = np.asarray(runstate.step.y_evo, dtype=np.float64)
        var_save['t_time'] = np.asarray(runstate.step.t_evo, dtype=np.float64)

    # 2. Atm dict — mirrors `vars(data_atm)`.
    atm_save = {}
    a_in = runstate.atm
    for f in a_in._fields:
        atm_save[f] = np.asarray(getattr(a_in, f))
    if md is not None:
        atm_save['Ti'] = np.asarray(md.Ti)
        atm_save['gas_indx'] = list(md.gas_indx)
        atm_save['pref_indx'] = int(md.pref_indx)
        atm_save['gs'] = float(md.gs)
        atm_save['sat_p'] = dict(md.sat_p)
        atm_save['sat_mix'] = dict(md.sat_mix)
        atm_save['r_p'] = dict(md.r_p)
        atm_save['rho_p'] = dict(md.rho_p)
        atm_save['fix_sp_indx'] = dict(md.fix_sp_indx)
    atm_save['conden_min_lev'] = {}

    # 3. Parameter dict — mirrors `vars(data_para)` from store.Parameters.
    para_save = {}
    if runstate.params is not None:
        p = runstate.params
        para_save['count'] = int(p.count)
        para_save['nega_count'] = int(p.nega_count)
        para_save['loss_count'] = int(p.loss_count)
        para_save['delta_count'] = int(p.delta_count)
        para_save['delta'] = float(p.delta)
        para_save['small_y'] = float(p.small_y)
        para_save['nega_y'] = float(p.nega_y)
        para_save['end_case'] = int(getattr(p, 'end_case', 0))
        para_save['solver_str'] = 'solver'
        para_save['switch_final_photo_frq'] = bool(
            getattr(p, 'switch_final_photo_frq', False)
        )
        para_save['where_varies_most'] = np.asarray(
            getattr(
                p,
                'where_varies_most',
                np.zeros_like(np.asarray(runstate.step.y, dtype=np.float64)),
            ),
            dtype=np.float64,
        )
        para_save['pic_count'] = int(getattr(p, 'pic_count', 0))
        para_save['fix_species_start'] = bool(p.fix_species_start)
    # Plotting-only master field. Runtime values live in live_ui, but the
    # public .vul schema should still expose the same parameter key.
    para_save['tableau20'] = _master_tableau20()
    para_save.setdefault('end_case', 0)
    para_save.setdefault('solver_str', 'solver')
    para_save.setdefault('switch_final_photo_frq', False)
    if runstate.step is not None:
        para_save.setdefault(
            'where_varies_most',
            np.zeros_like(np.asarray(runstate.step.y, dtype=np.float64)),
        )
    para_save.setdefault('pic_count', 0)
    para_save.setdefault('fix_species_start', False)
    if md is not None:
        para_save['start_time'] = float(md.start_time)

    return var_save, atm_save, para_save


class Output(object):
    """Per-run output: cfg copy, .vul writer, post-run plotters, progress prints.

    Set `VULCAN_HEADLESS_PLOT=1` to force the matplotlib Agg backend
    (useful in CI).
    """

    def __init__(self):

        output_dir, out_name, plot_dir = vulcan_cfg.output_dir, vulcan_cfg.out_name, vulcan_cfg.plot_dir

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)

        if os.path.isfile(output_dir+out_name):
            print ('Warning... the output file: ' + str(out_name) + ' already exists.\n')

    def print_prog(self, var, para):
        indx_max = np.nanargmax(para.where_varies_most)
        print ('Elapsed time: ' +"{:.2e}".format(var.t) + ' || Step number: ' + str(para.count) + '/' + str(vulcan_cfg.count_max) )
        print ('longdy = ' + "{:.2e}".format(var.longdy) + '      || longdy/dt = ' + "{:.2e}".format(var.longdydt) + '  || dt = '+ "{:.2e}".format(var.dt) )
        print ('from nz = ' + str(int(indx_max/ni)) + ' and ' + species[indx_max%ni])
        print ('------------------------------------------------------------------------' )

    def print_end_msg(self, var, para):
        print ("After ------- %s seconds -------" % ( time.time()- para.start_time ) + ' s CPU time')
        print (vulcan_cfg.out_name[:-4] + ' has successfully run to steady-state with ' + str(para.count) + ' steps and ' + str("{:.2e}".format(var.t)) + ' s' )
        print ('long dy = ' + f"{var.longdy:.6e}" + ' and long dy/dt = ' + f"{var.longdydt:.6e}" )

        print ('total atom loss:')
        for atom in vulcan_cfg.atom_list:
            if atom not in getattr(vulcan_cfg, 'loss_ex', []):
                print (atom + ': ' + f"{var.atom_loss[atom]:.4e}" + ' ')

        print ('negative solution counter:')
        print (para.nega_count)
        print ('loss rejected counter:')
        print (para.loss_count)
        print ('delta rejected counter:')
        print (para.delta_count)
        if getattr(vulcan_cfg, 'use_shark', False) == True:
            print ("It's a long journey to this shark planet. Don't stop bleeding.")
        print ('------ Live long and prosper \\V/ ------')

    def print_unconverged_msg(self, var, para, case):
        if case == 2:
            print ("After ------- %s seconds -------" % ( time.time()- para.start_time ) + ' s CPU time')
            print (vulcan_cfg.out_name[:-4] + ' did not reach steady-state:')
            print ('long dy = ' + str(var.longdy) + ' and long dy/dt = ' + str(var.longdydt) )
            print ('Integration stopped before converged...\nMaximal allowed runtime exceeded ('+ f"{vulcan_cfg.runtime:.1e}" + ' sec)')
        elif case == 3:
            print ("After ------- %s seconds -------" % ( time.time()- para.start_time ) + ' s CPU time')
            print (vulcan_cfg.out_name[:-4] + ' did not reach steady-state:')
            print ('long dy = ' + str(var.longdy) + ' and long dy/dt = ' + str(var.longdydt) )
            print ('Integration stopped before converged...\nMaximal allowed steps exceeded ('+ str(vulcan_cfg.count_max) + ' steps)')
        else:
            raise RuntimeError(f"Unconverged case undefined (case={case})")

        print ('total atom loss:')
        for atom in vulcan_cfg.atom_list:
            if atom not in getattr(vulcan_cfg, 'loss_ex', []):
                print (atom + ': ' + f"{var.atom_loss[atom]:.4e}" + ' ')
        print ('negative solution counter:')
        print (para.nega_count)
        print ('loss rejected counter:')
        print (para.loss_count)
        print ('delta rejected counter:')
        print (para.delta_count)

    def save_cfg(self, dname):
        output_dir, out_name = vulcan_cfg.output_dir, vulcan_cfg.out_name
        if not os.path.exists(output_dir):
            print ('The output directory assigned in vulcan_cfg.py does not exist.')
            print( 'Directory ' , output_dir,  " created.")
            os.mkdir(output_dir)

        # copy the vulcan_cfg.py file
        with open('vulcan_cfg.py' ,'r') as f:
            cfg_str = f.read()
        with open(dname + '/' + output_dir + "cfg_" + out_name[:-3] + "txt", 'w') as f: f.write(cfg_str)

    def save_out(self, *args, **kwargs):
        """Write the `.vul` pickle output.

        Canonical signature: `save_out(runstate, dname, photo_static=None,
        cfg=None)`. The legacy `(var, atm, para, dname, ...)` form still
        works.
        """
        # Resolve the polymorphic signature.
        if args and _is_runstate_arg(args[0]):
            return self._save_out_from_runstate(*args, **kwargs)
        return self._save_out_legacy(*args, **kwargs)

    def _save_out_from_runstate(self, runstate, dname,
                                photo_static=None, cfg=None):
        """Canonical .vul writer: reads everything from `runstate`."""
        cfg_mod = cfg if cfg is not None else vulcan_cfg
        var_save, atm_save, para_save = _synthesize_save_dicts(
            runstate, cfg_mod, photo_static=photo_static
        )

        output_dir, out_name = cfg_mod.output_dir, cfg_mod.out_name
        output_file = dname + '/' + output_dir + out_name

        if not os.path.exists(output_dir):
            print('The output directory assigned in vulcan_cfg.py does not exist.')
            print('Directory ', output_dir, " created.")
            os.mkdir(output_dir)

        with open(output_file, 'wb') as outfile:
            if cfg_mod.output_humanread:
                outfile.write(str(
                    {'variable': var_save, 'atm': atm_save, 'parameter': para_save}
                ))
            else:
                pickle.dump(
                    {'variable': var_save, 'atm': atm_save, 'parameter': para_save},
                    outfile, protocol=4,
                )

    def _save_out_legacy(self, var, atm, para, dname, photo_static=None,
                        runstate=None):
        """Legacy `(var, atm, para)` .vul writer; kept for hybrid oracle
        tests that share `(var, atm)` with master."""
        if runstate is not None:
            from state import runstate_to_store as _runstate_to_store
            _runstate_to_store(runstate, var, atm, para)

        output_dir, out_name = vulcan_cfg.output_dir, vulcan_cfg.out_name
        output_file = dname + '/' + output_dir + out_name

        if not os.path.exists(output_dir):
            print ('The output directory assigned in vulcan_cfg.py does not exist.')
            print( 'Directory ' , output_dir,  " created.")
            os.mkdir(output_dir)

        for key in var.var_evol_save:
            as_nparray = np.array(getattr(var, key))
            setattr(var, key, as_nparray)

        var_save = {'species':species, 'nr':nr}

        # Build the photo-static pytree lazily from (var, atm) when not
        # supplied by the caller; the cross-section dicts are then
        # synthesised at pickle time.
        if photo_static is None and bool(getattr(vulcan_cfg, "use_photo", False)):
            import photo_setup as _photo_setup
            photo_static = _photo_setup._build_photo_static_dense(var, atm)
            if hasattr(var, "sflux_din12_indx"):
                photo_static = photo_static.with_din12_indx(
                    int(var.sflux_din12_indx)
                )
        photo_dicts = (
            _synthesize_cross_dicts(photo_static)
            if photo_static is not None else None
        )

        for key in var.var_save:
            if key == 'k':
                # Build the legacy `{i: array(nz)}` dict from var.k_arr at
                # write time so plot_py/ scripts indexing `d['k'][i]` work.
                k_arr = np.asarray(var.k_arr, dtype=np.float64)
                var_save[key] = {i: k_arr[i].copy() for i in range(1, k_arr.shape[0])}
            elif photo_dicts is not None and key in photo_dicts:
                var_save[key] = photo_dicts[key]
            else:
                var_save[key] = getattr(var, key)
        if vulcan_cfg.save_evolution == True:
            # The runner already captured y_time/t_time at save_evo_frq cadence;
            # don't slice [::fq] again here.
            for key in var.var_evol_save:
                var_save[key] = getattr(var, key)

        with open(output_file, 'wb') as outfile:
            if vulcan_cfg.output_humanread == True: # human-readable form, less efficient
                outfile.write(str({'variable': var_save, 'atm': vars(atm), 'parameter': vars(para)}))
            else:
                pickle.dump( {'variable': var_save, 'atm': vars(atm), 'parameter': vars(para) }, outfile, protocol=4)

    # Post-run plotters (live UI lives in live_ui.py). Each saves a PNG to
    # plot_dir, optionally popping it up via PIL when `use_PIL=True`. plt
    # is imported lazily so non-plotting tests don't pull matplotlib in.

    def plot_end(self, var, atm, para):
        plt = _import_plt()
        plot_dir = vulcan_cfg.plot_dir
        colors = ['b','g','r','c','m','y','k','orange','pink','grey',
                  'darkred','darkblue','salmon','chocolate',
                  'mediumspringgreen','steelblue','plum','hotpink']
        ymix = np.asarray(var.ymix)
        plt.figure('mixing ratios')
        for color_index, sp in enumerate(vulcan_cfg.plot_spec):
            color = colors[color_index % len(colors)]
            if vulcan_cfg.plot_height == False:
                plt.plot(ymix[:, species.index(sp)], atm.pco/1.e6,
                         color=color, label=sp)
                plt.gca().set_yscale('log')
                plt.gca().invert_yaxis()
                plt.ylabel("Pressure (bar)")
                plt.ylim((vulcan_cfg.P_b/1.E6, vulcan_cfg.P_t/1.E6))
            else:
                plt.plot(ymix[:, species.index(sp)], atm.zmco/1.e5,
                         color=color, label=sp)
                plt.ylim((atm.zco[0]/1e5, atm.zco[0]/1e5))
                plt.ylabel("Height (km)")
        plt.title(str(para.count) + ' steps and '
                  + "{:.2e}".format(var.t) + ' s')
        plt.gca().set_xscale('log')
        plt.xlim(1.E-20, 1.)
        plt.legend(frameon=0, prop={'size':14}, loc=3)
        plt.xlabel("Mixing Ratios")
        plt.savefig(plot_dir + 'mix2.png')

        # use_live_plot keeps the figure on screen; use_PIL pops the PNG in
        # an external viewer; otherwise close so figures don't accumulate.
        if getattr(vulcan_cfg, "use_live_plot", False):
            plt.draw()
        elif getattr(vulcan_cfg, "use_PIL", False):
            from PIL import Image
            Image.open(plot_dir + 'mix2.png').show()
            plt.close()
        else:
            plt.close()

    def plot_evo(self, var, atm, plot_j=-1, plot_ymin=1e-20, dn=1):
        plt = _import_plt()
        plot_spec = vulcan_cfg.plot_spec
        plot_dir = vulcan_cfg.plot_dir
        plt.figure('evolution')
        ymix_time = np.array(np.asarray(var.y_time)
                             / atm.n_0[:, np.newaxis])
        for i, sp in enumerate(plot_spec):
            plt.plot(np.asarray(var.t_time)[::dn],
                     ymix_time[::dn, plot_j, species.index(sp)],
                     c=plt.cm.rainbow(float(i)/len(plot_spec)),
                     label=sp)
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.xlabel('time')
        plt.ylabel('mixing ratios')
        plt.ylim((plot_ymin, 1.))
        plt.legend(frameon=0, prop={'size':14}, loc='best')
        plt.savefig(plot_dir + 'evo.png')
        plt.close()

    def plot_TP(self, atm):
        plt = _import_plt()
        plot_dir = vulcan_cfg.plot_dir
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        if vulcan_cfg.plot_height == False:
            ax1.semilogy(atm.Tco, atm.pco/1.e6, c='black')
            ax2.loglog(atm.Kzz, atm.pico[1:-1]/1.e6, c='k', ls='--')
            plt.gca().invert_yaxis()
            plt.ylim((vulcan_cfg.P_b/1.E6, vulcan_cfg.P_t/1.E6))
            ax1.set_ylabel("Pressure (bar)")
        else:
            ax1.plot(atm.Tco, atm.zmco/1.e5, c='black')
            ax2.semilogx(atm.Kzz, atm.zmco[1:]/1.e5, c='k', ls='--')
            ax1.set_ylabel("Height (km)")
        ax1.set_xlabel("Temperature (K)")
        ax2.set_xlabel(r'K$_{zz}$ (cm$^2$s$^{-1}$)')
        plt.savefig(plot_dir + 'TPK.png')
        plt.close()
