"""Vendored I/O classes from VULCAN-master/op.py.

Source: ../VULCAN-master/op.py.
Contents:
- class ReadRate (op.py:49-781): network parser + rate-coef builder + photo
  bins/cross-section loader. Called from vulcan_jax.py for one-shot pre-loop
  setup (rate-constant assembly + photo cross-section binning).
- class Output (op.py:3131-3260, plotting methods + matplotlib import dropped):
  config copy + .vul writer + per-step progress printer. Called from
  vulcan_jax.py and outer_loop.OuterLoop.print_prog.

Verbatim copy of upstream method bodies. To re-sync: replace each method's
body with the corresponding op.py block; do not modify in-place.
"""

import numpy as np
import os
import pickle
import time

import vulcan_cfg
import chem_funs
from chem_funs import ni, nr  # number of species and reactions in the network
from vulcan_cfg import nz

species = chem_funs.spec_list


class ReadRate(object):
    """
    to read in rate constants from the network file and compute the reaction rates for the corresponding Tco and pco
    """

    def __init__(self):

        self.i = 1
        # flag of trimolecular reaction
        self.re_tri, self.re_tri_k0 = False, False
        self.list_tri = []


    def read_rate(self, var, atm):

        # Phase 22d retired `var.k`; the local `k` dict here is scratch
        # space the parser fills as it walks the network. The final values
        # are discarded -- `rates.setup_var_k` recomputes the dense rate
        # array from scratch right after this method returns. Kept for
        # bytewise compatibility with the upstream parser body below.
        k = {}
        Rf, Rindx, a, n, E, a_inf, n_inf, E_inf, k_fun, k_inf, kinf_fun, k_fun_new, pho_rate_index = \
        var.Rf, var.Rindx, var.a, var.n, var.E, var.a_inf, var.n_inf, var.E_inf, var.k_fun, var.k_inf, var.kinf_fun, var.k_fun_new,\
         var.pho_rate_index
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

                    # Note: make the defaut i=i
                    k_fun[i] = lambda temp, mm, i=i: a[i] *temp**n[i] * np.exp(-E[i]/temp)


                    if re_tri == False:
                        k[i] = k_fun[i](Tco, M)

                    # for 3-body reactions, also calculating k_inf
                    elif re_tri == True and len(columns)>=6:


                        kinf_fun[i] = lambda temp, i=i: a_inf[i] *temp**n_inf[i] * np.exp(-E_inf[i]/temp)
                        k_fun_new[i] = lambda temp, mm, i=i: (a[i] *temp**n[i] * np.exp(-E[i]/temp))/(1 + (a[i] *temp**n[i] * np.exp(-E[i]/temp))*mm/(a_inf[i] *temp**n_inf[i] * np.exp(-E_inf[i]/temp)) )

                        #k[i] = k_fun_new[i](Tco, M)
                        k_inf = a_inf[i] *Tco**n_inf[i] * np.exp(-E_inf[i]/Tco)
                        k[i] = k_fun[i](Tco, M)
                        k[i] = k[i]/(1 + k[i]*M/k_inf )


                    else: # for 3-body reactions without high-pressure rates
                        k[i] = k_fun[i](Tco, M)


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

                        k_fun[i] = lambda temp, mm, i=i: 1.932E3 *temp**-9.88 *np.exp(-7544./temp) + 5.109E-11*temp**-6.25 *np.exp(-1433./temp)
                        kinf_fun[i] = lambda temp, mm, i=i: 1.031E-10 * temp**-0.018 *np.exp(16.74/temp)
                        k_fun_new[i] = lambda temp, mm, i=i: (1.932E3 *temp**-9.88 *np.exp(-7544./temp) + 5.109E-11*temp**-6.25 *np.exp(-1433./temp))/\
                        (1 + (1.932E3 *temp**-9.88 *np.exp(-7544./temp) + 5.109E-11*temp**-6.25 *np.exp(-1433./temp)) * mm / (1.031E-10 * temp**-0.018 *np.exp(16.74/temp)) )

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

        k_fun.update(k_fun_new)

        # Phase 22d retired the dict-keyed `var.k` surface. The values
        # parsed here are immediately superseded by
        # `rates.setup_var_k -> rates.build_rate_array`, which writes the
        # dense `var.k_arr`. We keep the closures (`k_fun`, `kinf_fun`)
        # because legacy_io still emits them as part of `read_rate`'s
        # metadata return -- nothing downstream reads them post-22d.
        var.k_fun = k_fun
        var.kinf_fun = kinf_fun

        var.photo_sp = set(photo_sp)
        if vulcan_cfg.use_ion == True: var.ion_sp = set(ion_sp)

        return var



    # Phase 22e: `make_bins_read_cross` retired. The dense
    # `PhotoStaticInputs` pytree built by `photo_setup.populate_photo`
    # / `photo_setup._build_photo_static_dense` is the canonical
    # photo-input surface; the .vul writer synthesizes the legacy dict
    # views from it at pickle time (see `_synthesize_cross_dicts` below).
    # Tests that need master's dict view continue to use master's
    # `op.ReadRate().make_bins_read_cross` from sys.path.

def _import_plt():
    """Lazy import of matplotlib so tests that don't plot don't pay the cost."""
    import matplotlib
    matplotlib.use("Agg" if os.environ.get("VULCAN_HEADLESS_PLOT") else
                   matplotlib.get_backend())
    import matplotlib.pyplot as plt
    return plt


def _synthesize_cross_dicts(static) -> dict:
    """Build the legacy `var.cross*` dict views from a `PhotoStaticInputs`.

    Phase 22e: the .vul writer keeps publishing the same six photo dict
    keys the upstream `plot_py/` scripts index (`d['variable']['cross']
    [sp]`, `d['variable']['cross_J'][(sp,i)]`, etc.); the dicts are now
    rebuilt from the dense pytree at pickle time instead of read off
    `var`. Every value is wrapped in `np.asarray(..., dtype=np.float64)`
    so downstream consumers see plain ndarrays (not jax arrays) — same
    pattern as the Phase 22d k-synthesizer above.
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


class Output(object):
    """Vendored from VULCAN-master/op.py:3131-3454.

    Phase 19: live-output paths (plot_update / plot_flux_update /
    plot_evo_inter, save-movie hooks) were dropped from the supported
    surface; only the post-run plotters (plot_end / plot_evo / plot_TP)
    remain. Set `VULCAN_HEADLESS_PLOT=1` in the env to force the Agg
    backend (useful in CI).
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
        # Vendored from VULCAN-master/op.py:3160-3177.
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
        # Vendored from VULCAN-master/op.py:3179-3204.
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

    def save_out(self, var, atm, para, dname, photo_static=None):
        output_dir, out_name = vulcan_cfg.output_dir, vulcan_cfg.out_name
        output_file = dname + '/' + output_dir + out_name

        if not os.path.exists(output_dir):
            print ('The output directory assigned in vulcan_cfg.py does not exist.')
            print( 'Directory ' , output_dir,  " created.")
            os.mkdir(output_dir)

        # convert lists into numpy arrays
        for key in var.var_evol_save:
            as_nparray = np.array(getattr(var, key))
            setattr(var, key, as_nparray)

        # making the save dict
        var_save = {'species':species, 'nr':nr}

        # Phase 22e: when use_photo, the cross-section dict surface is
        # synthesized from the dense `PhotoStaticInputs` pytree at pickle
        # time. Callers that don't supply `photo_static` get a lazy build
        # from `(var, atm)` so legacy save_out callsites keep working
        # without explicit plumbing.
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
                # Phase 22d: synthesize the legacy `{i: array(nz)}` dict
                # view from the dense `var.k_arr` at write time so the .vul
                # schema (and plot_py/ scripts that index `d['k'][i]`) keep
                # working without the deprecated `var.k` attribute.
                k_arr = np.asarray(var.k_arr, dtype=np.float64)
                var_save[key] = {i: k_arr[i].copy() for i in range(1, k_arr.shape[0])}
            elif photo_dicts is not None and key in photo_dicts:
                var_save[key] = photo_dicts[key]
            else:
                var_save[key] = getattr(var, key)
        if vulcan_cfg.save_evolution == True:
            # The JAX OuterLoop already captures (y, t) at the configured
            # `save_evo_frq` cadence into var.y_time / var.t_time during
            # the run, so the master-style `[::fq]` re-slice would be a
            # double-slice. Just publish what the runner produced.
            for key in var.var_evol_save:
                var_save[key] = getattr(var, key)

        with open(output_file, 'wb') as outfile:
            if vulcan_cfg.output_humanread == True: # human-readable form, less efficient
                outfile.write(str({'variable': var_save, 'atm': vars(atm), 'parameter': vars(para)}))
            else:
                pickle.dump( {'variable': var_save, 'atm': vars(atm), 'parameter': vars(para) }, outfile, protocol=4)

    # ---- Post-run plotting (op.py:3262-3454, pruned to non-live methods) ----
    # Phase 19: live-window paths (plot_update / plot_flux_update /
    # plot_evo_inter, use_live_plot draw, use_PIL pop-ups, save-movie hooks)
    # were dropped. The remaining methods are post-run only — they save a
    # PNG to `plot_dir` and return. `plt` is imported lazily so non-plotting
    # tests don't pull matplotlib in.

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
        plt.savefig(plot_dir + 'mix.png')
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
