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
import scipy
from scipy import interpolate
import os
import pickle

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

        Rf, Rindx, a, n, E, a_inf, n_inf, E_inf, k, k_fun, k_inf, kinf_fun, k_fun_new, pho_rate_index = \
        var.Rf, var.Rindx, var.a, var.n, var.E, var.a_inf, var.n_inf, var.E_inf, var.k, var.k_fun, var.k_inf, var.kinf_fun, var.k_fun_new,\
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

        # store k into data_var
        # remeber k_fun has not removed reactions from remove_list
        var.k = k
        var.k_fun = k_fun
        var.kinf_fun = kinf_fun

        var.photo_sp = set(photo_sp)
        if vulcan_cfg.use_ion == True: var.ion_sp = set(ion_sp)

        return var


    def rev_rate(self, var, atm):

        rev_list = range(2,  var.stop_rev_indx, 2)
        # setting the rest reversal zeros
        for i in range(var.stop_rev_indx+1, nr+1,2):
            var.k[i] = np.zeros(nz)

        Tco = atm.Tco.copy()

        # reversing rates and storing into data_var
        print ('Reverse rates from R1 to R' + str(var.stop_rev_indx-2))
        print ('Rates greater than 1e-6:')
        for i in rev_list:
            if i in vulcan_cfg.remove_list:
                 var.k[i] = np.repeat(0.,nz)
            else:
                var.k_fun[i] = lambda temp, mm, i=i: var.k_fun[i-1](temp, mm)/chem_funs.Gibbs(i-1,temp)
                var.k[i] = var.k[i-1]/chem_funs.Gibbs(i-1,Tco)

            if np.any(var.k[i] > 1.e-6): print ('R' + str(i) + " " + var.Rf[i-1] +' :  ' + str(np.amax(var.k[i])) )
            if np.any(var.k[i-1] > 1.e-6): print ('R' + str(i-1) + " " + var.Rf[i-1] + ' :  ' + str(np.amax(var.k[i-1])) )

        return var


    def remove_rate(self, var):

        for i in vulcan_cfg.remove_list:
            var.k[i] = np.repeat(0.,nz)
            var.k_fun[i] = lambda temp, mm, i=i: np.repeat(0.,nz)

        return var

    def lim_lowT_rates(self, var, atm): # for setting up the lower limit of rate coefficients for low T
        for i in range(1,nr,2):
            if var.Rf[i] == 'H + CH3 + M -> CH4 + M':
                T_mask = atm.Tco <= 277.5
                k0 = 6e-29; kinf = 2.06E-10 *atm.Tco**-0.4 # from Moses+2005
                lowT_lim = k0 / (1. + k0*atm.M/kinf)
                print ("using the low temperature limit for CH3 + H + M -> CH4 + M")
                print ("capping "); print (var.k[i][T_mask]); print ("at "); print (lowT_lim[T_mask])
                var.k[i][T_mask] =  lowT_lim[T_mask]

            elif var.Rf[i] == 'H + C2H4 + M -> C2H5 + M':
                T_mask = atm.Tco <= 300
                print ("using the low temperature limit for H + C2H4 + M -> C2H5 + M")
                print ("capping "); print (var.k[i][T_mask]); print ("at "); print (3.7E-30)
                var.k[i][T_mask] = 3.7E-30 # from Moses+2005

            elif var.Rf[i] == 'H + C2H5 + M -> C2H6 + M':
                T_mask = atm.Tco <= 200
                print ("using the low temperature limit for H + C2H5 + M -> C2H6 + M")
                print ("capping "); print (var.k[i][T_mask]); print ("at "); print (2.49E-27)
                var.k[i][T_mask] = 2.49E-27 # from Moses+2005

        return var

    def make_bins_read_cross(self,var,atm):
        '''
        determining the bin range and only use the min and max wavelength that the molecules absorb
        to avoid photons with w0=1 (pure scatteing) in certain wavelengths
        var.cross stores the total absorption cross sections of each species, e.g. var.cross['H2O']
        var.cross stores the IDIVIDUAL photodissociation cross sections for each bracnh, e.g. var.cross_J[('H2O',1)], which is equvilent to var.cross['H2O'] times the branching ratio of branch 1
        '''
        photo_sp = list(var.photo_sp)
        ion_sp = list(var.ion_sp)
        absp_sp = photo_sp + ion_sp
        sp0 = photo_sp[0]

        cross_raw, scat_raw = {}, {}
        ratio_raw, ion_ratio_raw = {}, {}
        cross_T_raw = {}

        # In the end, we do not need photons beyond the longest-wavelength threshold from all species (different from absorption)
        sp_label = np.genfromtxt(vulcan_cfg.cross_folder+'thresholds.txt',dtype=str, usecols=0) # taking the first column as labels
        lmd_data = np.genfromtxt(vulcan_cfg.cross_folder+'thresholds.txt', skip_header = 1)[:,1] # discarding the fist column

        # for setting up the wavelength coverage
        threshold = {label: row for label, row in zip(sp_label, lmd_data) if label in species} # only include the species in the current network
        var.threshold = threshold

        # reading in cross sections into dictionaries
        for n, sp in enumerate(absp_sp):

            if vulcan_cfg.use_ion == True:
                try: cross_raw[sp] = np.genfromtxt(vulcan_cfg.cross_folder+sp+'/'+sp+'_cross.csv',dtype=float,delimiter=',',skip_header=1, names = ['lambda','cross','disso','ion'])
                except: print ('\nMissing the cross section from ' + sp); raise
                if sp in ion_sp:
                    try: ion_ratio_raw[sp] = np.genfromtxt(vulcan_cfg.cross_folder+sp+'/'+sp+'_ion_branch.csv',dtype=float,delimiter=',',skip_header=1, names = True)
                    except: print ('\nMissing the ion branching ratio from ' + sp); raise
            else:
                try: cross_raw[sp] = np.genfromtxt(vulcan_cfg.cross_folder+sp+'/'+sp+'_cross.csv',dtype=float,delimiter=',',skip_header=1, names = ['lambda','cross','disso'])
                except: print ('\nMissing the cross section from ' + sp); raise

            # reading in the branching ratios
            if sp in photo_sp: # excluding ion_sp
                try: ratio_raw[sp] = np.genfromtxt(vulcan_cfg.cross_folder+sp+'/'+sp+'_branch.csv',dtype=float,delimiter=',',skip_header=1, names = True)
                except: print ('\nMissing the branching ratio from ' + sp); raise

            # reading in temperature dependent cross sections
            if sp in vulcan_cfg.T_cross_sp:
                T_list = []
                for temp_file in os.listdir("thermo/photo_cross/" + sp + "/"):
                    if temp_file.startswith(sp) and temp_file.endswith("K.csv"):
                        temp = temp_file
                        temp = temp.replace(sp,''); temp = temp.replace('_cross_',''); temp = temp.replace('K.csv','')
                        T_list.append(int(temp) )
                        var.cross_T_sp_list[sp] = T_list
                for tt in T_list:
                    if vulcan_cfg.use_ion == True: # usually the T-dependent cross sections are only measured in the photodissociation-relavent wavelengths so cross_tot = cross_diss
                        cross_T_raw[(sp, tt)] = np.genfromtxt(vulcan_cfg.cross_folder+sp+'/'+sp+'_cross_'+str(tt)+'K.csv',dtype=float,delimiter=',',skip_header=1, names = ['lambda','cross','disso','ion'])
                    else: cross_T_raw[(sp, tt)] = np.genfromtxt(vulcan_cfg.cross_folder+sp+'/'+sp+'_cross_'+str(tt)+'K.csv',dtype=float,delimiter=',',skip_header=1, names = ['lambda','cross','disso'])
                # room-T cross section
                cross_T_raw[(sp, 300)] = cross_raw[sp]
                var.cross_T_sp_list[sp].append(300)

            if cross_raw[sp]['cross'][0] == 0 or cross_raw[sp]['cross'][-1] ==0:
                raise IOError ('\n Please remove the zeros in the cross file of ' + sp)

            if n==0: # the first species
                bin_min = cross_raw[sp]['lambda'][0]
                bin_max = cross_raw[sp]['lambda'][-1]
                # photolysis threshold
                try: diss_max = threshold[sp]
                except: print (sp + " not in threshol.txt"); raise

            else:
                sp_min, sp_max = cross_raw[sp]['lambda'][0], cross_raw[sp]['lambda'][-1]
                if sp_min < bin_min: bin_min = sp_min
                if sp_max > bin_max: bin_max = sp_max
                try:
                    if threshold[sp] > diss_max:
                        diss_max = threshold[sp]
                except: print (sp + " not in threshol.txt"); raise

        # constraining the bin_min and bin_max by the default values defined in store.py
        bin_min = max(bin_min, var.def_bin_min)
        bin_max = min(bin_max, var.def_bin_max, diss_max)
        print ("Input stellar spectrum from " + "{:.1f}".format(var.def_bin_min) + " to " + "{:.1f}".format(var.def_bin_max) )
        print ("Photodissociation threshold: " + "{:.1f}".format(diss_max) )
        print ("Using wavelength bins from " + "{:.1f}".format(bin_min) + " to " +  str(bin_max) )

        var.dbin1 = vulcan_cfg.dbin1
        var.dbin2 = vulcan_cfg.dbin2
        if vulcan_cfg.dbin_12trans >= bin_min and vulcan_cfg.dbin_12trans <= bin_max:
            bins = np.concatenate(( np.arange(bin_min,vulcan_cfg.dbin_12trans, var.dbin1), np.arange(vulcan_cfg.dbin_12trans,bin_max, var.dbin2) ))
        else: bins = np.arange(bin_min,bin_max, var.dbin1)
        var.bins = bins
        var.nbin = len(bins)

        # all variables that depend on the size of nbins
        # the direct beam (staggered)
        var.sflux = np.zeros( (nz+1, var.nbin) )
        # the diffusive flux (staggered)
        var.dflux_u, var.dflux_d = np.zeros( (nz+1, var.nbin) ), np.zeros( (nz+1, var.nbin) )
        # the total actinic flux (non-staggered)
        var.aflux = np.zeros( (nz, var.nbin) )
        # the total actinic flux from the previous calculation
        prev_aflux = np.zeros( (nz, var.nbin) )

        # staggered
        var.tau = np.zeros( (nz+1, var.nbin) )
        # the stellar flux at TOA
        var.sflux_top = np.zeros(var.nbin)


        # read_cross
        # creat a dict of cross section with key=sp and values=bins in data_var
        var.cross = dict([(sp, np.zeros(var.nbin)) for sp in absp_sp ]) # including photo_sp and ion_sp

        # read cross of disscoiation
        var.cross_J = dict([((sp,i), np.zeros(var.nbin)) for sp in photo_sp for i in range(1,var.n_branch[sp]+1)])
        var.cross_scat = dict([(sp, np.zeros(var.nbin)) for sp in vulcan_cfg.scat_sp])

        # for temperature-dependent cross sections
        var.cross_T = dict([(sp, np.zeros((nz, var.nbin) )) for sp in vulcan_cfg.T_cross_sp ])
        var.cross_J_T = dict([((sp,i), np.zeros((nz, var.nbin) )) for sp in vulcan_cfg.T_cross_sp for i in range(1,var.n_branch[sp]+1) ])

        #read cross of ionisation
        if vulcan_cfg.use_ion == True: var.cross_Jion = dict([((sp,i), np.zeros(var.nbin)) for sp in ion_sp for i in range(1,var.ion_branch[sp]+1)])

        for sp in photo_sp: # photodissociation only; photoionization takes a separate branch ratio file
            # for values outside the boundary => fill_value = 0
            inter_cross = interpolate.interp1d(cross_raw[sp]['lambda'], cross_raw[sp]['cross'], bounds_error=False, fill_value=0)
            inter_cross_J = interpolate.interp1d(cross_raw[sp]['lambda'], cross_raw[sp]['disso'], bounds_error=False, fill_value=0)
            inter_ratio = {} # excluding ionization branches

            for i in range(1,var.n_branch[sp]+1): # fill_value extends the first and last elements for branching ratios
                br_key = 'br_ratio_' + str(i)
                try:
                    inter_ratio[i] = interpolate.interp1d(ratio_raw[sp]['lambda'], ratio_raw[sp][br_key], bounds_error=False, fill_value=(ratio_raw[sp][br_key][0],ratio_raw[sp][br_key][-1]))
                except: print("The branches in the network file does not match the branchong ratio file for " + str(sp))

            # using a loop instead of an array because it's easier to handle the branching ratios
            for n, ld in enumerate(bins):
                var.cross[sp][n] = inter_cross(ld)

                # using the branching ratio (from the files) to construct the individual cross section of each branch
                for i in range(1,var.n_branch[sp]+1):
                    var.cross_J[(sp,i)][n] = inter_cross_J(ld) * inter_ratio[i](ld)

            # make var.cross_T[(sp,i)] and var.cross_J_T[(sp,i)] here in 2D array: nz * bins (same shape as tau)
            # T-dependent cross sections are usually only measured in the photodissociation-relavent wavelengths so cross_tot = cross_diss
            if sp in vulcan_cfg.T_cross_sp:

                # T list of species sp that have T-depedent cross sections (inclduing 300 K for inter_cross)
                T_list = np.array(var.cross_T_sp_list[sp])
                max_T_sp = np.amax(T_list)
                min_T_sp = np.amin(T_list)

                for lev, Tz in enumerate(atm.Tco): # looping z

                    Tz_between = False # flag for Tz in between any two elements in T_list
                    # define the interpolating T range
                    if list(T_list[T_list <= Tz]) and list(T_list[T_list > Tz]):
                        Tlow = T_list[T_list <= Tz].max() # closest T in T_list smaller than Tz
                        Thigh = T_list[T_list > Tz].min() # closest T in T_list larger than Tz
                        Tz_between = True

                        # find the wavelength range that are included in both cross_T_raw[(sp,Tlow)] and cross_T_raw[(sp,Thigh)]
                        ld_min = max( cross_T_raw[(sp,Tlow)]['lambda'][0], cross_T_raw[(sp,Thigh)]['lambda'][0] )
                        ld_max = min( cross_T_raw[(sp,Tlow)]['lambda'][-1], cross_T_raw[(sp,Thigh)]['lambda'][-1] )
                        inter_cross_lowT = interpolate.interp1d(cross_T_raw[(sp,Tlow)]['lambda'], cross_T_raw[(sp,Tlow)]['cross'], bounds_error=False, fill_value=0)
                        inter_cross_highT = interpolate.interp1d(cross_T_raw[(sp,Thigh)]['lambda'], cross_T_raw[(sp,Thigh)]['cross'], bounds_error=False, fill_value=0)
                        inter_cross_J_lowT = interpolate.interp1d(cross_T_raw[(sp,Tlow)]['lambda'], cross_T_raw[(sp,Tlow)]['disso'], bounds_error=False, fill_value=0)
                        inter_cross_J_highT = interpolate.interp1d(cross_T_raw[(sp,Thigh)]['lambda'], cross_T_raw[(sp,Thigh)]['disso'], bounds_error=False, fill_value=0)

                        for n, ld in enumerate(bins): # looping bins

                            # not within the T-cross wavelength range
                            if ld < ld_min or ld > ld_max:
                                var.cross_T[sp][lev, n] = var.cross[sp][n]
                                # don't forget the cross_J_T branches
                                for i in range(1,var.n_branch[sp]+1):
                                    var.cross_J_T[(sp,i)][lev, n] = var.cross_J[(sp,i)][n]

                            else:
                                # update: inerpolation in log10 for cross sections and linearly between Tlow and Thigh
                                log_lowT = np.log10(inter_cross_lowT(ld))
                                log_highT = np.log10(inter_cross_highT(ld))
                                if np.isinf(log_lowT ): log_lowT = -100. # replacing -inf with -100
                                if np.isinf(log_highT ): log_highT = -100.

                                inter_T = interpolate.interp1d([Tlow,Thigh], [log_lowT,log_highT], axis=0) # at wavelength ld, interpolating between Tlow and Thigh in log10
                                if inter_T(Tz) == -100: var.cross_T[sp][lev, n] == 0.
                                else: var.cross_T[sp][lev, n] = 10**(inter_T(Tz))

                                # update: inerpolation in log10 for cross sections and linearly between Tlow and Thigh
                                # using the branching ratio (from the files) to construct the individual cross section of each branch
                                for i in range(1,var.n_branch[sp]+1):
                                    J_log_lowT = np.log10(inter_cross_J_lowT(ld))
                                    J_log_highT = np.log10(inter_cross_J_highT(ld))
                                    if np.isinf(J_log_lowT): J_log_lowT = -100. # replacing -inf with -100
                                    if np.isinf(J_log_highT): J_log_highT = -100.

                                    inter_cross_J_T = interpolate.interp1d([Tlow,Thigh], [J_log_lowT,J_log_highT], axis=0)

                                    if inter_cross_J_T(Tz) == -100: var.cross_J_T[(sp,i)][lev, n] = 0.
                                    else: var.cross_J_T[(sp,i)][lev, n] = 10**(inter_cross_J_T(Tz)) * inter_ratio[i](ld) # same inter_ratio[i](ld) as the standard one above


                    elif not list(T_list[T_list < Tz]): # Tz equal or smaller than all T in T_list including 300K (empty list)

                        if min_T_sp == 300:
                            var.cross_T[sp][lev] = var.cross[sp] # using the cross section at room T
                            for i in range(1,var.n_branch[sp]+1):
                                var.cross_J_T[(sp,i)][lev] = var.cross_J[(sp,i)]
                        else: # min_T_sp != 300; T-cross lower than room temperature
                            # the wavelength range of cross_T_raw at T = min_T_sp
                            ld_min, ld_max = cross_T_raw[(sp,min_T_sp)]['lambda'][0], cross_T_raw[(sp,min_T_sp)]['lambda'][-1]
                            inter_cross_lowT = interpolate.interp1d(cross_T_raw[(sp,min_T_sp)]['lambda'], cross_T_raw[(sp,min_T_sp)]['cross'], bounds_error=False, fill_value=0)
                            inter_cross_J_lowT = interpolate.interp1d(cross_T_raw[(sp,min_T_sp)]['lambda'], cross_T_raw[(sp,min_T_sp)]['disso'], bounds_error=False, fill_value=0)
                            for n, ld in enumerate(bins): # looping bins
                                # not within the T-cross wavelength range
                                if ld < ld_min or ld > ld_max:
                                    var.cross_T[sp][lev, n] = var.cross[sp][n]
                                    # don't forget the cross_J_T branches
                                    for i in range(1,var.n_branch[sp]+1):
                                        var.cross_J_T[(sp,i)][lev, n] = var.cross_J[(sp,i)][n]
                                else:
                                    var.cross_T[sp][lev, n] = inter_cross_lowT(ld)
                                    # using the branching ratio (from the files) to construct the individual cross section of each branch
                                    for i in range(1,var.n_branch[sp]+1):
                                        var.cross_J_T[(sp,i)][lev, n] = inter_cross_J_lowT(ld) * inter_ratio[i](ld) # same inter_ratio[i](ld) as the standard one above

                    else: # Tz equal or larger than all T in T_list (empty list)
                        # the wavelength range of cross_T_raw[(sp,Thigh)]

                        if max_T_sp == 300:
                            var.cross_T[sp][lev] = var.cross[sp] # using the cross section at room T
                            for i in range(1,var.n_branch[sp]+1):
                                var.cross_J_T[(sp,i)][lev] = var.cross_J[(sp,i)]
                        else:  # the wavelength range of cross_T_raw at T = max_T_sp
                            ld_min, ld_max = cross_T_raw[(sp,max_T_sp)]['lambda'][0], cross_T_raw[(sp,max_T_sp)]['lambda'][-1]
                            inter_cross_highT = interpolate.interp1d(cross_T_raw[(sp,max_T_sp)]['lambda'], cross_T_raw[(sp,max_T_sp)]['cross'], bounds_error=False, fill_value=0)
                            inter_cross_J_highT = interpolate.interp1d(cross_T_raw[(sp,max_T_sp)]['lambda'], cross_T_raw[(sp,max_T_sp)]['disso'], bounds_error=False, fill_value=0)
                            for n, ld in enumerate(bins): # looping bins
                                # not within the T-cross wavelength range
                                if ld < ld_min or ld > ld_max:
                                    var.cross_T[sp][lev, n] = var.cross[sp][n]
                                    # don't forget the cross_J_T branches
                                    for i in range(1,var.n_branch[sp]+1):
                                        var.cross_J_T[(sp,i)][lev, n] = var.cross_J[(sp,i)][n]
                                else:
                                    var.cross_T[sp][lev, n] = inter_cross_highT(ld)

                                    # using the branching ratio (from the files) to construct the individual cross section of each branch
                                    for i in range(1,var.n_branch[sp]+1):
                                        var.cross_J_T[(sp,i)][lev, n] = inter_cross_J_highT(ld) * inter_ratio[i](ld) # same inter_ratio[i](ld) as the standard one above


        if vulcan_cfg.use_ion == True:
            for sp in ion_sp:
                if sp not in photo_sp:
                    inter_cross = interpolate.interp1d(cross_raw[sp]['lambda'], cross_raw[sp]['cross'], bounds_error=False, fill_value=0)

                inter_cross_Jion = interpolate.interp1d(cross_raw[sp]['lambda'], cross_raw[sp]['ion'], bounds_error=False, fill_value=0)
                ion_inter_ratio = {} # For ionization branches

                for i in range(1,var.ion_branch[sp]+1): # fill_value extends the first and last elements for branching ratios
                    br_key = 'br_ratio_' + str(i)
                    try:
                        ion_inter_ratio[i] = interpolate.interp1d(ion_ratio_raw[sp]['lambda'], ion_ratio_raw[sp][br_key], bounds_error=False, fill_value=(ion_ratio_raw[sp][br_key][0],ion_ratio_raw[sp][br_key][-1]))
                    except: print("The ionic branches in the network file does not match the branchong ratio file for " + str(sp))

                for n, ld in enumerate(bins):
                    # for species noe appeared in photodissociation but only in photoionization, like H
                    if sp not in photo_sp: var.cross[sp][n] = inter_cross(ld)
                    for i in range(1,var.ion_branch[sp]+1):
                        var.cross_Jion[(sp,i)][n] = inter_cross_Jion(ld) * ion_inter_ratio[i](ld)

        # reading in cross sections of Rayleigh Scattering
        for sp in vulcan_cfg.scat_sp:
            scat_raw[sp] = np.genfromtxt(vulcan_cfg.cross_folder + 'rayleigh/' + sp+'_scat.txt',dtype=float,\
            skip_header=1, names = ['lambda','cross'])

            # for values outside the boundary => fill_value = 0
            inter_scat = interpolate.interp1d(scat_raw[sp]['lambda'], scat_raw[sp]['cross'], bounds_error=False, fill_value=0)

            for n, ld in enumerate(bins):
                var.cross_scat[sp][n] = inter_scat(ld)


class Output(object):
    """Vendored from VULCAN-master/op.py:3131-3260, plotting methods dropped.

    matplotlib is not imported here. If `vulcan_cfg.use_plot_evo` or
    `use_plot_end` is set, save_out raises NotImplementedError loudly rather
    than silently skipping. VULCAN-JAX runs with both flags False by default.
    """

    def __init__(self):

        output_dir, out_name, plot_dir = vulcan_cfg.output_dir, vulcan_cfg.out_name, vulcan_cfg.plot_dir

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        if vulcan_cfg.use_save_movie == True:
            if not os.path.exists(vulcan_cfg.movie_dir): os.makedirs(vulcan_cfg.movie_dir)

        if os.path.isfile(output_dir+out_name):
            print ('Warning... the output file: ' + str(out_name) + ' already exists.\n')

    def print_prog(self, var, para):
        indx_max = np.nanargmax(para.where_varies_most)
        print ('Elapsed time: ' +"{:.2e}".format(var.t) + ' || Step number: ' + str(para.count) + '/' + str(vulcan_cfg.count_max) )
        print ('longdy = ' + "{:.2e}".format(var.longdy) + '      || longdy/dt = ' + "{:.2e}".format(var.longdydt) + '  || dt = '+ "{:.2e}".format(var.dt) )
        print ('from nz = ' + str(int(indx_max/ni)) + ' and ' + species[indx_max%ni])
        print ('------------------------------------------------------------------------' )

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

    def save_out(self, var, atm, para, dname):
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

        # plotting paths from VULCAN-master are not implemented in legacy_io.
        # If a config flips these on, fail loudly rather than silently skip.
        if vulcan_cfg.use_plot_evo == True or vulcan_cfg.use_plot_end == True:
            raise NotImplementedError(
                "use_plot_evo / use_plot_end are not implemented in VULCAN-JAX's "
                "legacy_io.Output. Set both to False, or wire matplotlib plotting "
                "back in if needed."
            )

        # making the save dict
        var_save = {'species':species, 'nr':nr}

        for key in var.var_save:
            var_save[key] = getattr(var, key)
        if vulcan_cfg.save_evolution == True:
            # slicing time-sequential data to reduce ouput filesize
            fq = vulcan_cfg.save_evo_frq
            for key in var.var_evol_save:
                as_nparray = getattr(var, key)[::fq]
                setattr(var, key, as_nparray)
                var_save[key] = getattr(var, key)

        with open(output_file, 'wb') as outfile:
            if vulcan_cfg.output_humanread == True: # human-readable form, less efficient
                outfile.write(str({'variable': var_save, 'atm': vars(atm), 'parameter': vars(para)}))
            else:
                pickle.dump( {'variable': var_save, 'atm': vars(atm), 'parameter': vars(para) }, outfile, protocol=4)
