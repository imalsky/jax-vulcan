# ==============================================================================
# Module contains classes that store all the variables used in VULCAN.
# Copyright (C) 2016 Shang-Min Tsai (Shami)
# ==============================================================================
# There are three classes: Variables, AtmData, and Parameters, which store the
# main physical variables, atmospheric variables, and numerical parameters,
# respectively.
#
# Phase 19: the typed pre-loop input schema is defined in `state.py`
# (`AtmInputs` / `RateInputs` / `PhotoInputs` / `RunState`) and is the public
# format that Phases 20+ will produce from JAX-native setup code. The classes
# below remain the legacy mutable containers; bridging adapters
# (`pytree_from_store` / `apply_pytree_to_store`) live in `state.py`. The
# per-step accumulator fields (`y_time`, `t_time`, `dy_time`, retry counters)
# are intentionally kept on the numpy host side — the runner does not read
# them and they are written only at host-sync points.
# ==============================================================================

import numpy as np
import scipy
import vulcan_cfg
from vulcan_cfg import nz
from chem_funs import ni, nr, spec_list  # number of species and reactions in the network

#from numba import jitclass
#from numba import f8 # f8: float64 = double

#spec_var = [('y',f8),('ymix',f8)  ]

#@jitclass(spec_var)
class Variables(object):
    """
    store the essential variables for calculation
    """
    def __init__(self, stellar_flux=None): # self means the created object instance
        # Phase 19: the historical `np.genfromtxt(vulcan_cfg.sflux_file, ...)`
        # block was extracted into `state.load_stellar_flux(cfg)`. Callers
        # can pre-load and pass the result via `stellar_flux=...` for the
        # "no host I/O in __init__" contract that the typed pytrees rely
        # on; legacy callers that pass nothing still get the file read here
        # so the existing test surface keeps working.
        # Phase 22d retired the dict-keyed `var.k` surface from VULCAN-JAX's
        # production path. The dense `(nr+1, nz)` `var.k_arr` is the source
        # of truth; the `.vul` writer synthesizes the legacy
        # `{i: array(nz)}` dict view at pickle time so plot_py/ scripts
        # that index `d['variable']['k'][i]` keep working unchanged.
        # `self.k = {}` is retained as an inert empty dict so oracle tests
        # that build state with VULCAN-JAX's `store.Variables()` and then
        # hand it to VULCAN-master's `op.ReadRate.read_rate` (which reads
        # `var.k` for local binding) continue to work; VULCAN-JAX's own
        # path neither reads nor writes this dict.
        self.k = {}
        self.k_arr = None
        self.y = np.zeros((nz, ni)) # current number density in the shape of (number of vertical levels, number of species)
        self.y_prev = np.zeros((nz, ni)) # number density at the previous step
        self.ymix = np.zeros((nz, ni)) # current mixing ratios
        self.y_ini = np.zeros((nz, ni)) # the initial number density 
        self.t = 0 # integration time
        self.dt = vulcan_cfg.dttry # time step
        self.dy = 1. # the max change of y from the previous step to current step 
        self.dy_prev = 1. # the max change of y from i-2 step to i-1 step 
        self.dydt = 1. # the max change of dy/dt
        self.longdy = 1. # the max change of y over a period of time 
        # for checking the steady state (dn in (10) in Tsai et al 2017)
        self.longdydt = 1. # the max change of dy/dt over a period of time 
        # (dn/dt in (11) in Tsai et al 2017)
        #self.slope_min = 0 # 
        
        self.dy_time = [] # storing dy at each step 
        self.dydt_time = []
        self.atim_loss_time = []
        self.ymix_time = [] # storing the mixing ratio at each step 
        self.y_time = [] # storing the number density at each step 
        self.t_time = [] # storing the time  at each step 
        self.dt_time = [] # storing the time step  at each step 
        self.atom_loss_time = [] # storing the loss of atoms at each step
        
        self.atom_ini = {}
        self.atom_sum = {}
        self.atom_loss = {}
        self.atom_loss_prev = {}
        self.atom_conden = {}  
        
        self.Rf = {}  # reaction equation
        self.Rindx = {}
        self.a, self.n, self.E, self.a_inf, self.n_inf, self.E_inf,= [{} for i in range(6)]
        self.k_fun, self.k_inf = [{} for i in range(2)] 
        self.photo_sp = set()  
        self.pho_rate_index, self.n_branch, self.wavelen = {}, {}, {}
        #if vulcan_cfg.use_ion == True:
        self.ion_rate_index, self.ion_branch, self.ion_wavelen, self.ion_br_ratio = {}, {}, {}, {}
        self.charge_list, self.ion_sp = [], set() # charge_list: list of species with non-zero charge; ion_sp: species subjected to photoionisation
        
        self.kinf_fun = {}
        self.k_fun_new = {}
        
        # the max change of the actinic flux (for convergence)
        # if photochemistry is off, the value remaines 0 for checking convergence
        self.aflux_change = 0.
        
        # The temporary wavelength range (nm) given by the stellar flux.
        # Later narrowed by `make_bins_read_cross` (legacy_io.ReadRate) using
        # the wavelength range of all photo-absorbing molecules.
        if stellar_flux is None:
            from state import load_stellar_flux
            stellar_flux = load_stellar_flux(vulcan_cfg)
        self.def_bin_min = stellar_flux.def_bin_min
        self.def_bin_max = stellar_flux.def_bin_max

        # Define what variables to save in the output file!
        self.var_save = ['k','y','ymix','y_ini','t','dt','longdy','longdydt',\
        'atom_ini','atom_sum','atom_loss','atom_conden','aflux_change','Rf'] 
        if vulcan_cfg.use_photo == True: 
            self.var_save.extend(['nbin','bins','dbin1','dbin2','tau','sflux','aflux','cross','cross_scat','cross_J', 'J_sp','n_branch'])
            if vulcan_cfg.T_cross_sp: self.var_save.extend(['cross_J','cross_T'])
            if vulcan_cfg.use_ion == True: self.var_save.extend(['charge_list', 'ion_sp', 'cross_Jion','Jion_sp', 'ion_wavelen','ion_branch','ion_br_ratio'])
        # 'ion_list' stores all the non-neutral species in build.atm whereas 'ion_sp' is for the species that actually have ionisation reactions in the network 
        self.var_evol_save = ['y_time','t_time']
        self.conden_re_list = []
        
        # Phase 22e: `self.threshold` and `self.cross_T_sp_list` removed
        # from defaults — both are write-once locals during photo setup.
        # `populate_photo` writes `var.threshold = _load_thresholds(...)`
        # before any consumer reads it; `cross_T_sp_list` is now a
        # function-local variable in `photo_setup._build_photo_static_dense`
        # (no readers downstream).
        
        # TEST 
        self.v_ratio = np.ones(nz)
        
        ### ### ### ### ### ### ### ### ### ### ###
        # List the names variables defined in op here!
        ### ### ### ### ### ### ### ### ### ### ###
        
        # Old stuff
        # self.yconv = 1.
        # self.yconv_prev = 1.
        # self.y_conden = np.zeros((nz, ni))  # to store the species removed by condensation      
        # photo data initiated in read_cross in op.py

class AtmData(object):
    """
    store the data of atmospheric structure  
    """
    def __init__(self):
        self.pco = np.logspace(np.log10(vulcan_cfg.P_b),np.log10(vulcan_cfg.P_t),nz) # pressure grids
        self.pico = np.empty(nz+1) # pressure grids at the interface
        self.dz = np.zeros(nz) # height grids
        self.dzi = np.zeros(nz-1) # height grids at the interface
        self.zco = np.zeros(nz+1) # not used in calculation
        self.zmco = np.empty(nz)  
        self.Tco = np.empty(nz)  # temperature grids
        self.Kzz = np.zeros(nz-1) 
        self.vz = np.zeros(nz-1) # vertical velocity
        self.M = np.empty(nz) # the number density of the third body
        self.n_0 = np.empty(nz) # the total number density
        self.Hp = np.empty(nz) # the scale height
        self.mu = np.empty(nz) # mean molecular weight
        self.ms = np.empty(ni) # molecular weight for every species
        self.Dzz = np.zeros((nz-1,ni)) # molecular diffusion (nz,ni)
        self.Dzz_cen = np.zeros((nz,ni)) # molecular diffusion (nz,ni)
        self.vm = np.zeros((nz,ni)) # molecular diffusion (nz,ni)
        self.vs = np.zeros((nz-1,ni)) # the settling velocity
        self.alpha = np.zeros(ni) # thermal diffusion factor = -0.25 for every species except for H and H2 (defined in mol_diff() in build_atm.py)
        self.gs = vulcan_cfg.gs # the gravitational acceleration at the surface or at 1 bar
        self.g = np.zeros(nz) # g(z) 
        
        self.top_flux = np.zeros(ni) # the assigned flux at the top boundary
        self.bot_flux = np.zeros(ni) # the assigned flux at the bottom boundary
        self.bot_vdep = np.zeros(ni) # the deposition velocity at the bottom boundary (surface sink)
        self.bot_fix_sp = np.zeros(ni) # the fixed mixing ratios for certain species at the bottom boundary
        
        self.sat_p = {}
        self.sat_mix = {}
        
        self.conden_min_lev = {} # the level of cold trap (sp dependent; for fixing the condensation only below the cold trap )
        
        # excluding non-gaseous species while computing ymix from y
        self.gas_indx = [_ for _ in range(ni) if spec_list[_] not in vulcan_cfg.non_gas_sp]
                    
        self.fix_sp_indx = {}
        if hasattr(vulcan_cfg, "fix_species"): # if fix_species is defined in vulcan_cfg
            for sp in vulcan_cfg.fix_species:
                self.fix_sp_indx[sp] = np.arange(spec_list.index(sp), spec_list.index(sp) + ni*nz, ni)
        
        # turning of df(e) while using charge conservation. it'll be very slow if not doing this
        if vulcan_cfg.use_ion == True: self.fix_e_indx = np.arange(spec_list.index('e'), spec_list.index('e') + ni*nz, ni)
        
        # TEST condensation excluding non-gaseous species
        self.r_p, self.rho_p = {}, {}
        if vulcan_cfg.use_condense == True:
            for sp in vulcan_cfg.r_p.keys():
                self.r_p[sp] = vulcan_cfg.r_p[sp]
            for sp in vulcan_cfg.rho_p.keys():
                self.rho_p[sp] = vulcan_cfg.rho_p[sp]  
            
            self.conden_status = np.zeros(nz, dtype=bool)
            
        # self.r_p['H2O_l_s'] = 0.01 # 100 micron
        # self.r_p['H2SO4_l'] = 1e-4 # 1 micron
        # self.r_p['NH3_l_s'] = 5e-5 # 0.5 micron
        # self.r_p['S8_l_s'] = 1e-4 # 1 micron
        # self.r_p['S2_l_s'] = 1e-4 # 1 micron
        # # particle density
        # self.rho_p['H2O_l_s'] = 0.9 # ice
        # self.rho_p['NH3_l_s'] = 0.7 # estimated
        # self.rho_p['H2SO4_l'] = 1.8302 #
        # self.rho_p['S8_l_s'] = 2.07 #
        # self.rho_p['S2_l_s'] = 2.0 # estimated
        

class Parameters(object):
    """
    store the overall parameters for numerical method and counters
    """
    def __init__(self):
        self.nega_y = 0
        self.small_y = 0
        self.delta = 0
        self.count = 0  # number of steps taken
        self.nega_count =0
        self.loss_count = 0
        self.delta_count= 0
        self.end_case = 0
        self.solver_str = '' # for assigning the name of solver
        self.switch_final_photo_frq = False
        self.where_varies_most = np.zeros((nz, ni)) # recording from where and what species flucating from convergence

        self.fix_species_start = False # flag for the start of fixing condensed species
        # Phase 19: `pic_count` and `tableau20` were dropped along with the
        # live-plotting/movie surface. They were only consumed by
        # `Output.plot_update` (now deleted).