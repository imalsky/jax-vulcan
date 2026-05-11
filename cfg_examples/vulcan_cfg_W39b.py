# =============================================================================
# Configuration file of VULCAN — WASP-39b evening terminator
#
# Mirrors Tsai et al. 2023 / Wogan et al. 2024 Figure 10b setup. Settings that
# differ from VULCAN's canonical hot-Jupiter cfg (cfg_examples/cfg_HD209.txt)
# are W39b-specific physics (gs, sl_angle, abundances, network, atm file) —
# numerical knobs are kept at canonical defaults so the run is comparable to
# the published Tsai/Wogan results.
# =============================================================================

# ====== Setting up the elements included in the network ======
atom_list = ['H', 'O', 'C', 'N', 'S']

# ====== Setting up paths and filenames for the input and output files  ======
network = 'thermo/SNCHO_photo_network.txt'
use_lowT_limit_rates = False
gibbs_text = 'thermo/gibbs_text.txt'
cross_folder = 'thermo/photo_cross/'
com_file = 'thermo/all_compose.txt'
atm_file = 'atm/atm_W39b_evening_TP_Kzz.txt'
sflux_file = 'atm/stellar_flux/sflux-W39b_Tsai2023.txt'
top_BC_flux_file = 'atm/BC_top.txt'
bot_BC_flux_file = 'atm/BC_bot.txt'
vul_ini = 'output/W39b-nominal.vul'
output_dir = 'output/'
plot_dir = 'plot/'
movie_dir = 'plot/movie/'
out_name =  'W39b.vul'

# ====== Setting up the elemental abundance ======
# 10x solar Lodders 2019 (matches Tsai 2023 / Wogan 2024 GasGiants notebook):
#   Solar (Lodders 2019): O=5.37e-4, C=2.95e-4, N=7.08e-5, S=1.41e-5,
#                         He=0.0838 (rel H)
# 10x solar => C/N/O/S scaled by 10, He unchanged.
#
use_solar = False
O_H = 5.37E-3
C_H = 2.95E-3
N_H = 7.08E-4
S_H = 1.41E-4
He_H = 0.0838
ini_mix = 'EQ'
fastchem_met_scale = 10.
fastchem_solar_abundance_file = 'fastchem_vulcan/input/solar_element_abundances_W39b.dat'

const_mix = {'CH4': 2*C_H, 'O2': 4.807e-4, 'He': He_H, 'N2': 0.5*N_H, 'H2': 1. - 8*C_H}

# ====== Setting up photochemistry ======
use_photo = True
r_star = 0.932               # solar radii (Mancini 2018)
Rp = 1.279 * 7.1492E9        # cm  (1.279 R_jup)
orbit_radius = 0.04828       # AU
sl_angle = 83 / 180. * 3.14159  # 83 deg (Tsai 2023)
f_diurnal = 1.
scat_sp = ['H2', 'He']
T_cross_sp = []

edd = 0.5
dbin1 = 0.1
dbin2 = 2.
dbin_12trans = 240.

# No photo_frq ramp on W39b. Canonical HD209 default is 100→5 (coarse
# photolysis early, fine photolysis near steady state) for speed, but on
# W39b that mid-run switch causes a `longdy` transient (master saw 0.47
# → 4.5e+3 between steps ~1000 and 1500) that bifurcates master's and
# JAX's iteration trajectories. Holding photo_frq=5 throughout costs ~30%
# more wall time but keeps both codes on the same trajectory.
ini_update_photo_frq = 5
final_update_photo_frq = 5

# ====== Setting up ionchemistry ======
use_ion = False
if not use_photo and use_ion:
    print('Warning: use_ion = True but use_photo = False')

# ====== Setting up parameters for the atmosphere ======
atm_base = 'H2'
rocky = False
nz = 150                     # canonical default
P_b = 7.6e6                  # 7.6 bar — bottom of the Tsai-2023-ready atm file
# P_t = 0.1 dyne/cm^2 = 1e-7 bar matches photochem GasGiant's
# TOA_pressure_avg=1e-7 bar default (set in extensions/gasgiants.py:39).
# VULCAN extends the atm above the climate-file top (5.5 dyne) isothermally
# at T=725.8K (last layer's T) — see build_atm.py:366. The extension covers
# the photochem region (1e-7..1e-5 bar) where Wogan Fig 10b plots SO/SO2.
P_t = 0.1
use_Kzz = True
use_moldiff = True
use_vm_mol = False
use_vz = False
atm_type = 'file'
Kzz_prof = 'file'
K_max = 1e5
K_p_lev = 0.1
vz_prof = 'const'
# Surface gravity for Mp = 0.28 M_jup, Rp = 1.279 R_jup:
# g = G * Mp / Rp^2 = 6.674e-8 * (0.28 * 1.898e30) / (1.279 * 7.1492e9)^2
gs = 422.
Tiso = 1000
para_warm = [120., 1500., 0.1, 0.02, 1., 1.]
para_anaTP = para_warm
const_Kzz = 1.E10
const_vz = 0

update_frq = 100

# ====== Setting up the boundary conditions ======
use_topflux = False
use_botflux = False
use_fix_sp_bot = {}
# diff_esc = [] (NOT ['H']). To make this an apples-to-apples comparison
# with photochem's GasGiant default, we match photochem's TOA boundary
# (`hydrogen-escape: type: none` in gasgiants.py:953 SETTINGS_TEMPLATE,
# i.e. zero-flux veff=0 for all species). VULCAN's HD189/HD209 default
# `diff_esc=['H']` is a stricter setup; using it here would give VULCAN
# a sink that photochem doesn't have, and the comparison would not be
# meaningful.
diff_esc = []
max_flux = 1e13

# ====== Reactions to be switched off  ======
remove_list = []

# == Condensation ======
use_relax = []
use_condense = False
use_settling = False
start_conden_time = 0
stop_conden_time = 1e5
condense_sp = []
non_gas_sp = []
r_p = {'H2O_l_s': 5e-3}
rho_p = {'H2O_l_s': 1}
fix_species = []
fix_species_time = 0
fix_species_from_coldtrap_lev = True
humidity = 1.
use_ini_cold_trap = False   # apply cold-trap clip to ini abundances (only used when use_condense=True)
use_sat_surfaceH2O = False  # pin surface H2O to saturation pressure (terrestrial atmospheres)

# ====== steady state check ======
st_factor = 0.5
conv_step = 500
conv_stall_window = 200

# ====== Setting up numerical parameters for the ODE solver ======
ode_solver = 'Ros2'
use_print_prog = True
use_print_delta = False
print_prog_num = 500
dttry = 1.E-10
trun_min = 1e2
runtime = 1.E22
# 30-min wall-clock backstop (end_case=4). Tightened yconv_min (see below)
# requires both codes to integrate the upper-atmosphere photochemistry
# cascade, which roughly doubles master's wall time vs the loose default.
wall_clock_max = 1800.0
dt_min = 1.E-14
dt_max = runtime*1e-5
dt_var_max = 2.
dt_var_min = 0.5
count_min = 120
count_max = int(3E4)            # canonical default
atol = 1.E-1
mtol = 1.E-22
mtol_conv = 1.E-20
pos_cut = 0
nega_cut = -1.
loss_eps = 1e-1
yconv_cri = 0.01                # canonical default
slope_cri = 1.e-4
yconv_min = 0.1                 # canonical default. Master uses 0.1 on
                                # W39b. The codegen chem_rhs (see
                                # VULCAN-JAX/CLAUDE.md "Numerical hygiene")
                                # would in principle allow tightening to
                                # 0.01, but kept at 0.1 for parity with
                                # master.
flux_cri = 0.1
flux_atol = 1.
# Canonical heavy-hydrocarbon ULP-floor offenders (per VULCAN-JAX/CLAUDE.md
# "Step-count and atom-conservation drift" note). Same list as the JAX
# vulcan_cfg_HD209.py default. Don't extend without proof a *new* species is
# gating; doing so silently masks bugs.
conver_ignore = ['C6H6', 'C2H2', 'C6H5', 'C2H', 'C2H4', 'C2H5', 'C2H6',
                 'C3H2', 'C3H3', 'C4H5', 'CH2NH', 'CH3NH2', 'H2CCO']

# ====== Setting up numerical parameters for Ros2 ODE solver ======
rtol = 0.2
post_conden_rtol = 0.1

# Adaptive rtol controller (only fires when use_adapt_rtol=True).
use_adapt_rtol = False
rtol_min = 0.0
rtol_max = 1.0
adapt_rtol_dec_period = 10
adapt_rtol_inc_period = 1000
adapt_rtol_dec = 0.75
adapt_rtol_inc = 1.25
adapt_rtol_loss_mul = 2.0
adapt_rtol_inc_loss_thresh = 2e-4

# Per-step retry cap inside the JIT'd runner.
batch_max_retries = 64

# Adaptive Ros2 step-size knobs.
step_size_safety = 0.9
step_size_zero_delta_frac = 0.01

# Photo-frequency switch thresholds.
photo_switch_longdy_thresh = yconv_min * 10.0
photo_switch_longdydt_thresh = 1e-6

hycean_pin_time = 1e6
loss_ex = []

# FastChem-driven equilibrium init Newton solver.
fastchem_newton_tol = 1e-12
fastchem_newton_max_iter = 50

use_fix_all_bot = False
use_fix_H2He = False
use_chunked_runner = False

# ====== plotting / output ======
plot_TP = False
use_live_plot = False
use_live_flux = False
use_plot_end = False
use_plot_evo = False
use_save_movie = False
use_flux_movie = False
plot_height = False
use_PIL = False
live_plot_frq = 10
save_movie_rate = live_plot_frq
y_time_freq = 1
plot_spec = ['H2O', 'H', 'CH4', 'CO', 'CO2', 'SO2', 'H2S', 'NH3']
output_humanread = False
use_shark = False
save_evolution = False
save_evo_frq = 10
