# =============================================================================
# Configuration file of VULCAN:
# =============================================================================

import os

# ====== Setting up the elements included in the network ======
# atom_list is import-frozen (jax_step bakes the reservoir-projection tables
# from it at the first `import vulcan_jax`). To run a different element set
# in-process (e.g. a sulfur network for WASP-39b SO2), set
# $VULCAN_JAX_ATOM_LIST (comma-separated) before that first import — the same
# pattern as $VULCAN_JAX_NETWORK below.
atom_list = [
    s for s in os.environ.get("VULCAN_JAX_ATOM_LIST", "H,O,C,N").split(",") if s
]
# ====== Setting up paths and filenames for the input and output files  ======
# input:
# The reaction network is parsed ONCE at import time (chem_funs / outer_loop
# module level), so it cannot be changed by mutating this attribute after
# `import vulcan_jax`. Callers that drive VULCAN-JAX in-process with a
# non-default network — e.g. the emulator's GPU-batched generation, which
# imports the package only once — set $VULCAN_JAX_NETWORK before that first
# import. The subprocess driver still overwrites this line with a literal.
network = os.environ.get("VULCAN_JAX_NETWORK", "thermo/NCHO_photo_network.txt")
use_lowT_limit_rates = False
gibbs_text = "thermo/gibbs_text.txt"  # (all the nasa9 files must be placed in the folder: thermo/NASA9/)
cross_folder = "thermo/photo_cross/"
com_file = "thermo/all_compose.txt"
atm_file = "atm/atm_HD189_Kzz.txt"  # TP and Kzz (optional) file
sflux_file = "atm/stellar_flux/sflux-HD189_Moses11.txt"  # the flux density at the stellar surface (alternative file: sflux-HD189_B2020.txt)
top_BC_flux_file = "atm/BC_top.txt"  # the file for the top boundary conditions
bot_BC_flux_file = "atm/BC_bot.txt"  # the file for the lower boundary conditions
vul_ini = "output/HD189-nominal.vul"  # the file to initialize the abundances for ini_mix = 'vulcan_ini'
# output:
output_dir = "output/"
plot_dir = "plot/"
movie_dir = "plot/movie/"
out_name = "HD189.vul"  # output file name

# ====== Setting up the elemental abundance ======
use_solar = True  # True: using the solar abundance from Table 10. K.Lodders 2009; False: using the customized elemental abundance.
# customized elemental abundance (only read when use_solar = False)
O_H = 6.0618e-4  # *(0.793)
C_H = 2.7761e-4
N_H = 8.1853e-5
S_H = 1.3183e-5
He_H = 0.09692
ini_mix = "EQ"  # Options: 'EQ', 'const_mix', 'vulcan_ini', 'table' (for 'vulcan_ini, the T-P grids have to be exactly the same)
fastchem_met_scale = 1.0  # scaling factor for other elements in fastchem (e.g., if fastchem_met_scale = 0.1, other elements such as Si and Mg will take 0.1 solar values)
fastchem_solar_abundance_file = "fastchem_vulcan/input/solar_element_abundances.dat"

# Initialsing uniform (constant with pressure) mixing ratios (only reads when ini_mix = const_mix)
const_mix = {
    "CH4": 2.7761e-4 * 2,
    "O2": 4.807e-4,
    "He": 0.09691,
    "N2": 8.1853e-5,
    "H2": 1.0 - 2.7761e-4 * 2 * 4 / 2,
}

# ====== Setting up photochemistry ======
use_photo = True
# astronomy input
r_star = 0.805  # stellar radius in solar radius
Rp = 1.138 * 7.1492e9  # Planetary radius (cm) (for computing gravity)
orbit_radius = 0.03142  # planet-star distance in A.U.
sl_angle = (
    48 / 180.0 * 3.14159
)  # zenith angle of the star, stored in radians (here 48 deg; ~58 deg is a common dayside-average choice)
f_diurnal = 1.0  # to account for the diurnal average of solar flux (i.e. 0.5 for Earth; 1 for tidally-locked planets)
scat_sp = ["H2", "He"]  # the bulk gases that contribute to Rayleigh scattering
T_cross_sp = []  # warning: slower start! available atm: 'CO2','H2O','NH3', 'SH','H2S','SO2', 'S2', 'COS', 'CS2'

edd = 0.5  # the Eddington coefficient
dbin1 = 0.1  # the uniform bin width < dbin_12trans (nm)
dbin2 = 2.0  # the uniform bin width > dbin_12trans (nm)
dbin_12trans = 240.0  # the wavelength switching from dbin1 to dbin2 (nm)

# the frequency to update the actinic flux and optical depth
ini_update_photo_frq = 100
final_update_photo_frq = 5

# ====== Setting up ionchemistry ======
use_ion = False
if not use_photo and use_ion:
    print("Warning: use_ion = True but use_photo = False")
# photoionization needs to run together with photochemistry


# ====== Setting up parameters for the atmosphere ======
atm_base = "H2"  # Options: 'H2', 'N2', 'O2', 'CO2 -- the bulk gas of the atmosphere: changes the molecular diffsion, thermal diffusion factor, and settling velocity
rocky = False  # for the surface gravity
nz = 150  # number of vertical layers
P_b = 1e9  # pressure at the bottom (dyne/cm^2)
P_t = 1e-2  # pressure at the top (dyne/cm^2)
use_Kzz = True
use_moldiff = True
use_vm_mol = False  # use upwind scheme for molecular diffusion -- under testing
use_vz = False
atm_type = (
    "file"  # Options: 'isothermal', 'analytical', 'file', or 'vulcan_ini' 'table'
)
Kzz_prof = "file"  # Options: 'const','file' or 'Pfunc' (Kzz increased with P^-0.4)
K_max = 1e5  # for Kzz_prof = 'Pfunc'
K_p_lev = 0.1  # for Kzz_prof = 'Pfunc'
vz_prof = "const"  # Options: 'const' or 'file'
gs = 2140.0  # surface gravity (cm/s^2)  (HD189:2140  HD209:936)
Tiso = 1000  # only read when atm_type = 'isothermal'
# setting the parameters for the analytical T-P from (126)in Heng et al. 2014. Only reads when atm_type = 'analytical'
# T_int, T_irr, ka_L, ka_S, beta_S, beta_L
para_warm = [120.0, 1500.0, 0.1, 0.02, 1.0, 1.0]
para_anaTP = para_warm
const_Kzz = 1.0e10  # (cm^2/s) Only reads when use_Kzz = True and Kzz_prof = 'const'
const_vz = 0  # (cm/s) Only reads when use_vz = True and vz_prof = 'const'

# frequency for updating dz and dzi due to change of mu
update_frq = 100

# ====== Setting up the boundary conditions ======
# Boundary Conditions:
use_topflux = False
use_botflux = False
use_fix_sp_bot = {}  # fixed mixing ratios at the lower boundary
diff_esc = []  # species for diffusion-limit escape at TOA
max_flux = 1e13  # upper limit for the diffusion-limit fluxes

# ====== Reactions to be switched off  ======
remove_list = []  # in pairs e.g. [1,2]

# == Condensation ======
use_relax = []
use_condense = False
use_settling = False
start_conden_time = 0
stop_conden_time = 1e5  # after this time to fix the condensable species
condense_sp = []
non_gas_sp = []
r_p = {"H2O_l_s": 5e-3}  # particle radius in cm (1e-4 = 1 micron)
rho_p = {"H2O_l_s": 1}  # particle density in g cm^-3
fix_species = []  # fixed the condensable species after condensation-evapoation EQ has reached
fix_species_time = 0
fix_species_from_coldtrap_lev = True
humidity = 1.0
use_ini_cold_trap = (
    False  # apply cold-trap clip to ini abundances (only used when use_condense=True)
)
use_sat_surfaceH2O = (
    False  # pin surface H2O to saturation pressure (terrestrial atmospheres)
)

# ====== steady state check ======
st_factor = 0.5
conv_step = 500
conv_stall_window = 200  # steps without longdy improvement before declaring stalled-convergence (end_case=1)

# ====== Setting up numerical parameters for the ODE solver ======
ode_solver = "Ros2"  # case sensitive
use_print_prog = True
use_print_delta = False
print_prog_num = 500  # print the progress every x steps
dttry = 1.0e-10
trun_min = 1e2
runtime = 1.0e22
dt_min = 1.0e-14
dt_max = runtime * 1e-5
dt_var_max = 2.0
dt_var_min = 0.5
count_min = 120
count_max = int(1e4)
atol = 1.0e-1  # Try decreasing this if the solutions are not stable
mtol = 1.0e-22
mtol_conv = 1.0e-20
pos_cut = 0
nega_cut = -1.0
loss_eps = 1e-1
yconv_cri = 0.01  # for checking steady-state
slope_cri = 1.0e-4
yconv_min = 0.1
flux_cri = 0.1
flux_atol = 1.0  # the tol for actinc flux (# photons cm-2 s-1 nm-1)
conver_ignore = [
    "C6H6",
    "C2H2",
    "C6H5",
    "C2H",
    "C2H4",
    "C2H5",
    "C2H6",
    "C3H2",
    "C3H3",
    "C4H5",
    "CH2NH",
    "CH3NH2",
    "H2CCO",
]  # heavy hydrocarbons that sit on the chem_rhs ULP cancellation floor and stall convergence; safe to exclude for hot-Jupiter bulk-chemistry runs

# ====== Setting up numerical parameters for Ros2 ODE solver ======
rtol = 0.2  # relative tolerence for adjusting the stepsize
post_conden_rtol = 0.1  # switched to this value after fix_species_time

# Adaptive rtol controller (only fires when use_adapt_rtol=True). Decreases
# rtol every adapt_rtol_dec_period accepted steps; conditionally increases
# every adapt_rtol_inc_period accepted steps when max column atom_loss is
# below adapt_rtol_inc_loss_thresh.
use_adapt_rtol = False
rtol_min = 0.0
rtol_max = 1.0
# Adaptive-rtol loss gate: the controller raises rtol only while the max column
# atom-loss sits below loss_criteria, and tightens loss_criteria by
# adapt_rtol_loss_mul when the loss stays high (outer_loop adaptive-rtol block).
loss_criteria = 5e-4
adapt_rtol_dec_period = 10
adapt_rtol_inc_period = 1000
adapt_rtol_dec = 0.75
adapt_rtol_inc = 1.25
adapt_rtol_loss_mul = 2.0
adapt_rtol_inc_loss_thresh = 2e-4

# Per-step retry cap inside the JIT'd runner. Mirrors master's force-accept
# fallback. 64 keeps the cap as a true safety guard while letting master's
# accept-criterion semantics control the trajectory in all reasonable configs.
batch_max_retries = 64

# Adaptive Ros2 step-size knobs (`outer_loop._step_size`).
step_size_safety = 0.9
step_size_zero_delta_frac = 0.01

# Gustafsson (1991) PI step-size controller (Hairer & Wanner II, section IV.2;
# ported from neoVULCAN's ode_solver.step_size). Off by default — pure
# I-control, the master-faithful behavior. When on, accepted steps refine
# h_factor with the previous accepted step's truncation error:
#   h_factor = safety * (rtol/delta)^(alpha/p) * (delta_prev/delta)^(beta/p)
# with p = 2 (the Ros2 error order), falling back to I-control on the first
# step and after any rejection.
use_pi_controller = False
pi_controller_alpha = 0.7
pi_controller_beta = 0.4

# Photo-frequency switch thresholds. `update_photo_frq` ramps from
# ini_update_photo_frq to final_update_photo_frq when both gates trip.
photo_switch_longdy_thresh = yconv_min * 10.0
photo_switch_longdydt_thresh = 1e-6

# Hycean H2/He snapshot pin time. Only active when H2 is not in fix_species.
hycean_pin_time = 1e6

# Atoms excluded from the loss-criteria check.
loss_ex = []

# FastChem-driven equilibrium initialization Newton solver (only used when
# ini_mix='EQ' or 'const_lowT').
fastchem_newton_tol = 1e-12
fastchem_newton_max_iter = 50

# Optional runner controls.
use_fix_all_bot = False
use_fix_H2He = False
use_chunked_runner = False
# Wall-clock budget (seconds) for one run. None/<=0 disables it; a positive
# value forces the chunked runner and bails between chunks once exceeded
# (reported as end_case=4, "runtime budget exceeded"). Off by default.
wall_clock_max = None

# ====== Setting up for output and plotting ======
# plotting:
plot_TP = False
use_live_plot = False
use_live_flux = False
use_plot_end = False
use_plot_evo = False
use_save_movie = False
use_flux_movie = False
plot_height = False
use_PIL = True
live_plot_frq = 10
save_movie_rate = live_plot_frq
y_time_freq = 1  #  storing data for every 'y_time_freq' step
plot_spec = ["H2O", "H", "CH4", "CO", "CO2", "C2H2", "HCN", "NH3"]
# output:
output_humanread = False
use_shark = False
save_evolution = False  # save the evolution of chemistry (y_time and t_time) for every save_evo_frq step
save_evo_frq = 10
