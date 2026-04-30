# ============================================================================= 
# Configuration file of VULCAN:  
# ============================================================================= 

# ====== Setting up the elements included in the network ======
atom_list = ['H', 'O', 'C', 'N']
# ====== Setting up paths and filenames for the input and output files  ======
# input:
network = 'thermo/NCHO_photo_network.txt'
use_lowT_limit_rates = False
gibbs_text = 'thermo/gibbs_text.txt' # (all the nasa9 files must be placed in the folder: thermo/NASA9/)
cross_folder = 'thermo/photo_cross/'
com_file = 'thermo/all_compose.txt'
atm_file = 'atm/atm_HD189_Kzz.txt' # TP and Kzz (optional) file
sflux_file = 'atm/stellar_flux/sflux-HD189_Moses11.txt' # sflux-HD189_B2020.txt This is the flux density at the stellar surface
top_BC_flux_file = 'atm/BC_top.txt' # the file for the top boundary conditions
bot_BC_flux_file = 'atm/BC_bot.txt' # the file for the lower boundary conditions
vul_ini = 'output/HD189-nominal.vul' # the file to initialize the abundances for ini_mix = 'vulcan_ini'
# output:
output_dir = 'output/'
plot_dir = 'plot/'
out_name =  'HD189.vul' # output file name

# ====== Setting up the elemental abundance ======
use_solar = True # True: using the solar abundance from Table 8. K.Lodders 2019; False: using the customized elemental abundance. 
# customized elemental abundance (only read when use_solar = False)
O_H = 6.0618E-4 #*(0.793)
C_H = 2.7761E-4
N_H = 8.1853E-5
S_H = 1.3183E-5
He_H = 0.09692
ini_mix = 'EQ' # Options: 'EQ', 'const_mix', 'vulcan_ini', 'table' (for 'vulcan_ini, the T-P grids have to be exactly the same)
fastchem_met_scale = 1. # scaling factor for other elements in fastchem (e.g., if fastchem_met_scale = 0.1, other elements such as Si and Mg will take 0.1 solar values)

use_ini_cold_trap = False
# Initialsing uniform (constant with pressure) mixing ratios (only reads when ini_mix = const_mix)
const_mix = {'CH4':2.7761E-4*2, 'O2':4.807e-4, 'He':0.09691, 'N2':8.1853E-5, 'H2':1. -2.7761E-4*2*4/2} 

# ====== Setting up photochemistry ======
use_photo = True
# astronomy input
r_star = 0.805 # stellar radius in solar radius
Rp = 1.138*7.1492E9 # Planetary radius (cm) (for computing gravity)
orbit_radius = 0.03142 # planet-star distance in A.U.
sl_angle = 48 /180.*3.14159 # the zenith angle of the star in degree (usually 58 deg for the dayside average)
f_diurnal = 1. # to account for the diurnal average of solar flux (i.e. 0.5 for Earth; 1 for tidally-locked planets) 
scat_sp = ['H2', 'He'] # the bulk gases that contribute to Rayleigh scattering
T_cross_sp = [] # warning: slower start! available atm: 'CO2','COS','CS2','H2O','H2O2','H2S','N2O','NH3','O2','SH','SO2'

edd = 0.5 # the Eddington coefficient 
dbin1 = 0.1  # the uniform bin width < dbin_12trans (nm)
dbin2 = 2.   # the uniform bin width > dbin_12trans (nm)
dbin_12trans = 240. # the wavelength switching from dbin1 to dbin2 (nm)

# the frequency to update the actinic flux and optical depth
ini_update_photo_frq = 100
final_update_photo_frq = 5

# ====== Setting up ionchemistry ======
use_ion = False
if use_photo == False and use_ion == True:
    print ('Warning: use_ion = True but use_photo = False')
# photoionization needs to run together with photochemistry


# ====== Setting up parameters for the atmosphere ======
atm_base = 'H2' #Options: 'H2', 'N2', 'O2', 'CO2 -- the bulk gas of the atmosphere: changes the molecular diffsion, thermal diffusion factor, and settling velocity
rocky = False # for the surface gravity
nz = 150   # number of vertical layers
P_b = 1e9  # pressure at the bottom (dyne/cm^2)
P_t = 1e-2 # pressure at the top (dyne/cm^2)
use_Kzz = True
use_moldiff = True
use_vm_mol = False # use upwind scheme for molecular diffusion -- under testing
use_vz = False
atm_type = 'file'  # Options: 'isothermal', 'analytical', 'file', or 'vulcan_ini' 'table'
Kzz_prof = 'file' # Options: 'const','file' or 'Pfunc' (Kzz increased with P^-0.4)
K_max = 1e5        # for Kzz_prof = 'Pfunc'
K_p_lev = 0.1      # for Kzz_prof = 'Pfunc'
vz_prof = 'const'  # Options: 'const' or 'file'
gs = 2140.         # surface gravity (cm/s^2)  (HD189:2140  HD209:936)
Tiso = 1000 # only read when atm_type = 'isothermal'
# setting the parameters for the analytical T-P from (126)in Heng et al. 2014. Only reads when atm_type = 'analytical' 
# T_int, T_irr, ka_L, ka_S, beta_S, beta_L
para_warm = [120., 1500., 0.1, 0.02, 1., 1.]
para_anaTP = para_warm
const_Kzz = 1.E10 # (cm^2/s) Only reads when use_Kzz = True and Kzz_prof = 'const'
const_vz = 0 # (cm/s) Only reads when use_vz = True and vz_prof = 'const'

# frequency for updating dz and dzi due to change of mu
update_frq = 100 

# ====== Setting up the boundary conditions ======
# Boundary Conditions:
use_topflux = False
use_botflux = False
use_fix_sp_bot = {} # fixed mixing ratios at the lower boundary
diff_esc = [] # species for diffusion-limit escape at TOA
max_flux = 1e13  # upper limit for the diffusion-limit fluxes
use_sat_surfaceH2O = False

# ====== Reactions to be switched off  ======
remove_list = [] # in pairs e.g. [1,2]

# == Condensation ======
use_relax = []
use_condense = False
use_settling = False
start_conden_time = 0
stop_conden_time = 1e5 # after this time to fix the condensable species
condense_sp = []     
non_gas_sp = []
r_p = {'H2O_l_s': 5e-3}  # particle radius in cm (1e-4 = 1 micron)
rho_p = {'H2O_l_s': 1} # particle density in g cm^-3
fix_species = []      # fixed the condensable species after condensation-evapoation EQ has reached  
fix_species_time = 0  
fix_species_from_coldtrap_lev = True
humidity = 1.

# ====== steady state check ======
st_factor = 0.5
conv_step = 500

# ====== Setting up numerical parameters for the ODE solver ====== 
ode_solver = 'Ros2' # case sensitive
use_print_prog = True
use_print_delta = False
print_prog_num = 500  # print the progress every x steps 
dttry = 1.E-10
trun_min = 2e8
runtime = 1.E8
dt_min = 1.E-14
dt_max = runtime*1e-5
dt_var_max = 2.
dt_var_min = 0.5
count_min = 120
count_max = int(1E4)
atol = 1.E-1 # Try decreasing this if the solutions are not stable
mtol = 1.E-22
mtol_conv = 1.E-20
pos_cut = 0
nega_cut = -1.
loss_eps = 1e-1
yconv_cri = 0.01 # for checking steady-state
slope_cri = 1.e-4
yconv_min = 0.1
flux_cri = 0.1
flux_atol = 1. # the tol for actinc flux (# photons cm-2 s-1 nm-1)

# === VULCAN-JAX outer loop (outer_loop.OuterLoop) ===
batch_steps = 1         # accepted Ros2 steps per JAX runner call. Held at
                        #   1 for semantic match with op.Integration; the
                        #   single-shot runner means this is effectively
                        #   ignored in production.
batch_max_retries = 64  # safety cap on inner accept/reject retries per
                        #   accepted step inside the JAX body. Set high so
                        #   the dt_min underflow path (master's only
                        #   force-accept trigger, op.py:2518) wins under
                        #   any reasonable (dttry, dt_var_min, dt_min):
                        #   for the defaults (1e-10, 0.5, 1e-14) underflow
                        #   fires at retry 14; configs with dttry=1e-3 fire
                        #   at retry 37. 64 keeps the cap as a true safety
                        #   guard against runaway loops without altering
                        #   master's accept-criterion semantics.
### use with caution
conver_ignore = [] # added 2023. to get rid off non-convergent species, e.g. HC3N without sinks

# ====== Setting up numerical parameters for Ros2 ODE solver ======
rtol = 0.2              # relative tolerence for adjusting the stepsize
post_conden_rtol = 0.1 # switched to this value after fix_species_time
use_adapt_rtol = False
rtol_min = 0.02
rtol_max = 2.5

# Adaptive rtol cadence + multipliers (op.py:836-852).
# Each `adapt_rtol_dec_period` accepted steps, if max|atom_loss| >= loss_criteria,
# rtol *= adapt_rtol_dec and loss_criteria *= adapt_rtol_loss_mul. Each
# `adapt_rtol_inc_period` accepted steps (count > 0), if max|atom_loss| <
# adapt_rtol_inc_loss_thresh, rtol *= adapt_rtol_inc.
adapt_rtol_dec_period = 10
adapt_rtol_dec = 0.75
adapt_rtol_loss_mul = 2.0
adapt_rtol_inc_period = 1000
adapt_rtol_inc = 1.25
adapt_rtol_inc_loss_thresh = 2e-4

# Hycean H2/He bottom-pin trip time (op.py:2935). After `t > hycean_pin_time`,
# the first accepted step snapshots the bottom-layer ymix for H2 and He and
# pins them at that value forever.
hycean_pin_time = 1e6

# Photo update_photo_frq ini→final switch thresholds (op.py:819-823).
# When longdy < photo_switch_longdy_thresh AND longdydt < photo_switch_longdydt_thresh,
# the per-iteration photo cadence switches from ini_update_photo_frq to
# final_update_photo_frq. Master uses `yconv_min*10.0` for the longdy gate.
photo_switch_longdy_thresh = yconv_min * 10.0
photo_switch_longdydt_thresh = 1e-6

# ====== Setting up for output and plotting ======
# Live-output flags. Each enables a host-side hook called between JIT'd
# step batches at `live_plot_frq` cadence (see live_ui.py). Setting any
# flag to True forces the chunked-runner path so the host can read
# state between chunks.
#   use_live_plot:  matplotlib mixing-ratio plot (op.plot_update)
#   use_live_flux:  matplotlib actinic/diffuse flux plot (requires use_photo)
#   use_save_movie: write `movie_dir/{N}.png` frames each plot tick
#   use_flux_movie: write `plot/movie/flux-{N}.jpg` frames each flux tick
use_live_plot = False
use_live_flux = False
use_save_movie = False
use_flux_movie = False
use_PIL = False           # post-run: open mix.png in a PIL viewer
movie_dir = 'plot/movie/' # destination for use_save_movie frames
live_plot_frq = 10        # accepted-step cadence for live UI updates

plot_TP = False
use_plot_end = True
use_plot_evo = False
plot_height = False
y_time_freq = 1  #  storing data for every 'y_time_freq' step
plot_spec = ['H2O', 'H', 'CH4', 'CO', 'CO2', 'C2H2', 'HCN', 'NH3' ]
# output:
output_humanread = False
use_shark = False
save_evolution = False   # save the evolution of chemistry (y_time and t_time) for every save_evo_frq step
save_evo_frq = 10
