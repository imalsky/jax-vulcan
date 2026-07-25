# VULCAN-JAX <-> VULCAN-master parity & bug guide

This is the single source of truth for where VULCAN-JAX intentionally diverges
from upstream VULCAN, and for confirmed bugs in VULCAN-master. It is also the
"bug guide" the standing rules point at.

## Policy (LLMs: read this first)

1. **The goal is parity with VULCAN-master.** VULCAN-JAX must reproduce
   VULCAN-master's science. Match master by default. Any intentional divergence
   must (a) fix a **confirmed, results-affecting** defect and (b) be recorded in
   this file with `file:line` on both sides. If a behavior is not listed here, do
   not "improve" it -- reproduce master.

2. **VULCAN-master is the oracle; do not edit or refactor it.** If you find a real
   bug in VULCAN-master, do **not** patch master. Document it here: under
   "Corrected in the JAX port" if the JAX port fixes it (also allowlist any
   data-file divergence in `tools/audit_master_parity.py`), or under "Bugs still
   present" if the port inherits it. Give `file:line` and the measured or
   estimated impact.

3. **Only real, results-affecting bugs belong here.** A "real bug" changes a
   number a user relies on, crashes, or silently corrupts a result on a path that
   can actually run. Do **not** log or report comment typos, stale docstrings,
   dead code, style, defensible approximations, or issues on paths no shipped
   config selects. The test is: *would this change a result someone trusts?* If
   no, drop it silently. Keeping this file short is deliberate -- it is meant to
   be read end to end without wading through trivia.

Conventions: locations are `file:line`. "master" = the workspace
`../VULCAN-master` validation oracle. The JAX port was ported from
`shami-EEG/VULCAN vm_branch @ 362cfa2`; a few entries note where that branch and
the workspace oracle differ. None of the items below affect the default
(gas-only, HD189) validated results unless stated.

## Corrected in the JAX port

Deliberate divergences that fix a confirmed master bug.

### C1 — CH2CN + H + M -> CH3CN + M low-pressure rate (data typo)
- **master:** `thermo/SNCHO_photo_network.txt` (R1131) `k0 = 1.00E-20` (and
  `thermo/SNCHO_photo_network_C3.txt`). Correct: `1.00E-29`. The 10.2025 typo fix
  was applied to master's **NCHO** network (the default) but missed on the
  **SNCHO** (sulfur) base file, which still carries `1.00E-20` on the workspace
  oracle. With `1e-20` the association never falls off (pinned at `k_inf`), wrong
  by up to ~1e7x at the model top; trace nitrile channel, small spectral effect.
- **JAX:** `src/vulcan_jax/thermo/SNCHO_photo_network.txt:520` -> `1.00E-29`.
  Allowlisted in `tools/audit_master_parity.py` (`KNOWN_THERMO_DIVERGENCES`).
- **Note:** master's DEFAULT config uses NCHO (already `1e-29`), so default master
  runs are unaffected; this divergence only appears on SNCHO/sulfur runs.

### C2 — S2 / S8 condensate molecular masses (copy-paste error)
- **master:** `op.py:1282` `45.019/Navo` (S2) and `op.py:1328` `360.152/Navo`
  (S8). Correct: 64.12 and 256.48 g/mol (2x/8x atomic S = 32.06,
  `all_compose.txt:126,129`). `45.019` is ~the HCS mass and `360.152 = 8x45.019`.
- **Effect:** the condensation rate scales with this mass, so master biased the
  S2 rate 0.702x and S8 1.404x (sulfur-cloud runs only).
- **JAX:** `src/vulcan_jax/conden.py::GAS_MASS_G_PER_MOL` S2 -> 64.12, S8 -> 256.48
  (S4 was already correct).

### C3 — H2O saturation vapour pressure = 0 at exactly 273 K
- **master:** `build_atm.py:844-874` (`sp_sat`) writes `(T<0)*ice + (T>0)*water`
  (T in Celsius); at 273.0 K both masks are False, so the value is exactly 0 (an
  artificial cold trap; neighbours are ~6111/6112 dyne/cm^2).
- **JAX:** `src/vulcan_jax/atm_setup.py::sat_p_jax` uses one
  `jnp.where(T_C < 0, ice, liquid)` -- identical except at the buggy point.

### C4 — sflux-epseri.txt surface-flux normalization (R_star multiplied, not divided)
- **master:** `atm/make_spectra_in_nm.py:25` converts the observed-at-Earth
  HST eps Eri UV spectrum (`obs_spectra/h_epseri_uvsum_spc.txt`) to stellar
  surface flux with `flux = F_earth*10.*(10.475*63241*au/r_sun*0.735)**2` --
  by operator precedence R_star = 0.735 R_sun MULTIPLIES where the conversion
  `F_surface = F_earth*(d/R_star)^2` divides. The shipped
  `atm/stellar_flux/sflux-epseri.txt` is therefore low by exactly
  `R_star^4 = 0.735^4 = 0.291843` (factor 3.426499 too small). The sibling
  builder `atm/stellar_flux/read_muscles_spectra_in_nm.py:38` uses the correct
  `(d/(r_sun*0.2064))**2` form for GJ1214, so this is a one-off
  parenthesization slip, verbatim in upstream exoclime/VULCAN (found by the
  2026-07-21 jwst-tool science audit, S2-01).
- **JAX:** `src/vulcan_jax/atm/stellar_flux/sflux-epseri.txt` rebuilt from the
  raw HST file with the corrected normalization; construction otherwise
  IDENTICAL to master's builder (positive-only filter, ERROR/DQ columns
  ignored, 3-decimal nm wavelengths, 2-sig-fig flux, 115.000-282.999 nm span,
  20 duplicate wavelengths retained). Every flux value is the master value
  times 3.4265 (2-sig-fig rounding); wavelength column byte-identical.
  Verified in `tools/audit_master_parity.py` (`KNOWN_SFLUX_RESCALES` +
  `_known_sflux_rescale_only`: wavelengths must match, every ratio must sit at
  the documented factor).
- **Impact:** only consumers selecting the eps Eri spectrum (jwst-tool's
  WASP-107 b default; no shipped VULCAN-JAX config uses it). Deliberately NOT
  fixed here: the 115-283 nm coverage (bands outside the file span are omitted
  from the photolysis grid by the standard clamp, master-identical) and the
  positive-only/DQ-blind construction -- measured photolysis-integral
  sensitivity of those choices is 2-6% for H2O/CH4/H2S/SO2/HCN (HO2 ~2x on
  the signed variant), small against the 3.43x normalization. jwst-tool side:
  decision record in `vulcan-jwst-tool/docs/audit_decisions_2026-07-21.md`,
  cache bust via `forward._VERSION` 22 (the UV file is cache-keyed by name,
  not content).

### C6 — H2S saturation-pressure unit conversion (mm Hg constant for a cm Hg formula)
- **upstream:** `build_atm.py:857` `saturate_p * 0.001333 * 1.e6` under upstream's own
  comment "from Giauque and Blue(1936) in cmHg" — the mm Hg constant for a cm Hg
  formula, 10x low. Same bug in shami-EEG vm_branch (`build_atm.py:920`).
- **JAX:** `src/vulcan_jax/atm_setup.py:943` `sat_p * 0.01333 * 1e6`. Anchor: at the
  H2S boiling point (212.8 K) the formula gives 76.1 cmHg -> 1.015 bar ~ 1 atm with
  0.01333; 0.1 atm with the upstream constant. **Workspace master is patched too**
  (the paper's comparison copy carries this fix), unlike C1-C4.
- Verified against upstream HEAD 2026-07-21. Previously recorded only in README.

### C7 — NH3 ice molecular weight (NH2's mass)
- **upstream:** `thermo/all_compose.txt:167` `NH3_l_s ... 16.023` — exactly NH2's
  mass (line 40), a copy-paste of the row above. Correct: 17.031. Same in vm_branch.
- **JAX:** vendored `all_compose.txt:167` -> 17.031. **Workspace master patched too.**
- Impact channel is mean molecular weight + molecular-diffusion mass only (both
  codes hardcode ~17 g/mol in the NH3 condensation RATE), and only when NH3_l_s is
  nonzero — real but tiny. Verified against upstream HEAD 2026-07-21; previously
  recorded only in README.

### C5 — Duplicated CH2_1 entry in the FastChem NASA-9 logK data
- **master:** `fastchem_vulcan/input/nasa9_logK_SNCHOTi.dat`,
  `nasa9_logK_SNCHOTi_ion.dat`, `nasa9_logK_SNCHOPTi.dat` each list the
  `CH2_1 : H 2 C 1` (singlet methylene) entry TWICE, with byte-identical
  coefficient lines. Still present on the workspace oracle.
- **JAX:** the vendored copies under `src/vulcan_jax/fastchem_vulcan/input/`
  keep one entry. Because the duplicate coefficients are identical, no
  measured benchmark impact (W39b V2-vs-V3 parity is 1.1e-9 median with the
  dedup in place on the JAX side only); recorded as a data-file divergence.
  Mechanism (verified 2026-07-21): FastChem's `init_add_species.cpp:132` has no
  duplicate check, so the second entry does enter the element-conservation sums;
  downstream, `build_atm.py`'s `np.genfromtxt(names=True)` renames the second
  output column `CH2_1_1` and VULCAN reads the first — trace-level at most.
  Upstream HEAD confirmed to carry the duplicate (byte-identical to workspace
  master; entries at lines 19 and 421 of SNCHOTi.dat, identical coefficients).
- **Note (same diff, separate):** the vendored `nasa9_logK_SNCHOTi.dat` also
  adds an SiO2 entry absent from master's copy; and
  `element_abundances_vulcan.dat` differs because it is a per-run scratch
  file rewritten by the EQ initialization, not a correction.

### C8 — Non-finite states scored as converged (solver-guard divergence, 2026-07-24)
- **master:** `op.py:1053` computes `longdy = np.amax(longdy[ymix>0]/ymix[ymix>0])`. On a poisoned
  state the boolean selection is empty and NumPy raises
  `ValueError: zero-size array to reduction operation maximum` — master crashes loudly.
- **JAX (before):** `outer_loop._conv_jax` masked with `jnp.where(s.ymix > 0, ...)`. `NaN > 0` is
  False, so every non-finite cell contributed exactly `0.0` to the max — the guard deleted the cells
  that most needed to be seen. Measured: a state correctly flagged unconverged at `longdy=0.03996`
  flipped to `longdy=0.0` (perfect score) when the one offending cell was set NaN; an all-NaN state
  read `0.0 < yconv_cri` and reported `end_case=1` "Integration successful".
- **JAX (now):** `src/vulcan_jax/outer_loop.py` forces `longdy=+inf` when any `y`/`ymix` cell is
  non-finite, reproducing master's "can never converge" semantics inside a jittable reduction. Both
  convergence routes stay shut (the normal branch, and the stall fallback which also requires
  `longdy < yconv_min`). Pinned by `tests/test_nonfinite_never_converges.py`.
- **Note:** this is a *guard* divergence, not a physics one — no converging run changes. The
  single-profile `cond_fn`/`_real_terminate` still carry no `isfinite` test; only the batched
  `body_fn_batch` sets `termination_reason=5`.

### C9 — Post-step clip fed the pre-step `ymix` (2026-07-24)
- **master:** `Ros2.solver` writes `var.y = sol` AND recomputes `var.ymix` from the new solution
  (`op.py:3028`, `op.py:3031-3034`); `one_step` then calls `clip` (`op.py:3139`), whose second rule
  (`op.py:2503`) `y[ymix<mtol & y<0] = 0` therefore tests the **post-solve** mixing ratio. Any cell
  with `y<0` has post-solve `ymix<0<mtol`, so master zeroes **every** negative unconditionally.
- **JAX (before):** `outer_loop` passed the **pre-step** `s.ymix`, which is positive, so that branch
  never fired. Measured on real HD189 Ros2 output: 267/204/204 negatives survived at
  `dt=1e4/1e8/1e12` (worst magnitude 6.99e9 cm^-3) where master's clip leaves zero in every case.
  Two consequences: negative number densities entered the carried state, and the `all_nonneg` accept
  gate rejected steps master provably accepts.
- **JAX (now):** both `_make_clip_fn` variants zero every negative, matching master's reduced rule.
- **Note:** **this changes accepted-step counts.** Any benchmark or Table 1 regeneration must be run
  after this change, not before.

### C10 — Diffusion-limited-escape term missing from the Jacobian diagonal (2026-07-24)
- **master:** adds `diff_lim[sp] = atm.top_flux[sp]/y[-1,sp]` to the top-layer diagonal in all three
  live variants (`op.py:2144-2148`, `2264-2268`, `2468-2472`), gated on a non-empty
  `vulcan_cfg.diff_esc` list — **not** on `use_topflux`.
- **JAX (before):** the escape flux reached the RHS (`jax_step._apply_diffusion_jax`, refreshed
  in-loop by `atm_refresh.update_phi_esc_jax`) but the `diag_d` assembly added only `bot_vdep`;
  `grep -rn diff_lim src/` returned nothing, so the Jacobian was inconsistent with the residual it
  linearizes. Live on two shipped configs: `HD209.yaml` (`diff_esc: [H]`) and `Earth.yaml`
  (`[H2, H]`), both `use_topflux: false`. Measured on the HD209 state, the omitted element is
  **1.841x** the Rosenbrock diagonal at `dt=1 s` and **1.841e8x** at `dt=1e8 s` — dominant as `dt`
  grows toward `dt_max`.
- **JAX (now):** a `diff_esc_mask` (ni,) bool rides `AtmStatic` beside `top_flux`, built identically
  in `make_atm_static` and `atm_jax.build_atm_static`, and the term is added with master's
  `if y[-1,sp] > 0` guard. Verified no-op where `diff_esc` is empty (mask all-False for HD189/W39b,
  flags `H` for HD209), so master-parity numbers on those columns are untouched.

## Bugs still present in the JAX port (inherited from master)

Confirmed master bugs the port faithfully carries. None affect the default
(gas-only, HD189) validated results -- they bite the convergence machinery or the
non-default condensation/photochemistry paths.

### Unweighted atom-conservation diagnostic
- **Where:** master `op.py:2551-2569` (`ODESolver.loss`), `build_atm.py:287-292`;
  JAX `src/vulcan_jax/outer_loop.py:285-290` (`_compute_atom_loss`).
- **What:** "atom loss" is `sum over layers of compo*y` with **no `dz` weighting**,
  so on a non-uniform grid it can read exactly-conservative vertical redistribution
  as loss. It feeds step acceptance and adaptive rtol.
- **Status:** deferred. A hot-path convergence heuristic; a `dz`-weighted column
  inventory diverges from master and needs a re-baseline. JAX matches master to
  ~0.02 dex and validates conservation via the reservoir projection
  (`test_atom_conservation_projection.py`), so it is not producing wrong science
  on tested cases.

### Two-stream particular-solution pole
- **Where:** master `op.py` `compute_flux` (`ll = -w0/(1/mu^2 - (1-w0)/edd^2)`);
  JAX `src/vulcan_jax/photo.py:197`.
- **What:** `ll` is singular at `w0 = 1 - edd^2/cos^2(sl_angle)`. With `edd=0.5`
  the pole lies inside [0,1] for `sl_angle <= 60 deg`: **reachable** at the 48 deg
  configs (HD189 / HD209 / default; pole at `w0 ~ 0.44`) and Earth (58 deg,
  `w0 ~ 0.11`); **not reachable** at the 83 deg W39b config (pole at `w0 = -15.8`).
  Near the pole `g_p`/`g_m` spike and can flip sign, corrupting the DIFFUSE actinic
  flux (the direct Beer's-law beam is unaffected). master's `ll` clip is dead code
  (applied AFTER `g_p`/`g_m`, and `ll` is never read again); JAX has no `ll` clip
  (it guards `chi` instead). Neither regularizes the blow-up.
- **Impact:** only the scattered-flux correction, only for `w0` within a thin band
  of the pole. Quantified over the converged HD189/HD209 runs (150 layers x 2588
  wavelength bins, `jax_paper/scripts/two_stream_pole_margin.py`): `w0` spans [0,1]
  and does reach the pole (closest cell `w0 = 0.44161`; `min|1/mu^2-(1-w0)/edd^2|
  ~ 1e-4`), with ~0.1-0.5% of layer/wavelength cells in a thin band of it, where
  `|ll| = w0/margin` spikes to ~5200 (HD189) / ~3580 (HD209) vs O(1) elsewhere.
  Because it touches only the diffuse actinic flux and both codes carry it
  identically, the gas-only parity comparison is unbiased; the effect on the
  absolute converged abundances remains unquantified (no evidence it moves the
  validated results). Not discussed in the paper (small, shared, inherited).
- **Status:** DEFERRED, and a fix is NOT a drop-in. The correct fix is the analytic
  resonant limit (or a stable boundary-value solve) and needs its own RT
  validation. A naive `|denominator|` floor or `g_p`/`g_m` clip changes the
  near-pole physics and would break parity with master there, so do this as a
  deliberate, validated task -- not a reflexive clip. (master's own `min/max` clip
  on `ll` does not even help: near the pole `|ll|` is large but well below `1e10`,
  so the clip never fires on the damaging values.)

### Condensate handling in the solver (active-condensation only)
Three inherited condensation-path issues; all no-op when `non_gas_sp` is empty
(the default, `use_condense=False`), so they only matter for cloud runs. The
sibling JWST-tool and retrieval both refuse condensation on their gradient paths,
so these do not reach any inference result.
- **Condensate mass in the gas mean molecular weight** — master `op.py:2544-2546`,
  `build_atm.py:522` (`mean_mass` sums every species; the gas-only `exc_conden`
  sum at `build_atm.py:267` is used for normalization but NOT for `mu`); JAX
  `atm_setup.py:363`. Condensate mass can pollute the gas `mu` (scale height) when
  condensation is active. Fix: restrict the `mu` sum to `gas_indx`. This is the one
  physically-wrong item of the three.
- **Condensates excluded from the convergence metric** — master `op.py:1046-1063`;
  JAX `outer_loop.py:649` (`condense_zero_conv_mask` zeroes `non_gas_sp` columns in
  `longdy`). A cloud column can be declared steady while condensates still evolve.
  (Defensible: pinned-to-saturation species are set, not relaxing, so a `longdy`
  residual on them is not a steady-state signal -- documented as a known trade-off,
  not clearly a bug.)
- **Condensates excluded from the local error norm** — master `op.py:3023-3037`;
  JAX `outer_loop.py:721-751` (`non_gas_present` gas-only error denominator). The
  adaptive step can accept a step whose largest error is in a settling condensate.

### Off-path data typo
- `src/vulcan_jax/thermo/SNCHO_photo_network_C3.txt` still carries the C1 CH2CN
  `1.00E-20` typo. No shipped config or sibling repo selects this variant, so it
  changes nothing today; fix it if C3 chemistry is ever activated.

## Master-only, already better in the JAX port

Not a JAX bug; recorded so no one "fixes" JAX to match master's weaker behavior.

### Optical-depth vs single-scattering-albedo opacity inconsistency
- **master:** `compute_tau` sums absorption over `photo_sp ∪ ion_sp` with
  T-dependent cross sections; `compute_flux`'s `w0` uses only `photo_sp` with the
  T-INDEPENDENT table (`op.py` ~2621-2672). So the optical depth and the single-
  scattering albedo are built from different opacities.
- **JAX:** `compute_tau_jax` and `compute_flux_jax` build absorption from the SAME
  `PhotoData` arrays (`absp_idx` + `absp_T_idx` + `scat_idx`), so `tau` and `w0`
  are self-consistent (`src/vulcan_jax/photo.py:106-184`). Minor (`ion_sp` empty by
  default; only the scattered flux is affected), but JAX is the correct one -- do
  not "restore parity" by reintroducing the inconsistency.

## Scope / verification

- C1/C2/C3 verified against the fetched `vm_branch` source and the workspace
  oracle; they keep `tests/test_conden_profile_builder.py` green and H2O
  saturation continuous through 273 K. The still-present items were verified to
  exist in the JAX code.
- Deliberately NOT logged (not real, or off the active paths): stale generated-
  kernel / `-n` regeneration workflow (JAX uses content-hashed codegen + a
  fail-fast network guard), the atomic-P / pressure column collision (JAX reads
  `fc["P_1"]`), the dense ~1 GiB Jacobian (JAX bands it), the dark-column `nanmax`
  crash (JAX is vectorized), FastChem I/O concurrency (per-run isolation), the
  `NH3 -> NH3_l` condensation mismatch (JAX uses `NH3_l_s`), plus approximations,
  data-file duplicates, and documentation/packaging items.
- Not a bug, do not "fix": FastChem retains rocky elements (Mg/Si/Fe/…) at solar
  in its equilibrium (`build_atm.py` solar and customized branches both), so a few
  percent of O can sit in untracked gas species. This is a modeling choice, not a
  defect, and it is identical in both config branches. VULCAN-JAX ships a rocky-
  suppressed abundance file, which is a legitimate initial-condition difference for
  the truncated NCHO/SNCHO networks. (There is no `use_other_ele` flag in either
  codebase.)


## 2026-07-20 — atm_type='table' stale pico (upstream, faithfully ported, NOT fixed)

Moved from VULCAN-JAX/CLAUDE.md during the memory-threshold
consolidation; this is the authoritative record. Policy fit: real,
results-affecting upstream bug, divergence documented, master left
untouched (do-not-refactor oracle).

### atm_type='table' stale pico (full record) (verbatim pre-consolidation text)

**KNOWN ISSUE — `atm_type='table'` stale `pico` (latent UPSTREAM bug, faithfully ported; NOT being fixed this release per Isaac, 2026-06-17).** In `table` mode, setup runs `f_pico` (pico from the original logspace `P_b..P_t` grid) BEFORE `load_TPK` overwrites `pco` from the table file, and never recomputes `pico` — so `f_mu_dz` integrates `dz`/`dzi`/`pref_indx` from a stale `pico` (g/Hp ~1%, dzi ~12% off when the table grid differs from logspace; masked when it matches, which is the common restart case). **VULCAN-master has the identical bug** (`vulcan.py:118`→`120`→`148`; `build_atm.py:406` pco rewrite, `:530/554/562` stale-pico integration), so VULCAN-JAX's production path is a faithful port; `build_atm_static` recomputes `pico` self-consistently (the deviation). DECISION: keep production matching master for parity; do NOT silently fix VULCAN-JAX alone (would diverge from master); the real fix is one line upstream. Revisit only if upstream fixes the `f_pico`/`pco` ordering.
