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

2b. **"What does upstream do?" is answered by FETCHING upstream, never by reading
   `../VULCAN-master/`.** That directory is an unversioned copy (`git rev-parse`
   fails in it) and it is **contaminated**: verified 2026-07-30, it contains
   VULCAN-JAX's own stall detector (`store.py:204-209`, `op.py:1078-1104`), its
   `conv_stall_window` knob (`vulcan_cfg.py:141`,
   `cfg_examples/vulcan_cfg_HD189.py:122`), its `wall_clock_max`/`end_case=4`
   exit (`op.py:1119-1126`), and a 13-species `conver_ignore`
   (`vulcan_cfg.py:183`) that exists in no upstream repository. Citing it as
   upstream is how a wrong config change shipped on 2026-07-26 (`eebc8a5`) and
   2026-07-30 (`2fdc66b`). Fetch instead:
   `raw.githubusercontent.com/exoclime/VULCAN/master/<path>` for VULCAN 2, and
   `shami-EEG/VULCAN` `vm_branch` for VULCAN 3 features.
   `tools/audit_master_parity.py` now refuses to run against a checkout carrying
   VULCAN-JAX-only identifiers, and `tests/test_default_master_parity.py` skips
   (loudly, not passes) rather than reporting a circular parity verdict.
   The copy is still useful as a *numerical* oracle for per-step kernel
   comparisons, which do not depend on the contaminated convergence/config code.

3. **Only real, results-affecting bugs belong here.** A "real bug" changes a
   number a user relies on, crashes, or silently corrupts a result on a path that
   can actually run. Do **not** log or report comment typos, stale docstrings,
   dead code, style, defensible approximations, or issues on paths no shipped
   config selects. The test is: *would this change a result someone trusts?* If
   no, drop it silently. Keeping this file short is deliberate -- it is meant to
   be read end to end without wading through trivia.

Conventions: locations are `file:line`. "master" = the workspace
`../VULCAN-master` validation oracle **for per-step numerical comparisons only**
(see Policy 2b — it is not a provenance source for any config or convergence
question). "fetched master" / "fetched vm_branch" mean the files pulled from
`raw.githubusercontent.com` on the date given. The JAX port was ported from
`shami-EEG/VULCAN vm_branch @ 362cfa2`; a few entries note where that branch and
the workspace oracle differ. None of the items below affect the default
(gas-only, HD189) validated results unless stated.

## Corrected in the JAX port

Deliberate divergences that fix a confirmed master bug. All are live in the
code today: each one is why a VULCAN-JAX file differs from master right now.

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

### C12 — FastChem element row order vs hard-coded C++ slots (P/S swap) (logged 2026-07-29)
- **Mechanism:** FastChem builds its `elements` vector in **abundance-file row
  order** — `init_read_files.cpp:204` calls `addAtom(symbol)` once per line and
  `init_add_species.cpp:55-57` does `elements.push_back` +
  `index = elements.size()-1`. But `mass_action_constant.cpp:380-399` subtracts the
  per-element NASA-9 reference polynomial by **hard-coded slot index** under its own
  comment "in the order of the element_abundances.dat file": `index_P = 5`,
  `index_S = 6`. Abundance *values* are set by symbol
  (`init_read_files.cpp:202 -> setElementAbundance`) and are therefore always right;
  what the row order controls is which reference polynomial each stoichiometric
  coefficient is multiplied by.
- **upstream:** `fastchem_vulcan/input/solar_element_abundances.dat` lists
  `... O, S, P, Si ...` — S at slot 5, P at slot 6, the exact transpose of the C++.
  So every S-bearing molecule's `log_K` is built with `log_P` and every P-bearing
  one with `log_S`. Introduced by shami-EEG `604ca6e` (2025-12-17, "PHO network
  running"), whose diff is literally a two-row `P`/`S` swap in that file. Verified
  present in **exoclime/VULCAN HEAD** as of 2026-07-29, not just the shami fork.
- **Reachable path:** `ini_mix='EQ'` only (FastChem sets the initial mixing ratios;
  it is not on the kinetics path). Live on `W39b.yaml`, which is
  `ini_mix: EQ` on an SNCHO network. `build_atm.py:82-129` rewrites the file
  **preserving its row order**, so the config layer cannot mask or fix this.
- **JAX:** `src/vulcan_jax/fastchem_vulcan/input/solar_element_abundances.dat:13-`
  ships the canonical `C,H,He,N,O,P,S,...` order the C++ expects (fixed in
  `45cc4c7`, 2026-06-08, alongside the port-introduced C/H/He regression).
  Guarded two ways so a future reorder fails loudly rather than silently:
  `runtime_validation.py:50,126` (`_FASTCHEM_ELEMENT_ORDER`, a row-ORDER check, not
  values-only) and `tests/test_fastchem_element_order.py` (pins the C++ indices, the
  validator constant, the shipped files, plus an end-to-end "CO forms" probe).
  `ini_abun.py:192-199` documents that `fc_list` order is inert by contrast.
- **Measured impact (2026-07-29, direct two-way FastChem run, identical binary and
  element->value map, row order the only difference).** The error lands almost
  entirely on whichever species is *not* the dominant reservoir, because element
  conservation pins the reservoir and the polynomial error is absorbed by the
  minor one:
  - **On the real W39b evening column** (nz=150, W39b.yaml abundances, 7.6 bar /
    2246 K to 4e-6 bar / 726 K): every molecular species is **unaffected** —
    SO2, H2S, SO, S2, COS, CS all <= 0.004 dex, H2O/CO/CH4/CO2/NH3/HCN at 0.000 dex.
    **Atomic S is wrong by 2.6-3.1 dex in all 150 layers** (median 2.58 dex across
    the 1e-4..1e-2 bar transit slab).
  - **Only where atomic S is itself a major reservoir** (hot *and* rarefied,
    T >~ 2000 K at P <~ 1e-2 bar — off the W39b profile) does it reach the
    molecules: up to 0.85 dex on SO2/H2S, 1.7 dex on S2, 3.4 dex on S4.
  - Because this is an **initial condition** and VULCAN integrates to a
    photochemical steady state — and atomic S has a very short chemical timescale —
    the converged W39b science is not expected to move. **Not quantified:** the
    effect of the wrong initial atomic-S on the convergence path/step count.
- **Upstream is not fixed and now believes it is.** shami-EEG `8970337`
  (2026-07-28) reordered `fc_list` to `C,N,O,P,S,...` with the comment "need to be
  exactly the same order as element_abundances_vulcan.dat". `fc_list` is used only
  for membership (`sp in fc_list`, `build_atm.py:113`), so that edit is **inert**,
  and it does not touch the abundance file that actually sets the slots (membership
  test at upstream `build_atm.py:114`). O1 remains open upstream. If reporting this,
  point at the .dat row order, not `fc_list`.
- **Workspace-oracle note:** `../VULCAN-master`'s copy of this file **has been
  patched** to the corrected order, so `tools/audit_master_parity.py`'s
  byte-identity check on the abundance file passes against a *patched* oracle. A
  fresh upstream clone will trip that check; the drift message is expected and this
  entry is the explanation. See "Workspace-oracle patches" below.

### C13 — Earth example lists argon, which no network contains (logged 2026-07-30)
- **master:** `cfg_examples/vulcan_cfg_Earth.py:6` puts `'Ar'` in `atom_list` and
  line 39 puts `'Ar':9.34e-3` in `const_mix`, but the network that same file
  selects (`thermo/SNCHO_full_photo_network.txt`, 99 species) has no `Ar`
  species. `build_atm.py:200` does
  `y_ini[:,species.index(sp)] = gas_tot*const_mix[sp]` over every `const_mix`
  key, so the run dies on `ValueError: 'Ar' is not in list` during setup. The
  shipped upstream Earth example therefore does not run at all.
- **JAX:** VULCAN-JAX no longer ships an Earth config at all (removed
  2026-07-30: it converged in neither code, see `validation.md`). While it did,
  it dropped `Ar` from both `atom_list` and `const_mix` and changed nothing else.
  `runtime_validation.py` still rejects any config whose `const_mix` names a
  non-network species, with this finding as the message, so a user writing their
  own Earth case is told upfront rather than crashing in setup.
- **Effect:** the `const_mix` total falls from 0.98974 to 0.98040. Neither sums
  to 1 — upstream leaves the remainder unassigned either way — so this shifts
  the initial condition by 0.93% of the column and lowers the initial mean
  molecular weight slightly, argon being heavier than the N2 it sat beside.
  Argon is chemically inert and absent from the network, so it could not have
  participated in any reaction; the loss is in the initial normalization only,
  and Earth integrates to a photochemical steady state from there.
- **Note:** the alternative (fold 9.34e-3 into N2) preserves the total but
  misstates the composition, so the deficit was kept visible instead.

### C14 — photolysis reaction index read from the file's id column (logged 2026-07-30)
- **master:** the id column of a network file is *output*, not input.
  `make_chem_funs.py:71-72` rewrites every reaction line as
  `'{:<4d} {:s}'.format(i, ...)` — the comment says "updating the numerical
  index in the network (1, 3, ...)" — and line 109
  (`with open(vulcan_cfg.network, 'w+') as f: f.write(new_network)`) writes the
  file back **in place**. Only after that does `op.py:245` do
  `pho_rate_index[(columns[0], int(columns[1]))] = Rindx[i]`, reading back the
  number it just wrote. So in master's workflow the id column always equals the
  parser position, and reading either is the same thing.
- **JAX:** VULCAN-JAX deliberately does not rewrite vendored network files, so
  the id column is whatever the file shipped with. It is stale in **6 of the 18
  vendored networks** — a file fetched from a remote or hand-edited but never run
  through upstream has never been renumbered. `network.py:288` already indexed by
  `parser_i` (position) and `rates.apply_remove_list` masks positionally, but
  `legacy_io.py` had been ported verbatim from `op.py:245` and used the id
  column, so the two parsers disagreed on exactly those files.
- **Effect:** `op_jax.compute_J` writes `var.k_arr[ridx, :]`, and `k_arr` is a
  dense `[nr+1, nz]` array. Three networks raised `IndexError` (e.g.
  `NCHO_photo_network_lowT.txt`: 16 photolysis rows past `nr = 674`), and three
  more — `CHO_photo_network.txt` (36 rows), `NCHO_full_photo_network.txt` (57),
  `TiSNCHO_photo_network.txt` (65) — had every photolysis id in range but wrong,
  so each J landed on an unrelated reaction with no error at all. Master's
  `var.k` is a dict, so the same stale file there silently drops or misassigns
  the rate instead of raising.
- **Fix:** `legacy_io.py` now indexes `pho_rate_index` / `ion_rate_index` by the
  parser position, matching `network.py`, and `_warn_stale_reaction_ids` raises a
  `RuntimeWarning` naming the file when its ids disagree — because a
  `cfg.remove_list` written by reading ids off an un-renumbered file selects the
  wrong reactions in **both** codes. Pinned by
  `tests/test_network_reaction_ids.py`.
- **No shipped result moves:** every network selected by a shipped config
  (`NCHO_photo_network.txt`, `SNCHO_photo_network.txt`,
  `SNCHO_photo_network_2025.txt`, `SNCHO_full_photo_network.txt`) is renumbered,
  so position and id already agreed there. A test pins that too.

## Live constraints left behind by fixed port regressions

The regressions themselves are fixed and are no longer described here. What
survives is the invariant each fix left behind.

### C11 — `batch_max_retries` must be 110, not 64
All five shipped configs cite this anchor; keep it. A withdrawn K2-18b config
omitted the key and silently inherited the code default of 64, which is exactly
the failure this anchor exists to prevent, so declare it in every new config. `lax.while_loop` cannot retry
unboundedly, so the runner carries `batch_max_retries` as a deadlock backstop and
force-accepts on `dt_underflow | retry_exhausted` (`outer_loop.py:1097-1103`).
Master's give-up condition is a dt floor with no retry count (`op.py:2549-2560`).
Walking `dt` from `dt_max=1e17` to `dt_min=1e-14` at `dt_var_min=0.5` takes 103
rejects, so any cap below that makes the COUNT fire first and force-accepts a step
master would still be halving. 110 keeps `dt_underflow` the operative trigger.
**If `dt_max`, `dt_min` or `dt_var_min` change, redo this arithmetic and update all
five configs.**

### Known gap — no NaN-specific termination on the single-profile path
`_conv_jax` (`outer_loop.py:918`) forces `longdy=+inf` on any non-finite `y`/`ymix`,
so a poisoned state can never be scored as converged. But `_real_terminate`
(`:957`) and `cond_fn` (`:1004`) carry no `isfinite` test of their own, so a
single-profile NaN run exhausts its budget and reports budget exhaustion rather
than a NaN reason. Only the batched path sets `termination_reason=5`
(`:1690-1693`).

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

### `atm_type='table'` stale `pico`
- **Where:** master `vulcan.py:118->120->148`, `build_atm.py:406` (pco rewrite),
  `:530/554/562` (stale-pico integration); JAX production setup path.
- **What:** in `table` mode setup runs `f_pico` (pico from the logspace `P_b..P_t`
  grid) BEFORE `load_TPK` overwrites `pco` from the table, and never recomputes
  `pico`. So `f_mu_dz` integrates `dz`/`dzi`/`pref_indx` from a stale `pico`:
  g/Hp ~1% off, `dzi` up to ~12% off when the table grid differs from logspace.
  Masked when the grids match, which is the common restart case.
- **Not live on any shipped config** — all six use `atm_type: file`. The on-graph
  `atm_jax.build_atm_static` (the retrieval path) recomputes `pico`
  self-consistently, so it does not carry this.
- **Status:** deferred, deliberately. Keep production matching master for parity;
  do NOT silently fix VULCAN-JAX alone. The real fix is one line upstream. Revisit
  if upstream fixes the `f_pico`/`pco` ordering.

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

## Workspace-oracle patches (read before trusting a parity result)

`../VULCAN-master` is the do-not-refactor oracle, but it is **not pristine
upstream** — three upstream defects were patched into the workspace copy so the
paper's comparison runs are apples-to-apples. Anything measured against it is
measured against a partially-corrected VULCAN. Re-cloning upstream will change
these files and trip `tools/audit_master_parity.py`.

| Item | Workspace master | exoclime/VULCAN HEAD (2026-07-29) |
|---|---|---|
| C6 H2S cmHg constant (`build_atm.py`) | `0.01333` (patched, `:857`) | `0.001333`, 10x low (`:858`) |
| C7 `NH3_l_s` mass (`thermo/all_compose.txt:167`) | `17.031` (patched) | `16.023` (NH2's mass) |
| C12 FastChem element row order | `...O,P,S...` (patched) | `...O,S,P...` (swapped) |

Still un-patched in the workspace oracle, so parity runs carry them on both sides:
C1 (SNCHO CH2CN `1.00E-20`, `thermo/SNCHO_photo_network.txt:520`), C2 (S2/S8 masses
`op.py:1244,1290`), C3 (H2O saturation zero at 273 K, `build_atm.py:809`), C5
(duplicate `CH2_1`).

## Upstream defects NOT inherited by VULCAN-JAX

Recorded so they are not re-investigated and so they can be reported upstream.
Each is confirmed present in exoclime/VULCAN HEAD as of 2026-07-29; none is a
VULCAN-JAX bug, so none needs a fix here.

| Upstream defect | Location (upstream) | Why JAX is clear |
|---|---|---|
| FastChem uses fixed shared I/O filenames, and `rm`s its own output; two concurrent runs in one checkout clobber each other | `build_atm.py:129,139,154,170` | per-run isolation + `fcntl.flock` (`ini_abun`), `$VULCAN_JAX_FASTCHEM_DIR` per-worker trees |
| Shipped Earth example sets `const_mix` `Ar` and `atom_list` `Ar`, but `SNCHO_full_photo_network.txt` has no Ar species, so the example raises on init | `cfg_examples/vulcan_cfg_Earth.py:6,39` | `Earth.yaml` does not request Ar |
| Flux-convergence test calls `np.nanmax` on an empty selection for a fully dark/filtered column | `op.py:2737` | explicitly guarded: `outer_loop.py:538` `jnp.where(jnp.any(mask), max, 0.0)`; `op_jax.py:79-88` has the `else: 0.0` branch |
| `S4` has a saturation-pressure branch but is absent from `sat_sp_list`, so selecting S4 condensation raises before the implemented code runs | `build_atm.py:788` vs `:833` | `S4` present in both `atm_setup.py:884` and `conden.py:51,68,77` |
| Kinetic NH3 condensation looks for reaction label `NH3 -> NH3_l`; the low-T network defines `NH3 -> NH3_l_s`, so the rate never attaches | `op.py:1151` vs `thermo/NCHO_photo_network_lowT_Jupiter.txt:441` | JAX matches on `NH3_l_s` |
| `check_conserv` runs `np.genfromtxt(dtype=None)` then `str(sp)`; under NumPy < 2 byte-string behavior names become `b'OH'` and the post-codegen conservation check raises, leaving an unchecked generated kernel | `make_chem_funs.py:719-725` | JAX uses content-hashed codegen + a fail-fast network guard; also does not reproduce under NumPy 2.3.5 |

## Provenance

Upstream-side items above were cross-checked against a collaborator audit,
`~/Desktop/VULCAN_original_code_error_audit.md` (2026-07-29), which also lists 16
historical upstream defects that upstream has already fixed. Those are upstream
history, not parity items, and are deliberately not restated here.

Verified independently 2026-07-29: the C12 mechanism and its `604ca6e` introduction;
C1/C2/C3/C6/C7 still open at exoclime/VULCAN HEAD; every row of the not-inherited
table. One correction to that audit: it lists the dark-column `nanmax` crash as
"inherited in JAX" — it is not, both JAX paths guard the empty selection.

## Scope / verification

- C1/C2/C3 verified against the fetched `vm_branch` source and the workspace
  oracle; they keep `tests/test_conden_profile_builder.py` green and H2O
  saturation continuous through 273 K. The still-present items were verified to
  exist in the JAX code.
- Upstream defects that JAX does not inherit now have their own section above
  ("Upstream defects NOT inherited by VULCAN-JAX") — dark-column `nanmax`, FastChem
  I/O concurrency, `NH3 -> NH3_l`, `S4` saturation, the Earth `Ar` example, and the
  `check_conserv` byte-string failure. Also deliberately NOT logged: the stale
  generated-kernel / `-n` regeneration workflow (JAX uses content-hashed codegen +
  a fail-fast network guard), the atomic-P / pressure column collision (JAX reads
  `fc["P_1"]`), the dense ~1 GiB Jacobian (JAX bands it), plus approximations,
  data-file duplicates, and documentation/packaging items.
- **Initial elemental composition differs from upstream, and it dominates any
  cross-code comparison (recorded 2026-07-30).** Two separate, deliberate
  divergences live in the same file,
  `fastchem_vulcan/input/solar_element_abundances.dat`:
  (a) **Lodders 2019 (Wogan & Tsai 2023) values instead of upstream's Lodders
  2009.** For the C-H-N-O networks the only changed value is helium,
  `He 10.9864` (upstream) -> `10.9232` (JAX), i.e. He/H lower by 0.063 dex.
  Sulfur also moves, `7.12` -> `7.1492`. C, H, N and O are identical.
  (b) rocky suppression to `-3.0` (the bullet below).
  **Measured consequence, HD 189733 b, both codes converged, everything else
  matched (same network, same cfg to 1 UI flag, same TP/Kzz/stellar files):**
  median relative difference across all species above a 1e-12 floor is
  **2.0e-01**, and it is present in the deep well-mixed column, not just the
  photochemical upper atmosphere: inert He differs by a uniform 11.6%, H2O by
  25%, CO2 by 42% below 1 bar. For scale, upstream compared against *itself*
  (same code, one reaction changed, different step counts) agrees to
  **7e-06** median. So the composition file, not the solver, is what separates
  the two codes on a free-convergence run.
  **This is an initial-condition choice, not a defect** — the rocky suppression
  is deliberate for the truncated networks, and the Lodders update is a data
  refresh. But it means *"VULCAN-JAX vs VULCAN 2.0"* numbers are only meaningful
  if this file is matched first. Any parity figure or step-count comparison must
  say which composition it used.
- Not a bug, do not "fix": FastChem retains rocky elements (Mg/Si/Fe/…) at solar
  in its equilibrium (`build_atm.py` solar and customized branches both), so a few
  percent of O can sit in untracked gas species. This is a modeling choice, not a
  defect, and it is identical in both config branches. VULCAN-JAX ships a rocky-
  suppressed abundance file, which is a legitimate initial-condition difference for
  the truncated NCHO/SNCHO networks. (There is no `use_other_ele` flag in either
  codebase.)
