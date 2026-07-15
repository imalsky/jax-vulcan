# Corrections to the original VULCAN code

Complete, located record of the correctness/quality issues found in upstream
VULCAN (`shami-EEG/VULCAN` `vm_branch` @ `362cfa2`, and the byte-identical parts
of the local `VULCAN-master` oracle), plus the VULCAN-JAX release issues found in
the same audit pass. For each item: **where** it is (file:line as cited by the
audit against `362cfa2`), what is wrong, and how VULCAN-JAX handles it.

VULCAN-JAX's default target is *scientific parity* with VULCAN-master, so the few
places it deliberately diverges (the fixes below) are scoped to keep the parity
story auditable (`tools/audit_master_parity.py` knows about the data-file ones).

**Verification legend:** [V] = independently verified against the fetched source
and/or the JAX code during this pass; [A] = assessed from the audit + VULCAN-JAX
architecture, not line-by-line re-verified.

---

## Part A. Upstream VULCAN (vm_branch) — corrections VULCAN-JAX applies

Three confirmed upstream errors that VULCAN-JAX had inherited and now fixes:

### C1 — CH2CN + H + M -> CH3CN + M low-pressure rate (data typo) [V]
- **Upstream:** `thermo/SNCHO_photo_network.txt` (reaction R1131 upstream / line
  520 in JAX numbering): low-pressure coefficient `k0 = 1.00E-20`.
- **Correct:** `1.00E-29`. Master's own NCHO header documents "10.2025 correct
  the typo in swapping k0 and k_inf for R1131 CH2CN + H + M -> CH3CN + M"; it was
  applied to NCHO but missed on the sulfur SNCHO network. vm_branch fixed both.
- **Why it matters:** with `1e-20` the association never falls off (crossover
  density ~1e10 cm^-3 is above the atmosphere), pinning it at `k_inf` everywhere;
  the effective rate is wrong by up to ~1e7x at the model top. Trace nitrile
  channel, so spectral impact is expected small but was not measured.
- **JAX fix:** `src/vulcan_jax/thermo/SNCHO_photo_network.txt:520` -> `1.00E-29`
  (also present correctly in JAX NCHO/SNCHO_full). Allowlisted in
  `tools/audit_master_parity.py` (`KNOWN_THERMO_DIVERGENCES`). NOTE: the
  non-active variant `SNCHO_photo_network_C3.txt` still carries the typo.

### C2 — S2 / S8 condensate molecular masses (propagated copy-paste) [V]
- **Upstream:** `op.py:1282` `m = 45.019/Navo` (S2) and `op.py:1328`
  `m = 360.152/Navo` (S8).
- **Correct:** 64.12 and 256.48 g/mol (2x/8x atomic S = 32.06 in
  `thermo/all_compose.txt:126,129`). `45.019` is ~the HCS mass and
  `360.152 = 8 x 45.019`, so S8 was derived from the wrong S2.
- **Why it matters:** the condensation rate is proportional to this mass, so the
  upstream literals biased the S2 rate by 0.702x and S8 by 1.404x. Sulfur-cloud
  runs only (non-default).
- **JAX fix:** `src/vulcan_jax/conden.py::GAS_MASS_G_PER_MOL` S2 -> 64.12,
  S8 -> 256.48. `tests/test_conden_profile_builder.py` reads the mass from this
  constant so it stays green. (S4 was already correct at `32.06*4`.)

### C3 — H2O saturation vapour pressure = 0 at exactly 273 K [V]
- **Upstream:** `build_atm.py:844-874` (`sp_sat`) writes `(T<0)*ice + (T>0)*water`
  (T in Celsius). At T = 273.0 K both masks are False -> saturation pressure is
  exactly 0, an artificial cold trap / discontinuity (neighbours ~6111, ~6112).
- **JAX fix:** `src/vulcan_jax/atm_setup.py::sat_p_jax` uses a single
  `jnp.where(T_C < 0, ice, liquid)` -- identical to upstream everywhere except the
  buggy point, and it removes a spurious non-smoothness on the differentiable curve.

---

## Part A (continued). Full upstream findings, located

Category key: **bug** = concrete wrong result; **numerical** = invalid/at-risk
method; **approx** = defensible approximation to document, not a defect;
**data** = data-file hygiene; **repro** = reproducibility/provenance;
**overstated** = the audit's severity is higher than the code warrants.

| ID | Upstream location (`362cfa2`) | What is wrong | Kind | VULCAN-JAX status |
|---|---|---|---|---|
| F-01 | `vulcan.py:63-84`, `op.py:89-189`/`233-250`, `chem_funs.py:1447-1452` | committed `chem_funs.py` is for a different network than configured; `-n` skips regen; `os.system` generation unchecked | overstated (repro) | **Avoided** [V]: content-hash codegen + fail-fast network guard; normal `vulcan.py` regenerates. Only `-n`/direct-import/silent-gen-failure bite upstream |
| F-02 | `build_atm.py:79-143` (esp `123-126`), `fastchem_src/init_read_files.cpp:189-204` | `use_other_ele=False` ignored for solar; custom path writes `log10(0) = -inf` | bug | **N/A** [A]: JAX has no `use_other_ele` (config schema now rejects it); `-inf` is the upstream FastChem C++ path |
| F-03 | `build_atm.py:133-168`, `model_main/model_main.cpp:205-254` | per-layer FastChem `ok/fail` in `monitor_output.dat` never parsed | bug | **Likely inherited** [A]: JAX FastChem wrapper also doesn't parse the monitor file; candidate hardening |
| F-04 | `build_atm.py:145-168`, `model_main.cpp:174-180` | atomic P collides with the FastChem pressure column (`fc['P']`) | bug | **Avoided** [V]: `ini_abun.py:307` reads `fc["P_1"]` (the renamed species column) |
| F-05 | `op.py:996-1003` + RHS gates `1566`/`1668`/`1765`/`1862`/`1969`, `cfg_examples/vulcan_cfg_Earth.py:97-101` | diffusion-limited escape has no effect unless `use_topflux=True` | bug (boundary) | **Assess** [A]: JAX escape/top-flux gating not re-verified; check before relying on escape |
| F-06 | `op.py:2181-2186`/`2300-2306`/`2504-2510` | escape Jacobian adds `top_flux/y` but omits `/dz` (~4e6x too large) | numerical | **Assess** [A]: check JAX escape Jacobian if escape is used |
| F-07 | `op.py:3023-3037` (`Ros2.solver`) | error mask zeroes the bottom-layer / cloud error estimate for *all* species there | numerical | **Related** [A]: JAX has its own delta error-control masks (see CLAUDE.md); document |
| F-08 | `build_atm.py:287-292`, `op.py:2551-2569`/`835-850`/`3241-3244` | "atom loss" = `sum(n_i)` with no `dz`; misreads redistribution as loss; feeds accept + adaptive rtol | numerical | **Inherited** [V]: `outer_loop._compute_atom_loss` also unweighted. Deferred (hot-path convergence heuristic; a `dz`-weighted change diverges from master + needs re-baseline). Conservation still validated via the reservoir projection |
| F-09 | `op.py:1046-1063` (`conv`) | cloud convergence declared while condensates still evolve (non-gas set to zero) | numerical | **Document** [A]: relates to `conver_ignore`/convergence gating |
| F-10 | `chem_funs.py:15586-15591` (`neg_symjac`) | dense `(ni*nz)^2` float64 Jacobian (~1.1 GiB checked / ~0.93 GiB active) before banding | performance | **Avoided** [V]: JAX uses `block_thomas_diag_offdiag` banded solve; no dense global Jacobian |
| F-11 | `op.py:1281-1288`/`1327-1334`, `all_compose.txt:169-170` | S2/S8 condensate masses 45.019/360.152 | bug | **Fixed** [V] -> C2 |
| F-12 | `op.py:2544-2546`, `build_atm.py:575-579` (`mean_mass`) | condensate mass enters the gas mean molecular weight | bug | **Inherited** [V]: `compute_mean_mass` sums all provided species. Deferred (only bites with active condensation; needs a `gas_indx` restriction across callers) |
| F-13 | `op.py:2690-2750` (esp `2731-2733`), `vulcan_cfg.py:47`,`137` | two-stream `ll = -w0 / (1/mu^2 - (1-w0)/edd^2)` is singular at w0 ~ 0.44 (default `edd`, 48 deg) | numerical | **Inherited** [V]: `photo.py:197`. Deferred (needs analytic resonant limit / stable BVP + RT validation; NaN guard exists) |
| F-14 | `op.py:3284-3336` (`save_out`) | output can't identify the executed kernel/data (no hashes) | repro | **Partly avoided** [A]: JAX writes the resolved config YAML, but no full dependency-hash manifest |
| F-15 | `op.py:1230-1233` (`conden`), `thermo/NCHO_photo_network_lowT_Jupiter.txt:439-442` | code matches `NH3 -> NH3_l` but the network uses `NH3_l_s`, so kinetic NH3 condensation is skipped; tests truthiness of the whole `use_relax` list | bug | **Assess** [A]: check JAX conden reaction matching if NH3 kinetic condensation is used |
| F-16 | `build_atm.py:844-874` (`sp_sat`) | H2O saturation = 0 at exactly 273 K | bug | **Fixed** [V] -> C3 |
| F-17 | `all_compose.txt` (dup keys C4H2/C2H4O/CH3O2/CH3OOH/CH3NO2/HCS), `build_atm.py:573` | duplicate species keys (first-match); HCS masses conflict 45.178 vs 45.079; `NH3_l_s` = 16.023 vs ~17.031 | data | **Documented** [V dup exist]: JAX condensation uses its own `conden.py` masses (NH3=17.0), so the table `NH3_l_s` value is off-path. Data curation needs source check |
| F-18 | `op.py:2659-2676` vs `2700-2706` | optical depth includes photo+ion+T-dependent cross sections; the scattering-albedo denominator uses only photo + room-T `var.cross` | scientific | **Assess** [A]: check JAX `compute_tau`/`compute_flux` opacity consistency |
| F-19 | `thermo/photo_cross/HCO/HCO_cross.csv`, `build_atm.py:690-735` | duplicate wavelengths (235/240/248/248.5/250, 3x at 260 nm) -> interpolation jumps ~46% | data | **Documented** [A]: JAX vendors the same CSVs; curation needs source check |
| F-20 | `op.py:2810-2823` | `nanmax` over an all-below-`flux_atol` array crashes a dark/shielded column | robustness | **Avoided** [V]: JAX photo kernels are vectorized; no host `nanmax` over a possibly-empty set |
| F-21 | `op.py:575-588`, `build_atm.py:706-712` | bin-split index `-1` when the transition is out of range; the interval between the two bins is dropped from the split quadrature | numerical | **Document** [A]: check JAX photo binning edge handling |
| F-22 | `op.py:2207-2245` | transport Jacobian omits cross-species derivatives through the total density (frozen-total W-method) | approx | **Design** [A]: JAX uses an analytical Jacobian; document the approximation rather than "fix" |
| F-23 | `op.py:3023-3025` (`Ros2.solver`) | fixed bottom mixing ratios overwritten after the step, not imposed in the stage equations (untracked reservoir) | numerical | **Document** [A]: relates to JAX `fix_species` handling |
| F-24 | `op.py:2574-2601` (`step_reject`) | at min dt, repeatedly-negative states are clipped and the run advances (warns) | numerical | **Document** [A]: JAX replicates per-step clipping; a strict-mode/ledger is the upgrade |
| F-25 | `build_atm.py:41-69`/`207-212` | `abun_lowT` normalization omits the solved NH3; a disabled `abun_highT` omits N2; He assigned as residual | legacy bug | **Document** [A]: JAX `ini_abun` modes derive from explicit constraints |
| F-26 | `thermo/SNCHO_full_photo_network_2025.txt:580`,`600` | `H + CS + M` repeats all six parameters of `NH2 + CH3 + M` (uncited) | data | **Documented** [V line]: strong copy-paste; verify provenance before changing scientific data |
| F-27 | `build_atm.py:354-389` (`load_TPK`) | T/Kzz interpolated linearly in P (not log-P) | approx | **Document** [A]: modelling choice; state the coordinate |
| F-28 | `build_atm.py:641-668`, `914-920` | Stokes settling without slip correction; CO2 reuses N2 viscosity; H2S vapour fit extrapolated past 164.9-213.2 K | approx | **Document** [A]: enforce/annotate validity ranges |
| F-29 | `op.py:2683-2686` (`compute_flux`) | upward lower-boundary radiation hardcoded to 0; no surface albedo | assumption | **Document** [A]: fine for deep giant-planet boundaries; label it |
| F-30 | `build_atm.py:79-168` | fixed FastChem I/O paths; concurrent runs collide | repro | **Avoided** [A]: `$VULCAN_JAX_FASTCHEM_DIR` gives per-run isolation |
| F-31 | `build_atm.py:170-180` (`vulcan_ini`) | restart copies arrays by index; no pressure-grid/species check | validation | **Document** [A]: check JAX `vulcan_ini` restart |
| F-32 | `build_atm.py:828-840` (`BC_flux`) | `use_fix_sp_bot` is a dict compared to `True`, so the branch is dead | maintainability | **Document** [A]: JAX boundary config |
| F-33 | `build_atm.py:492-537`, `vulcan.py:114-118` | high-T cut mutates `self.P_b` not `vulcan_cfg.P_b`; saved config predates the mutation; a layer index labelled `nz` | repro | **Partly avoided** [A]: JAX `dump_config` writes the resolved config; check the high-T-cut grid is included |
| F-34 | `build_atm.py:234-260` | cold-trap init clips supersaturated gas without moving it to the condensate (open system) | initialization | **Document** [A]: JAX `use_ini_cold_trap`; label as init-only |
| F-35 | `make_chem_funs.py:749-766` | duplicate-reaction checker compares species *sets*, losing stoichiometric multiplicity | test | **Document** [A]: JAX has its own `make_chem_funs`; independent multiset scan passed the audit |
| F-36 | `README.md:13-31`, `build_atm.py:8`, `op.py:3239`, `vulcan_cfg_README.txt:94`, `vulcan_cfg.py:45` | Astropy missing from requirements; no CI/lock; false Py2.7 claim; typos; wrong BC filename; planet mass labelled cm not g | docs/pkg | **N/A** [A]: JAX has its own packaging/CI; `vulcan_cfg.py:45` is a comment-only unit typo |

Audit "positive checks" (not defects): element balance holds for all networks;
the generated gas-chemistry Jacobian matches central differences to ~1e-10; the
`GM/R^2` gravity form is correct; C6H6 branch sums >1 are documented two-photon
conventions.

---

## Part B. VULCAN-JAX release issues (toolchain audit), located

These are defects in the VULCAN-JAX / sibling release (not the original code),
found in the same pass and already fixed + pushed (jax-vulcan `e841d77`,
vulcan-retrieval `444291d`, vulcan-jwst-tool `538a196`). Recorded here so the
guide is complete.

| ID | Location | What was wrong | Fix |
|---|---|---|---|
| F04 | `config.py::load_config`; `README.md:289`,`364` | removed `gs` and misspelled keys silently accepted; README still said `gs` overrides | strict schema validation (reject removed/unknown keys); README corrected |
| F05 | `tools/audit_master_parity.py:213-232`,`254-279` | parity tool exited on the deleted `vulcan_cfg.py`/`cfg_examples/` | loads HD189 YAML; allowlists `use_vm_mol`/`conver_ignore`; validates Mp/Rp->gs |
| F03 | `_version.py` (x3), `pyproject.toml` floors, `vulcan-retrieval/requirements-hpc.txt:23`, `validate_env.py` | versions unbumped despite the breaking `load_config` API; HPC pin was the pre-`load_config` parent | versions 0.2.0/0.9.0; floors >=0.2.0/>=0.9.0; pin -> `e841d77`; added a `load_config` capability probe |
| F10 | `vulcan-retrieval/pipeline.py:721-728` | `set_observations` allowed a post-JIT swap (stale baked-in observations) | raises on a second call |
| F13 | `vulcan-jwst-tool/forward.py:113` | model-cache `_VERSION` not bumped after the network change | `_VERSION` 8 -> 9 |
| F15 | `vulcan-jwst-tool/app.py:82`,`302`,`373` | "always weaker", "cloud deck", "3 sigma standard evidence threshold" overstated the metric | softened to "usually", "power-law cloud/haze opacity", "not Bayesian evidence" |
| F16 | `docs/file_organization.md` | stale `vulcan_cfg.py`/`cfg_examples`/`test_oracle`/`oracle_baselines` refs | updated to the YAML config surface |
| F17 | `vulcan-jwst-tool/tests/unit/test_binning.py`; `tests/test_cli_smoke.py` | `np.trapezoid` (NumPy-2-only) under a 1.26 floor; CLI test read pytest argv + mutated the wrong singleton | version-agnostic `trapz` shim; deliver overrides via a resolved YAML + isolated argv |
| F02 | `NCHO_photo_network.txt:193` (+ SNCHO_full) | audit called the extra `NH3 + CH -> NH2 + CH2` reaction a defect | NOT a bug: it is present in VULCAN-master too; JAX mirrors its oracle. Left as-is |

Also flagged for your decision (not changed): the `conver_ignore` doc-vs-config
mismatch (VULCAN-JAX CLAUDE.md/README say the big hydrocarbon list; the shipped
configs use `[HC3N]`), and the `SNCHO_photo_network_C3.txt` CH2CN typo (off-path).

---

## Verification done this pass

C1/C2/C3 verified against the fetched `vm_branch` source and the local oracle;
C2/C3 keep `tests/test_conden_profile_builder.py` green (7 passed) and H2O
saturation continuous through 273 K. Full conden/atm/sat regression suite green.
Not run: reconverged W39b spectra A/B for C1, and active-condensation end-to-end
for C2/C3 (slow subprocess). Upstream locations are as cited by the audit against
`362cfa2`; the [A]-marked JAX dispositions are architectural assessments, not
line-by-line re-verified.
