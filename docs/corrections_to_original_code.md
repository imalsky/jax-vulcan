# Corrections to the original VULCAN code

This note records scientific/correctness issues found in upstream VULCAN
(`shami-EEG/VULCAN` `vm_branch` @ `362cfa2`, and the byte-identical portions of
the local `VULCAN-master` oracle) and how VULCAN-JAX handles each. It is the
provenance record for the small number of places where VULCAN-JAX **intentionally
diverges** from upstream to fix a defect, and a triage of the remaining upstream
findings that VULCAN-JAX either already avoids by design, faithfully ports, or
leaves for a validated future change.

VULCAN-JAX's default target is *scientific parity* with VULCAN-master. The
divergences below are deliberate corrections of confirmed upstream errors, each
scoped to keep the parity story auditable (`tools/audit_master_parity.py` knows
about the data-file ones).

## Corrected in VULCAN-JAX (deliberate divergence from upstream)

### C1 — CH2CN + H + M -> CH3CN + M low-pressure rate (data typo)
- **Upstream:** `thermo/SNCHO_photo_network.txt` low-pressure coefficient `k0 = 1.00E-20`.
- **Correct:** `1.00E-29`. This is master's *own* documented typo fix: the NCHO
  header records "10.2025 correct the typo in swapping k0 and k_inf for R1131
  CH2CN + H + M -> CH3CN + M", and master applied it to NCHO but not to the
  sulfur SNCHO network. vm_branch fixed both; VULCAN-JAX's own NCHO/SNCHO_full
  already carried `1.00E-29`.
- **Impact:** with `1e-20` the association never falls off (crossover density
  ~1e10 cm^-3 is above the atmosphere), pinning it at `k_inf` everywhere; the
  effective rate is wrong by up to ~1e7x at the model top. It is a trace nitrile
  channel, so the spectral impact is expected to be small but was not measured.
- **Fix:** `src/vulcan_jax/thermo/SNCHO_photo_network.txt` -> `1.00E-29`.
  Recognized as a known-OK divergence by `tools/audit_master_parity.py`
  (`KNOWN_THERMO_DIVERGENCES`). Note: the non-active variant
  `SNCHO_photo_network_C3.txt` still carries the typo (no active path uses it).

### C2 — S2 / S8 condensate molecular masses (propagated copy-paste)
- **Upstream:** `op.py` hardcodes `m = 45.019/Navo` for S2 (op.py:1282) and
  `m = 360.152/Navo` for S8 (op.py:1328).
- **Correct:** 64.12 and 256.48 g/mol (2x and 8x the atomic-S mass 32.06 in
  `thermo/all_compose.txt`). `45.019` is ~the HCS mass and `360.152 = 8 x 45.019`,
  so the S8 value was derived from the already-wrong S2 value.
- **Impact:** the condensation rate is proportional to this mass, so upstream
  biased the S2 rate by 0.702x and the S8 rate by 1.404x. Only affects sulfur
  cloud runs (`condense_sp` containing S2/S8), which are non-default.
- **Fix:** `src/vulcan_jax/conden.py::GAS_MASS_G_PER_MOL` S2 -> 64.12, S8 -> 256.48.
  `tests/test_conden_profile_builder.py` reads the mass from this constant, so it
  stays green; the runtime condensation test checks conservation/structure, not
  the exact rate. (S4 was already correct at `32.06*4`.)

### C3 — H2O saturation vapour pressure zero at exactly 273 K
- **Upstream:** `build_atm.py::sp_sat` writes
  `(T<0)*ice + (T>0)*water` (T in Celsius). At T = 273.0 K both masks are False,
  so the saturation pressure evaluates to exactly 0 -- an artificial cold trap /
  discontinuity (neighbouring values are ~6111 and ~6112 dyne/cm^2).
- **Fix:** `src/vulcan_jax/atm_setup.py::sat_p_jax` uses a single
  `jnp.where(T_C < 0, ice, liquid)`. Identical to upstream everywhere except the
  single buggy point (now returns the liquid value ~6112 at 0 C), and it removes
  a spurious non-smoothness on the differentiable saturation curve.

## Already avoided by VULCAN-JAX's design (no change needed)

| Upstream finding | Why it does not apply to VULCAN-JAX |
|---|---|
| **Stale generated kernel** (committed `chem_funs.py` is for a different network than configured; `-n` skips regeneration; `os.system` generation is unchecked) | VULCAN-JAX generates the RHS from the network **content hash** (`make_chem_funs.build_chem_rhs`, cached at `__pycache__/chem_rhs_codegen_<hash>.py`), and `state._build_pre_loop_runstate` **fails fast** if `cfg.network` differs from the import-locked network. There is no stale-artifact path and no unchecked shell-out. |
| **Atomic-P collides with the FastChem pressure column** (`fc['P']` reads pressure) | `ini_abun.py` reads `fc["P_1"]` (the `genfromtxt`-renamed atomic-P column), so the collision is handled. (No shipped network uses atomic P regardless.) |
| **`nanmax` on an all-below-threshold flux array crashes a dark column** | The JAX photo kernels are vectorized (`photo.py`); there is no host-side `nanmax` over a possibly-empty selection. |

## Inherited, but not a minimal/safe fix (documented, deferred)

- **Unweighted "atom loss" diagnostic.** `outer_loop._compute_atom_loss` computes
  `Σ_z compo[i,a]*y[z,i]` without `dz` weighting, matching upstream
  `op.loss`/`build_atm.ele_sum`. On a non-uniform grid this can read pure vertical
  redistribution as atom loss. It feeds step acceptance and adaptive rtol, so
  changing it to a `dz`-weighted column inventory is a hot-path convergence change
  that diverges from master and would need a full re-baseline -- not a minimal fix.
  VULCAN-JAX nonetheless validates conservation with the reservoir projection
  (`test_atom_conservation_projection.py`) and matches master to ~0.02 dex, so the
  diagnostic's imperfection is not producing wrong science on tested cases.
- **Condensate mass in the gas mean molecular weight.** `compute_mean_mass` sums
  over every species passed to it; when condensation is active and condensate
  columns carry mass, they can enter the gas `mu`. Only bites with active
  condensation (off in production paths). A correct fix restricts the sum to
  `gas_indx` across `compute_mean_mass` and its callers; deferred until a
  condensation-on validation pass.
- **Two-stream particular-solution pole.** `photo.py` (and upstream `op.compute_flux`)
  form `ll = -w0 / (1/mu^2 - (1-w0)/edd^2)`, which is singular at a physically
  ordinary single-scattering albedo (w0 ~ 0.44 for the default `edd=0.5`, 48 deg).
  A NaN guard exists but does not regularize the near-pole blow-up. The correct fix
  is the analytic resonant limit or a stable BVP solve; that is an RT-numerics
  change needing its own validation, not a minimal edit.

## Upstream data curation flagged, not blindly changed

These need verification against the primary data source before editing (changing
scientific data on a hunch is worse than documenting it):

- **`thermo/all_compose.txt` duplicate species keys** (C4H2, C2H4O, CH3O2,
  CH3OOH, CH3NO2, HCS) resolved by first-match `list.index`; HCS masses disagree
  (45.178 vs 45.079), and `NH3_l_s` is listed as 16.023 g/mol vs the ~17.031
  implied by N+3H. VULCAN-JAX condensation uses its own `conden.py` masses
  (NH3=17.0), so the `NH3_l_s` table value is not on the active condensation path.
- **Duplicate wavelengths in photo cross-section CSVs** (e.g. `HCO_cross.csv` at
  260 nm), which make interpolation pick an endpoint and jump ~46%. Curation
  against the source data is required.
- **`H + CS + M` rate in `SNCHO_full_photo_network.txt`** repeats all six
  parameters of `NH2 + CH3 + M` (uncited), strong copy-paste evidence; the reduced
  active network uses a different explicit estimate. Verify provenance first.

## Upstream approximations and design choices (not defects)

Not bugs; they are modelling choices that should be documented rather than
"fixed": the frozen-total (W-method) transport Jacobian, linear-in-pressure T/Kzz
interpolation, Stokes settling without slip correction, the zero upward
lower-boundary radiation, and the `atom_loss` heuristic above. Several of these
VULCAN-JAX already improves or documents (e.g. `build_atm_static` recomputes a
self-consistent structure; the reverse-mode adjoint is explicit about its scope).

## Upstream-only (workflow / packaging / FastChem C++)

Findings about the upstream run/reproducibility workflow do not map to VULCAN-JAX,
which has its own: content-hashed codegen, `load_config` + strict YAML schema,
per-run isolation for FastChem (`$VULCAN_JAX_FASTCHEM_DIR`), fail-fast validation,
and resolved-config provenance output. The FastChem C++ `-inf` abundance token,
ignored per-layer monitor status, and fixed FastChem I/O paths are upstream
process issues; VULCAN-JAX's `ini_abun`/`_run_fastchem_locked` path and the
`audit_master_parity` FastChem checks cover the parts that matter for the port.

## Verification

- C1/C2/C3 verified against the fetched `vm_branch` source and the local oracle.
- C2/C3 confirmed to keep `tests/test_conden_profile_builder.py` green (7 passed)
  and H2O saturation continuous through 273 K.
- Not run here: reconverged W39b spectra A/B for C1, and any active-condensation
  end-to-end for C2/C3 (the runtime condensation suite is a slow subprocess).
