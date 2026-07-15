# Corrections to the original VULCAN code

Where VULCAN-JAX diverges from upstream VULCAN (`shami-EEG/VULCAN` `vm_branch` @
`362cfa2`) to fix a confirmed defect, and the substantive upstream bugs the port
still carries. Trivia (typos, dead code, documentation, defensible approximations)
and upstream bugs VULCAN-JAX already avoids by design are out of scope here.
Locations are `file:line`; upstream lines are against `362cfa2`.

VULCAN-JAX's default target is scientific parity with VULCAN-master, so the fixes
below are deliberate divergences (`tools/audit_master_parity.py` knows about the
data-file one).

## Corrected in the JAX port

### C1 — CH2CN + H + M -> CH3CN + M low-pressure rate (data typo)
- **Upstream:** `thermo/SNCHO_photo_network.txt` (R1131): `k0 = 1.00E-20`.
- **Correct:** `1.00E-29`. Master's NCHO header records this as a 10.2025 typo
  fix that was applied to NCHO but missed on the sulfur network; vm_branch fixed
  both. With `1e-20` the association never falls off (pinned at `k_inf`), wrong by
  up to ~1e7x at the model top; trace nitrile channel, so small spectral effect.
- **JAX:** `src/vulcan_jax/thermo/SNCHO_photo_network.txt:520` -> `1.00E-29`.
  Allowlisted in `tools/audit_master_parity.py` (`KNOWN_THERMO_DIVERGENCES`).

### C2 — S2 / S8 condensate molecular masses (copy-paste error)
- **Upstream:** `op.py:1282` `45.019/Navo` (S2) and `op.py:1328` `360.152/Navo`
  (S8). Correct: 64.12 and 256.48 g/mol (2x/8x atomic S = 32.06,
  `all_compose.txt:126,129`). `45.019` is ~the HCS mass and `360.152 = 8x45.019`.
- **Effect:** the condensation rate scales with this mass, so upstream biased the
  S2 rate 0.702x and S8 1.404x (sulfur-cloud runs only).
- **JAX:** `src/vulcan_jax/conden.py::GAS_MASS_G_PER_MOL` S2 -> 64.12, S8 -> 256.48
  (S4 was already correct).

### C3 — H2O saturation vapour pressure = 0 at exactly 273 K
- **Upstream:** `build_atm.py:844-874` (`sp_sat`) writes `(T<0)*ice + (T>0)*water`
  (T in Celsius); at 273.0 K both masks are False, so the value is exactly 0 (an
  artificial cold trap; neighbours are ~6111/6112 dyne/cm^2).
- **JAX:** `src/vulcan_jax/atm_setup.py::sat_p_jax` uses one
  `jnp.where(T_C < 0, ice, liquid)` -- identical except at the buggy point.

## Still present in the JAX port

Inherited upstream defects VULCAN-JAX has not yet fixed. None affect the default
(gas-only, HD189) validated results -- they bite the convergence machinery or the
non-default condensation/photochemistry paths.

### Unweighted atom-conservation diagnostic
- **Where:** upstream `op.py:2551-2569` (`ODESolver.loss`), `build_atm.py:287-292`;
  JAX `src/vulcan_jax/outer_loop.py:285-290` (`_compute_atom_loss`).
- **What:** "atom loss" is `sum over layers of compo*y` with **no `dz` weighting**,
  so on a non-uniform grid it can read exactly-conservative vertical redistribution
  as loss. It feeds step acceptance and adaptive rtol.
- **Status:** deferred. It is a hot-path convergence heuristic; a `dz`-weighted
  column inventory diverges from master and needs a re-baseline. In practice JAX
  matches master to ~0.02 dex and validates conservation via the reservoir
  projection (`test_atom_conservation_projection.py`), so it is not producing
  wrong science on tested cases -- but it is the correct thing to fix eventually.

### Two-stream particular-solution pole
- **Where:** upstream `op.py:2731-2733` (`compute_flux`); JAX
  `src/vulcan_jax/photo.py:197`.
- **What:** `ll = -w0 / (1/mu^2 - (1-w0)/edd^2)` is singular at a physically
  ordinary single-scattering albedo (`w0 ~ 0.44` for the default `edd=0.5`, 48
  deg). A NaN guard exists but does not regularize the near-pole blow-up, so
  actinic flux / photolysis can overflow or flip sign as composition passes
  through that optical state.
- **Status:** deferred. The fix is the analytic resonant limit or a stable
  boundary-value solve, which needs its own RT validation.

### Condensate handling in the solver (active-condensation only)
Three inherited condensation-path issues; all no-op when `non_gas_sp` is empty
(the default), so they only matter for cloud runs.
- **Condensate mass in the gas mean molecular weight** -- upstream
  `op.py:2544-2546`, `build_atm.py:575-579`; JAX `atm_setup.py:363`
  (`compute_mean_mass` sums every species it is given). Condensate mass can pollute
  the gas `mu` (scale height) when condensation is active. Fix: restrict the sum to
  `gas_indx` across `compute_mean_mass` and its callers.
- **Condensates excluded from the convergence metric** -- upstream `op.py:1046-1063`;
  JAX `outer_loop.py:649` (`condense_zero_conv_mask` zeroes `non_gas_sp` columns in
  `longdy`). A cloud column can be declared steady while condensates still evolve.
- **Condensates excluded from the local error norm** -- upstream `op.py:3023-3037`;
  JAX `outer_loop.py:721-751` (`non_gas_present` gas-only error denominator). The
  adaptive step can accept a step whose largest error is in a settling condensate.

### Off-path data typo
- `src/vulcan_jax/thermo/SNCHO_photo_network_C3.txt` still carries the C1 CH2CN
  `1.00E-20` typo. No shipped config or sibling repo selects this variant, so it
  changes nothing today; fix it if C3 chemistry is ever activated.

## Scope / verification

- C1/C2/C3 verified against the fetched `vm_branch` source; they keep
  `tests/test_conden_profile_builder.py` green (7 passed) and H2O saturation
  continuous through 273 K. The still-present items were verified to exist in the
  JAX code this pass.
- Not fixed here because they do not apply or do not change results: the stale
  generated-kernel / `-n` regeneration workflow (JAX uses content-hashed codegen +
  a fail-fast network guard), the atomic-P / pressure column collision (JAX reads
  `fc["P_1"]`), the dense ~1 GiB Jacobian (JAX bands it), the dark-column `nanmax`
  crash (JAX is vectorized), FastChem I/O concurrency (per-run isolation), the
  `NH3 -> NH3_l` condensation mismatch (JAX correctly uses `NH3_l_s`), plus the
  upstream approximations, data-file duplicates, and documentation/packaging items,
  which are either defensible or off the active paths.
- Not re-verified in the JAX port (check before relying on those paths): ignored
  per-layer FastChem convergence status (`op.py` monitor file), the escape
  RHS/Jacobian (`op.py:996-1003`/`2181-2186`), and the optical-depth vs
  single-scattering-albedo opacity consistency (`op.py:2659-2676` vs `2700-2706`).
