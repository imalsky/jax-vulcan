# Validation against VULCAN 2.0

This page records how closely VULCAN-JAX reproduces VULCAN 2.0, what is
compatible, and where the two codes differ on purpose. It was moved out of the
README in 2026-07 to keep that file short.

Defects that VULCAN-JAX corrects in the original code are listed separately in
[`corrections_to_original_code.md`](corrections_to_original_code.md), which also
states the parity policy for divergences.

## Per-component agreement

Measured on the default `NCHO_photo_network` (69 species, 878 reactions, 439 of
them forward) against the VULCAN 2.0 oracle. Each row names the test that
reproduces it; re-run those tests for current numbers on your own host.

| Layer | Maximum relative error | Backing test |
|---|---|---|
| Forward rate coefficients (439 forward) | bit-exact | `test_rates` |
| Reverse rates, Gibbs-derived | 1.4e-14 | `test_gibbs` |
| Atmosphere structure (`pco`, `Tco`, `Kzz`, `M`) | bit-exact | `test_default_master_parity` |
| Initial abundances, FastChem path | bit-exact | `test_default_master_parity` |
| Chemistry right-hand side | about 2e-13 worst cell; bulk species about 1e-16 | `test_chem_rhs_codegen` |
| Chemistry Jacobian, analytical vs `jacrev` | 2.8e-15 | `test_chem_jac_sparse` |
| Diffusion operator | about 1e-5, bound by floating-point cancellation; Jacobian blocks bit-exact | `test_diffusion` |
| Upwind molecular diffusion (`use_vm_mol`) | drift `vm` bit-identical to the reference formula; operator about 2.7e-6, cancellation-bound | `test_diffusion_variants`, `test_diffusion_production_kernel` |
| Block-Thomas solver | 3e-15 | `test_block_thomas_diag` |
| Single Ros2 step | 1.6e-9 | `test_ros2_step` |
| Photochemistry kernels (tau, flux, J) | 7e-16, 1.2e-11, 6.8e-12 | `test_photo` |

## Converged-run agreement

Forcing both codes through the same accepted Ros2 steps from the same
HD 189733b state, the maximum relative difference stays between about 1e-15 and
2e-13 over a 1-20 step integration. That is the narrowest test: it checks only
that one step evaluates the same way in both codes.

For fully converged runs, the median fractional difference in volume mixing ratio
across all species is:

| Planet | Median fractional difference |
|---|---|
| WASP-39b | 1.1e-9 |
| HD 189733b | 3.3e-6 |
| HD 209458b | 1.4e-4 |

The largest single-cell differences reach order unity, but only in trace species
sitting at the numerical floor. For cells with a mixing ratio above 1e-10, the
largest single-cell difference in any run is 5.2e-4.

**These numbers only mean anything with the composition matched.** VULCAN-JAX and
upstream ship different `solar_element_abundances.dat` files; unmatched, HD 189733b
disagrees by ~20% median rather than 3.3e-6. See "Elemental abundances are a
config choice" below.

Maximum atom-sum drift is 2.6e-5 for HD 189733b, 3.1e-4 for HD 209458b, and
1.7e-5 for WASP-39b. All are far below VULCAN 2.0's default fractional
atomic-loss budget of 1e-1.

## Compatibility surface

| Surface | Compatible |
|---|---|
| Config format | YAML in `configs/*.yaml`, with the same knob names as VULCAN 2.0 |
| Network, atmosphere, and cross-section files | Yes, same parsers and vendored data |
| `.vul` output schema | Yes, same public keys, shapes, and dtypes |
| `plot_py/` scripts | Yes, unchanged |
| Solver | Ros2 only. The non-Ros2 solvers were dead code in VULCAN 2.0 |

## Deliberate differences

- **Live plotting fires between JIT-compiled step batches.** The cadence is
  faithful, but the call site is not identical.
- **The output writer synthesizes `J_sp` and `Jion_sp` at pickle time** rather
  than accumulating them incrementally.
- **A convergence stall fallback** (`use_conv_stall`) exists but is **off in
  every shipped config** and has no upstream counterpart. Measured 2026-07-30, it
  never fires on any shipped case, so nothing shipped decides convergence on a
  criterion VULCAN 2.0 lacks. `termination_reason` (0 running, 1 converged,
  2 runtime, 3 step count, 4 stall, 5 non-finite) reports which exit was taken;
  `end_case` cannot, because it reports 1 for both a normal and a stall
  convergence.
- **Config validation is stricter.** Non-network `const_mix` keys and unsupported
  `condense_sp` entries fail at validation time with an explanation. VULCAN 2.0
  crashes deep in setup for the first and silently zero-rates the second.
- **`use_print_delta` is not supported.** A per-step host print is impractical
  inside the JIT-compiled runner. The key is removed from the config surface and
  rejected with a migration message, as are `fix_species_time` (the pin gate is
  `stop_conden_time`) and `gs`.

### Upwind molecular diffusion

`use_vm_mol` uses the interface-centered drift velocity from
`op.update_mu_dz`, with shape `(nz-1, ni)` and a harmonic-mean interface scale
height.

The drift `vm` is **recomputed in the loop**, every `update_frq` steps, from the
live mean molecular weight. This mirrors upstream's "also update vm" behavior. It
is deliberately not frozen at setup the way `Dzz` and `Ti` are, because `vm`
depends on `1/Hpi` and `Hpi` tracks the mean molecular weight.

Freezing it was an earlier bug, and it mattered: it biased a
molecular-diffusion-dominated upper atmosphere at low `Kzz` by up to about
1.7 dex. The refresh collapses the gap to 0.16 dex or less, which is the
stiff-regime convergence floor, and it matches upstream's step count.

Two smaller notes. VULCAN-JAX keeps `vm` consistent at the bottom boundary
(`j=0`) in every mode; upstream `op.diffdf_settling_vm` drops `vm` at `j=0`,
which is inconsistent with its own `op.diffdf_vm`, so the doubly-non-default
combination of `use_vm_mol` with `use_settling` differs from upstream at that one
cell. VULCAN-JAX also ports the correct `axis=0` layer-averaging form from
`op.update_mu_dz`; the copy of that formula in `build_atm.py` omits `axis=0`,
which is a latent species-mixing bug.

### Hybrid molecular diffusion

`use_hybrid_vm_mol` is **off in the VULCAN 2 parity configs and on only in
`HD189_vulcan3.yaml`**. It converges on the robust upwind
scheme, then switches to central difference and finishes, matching the sequence
upstream describes: first-order upwind, then convergence, then central difference
for at most 2000 further steps.

Because the runner is one JIT-compiled `lax.while_loop`, this is implemented as an
**in-loop phase flip** rather than a host-side two-stage driver. A carry blend
factor `hybrid_use_vm` starts at 1.0 for upwind and flips to 0.0 for central the
first time phase 0 reaches the convergence criteria. The flip resets the
convergence trackers and extends the step and runtime budgets the same way the
reference two-stage driver does.

The returned state is a central-difference fixed point, so forward-mode `jvp`
through the runner and the steady-state adjoint both apply to it unchanged. Set
`use_vm_mol=False` for a fixed central-difference scheme, which is what batched
emulator generation wants, because it is deterministic.

## Numerical notes

### Chemistry right-hand-side parity

`make_chem_funs.build_chem_rhs(net)` emits per-network code in the same order as
VULCAN 2.0's SymPy-generated `chemdf`: paired reactions, stoichiometry-repeated
multiply chains, asymmetric third-body M, and products-before-reactants
accumulation. It is faithful to about 1 unit in the last place per multiply chain.

### Atom conservation projection

JIT compilation lets XLA reorder floating-point operations, so the large
production and loss terms that should cancel exactly do not round identically.
Fused multiply-add rewriting breaks the stoichiometric null space of the
chemistry right-hand side.

`jax_step._project_chem_rhs` distributes the per-layer atom residual, about 5e-13
relative per step, across one abundant reservoir species per conserved element:
H2, H2O, CO, N2, and H2S on the sulfur network. This enforces exact conservation
of H, O, C, and N, plus S where the network carries it. The reservoir-to-atom
pairing is selected dynamically from `atom_list` via `jax_step._ATOM_RESERVOIRS`,
so any atom subset with an abundant reservoir is conserved rather than a
hardcoded set. Overhead is about 3% per step.

This is not only a diagnostic fix. Because the corrected right-hand side is used
inside each Rosenbrock stage, the guard changes the integrated solution. For an
HD 189733b test, disabling the projection raises the median fractional difference
against VULCAN 2.0 from 7.3e-6 to 3.9e-4.

VULCAN 2.0 has no equivalent correction. Its atom drift is comparable in
magnitude but arises from Python evaluation order instead.

### float64 is required

Rate constants span about 50 orders of magnitude, and float32 fails silently.
`jax_enable_x64 = True` is set at import time.

## Test-suite notes

The suite imports the **installed** `vulcan_jax`, because the package uses a src
layout. Development therefore requires an editable install. A non-editable
install would shadow the checkout and silently test stale code, so
`tests/conftest.py` fails collection with a clear message when `import
vulcan_jax` resolves outside the repo's `src/`.

Tests that compare against VULCAN 2.0 need the sibling `../VULCAN-master/`
checkout and skip cleanly when it is absent. They run the VULCAN 2.0 imports in
isolated subprocesses.

The slowest test is `tests/test_nh3_conden_batch_subprocess.py`, about 10 minutes
cold. A fresh subprocess parses and compiles the 1141-reaction low-temperature
Jupiter network to prove that batched NH3 condensation matches solo runs end to
end. JAX's persistent compilation cache makes identical re-runs much cheaper.

### The K2-18b configuration was removed on 2026-07-30

It was ported from a collaborator-supplied case and never converged. Measured on
the full 30000-step budget:

| scheme | steps | longdy | gating cell | verdict |
|---|---|---|---|---|
| hybrid upwind to central (as supplied) | 31002 | 25.80 | z=120, C3H2 | not converged |
| pure central difference | 30001 | 0.459 | (not reported) | not converged |
| upstream VULCAN 2.0, 3000-step budget | 3000 | 1.00 | z=0, OH | not converged |

The criterion is `yconv_cri = 0.01`, and at the end of the central run
`longdy/dt` is still 4.8 against a `slope_cri` of 1e-4, so it is not asymptoting
either. Atom conservation is fine throughout (worst 5.6e-02 on nitrogen), so this
is a genuine convergence failure rather than a misread diagnostic.

**The hybrid scheme is what makes it far worse, and the mechanism is exact.**
Isolating the two VULCAN 3 knobs at a 3000-step budget:

| variant | upwind/hybrid | high_temp_cut | longdy | gating cell |
|---|---|---|---|---|
| as supplied | on | on | 74111.21437235552 | z=124, H |
| high_temp_cut off | on | off | 74111.21437235552 | z=124, H |
| upwind off | off | on | 2.8486787705982635 | z=122, NS |
| both off | off | off | 2.8486787705982635 | z=122, NS |

The pairs are identical to every digit, so `high_temp_cut` contributes nothing
(this planet peaks near 2060 K, below the 3500 K threshold, so the cut never
fires). The upwind/hybrid scheme is entirely responsible.

The mechanism shows in the accepted-step counts. Phase 0 (upwind) is not allowed
to end a run: it flips to central difference when it converges, and the central
phase then gets `count + 2000`. If phase 0 instead exhausts its budget, the flip
still happens but the central phase gets only `count + 1000`. On this case phase 0
never converges, so the flip always fires on the budget:

    count_max 3000  -> 4002 accepted   (3000 + 1000)
    count_max 30000 -> 31002 accepted  (30000 + 1000)

The central phase therefore always starts from an unconverged upwind state with
half the intended budget. This is a faithful port of the upstream logic, not a
port defect; the scheme simply does not suit this planet.

Three provenance items were closed before the case was withdrawn:

- `remove_list` resolves correctly **by position** against
  `SNCHO_photo_network_2025.txt` (ni=93, nr=1192): 315/316 =
  `NH3 + CH -> NH2 + CH2` forward/reverse, 817/818 = `SH + NO2 -> HSO + NO`.
  Neither is a DMS reaction, despite the source config's "noDMS" name. See
  correction C14 for why position rather than the printed id is canonical.
- The missing `atm/BC_bot_SdepOnly_noSorg.txt` was inert, verified against the
  code path rather than asserted: `atm_setup.py:859` opens `bot_BC_flux_file`
  only when `use_botflux` is true (false here), and the second reader at line
  871 is gated on `use_fix_sp_bot` being the literal `True` (this config set
  `{}`).
- S8 condensation was not intended. `S8_l_s` appeared in `non_gas_sp` because
  both `S8` and `S8_l_s` are real network species that must be kept out of the
  mixing-ratio normalization, but `condense_sp` listed only `H2O`.

Two defects found in the config while checking it, both now moot but worth
recording as failure modes: it omitted `batch_max_retries` and silently took the
code default 64 instead of 110, and it enabled `use_adapt_rtol` while declaring
none of the six controller constants, so it silently took code defaults including
a decrease factor of 0.75, which is upstream master's value rather than
vm_branch's 0.5.

### The Earth configuration was removed on 2026-07-30

It did not converge, and neither does upstream VULCAN's own Earth example.

Upstream's example cannot even run as shipped, for two independent reasons. It
lists `Ar` in `atom_list` and `const_mix` while `Ar` is in no reaction of the
SNCHO network, so `build_atm.ini_y` dies at `build_atm.py:200` with
`ValueError: 'Ar' is not in list` (correction C13). It also omits `conver_ignore`,
`rtol_min`, `rtol_max` and `use_adapt_rtol`, all of which `op.py` reads, so it
raises `AttributeError` before that. Upstream's HD189 example has the same
missing-key defect.

Made runnable (Ar dropped, the four keys taken from upstream's own default
`vulcan_cfg.py`) and run at a 2000-step budget matched to VULCAN-JAX:

| code | longdy at step 2000 | gating cell | verdict |
|---|---|---|---|
| VULCAN 2.0 | 0.880 | z=81, C2H4 | "Integration not completed" |
| VULCAN-JAX | 54.33 | z=116, atomic C | step cap, not converged |

The convergence criterion is `yconv_cri = 0.01`, so neither is close. Shipping a
configuration that converges in neither code is misleading, so it was removed
rather than kept with a warning.

**One thing is recorded but not explained.** VULCAN-JAX with `diff_esc` emptied
lands at `longdy = 0.879` on the same cell (z=81, C2H4) as upstream reaches with
`diff_esc` active. The escape formula itself is not the difference: VULCAN-JAX's
`atm_refresh.update_phi_esc_jax` is algebraically identical to upstream's
`op.update_phi_esc` (same expression, same `-max_flux` floor) and both refresh on
the same `update_frq` cadence. So the near-coincidence is unexplained rather than
diagnosed, and it may simply be that both runs settle on the same
slowest-converging species. `HD209.yaml` is the only remaining shipped config
using `diff_esc`, and it converges and agrees with upstream to 1.4e-4 median, so
nothing shipped is known to be affected.

### Elemental abundances are a config choice

VULCAN-JAX and upstream VULCAN do not start from the same composition, and this
is the single most common way to get a wrong cross-code number.

| file | contents |
|---|---|
| `fastchem_vulcan/input/solar_element_abundances.dat` | Lodders 2019, rocky elements (P, Si, Ti, V, Cl, K, Na, Mg, F, Ca, Fe) suppressed to -3.0. **The default.** |
| `fastchem_vulcan/input/solar_element_abundances_lodders2009.dat` | Lodders 2009, every element at solar. **What upstream ships.** |

`fastchem_solar_abundance_file` selects, and every shipped config declares it
explicitly. `runtime_validation` accepts either preset exactly and rejects
anything else, naming both, because a hand-edited composition otherwise fails
silently.

Suppression is right for the shipped networks: they have no Mg, Si or Fe species,
so at solar dex FastChem locks oxygen into MgO/SiO2/FeO that the kinetics can
never release. For a C-H-N-O network the only value that actually differs is
**helium** (10.9864 upstream, 10.9232 here), which is inert, which is why the
difference hid as a flat offset in the deep column rather than as anything that
looked like a chemistry bug.

Measured on HD 189733b, both codes converged, floor 1e-12:

| | median | max |
|---|---|---|
| upstream defaults | 2.0e-01 | 8.8e+01 |
| + VULCAN-JAX network | 2.0e-01 | 8.8e+01 |
| **+ network AND composition** | **3.8e-06** | 4.5e-02 |
| control: upstream vs upstream | 7.0e-06 | 1.7e-02 |

Selecting the upstream preset in VULCAN-JAX reproduces the expected direction at
the deep layer: He +13.1%, H2O -19.3%, CO2 -29.1%, CO -14.1%, H2 -2.2%. It also
converges in 911 accepted steps instead of 1495, so composition moves convergence
as well as abundances.

The upstream preset carries upstream's values verbatim with **one** deviation:
P and S are swapped back into the order FastChem's own
`mass_action_constant.cpp` expects (`index_P=5`, `index_S=6`). Upstream ships
them the other way round; that is correction C12, and copying the file verbatim
would have shipped the bug.
