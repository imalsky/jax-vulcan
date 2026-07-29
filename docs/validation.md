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
| HD 189733b | 7.3e-6 |
| HD 209458b | 1.4e-4 |

WASP-39b converges in the same 4548 accepted steps in both codes. The largest
single-cell differences reach order unity, but only in trace species sitting at
the numerical floor. For cells with a mixing ratio above 1e-10, the largest
single-cell difference in any run is 2.5e-2.

Maximum atom-sum drift is 2.5e-5 for HD 189733b, 3.1e-4 for HD 209458b, and
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
- **A convergence stall fallback** (`conv_stall_window`) handles
  heavy-hydrocarbon oscillation.
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

`use_hybrid_vm_mol` is on by default together with `use_vm_mol`. It converges on
the robust upwind scheme, then switches to central difference and finishes.

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

### The K2-18b config is a ported collaborator case, not a validated benchmark

`configs/K2-18b.yaml` is a port of a collaborator-supplied legacy `vulcan_cfg`
file (100x-Neptune metallicity, apoastron, H2O clouds, DMS reactions removed).
It exercises the condensation and settling path that no other shipped config
turns on: `use_condense` and `use_settling` with `condense_sp: [H2O]`,
`non_gas_sp: [H2O_l_s, S8_l_s]`, `humidity: 0.8`, `use_ini_cold_trap`,
`fix_species` on H2O plus its condensate, and `use_adapt_rtol` with
`post_conden_rtol: 1.5`.

It is covered by `tests/test_cfg_examples.py` (pre-loop setup) and has been run
locally for 1402 accepted steps (t = 9.6e2 s, 89 delta-rejections, 57 s wall):
finite throughout, initial cold trap placed at level 48, H2O condensate reaching
a 3.9e-3 mixing ratio. The run was launched with `count_max = 400`, which is
**not** the count it stopped at — `use_vm_mol` + `use_hybrid_vm_mol` are both on
here, and the hybrid phase flip re-seeds the dynamic step budget
(`count_max_dyn = count + 2000` on a phase-0 convergence), so the static cap is
not the operative terminator. It is **not** a
validated benchmark — there is no VULCAN 2.0 cross-check for it, and three
provenance items are unresolved:

- `remove_list: [315, 316, 817, 818]` resolves in the vendored
  `SNCHO_photo_network_2025.txt` to `NH3 + CH -> NH2 + CH2` and
  `SH + NO2 -> HSO + NO` (plus their reverse slots), which are not DMS
  reactions despite the source config's "noDMS" name. The author's copy of the
  2025 network is presumably indexed differently.
- The source config's `bot_BC_flux_file` (`atm/BC_bot_SdepOnly_noSorg.txt`) was
  not supplied. Inert as configured, since `use_botflux` is false.
- `S8_l_s` is in `non_gas_sp` (so it settles, and carries `r_p`/`rho_p`) but
  `S8` is not in `condense_sp`. `op.conden` gates each condensation reaction on
  the **gas-phase** name, so `S8 -> S8_l_s` (printed index 1069) never activates
  and `S8_l_s` — whose only source is that reaction — stays identically zero.
  Reproduced in both codes; whether it is intended is a question for the author.

Four further divergences from upstream defaults were checked and are recorded
here so they are not re-diagnosed as port bugs:

- **`use_relax: [H2O]` short-circuits the H2O condensation kinetics.**
  `op.conden` zeroes `k[re]`/`k[re+1]` for a species in `use_relax` and routes it
  through `h2o_conden_evap_relax` (implicit-Euler relaxation toward
  `sat_p * humidity`) instead. The case therefore exercises the relaxation path,
  not kinetic condensation. Ported faithfully (`conden.py` gives relax-shorted
  rows `coeff = 0.0`).
- **`mtol: 1e-14` / `mtol_conv: 1e-16` are 8 and 4 orders looser than upstream's
  `1e-22` / `1e-20`.** `mtol` masks entries out of the Ros2 embedded
  truncation-error estimate (`delta = |sol - yk2|`, consumed as
  `h_factor = 0.9*(rtol/delta)^0.5`) and clips negative trace entries; it does
  **not** freeze the species. At the EQ-initialised state 83.7% of
  species-levels fall below `1e-14` (vs 76.6% below `1e-22`), and 17 of 93
  species lie entirely below it — those are integrated without step-size error
  control. Faithful to the config as written, but a deliberate accuracy
  trade the author should confirm.
- **`high_temp_cut: true` never fires here.** The atmosphere file peaks at
  2059.9 K, below `high_temp_cut_K: 3500`, so no layer satisfies the cut.
- **`use_vm_mol` / `use_hybrid_vm_mol` are both true**, where upstream ships
  `use_vm_mol = False` ("under testing") and has no `use_hybrid_vm_mol` or
  `high_temp_cut` knob at all — the source config descends from a vm_branch
  fork. This is also why `count_max` is not the operative terminator above.

Two knobs in the source config (`no_photo_ini_conden`, `use_other_ele`) appear
nowhere in upstream `op.py`, `build_atm.py`, `store.py`, or `vulcan_cfg.py`, so
they were inert in the author's runs too. `fix_species_time` is likewise dead
upstream — `op.py` gates the pin on `stop_conden_time` only — which is why
dropping it on import costs nothing. `conver_ignore: [HC3N]` is correct and
needs no change: HC3N has five sources and zero sinks in this network, exactly
the case upstream's own comment cites.

### The Earth config cannot run, in either code

`configs/Earth.yaml` ships but does not run, and it does not run in VULCAN 2.0
either. It lists Ar in `atom_list` and `const_mix`, but Ar appears in no reaction
of the SNCHO network, so it is not a network species. VULCAN 2.0's
`build_atm.ini_y` calls `species.index(sp)` unconditionally and fails with
`ValueError: 'Ar' is not in list` at `build_atm.py:200`. This is reproduced end to
end on the shipped Earth example.

Inert background gases without network reactions were never live VULCAN 2.0
physics, so VULCAN-JAX does not invent them. `runtime_validation` rejects such a
`const_mix` upfront with an explanation instead of failing mid-setup; see
`tests/test_validation_const_mix_conden.py`. The config is kept verbatim as
upstream ships it. Running it means removing Ar from `const_mix` and `atom_list`.
Note that VULCAN 2.0 would additionally poison its atom-conservation diagnostics
with NaN for any `atom_list` atom carried by no species, through a 0/0 in
`atom_loss`.
