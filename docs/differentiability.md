# Differentiability

This page states what VULCAN-JAX can differentiate, how to do it, and how
accurate the result is. It was moved out of the README in 2026-07 to keep that
file short.

## The rule

A quantity is differentiable **when it reaches the runtime as a JAX array**. That
happens in one of two ways:

1. You supply it directly into the runtime pytrees (`AtmStatic`, `RateInputs`,
   the initial `y`, and most of `PhotoStaticInputs`).
2. An **on-graph builder** produces it. `rates_jax` builds `T -> k`.
   `atm_jax.build_atm_static` builds the whole atmosphere structure (`pco`,
   `Tco`, gravity, composition to `M`, `dz`, `Hp`, `Dzz`, `vm`, `vs`).

A scalar parameter that a host-side setup formula expands into those arrays
becomes differentiable once that formula is on the graph. After
`build_atm_static`, this covers the atmosphere cascade.

Drive the inner `integ._runner`, not `OuterLoop.__call__`. The public call copies
state to the host to write `.vul` output, which breaks tracing.

## Condensation is not differentiable through

With `use_condense=True` the converged state comes from a finite condensation
window plus a `fix_species` pin that snapshots the condensate reservoir at a
transient moment. That snapshot is not a smooth steady state. The pinned-species
forward-mode tangent disagrees with re-converged finite differences at order
unity (about 0.91 relative). The set of actively condensing layers and the NH3
cold-trap level also switch discretely with temperature.

The low-level kernels (`conden.sat_p_jax`, `conden.build_conden_profile`) stay
differentiable. The completed pinned model does not:

- `steady_state_input_sensitivity` refuses a condensation state.
- `steady_state_reaction_sensitivity` returns only a ranking that is conditional
  on the frozen reservoir.
- There is no supported Fisher or retrieval-inference path through condensation.

The project-level contract document `condensation_differentiation.md` records the
full scope and rationale. It lives in the umbrella project's shared `docs/`
directory, which is present only in the multi-repo local checkout.

## What you can differentiate (forward mode, end to end)

| Physical input | How |
|---|---|
| Reaction rates `k`, forward and reverse | Supply `k_arr`. For a reaction ranking use `steady_state_reaction_sensitivity` |
| Temperature `T` (per-layer array) | `atm_jax.build_atm_static` rebuilds `M`, `dz`, `Hp`, `Dzz`, `vm`, `vs` from `Tco`. Also rebuild `k(T)` with `rates_jax.build_rate_array` |
| Surface gravity, planet radius `Rp` | `build_atm_static`. `gs` is resolved as `G*Mp/Rp^2` by `atm_setup.surface_gravity` and enters the graph as the resolved leaf. There is no `gs` knob |
| Pressure grid (`P_b`, `P_t`) | `atm_jax.pco_from_endpoints(P_b, P_t, nz)` gives the `pco` leaf, which reaches `M`, `Dzz`, and `dz` |
| Molecular and thermal diffusion `Dzz`, `vm`, `vs` | `build_atm_static` carries the Moses `T -> Dzz` fit, `vm`, and the Cloutman settling formulae on the graph |
| Arrhenius coefficients, NASA-9 thermodynamic data | `rates_jax.build_rate_array(..., rate_coeffs={"a": ...})` and `nasa9_coeffs`. One hardcoded Troe row is excepted |
| Eddy diffusion `Kzz`, advection `vz` | `atm._replace(Kzz=...)`, or `atm_setup.kzz_profile_jax` for the deep and maximum `Kzz` |
| Boundary fluxes, deposition velocity | Supply `top_flux`, `bot_flux`, `bot_vdep` |
| Initial abundances `y0` | Perturb `y0` directly |
| Metallicity `[M/H]`, C/O ratio | A `y0` tangent (see below) |

### Metallicity and C/O are y0 tangents

A converged closed column forgets the initial speciation. The steady state
depends on the conserved elemental totals, not on how those atoms were first
distributed. So the correct metallicity derivative scales the metal-bearing
initial abundances, and the correct C/O derivative scales C-bearing species
against O-bearing ones.

```python
import jax, jax.numpy as jnp
from vulcan_jax import composition

# compo_array column 0 is H in the default atom_list order, so [:, 1:] is metals.
metal = jnp.asarray((composition.compo_array[:ni, 1:].sum(1) > 0).astype(float))

def run_from_y0(y0):
    final = integ._runner(state0._replace(y=y0), atm)
    return final.y / final.y.sum(1, keepdims=True)

_, dlnVMR_dlnZ = jax.jvp(run_from_y0, (y0,), (y0 * metal[None, :],))
```

This is the derivative behind the published `d ln SO2 / d ln Z = 2.6` result for
WASP-39b.

### Building the differentiable atmosphere

```python
phys, spec = atm_jax.make_physical_inputs(cfg, var, atm, species_list)
atm_static = atm_jax.build_atm_static(phys._replace(Tco=new_T), spec)
```

`build_atm_static` reproduces the production `make_atm_static` field for field to
machine precision for the default configuration, which is `atm_type` `file`,
`analytical`, or `isothermal` with `use_moldiff` on. That is what the runner
uses. See `examples/grad_physical_example.py`.

Two non-default modes differ, and in both cases `build_atm_static` is the more
self-consistent of the two. With `atm_type='table'` it recomputes the interface
pressures from the rewritten grid, where production keeps a stale `pico`. With
`use_moldiff` off it computes `Ti` and `Hpi` as interface averages, where
production leaves them at legacy defaults; that difference is inert at runtime.

### Condensation follows a live temperature profile

`conden.make_conden_spec` extracts the temperature-independent metadata once per
config on the host. `conden.build_conden_profile(spec, Tco, pco, n_0, Dzz)` then
rebuilds every temperature- and structure-dependent condensation array on the
graph: saturation number densities, the growth and diffusion `Dg` terms, the H2O
and NH3 relaxation inputs, the NH3 cold-trap index, and the fix-species
saturation mixing ratios.

The builder is jit-, vmap-, and jvp-compatible, and the runner already reads
these arrays from the `ProfileVars` carry every step.
`OuterLoop._build_conden_static` delegates to the same builder, so host setup and
on-graph rebuild share one implementation.

The cold-trap index is an `argmin`, so it is an integer with no tangent. A
temperature tangent moves the saturation curves smoothly, but the active-layer
set and the cold-trap index change layer by layer. Forward-mode derivatives are
therefore valid only away from those switches, the same caveat as any phase
boundary.

## What you cannot differentiate yet

| Blocked input | Why | What to do instead |
|---|---|---|
| `d/d T_irr` through the Heng et al. (2014) profile | `analytical_TP_H14` is on the graph, but forward mode through `jax.scipy.special.expn` is very slow over a deep column | Differentiate the per-layer `Tco` leaf, or use a cheaper `T(P)` parameterization |
| Stellar flux scale and spectrum | `sflux_top` and the room-temperature cross sections `cross_J` and `absp_cross` are closure-baked into `outer_loop._make_photo_branch`, not read from a runtime pytree | Not exposed. This needs a runner-level input, not a pytree field |
| The cross-section temperature rebake | `photo_setup._bin_T_dependent` re-interpolates cross sections per layer on the host at setup | The temperature-dependent cross sections do ride the carry (`s.pv.p_cross_J_T`, `p_absp_T_cross`), so they are differentiable as arrays. The static cross sections and the `T`-to-cross-section map are not |

FastChem is a hard wall because it is a subprocess: the scalar map from `[M/H]`
to the equilibrium speciation at `t=0` is not differentiable. This rarely
matters, because a converged closed column forgets the initial speciation and the
`y0` tangents above are the scientifically correct derivatives.

The `const_lowT` Newton residual (`ini_abun._abun_lowT_residual`) is
differentiable with respect to the elemental ratios `O_H`, `C_H`, `He_H`, and
`N_H` for the reduced H2/H2O/CH4/He/NH3 system. The shipped `ini_abun` entry
point reads them as Python floats, so call the solver directly with JAX arrays to
get that gradient.

Host-side file readers (`photo_setup.py`, `composition.py`, and the CSV loaders
in `atm_setup.py`) are not differentiable by design. Build the corresponding
pytree directly with JAX arrays instead.

## Forward mode

`lax.while_loop` supports `jvp`, so one forward pass differentiates the whole
converged integration.

```python
import jax
from vulcan_jax.jax_step import make_atm_static

state0 = integ._pack_state_from_runstate(rs)
atm    = make_atm_static(data_atm, ni, nz, cfg=integ._cfg)

def run(Kzz):
    final = integ._runner(state0, atm._replace(Kzz=Kzz))
    return final.y / final.y.sum(axis=1, keepdims=True)

ymix, dymix = jax.jvp(run, (atm.Kzz,), (atm.Kzz,))
```

This is validated end to end on a full HD 189733b production run with
photochemistry on and about 1300 accepted steps. The `jvp` tangent matches
re-converged centered finite differences to better than 0.1% on the responding
levels, with correlation above 0.9999. The route never inverts `df/dy`, so it
stays well posed where the reverse-mode adjoint does not. See
`examples/grad_jvp_example.py`.

**Temperature gradients need the rate rebuild.** The runner's `k_arr` is frozen
at setup by the host-side NumPy `rates.build_rate_array`, so a `d/dT` jvp must
rebuild it on the graph with `rates_jax.build_rate_array`, which is bit-exact to
about 5e-14 against the NumPy build. `atm_jax.build_atm_static` rebuilds `M`,
`dz`, `Hp`, and `Dzz(T)`, so those are no longer frozen. Only the host-side
cross-section temperature interpolation stays frozen, and that is second order.
Forward-mode `d/dT` is validated against finite differences: HD 189733b dominant
species to 3-4 significant figures, and WASP-39b SO2 to correlation 1.0.

## Reverse mode: the steady-state adjoint

Reverse mode answers the many-inputs, one-output question: which reactions set
the converged abundance of a species. One adjoint solve returns
`dL/d ln k_r` for every reaction, where finite differences would need one
re-converged model each.

```python
import jax.numpy as jnp
from vulcan_jax import composition
from vulcan_jax.steady_state_grad import steady_state_reaction_sensitivity

def loss(y):                       # log10 SO2 mixing ratio at its peak layer L
    return jnp.log10(y[L, so2] / y[L].sum())

dL_dlnk = steady_state_reaction_sensitivity(   # shape (nr+1,)
    loss, y_star, k_arr, atm, net,
    compo_array=composition.compo_array[:ni], dz=dz,
    integ=integ, converged_state=final_state,
)
```

### How it works

`lax.while_loop` blocks `vjp`, so this is the steady-state adjoint of the body
map, not backpropagation through the loop. At convergence `G(y*) = y*`, and
`(I - dG/dy)^T z = v` is solved with the integrator's own regularized step as the
operator, in log-abundance coordinates, with the conserved-mass null space
deflated.

The solve uses LGMRES. That choice is measured, not incidental: restarted GMRES
oscillates on this operator and a raw Neumann iteration diverges, because the
operator is indefinite and singular. Earlier attempts that took the adjoint of
the bare residual `df/dy` all failed. On a closed column that residual is both
singular, from mass conservation, and severely ill-conditioned, from stiff
chemistry. That is why the solver-map route exists.

### Accuracy

- **Default `solver_map="renorm"` reaches percent level.** HD 189733b CH4 lands
  about 0.7% from finite differences, because `y*` is a roughly 1e-9 fixed point
  of the renormalized map the loop actually iterates. HD 209458b forward rows
  improve from 35% to 1%. A 2026-07-03 campaign across HD 189733b, HD 209458b,
  and WASP-39b, checked against both re-converged finite differences and
  forward-mode `jvp` (which agree to 0.02%), showed the residual error is a
  linearized-map effect. It is not a convergence-criterion mismatch: finite
  differences are invariant across `yconv` from 1e-2 to 1e-4, and stricter
  convergence, a `body_dt` scan, and a larger LGMRES budget do not move it.
- **With photochemistry on, supply the runner context.** The default
  `photo_recompute_k="auto"` reaches percent level when given
  `runner_photo_static=integ._photo_static` and `converged_state=final`, or
  `integ=integ` and `converged_state=final`. It rebuilds `J(y)` through the
  runner's two-stream radiative transfer on each application, so the operator
  carries `dJ/dy`. Without it, frozen photolysis leaves those rows at leading
  order, about 11% off. With the default renorm map, WASP-39b SO2 dominant rows
  reach 0.2% and 0.1%. The cost is one radiative-transfer solve per Krylov
  matrix-vector product. Pass `photo_recompute_k=None` only to reproduce the
  older frozen-photolysis result.
- **Legacy `solver_map="bare"` reaches a few percent.** It linearizes the raw
  Ros2 step, for which `y*` is only a roughly 1e-4 fixed point: HD 189733b CH4
  about 6.6%, WASP-39b OH and H2 about 11%. It is kept only to reproduce
  pre-2026-07 behavior.
- **One case stays hard.** HD 209458b CH2OH near-equilibrium reverse rows remain
  ill-conditioned, with an LGMRES residual near 0.1 even with the renorm map.
  The default diagnostics flag them. Treat those as a ranking, and use forward
  mode where an exact value is needed.

### The `body_dt` probe step

`body_dt` sets the solver regime. It is an adjoint-only knob; the forward model
is untouched. The default `body_dt=1e7` sits in the measured low-residual regime:
on HD 189733b the residual is 0.04-0.15 and the twin ensemble lands 0.3-6% from
finite differences, with a mean of 3.5%.

At `body_dt` of 1e8 or more, which was the old default, the solve stagnates. The
residual rises to 0.2-0.7, because the body map has unstable top-layer H and H2
eigenmodes and the matrix-vector product's floating-point floor grows with `dt`;
single-solve magnitudes then bounce by about 25%. At `body_dt` near 3e6 the solve
converges fully but deterministically underweights slow chemistry, biasing the
result by about 28%.

The safe window is column-dependent. Scan a few values and keep the lowest
`info["resid"]`; the map is recorded in the comment on `BODY_MAP_DT`.

The gradient is returned as the mean over an `n_solves` twin ensemble, three by
default, using deterministic seeded right-hand-side perturbations. The twin
spread in `info["ensemble_spread"]` is the honest error bar on the magnitude. The
**ranking** is robust in every non-divergent regime: dominant reactions stand one
to two orders of magnitude above the noise with stable signs.

### Cross-regime validation (2026-07-02)

WASP-39b with the SNCHO network, photochemistry on, 1150 reactions, and an SO2
loss is an easy regime. Residuals are 0.005-0.05 at every `body_dt` from 3e6 to
1e8, answers are `dt`-insensitive to better than 1%, twin spread is about 6e-4,
and the ranking reproduces the paper exactly.

On HD 189733b three loss regimes degrade, and the default-on diagnostics flag all
three:

- **Buffered species** (H2O and CO in the mid column). The spread warns on the
  twin-noisy tail, but the insensitivity conclusion itself is robust.
- **Upper-atmosphere losses.** True stagnation; the median residual warns.
- **Losses coupled to the unstable top-layer H and H2 modes.** Residuals are
  tiny, but `ensemble_spread` is about 0.9 and `info["pair_antisym"]` is about 1.
  That forward/reverse pair-antisymmetry check catches internal inconsistency
  that residuals miss.

Mid-column composition losses, which are the design use case, are safe because
their cotangent is orthogonal to the unstable subspace.

### Physical-input gradients through the same solve

`steady_state_input_sensitivity(loss, y_star, k_arr, atm, net, p0, rebuild, ...)`
returns `dL/dp` for an arbitrary input pytree, for example a full `(nz,)`
temperature profile, in one adjoint solve plus one VJP. It needs a
differentiable `rebuild(p) -> (k(p), atm(p))`, which `rates_jax` plus `_replace`
provides; non-thermal rows are spliced in frozen.

The rebuild is consistency-checked at `p0` and warns or refuses on a mismatch.
The renormalization is differentiated through `atm(p).M`, because temperature
moves the rebalance. The function returns the chemistry path
`dL/dy* * dy*/dp`; a spectrum loss's direct `dL/dp` term, such as temperature in
the radiative transfer, is added separately by the caller. Forward-mode `jvp`
remains the exact route for a handful of directions.

### What the body map contains

The body map is `ros2_step`, plus renormalization, plus photochemistry, plus
optional `body_terms`. Build the last with
`make_body_terms(integ, converged_state, atm_static)`, which also returns the
correctly spliced `atm`, including a live `vm` when `use_vm_mol` is on.

`body_terms` carries the per-step processes that a non-default config turns on:

- The in-window **condensation** composite. Condensation and evaporation rate
  rows are recomputed from `y`, giving the `dk/dy` feedback that is analogous to
  the photolysis `dJ/dy`, plus the H2O and NH3 relaxation kernels and the
  gas-only partial rebalance.
- The **`fix_species`** pins, for species clamped inside the Ros2 step.
- The **layer-0 boundary pins** (`use_fix_all_bot`, `use_fix_sp_bot`, and a
  tripped hycean H2-He).

Everything else stays outside the linearization: clipping is the identity almost
everywhere, ion charge balance is unsupported and raises, and the escape-flux
recompute and the composition-to-mu atmosphere refresh are frozen and second
order.

**A fingerprint guard raises rather than return a silently wrong gradient.** A
state converged with condensation active is refused without matching terms,
active ion rows are always refused, and frozen photolysis warns.

Before trusting the gradient on any non-default config, run:

```python
audit_adjoint_scope(y_star, k_arr, atm, net, cfg=..., final_state=...,
                    loss_fn=..., body_terms=...)
```

It classifies every dropped process for that config as error, warning, or info,
confirms the converged geometry was spliced into `atm`, and measures the
**per-cell** fixed-point defect `|G(y*) - y*|/y*` of the exact map the solver
uses. That per-cell measurement matters: a pinned bottom trace row can be 100%
off while the global max-norm `fp_err` reads 1e-9. It also reports the defect
inside the loss's own footprint.

### Diagnostics

Diagnostics are on by default. Warnings fire on a poor LGMRES residual, a loose
fixed point, or a large twin-ensemble spread. An LGMRES breakdown or a
rank-deficient deflation basis raises instead of returning garbage.

`info["null_quality"]` reports how null the deflated conserved-mass directions
actually are, relative to the operator's scale. It is about 3e-5 on a healthy
closed HD 189733b column. The atom-count vectors are only approximately null,
because the diffusion discretization is not exactly conservative under the `dz`
weights. A value of order unity means conservation is broken, for example by open
boundary fluxes.

The solve itself is host-side SciPy LGMRES, because JAX has no LGMRES. It runs
once after convergence, off the hot path, and warm-start cycles stop early once
SciPy reports convergence.

See `examples/grad_reverse_example.py` and
`tests/test_steady_state_reaction_sensitivity.py`.
