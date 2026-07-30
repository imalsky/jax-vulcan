# Verified facts — safe to send Shami

Everything here was established by fetching or by running, on 2026-07-30, and is
reproducible from this artifact directory. Nothing here is inferred from a
comment, a commit message, or the local `VULCAN-master` copy.

## A. Provenance, established by HTTP fetch from the real remotes

A1. `exoclime/VULCAN` master `vulcan_cfg.py` and `shami-EEG/VULCAN` master
`vulcan_cfg.py` are **byte-identical** (md5 `be64301868912c77ee1105ef52e0bbed`).

A2. `conver_ignore` upstream:
- `exoclime/VULCAN` master `vulcan_cfg.py:151` = `[]`
- `exoclime/VULCAN` master `cfg_examples/vulcan_cfg_HD189.py:148` = `[]`
- `shami-EEG/VULCAN` master `vulcan_cfg.py:151` = `[]`
- `shami-EEG/VULCAN` vm_branch `vulcan_cfg.py:157` = `['HC3N']`

A3. The 13-heavy-hydrocarbon `conver_ignore` list that VULCAN-JAX was shipping
exists in **no** upstream repository, on either remote, on either branch.

A4. Adaptive rtol. The only difference between master and vm_branch is the
decrease factor. Both use the same periods and the same increase factor:

| | master `op.py:836-848` | vm_branch `op.py:836-848` |
|---|---|---|
| decrease every | `count % 10` | `count % 10` |
| decrease factor | 0.75 | **0.5** |
| increase every | `count % 1000` | `count % 1000` |
| increase factor | 1.25 | 1.25 |

This does **not** match the values in the 2026-07-14 email
(`adapt_rtol_inc_period = 500`, `adapt_rtol_dec = 0.5`, `adapt_rtol_inc = 1.5`):
only `dec = 0.5` is in the branch. VULCAN-JAX had been carrying 500 / 0.5 / 1.5,
which matches neither remote. Those knobs are inert in every VULCAN-JAX config
that sets them, because `use_adapt_rtol` is false there.

A5. `high_temp_cut_K = 3500.0` and `high_temp_cut_P = 1e6` confirmed against
current vm_branch `vulcan_cfg.py:150-151`.

A6. `exoclime/VULCAN` master `cfg_examples/vulcan_cfg_HD189.py` does not define
four attributes that master's own `op.py` reads: `use_adapt_rtol`, `rtol_min`,
`rtol_max`, `use_fix_all_bot`. A pristine clone therefore cannot run that example
as shipped. (Verified by cloning `exoclime/VULCAN` at HEAD `8970337` and running
it.)

## B. Measured, VULCAN-JAX, controlled comparisons

Physical inputs identical within each config; only the named knob varies. The
JAX-only stall fallback is explicitly disabled unless stated.

B1. `conver_ignore` sensitivity, accepted steps to convergence:

| config | `[]` (master) | `['HC3N']` (vm_branch) | 13 + HC3N |
|---|---|---|---|
| HD189 | 1495 | 1495 | 1296 |
| HD209 | 1206 | 1206 | 1206 |
| WASP-39b | 1202 | 1202 | 1202 |

`[]` and `['HC3N']` give identical results everywhere tested: same step count,
same final `longdy`, same controlling species and level. Only HD189 responds to
the 13-species list, and only by removing species from the convergence metric.

B2. Every run in B1 exits on the normal convergence criterion, never the stall
fallback.

B3. HD189 convergence is gated by atomic **C** at 5.0e-7 bar under `[]` and
`['HC3N']`. HC3N is not the controlling species on this network, which is why
adding it to the ignore list changes nothing.

B4. WASP-39b converges in 1202 steps under all three lists and with the stall
fallback both on and off. It is completely insensitive to this choice.

B5. Molecular-diffusion schemes, HD189, VULCAN 2 parity config:

| scheme | accepted steps | ends in |
|---|---|---|
| central | 1495 | central |
| pure upwind | 1495 | upwind |
| hybrid (upwind → central) | 2102, switch at step 1500 | central |
| VULCAN 3 preset | 2820 | central |

B6. Final-state difference against the pure-central run, relative, over an
abundance floor of 1e-12, for H2O / CH4 / CO / HCN / C2H2 / H:

| scheme | median | max |
|---|---|---|
| pure upwind | 2.5e-06 | **2.6e+00** |
| hybrid | 2.3e-06 | **1.7e-04** |

The hybrid returns the central-difference answer. Pure upwind does not.

B7. Zhang, Shia & Yung (2013) analytic diffusive-separation benchmark, driving
the production VULCAN-JAX kernels:
- central (= converged hybrid): **0.8%** max fractional error against the
  analytic solution
- pure upwind: **45.7%** (numerically diffusive, under-separates)
- VULCAN-JAX vs the VULCAN 2.0 operator (`op.diffdf` / `op.diffdf_vm`):
  **0e+00** — exact agreement, so the port is faithful
- central goes negative above cell Péclet ≈ 2; upwind stays positive at every
  Péclet tested (0.3 to 5.1)

B8. Atom conservation, max |atom_loss|, converged parity runs:
HD189 2.6e-05, HD209 5.1e-05, WASP-39b 8.4e-07. All finite, no negative cells.

## B9. Independent cross-code check against a pristine upstream clone

A clean `exoclime/VULCAN` clone (HEAD `8970337`) was built and run. HD 189733 b,
converged both sides, abundance floor 1e-12, relative difference vs VULCAN-JAX:

| upstream run | steps | median | p90 | max |
|---|---|---|---|---|
| upstream defaults | 1131 | 1.993e-01 | 8.089e-01 | 8.765e+01 |
| + VULCAN-JAX network | 1081 | 1.993e-01 | 8.078e-01 | 8.767e+01 |
| **+ network AND composition matched** | 1600 | **3.756e-06** | 2.365e-03 | 4.480e-02 |
| *control:* upstream vs upstream | 1131 vs 1081 | 7.0e-06 | 5.3e-04 | 1.7e-02 |

Deep well-mixed column (p > 1 bar), fully matched, median relative difference:
H2 4.5e-10, He 1.3e-09, H2O 1.2e-07, CH4 1.5e-06, CO 1.4e-06, CO2 1.2e-06,
N2 1.4e-07, NH3 1.3e-07.

Configuration verified knob-by-knob: of 122 settings shared between upstream's
`cfg_examples/vulcan_cfg_HD189.py` and VULCAN-JAX's `HD189.yaml`, only
`use_live_plot` differs.

B10. The 20% out-of-the-box difference is entirely the elemental composition.
`fastchem_vulcan/input/solar_element_abundances.dat`: VULCAN-JAX ships Lodders
2019 (Wogan & Tsai 2023) with the rocky elements suppressed to -3.0; upstream
ships Lodders 2009 with them at solar. For C-H-N-O the only changed value is
helium, `10.9864` -> `10.9232`. Helium is inert, which is why it appeared as a
uniform 11.6% offset. Tightening VULCAN-JAX's convergence 10x changed nothing,
and matching the network alone changed nothing.

## C. Things that are NOT safe to claim

C1. **No cross-code number measured against the local `VULCAN-master` copy is
valid.** That copy is not a git checkout and contains VULCAN-JAX's own stall
detector, its `conv_stall_window` knob, its `wall_clock_max` exit, the
13-species list, the extra network reaction, and the modified composition file.
This affects the paper's "HD189 1296 matches exactly" sentence and the WASP-39b
"both codes independently accept 1202 steps" footnote. Use B9 instead, and state
that the composition was matched.

C2. The HD189 step count in the paper (1296) came from the 13-species list. The
VULCAN 2 parity number is 1495. The WASP-39b 1202 is unaffected (B4).

C3. `high_temp_cut` is **not** validated by K2-18b: that atmosphere peaks at
2059.92 K and never reaches the 3500 K threshold, so the cut cannot fire there.

C4. Earth does not converge and its atom conservation is badly violated
(max |atom_loss| ≈ 3e+03 at the 20000-step cap). It is not a usable result under
either `conver_ignore` value.

C5. K2-18b is an unresolved diagnostic case. Provenance work on the supplied
configuration (the `remove_list` index mapping, the missing bottom-boundary file,
S8) has not been done.
