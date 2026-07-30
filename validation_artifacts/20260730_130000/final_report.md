# VULCAN-JAX cleanup and validation — final report

Session date: 2026-07-30. Repository: `imalsky/jax-vulcan`, branch `main`,
pre-edit baseline `2fdc66b`.

---

## 1. Executive conclusion

Two claims that the repository was making about itself were false, and both had
the same root cause: **the local `../VULCAN-master/` directory was being used as
"upstream", and it is not upstream.** It is an unversioned copy that had
VULCAN-JAX's own stall detector, its `conv_stall_window` knob, its
`wall_clock_max` exit and a 13-species `conver_ignore` list written into it.
Reading a value back out of that copy and calling it "master parity" is
circular, and that is exactly what happened in commits `eebc8a5` and `2fdc66b`.

Fixed. The shipped planet configurations are now VULCAN 2 parity configurations
whose convergence settings match `exoclime/VULCAN` master as *fetched*, and a
new `HD189_vulcan3.yaml` holds the `vm_branch` numerics explicitly. The JAX-only
stall fallback has an off switch for the first time and is off in every parity
config. The parity audit tool now refuses to run against a contaminated oracle.

The most consequential number to change: **HD 189733 b's VULCAN 2 parity step
count is 1495, not the 1296 in the paper.** 1296 came from the local-only
13-species list.

Because the old oracle could not be trusted, a pristine `exoclime/VULCAN` clone
was built and run for the first time. That produced the project's **first
genuinely independent cross-code check**, and one new finding that matters more
than anything else here: **VULCAN-JAX and upstream VULCAN do not start from the
same elemental composition** (VULCAN-JAX ships Lodders 2019 with rocky elements
suppressed; upstream ships Lodders 2009 at solar). Out of the box the two codes
differ by 20% median on HD 189733 b. With the composition matched they agree to
**3.8e-06 median**, and to parts in 1e9 for inert species in the deep column.
So the port is sound, but every past "VULCAN-JAX vs VULCAN 2.0" number was
comparing two different problems. Details in section 6.

---

## 2. Exact provenance

All upstream statements below come from HTTP fetches on 2026-07-30, saved under
`upstream_fetch/`. Nothing is taken from the local copy.

| source | fetched |
|---|---|
| `exoclime/VULCAN` master | `vulcan_cfg.py`, `cfg_examples/vulcan_cfg_HD189.py`, `op.py`, `build_atm.py`, `store.py`, `make_chem_funs.py`, `thermo/*` |
| `shami-EEG/VULCAN` master | `vulcan_cfg.py` |
| `shami-EEG/VULCAN` vm_branch | `vulcan_cfg.py`, `op.py` |
| pristine clone | `exoclime/VULCAN` HEAD `8970337`, cloned and run |

`exoclime` master and `shami-EEG` master `vulcan_cfg.py` are byte-identical
(md5 `be64301868912c77ee1105ef52e0bbed`).

**The local `../VULCAN-master/` copy is contaminated.** Diffed against the
fetched files (patches in `patches/`):

| file | what it carries that is VULCAN-JAX's, not upstream's |
|---|---|
| `store.py:204-209` | `longdy_seen_min` / `count_since_new_min` stall state |
| `op.py:1078-1104` | the 5%-improvement stall predicate, ending `end_case = 1` |
| `op.py:1119-1126` | `wall_clock_max` -> `end_case = 4` |
| `vulcan_cfg.py:141`, `cfg_examples/vulcan_cfg_HD189.py:122` | `conv_stall_window = 200` |
| `vulcan_cfg.py:183`, `cfg_examples/vulcan_cfg_HD189.py:149` | the 13-species `conver_ignore` |
| `thermo/NCHO_photo_network.txt` | one reaction absent from every upstream branch |
| `fastchem_vulcan/input/solar_element_abundances.dat` | Lodders 2019 + rocky suppression |

`build_atm.py` and `make_chem_funs.py` also differ, but those are the already
documented deliberate fixes and are not the problem. The directory was **not**
modified by this session; it stays read-only.

---

## 3. Source-of-truth and default policy applied

1. Fetched `exoclime/VULCAN` master is the truth for VULCAN 2 baseline and
   parity behaviour.
2. Fetched `shami-EEG/VULCAN` vm_branch is the truth for VULCAN 3 features.
3. A locally patched checkout is never upstream.
4. VULCAN 2 parity configurations match (1) on every convergence knob.
5. VULCAN 3 behaviour lives in a clearly named preset and cannot silently alter
   a parity run.

---

## 4. Changes made

Full file-by-file account in `changes_made.md`. Summary:

- `default/HD189/HD209/W39b.yaml`: `conver_ignore` 13-species -> `[]`;
  new `use_conv_stall: false`; false attribution comment replaced.
- `HD189_vulcan3.yaml`: new explicit VULCAN 3 preset (vm_branch numerics,
  every line cited).
- `K2-18b.yaml`: declares `use_conv_stall: true` and `batch_max_retries: 110`
  (it was silently taking the code default 64, unlike every other config).
- `outer_loop.py` / `state.py` / `legacy_io.py`: `use_conv_stall` gate;
  `termination_reason` recorded on the single-profile path and written to the
  `.vul` file, the API and stdout.
- `tools/audit_master_parity.py`: refuses a contaminated oracle.
- `benchmarks/zhang2013_moldiff_benchmark.py`: repaired (it could not run) and
  extended to emit PNG/PDF/CSV.
- `tests/`: new `test_conv_stall_gate.py`; the circular parity assertion
  rewritten; config coverage now globs.
- Docs: `corrections_to_original_code.md` policy 2b + the composition entry;
  `vulcan_jax_notes.md` new entry + superseded banners; `README.md` config
  table and a parity-vs-V3 section; `CLAUDE.md` rules.

---

## 5. Changes considered and deliberately NOT made

- **`conver_ignore: ['HC3N']` in the parity configs.** It is what Shami uses and
  what vm_branch ships, and it measures identically to `[]`. But upstream master
  ships `[]`, and the policy is that parity configs match master. `['HC3N']` is
  in the V3 preset instead.
- **Retuning Earth or K2-18b.** Both hit their caps. The plan forbids tuning for
  convergence alone, and neither has a diagnosed cause.
- **Changing `adapt_rtol_inc_period`/`adapt_rtol_inc`.** They match neither
  remote, but they are inert (`use_adapt_rtol` false) and the intended values are
  genuinely ambiguous (Shami's email disagrees with his branch). Question for him.
- **Editing the vendored networks with rate defects** (`SNCHO_photo_network_C3`,
  `SNCHO_DMS_photo_network_Tsai2024`). Recorded, not silently patched; no shipped
  config selects them.
- **Editing `jax_paper`.** The numbers that need changing depend on section 6.
- **Fixing `make_spectra_in_nm.py`.** The shipped data file is already correct;
  the correct normalisation was not independently re-derived here.

---

## 6. VULCAN 2 parity result — and the finding that qualifies it

A pristine `exoclime/VULCAN` clone (HEAD `8970337`) was built and run for the
first time. Minimal run-enablement patch saved as
`patches/upstream_clone_run_enablement.patch`; it changes no physics (a
numpy>=1.24 `genfromtxt` bytes fix, a portable FastChem compiler flag, and the
four cfg attributes upstream's own HD189 example fails to define).

| run | steps | final `longdy` | wall |
|---|---|---|---|
| upstream, upstream network | 1131 | 1.63e-02 | 183.6 s |
| upstream, VULCAN-JAX network | 1081 | 9.86e-02 | 168.2 s |
| VULCAN-JAX parity config | 1495 | 9.17e-02 | ~35 s |

Configuration was verified knob-by-knob: of 122 shared settings between
upstream's `cfg_examples/vulcan_cfg_HD189.py` and VULCAN-JAX's `HD189.yaml`,
**exactly one differs**, `use_live_plot` (a UI flag).

**Steady-state comparison, abundance floor 1e-12:** median relative difference
**2.0e-01**, 90th percentile 8.1e-01, max 8.8e+01.

That is far too large for a port that agrees to 1e-15..2e-13 per step. Three
controls were run to find out why:

1. **Chemistry held identical** (upstream re-run with VULCAN-JAX's network):
   median unchanged at 1.993e-01. The network is *not* the cause.
2. **VULCAN-JAX convergence tightened 10x** (`yconv_min` 0.1 -> 0.01, converging
   at `longdy` 9.8e-03 in 1565 steps): median unchanged at 1.993e-01. The
   stopping criterion is *not* the cause, and VULCAN-JAX is at a genuine fixed
   point.
3. **Upstream against itself** (same code, one reaction changed, 1131 vs 1081
   steps, `longdy` 1.6e-02 vs 9.9e-02): median **7.0e-06**, max 1.7e-02. So
   VULCAN 2.0 reproduces itself to parts in 1e6 across that variation.

The disagreement is present in the deep, well-mixed column where photochemistry
is irrelevant, and **inert helium differs by a uniform 11.6%**. Helium has no
chemistry, so that is an initial-condition difference, not a solver difference.

**Cause: the two codes do not start from the same elemental composition.**
`fastchem_vulcan/input/solar_element_abundances.dat` differs:

- VULCAN-JAX uses **Lodders 2019** (Wogan & Tsai 2023): `He 10.9232`;
  upstream uses **Lodders 2009**: `He 10.9864`. For C-H-N-O networks helium is
  the only changed value (C, H, N, O identical; S also moves, 7.12 -> 7.1492).
- VULCAN-JAX **suppresses the rocky elements** (P, Si, Ti, V, Cl, K, Na, Mg, F,
  Ca, Fe) to -3.0 because the truncated networks have no species for them.
  Upstream keeps them at solar, which sequesters oxygen into MgO/SiO2/FeO.

Both are deliberate. The rocky suppression is already recorded in
`docs/corrections_to_original_code.md` as "a legitimate initial-condition
difference", and Isaac described it to Shami on 2026-07-15 ("photochem …
suppresses the rocky elements that the network doesn't treat"). The Lodders 2019
value change was **not** recorded anywhere; it is now.

The signature matches: more available oxygen and less helium give H2O +25% and
CO2 +42% below 1 bar, exactly the direction observed.

**Confirmed by direct test.** Upstream was re-run a third time with BOTH the
network and the composition file matched to VULCAN-JAX (1600 accepted steps,
268.7 s):

| upstream run | median | p90 | max |
|---|---|---|---|
| upstream defaults (1131 steps) | 1.993e-01 | 8.089e-01 | 8.765e+01 |
| + VULCAN-JAX network (1081 steps) | 1.993e-01 | 8.078e-01 | 8.767e+01 |
| **+ network AND composition (1600 steps)** | **3.756e-06** | 2.365e-03 | 4.480e-02 |

Matching the composition drops the median disagreement by a factor of ~53,000.
In the deep well-mixed column the two codes agree to parts in 1e9 for the inert
species and parts in 1e6 for the majors:

| | H2 | He | H2O | CH4 | CO | CO2 | N2 | NH3 |
|---|---|---|---|---|---|---|---|---|
| median rel. diff. | 4.5e-10 | 1.3e-09 | 1.2e-07 | 1.5e-06 | 1.4e-06 | 1.2e-06 | 1.4e-07 | 1.3e-07 |

**Conclusion. VULCAN-JAX reproduces VULCAN 2.0 to 3.8e-06 median on a free
convergence of HD 189733 b, once the two codes are given the same elemental
composition.** This is the first genuinely independent cross-code check the
project has (every earlier one used the contaminated local copy). Step counts
still differ (1495 vs 1600, 7%), which is the expected roundoff-driven
difference in the adaptive step sequence and is already discussed in the paper.

The composition file is therefore not a defect but it **is** a required part of
the experimental setup: no VULCAN-JAX-vs-VULCAN-2.0 number means anything unless
it is matched, and no past comparison matched it.

---

## 7. VULCAN 3 hybrid result

HD 189733 b, parity config plus the diffusion scheme varied:

| scheme | steps | ends in | reason |
|---|---|---|---|
| central | 1495 | central | converged |
| pure upwind | 1495 | upwind | converged |
| hybrid | 2102, switch at step 1500 | **central** | converged |
| `HD189_vulcan3` preset | 2820 | **central** | converged |

Final-state difference against pure central, floor 1e-12, over
H2O/CH4/CO/HCN/C2H2/H:

| scheme | median | max |
|---|---|---|
| pure upwind | 2.5e-06 | 2.6e+00 |
| hybrid | 2.3e-06 | 1.7e-04 |

The hybrid returns the central-difference answer; pure upwind does not. The
returned `hybrid_use_vm = 0.0` confirms the run ends in the central phase, so the
converged state is a central fixed point (which is what the forward-mode AD and
the steady-state adjoint rely on).

---

## 8. `conver_ignore` provenance and final policy

Provenance in `email_facts.md` A2/A3. Policy: parity configs `[]`, V3 preset
`['HC3N']`, 13-species list dropped entirely (it reproduces a step count, which
is not a reason).

Measured sensitivity, stall disabled:

| config | `[]` | `['HC3N']` | 13 + HC3N |
|---|---|---|---|
| HD189 / default | 1495 | 1495 | 1296 |
| HD209 | 1206 | 1206 | 1206 |
| W39b | 1202 | 1202 | 1202 |

`[]` and `['HC3N']` are identical everywhere tested, so the change is free.
W39b is the base config for `vulcan-forward`, `vulcan-retrieval` and
`vulcan-jwst-tool`, and it is insensitive to both this knob and the stall flag,
so the downstream repositories are unaffected. They also override
`conver_ignore` explicitly rather than inheriting it.

---

## 9. `conv_stall` policy and tests

JAX-only; absent from master and vm_branch. Now gated by `use_conv_stall`,
default false in parity configs, true in the V3 preset and K2-18b. When off the
predicate folds away at trace time. `conv_stall_window` is documented as a
lookback, not an off switch (it must be >= 1 and a smaller value fires sooner).

`termination_reason` (0/1/2/3/4/5) is now recorded on the single-profile path and
surfaced in `RunState.params`, the `.vul` `'parameter'` dict and stdout, because
`end_case` reports 1 for both a normal and a stall convergence.

`tests/test_conv_stall_gate.py` proves enabled fires (reason 4), disabled does
not (falls through to reason 3), the gates hold, and `window: 0` is rejected.
**No parity run in this report used the stall fallback**; every one exited
`termination_reason = 1`.

---

## 10. `high_temp_cut` and adaptive `rtol`

`high_temp_cut` confirmed to raise `P_b` and regrid, not clip temperature
(`atm_setup.py:64-90`, `1003-1044`); exact no-op when disabled. `3500 K` /
`1e6 dyn cm^-2` confirmed against fetched vm_branch. Enabled in the V3 preset,
off in parity. **It is not validated by K2-18b** — that atmosphere peaks at
2059.92 K and never reaches the threshold. Existing tests do exercise profiles
above 3500 K (4200/4000/4500 K) but only through the pure re-grid helper.

Adaptive `rtol`: fetched master and vm_branch differ only in the decrease factor
(0.75 vs 0.5); periods and increase factor are identical (10 / 1000 / 1.25). The
shipped 500 / 1.5 match neither and are inert. Left alone; question for Shami.

---

## 11. Fixed-bug ledger

| item | classification |
|---|---|
| 13-species `conver_ignore` credited to master | (1) JAX-port regression, fixed |
| `use_conv_stall` had no off switch | (3) JAX-only feature, now named/gated/tested |
| stall exit indistinguishable from normal | (3) JAX-only, now reported |
| audit tool trusted a contaminated oracle | (1) fixed |
| Zhang benchmark could not run (`diff_esc_mask`) | (1) fixed |
| `K2-18b` silently used `batch_max_retries=64` | (1) fixed |
| Lodders 2019 + rocky suppression | (3) intentional, now documented |
| extra `NH3 + CH` reaction in the NCHO network | (4) inherited, unresolved |
| `SNCHO_photo_network_C3` / `_DMS_Tsai2024` rate defects | (4) inherited, recorded |
| eps Eri builder script | (4) inherited, data file already correct |

---

## 12. K2-18b classification: **C — unresolved diagnostic case**

Runs end to end, 31002 steps against a 30000 cap, `end_case = 3`,
`termination_reason = 3`, `longdy = 25.8`, 7923 delta rejects,
max |atom_loss| 5.5e-02, all finite, no negative cells, 1461 s. Stalls on
**C3H2** at 5.6e-08 bar / 207 K.

Not validated, not a headline figure. Its non-convergence is **not** entangled
with the `conver_ignore` decision after all: C3H2 was in the disputed
13-species list, but K2-18b ships `['HC3N']` and was never using that list.
Provenance work (the `remove_list` index mapping, the missing bottom-boundary
file, S8) remains undone — see `unresolved_items.md` U4.

---

## 13. Test results

| | passed | skipped | failed |
|---|---|---|---|
| baseline (`2fdc66b`, before any edit) | 242 | 5 | 0 |
| final | 252 | 6 | 0 |

Ten new passing tests: eight in `tests/test_conv_stall_gate.py`, one
`test_audit_refuses_a_contaminated_oracle`, one
`test_expected_gs_covers_every_shipped_config`.

The sixth skip is new and is deliberate:
`test_audit_master_parity_with_staged_stock_fastchem` now **skips instead of
passing**, naming every reason, because the sibling `../VULCAN-master` is not
pristine upstream. Previously it asserted the audit was clean against that
checkout, which is the circularity this session was fixing. Skipped is not
passed, and the skip message says so.

The other five skips are the pre-existing ones (two `H2O_l_s`-not-in-network
config-matrix cases, three `VULCAN_JAX_RUN_SLOW=1` gates).

Quality gates: `git diff --check` clean, `ruff check` clean, `ruff format
--check` clean, `--help` works, every shipped YAML loads through the strict
loader, resolved configs round-trip (`resolved_configs/`).

---

## 14. Timing table

Apple M3 Pro, 12 cores, quiet machine (load avg 2.5), lid open, under
`caffeinate -dims`. Fresh subprocess per measurement. Cold = empty XLA cache
directory; warm = same directory once populated. Three measured warm repeats,
all reported. Nothing was tuned after seeing a result. Full data in
`timings.csv`.

| configuration | mode | wall (s) | setup (s) | integrate (s) | steps | rejects | reason |
|---|---|---|---|---|---|---|---|
| `HD189` (VULCAN 2 parity) | cold | 38.21 | 3.11 | 34.57 | 1495 | 37 | 1 |
| | warm 1 | 35.82 | 3.03 | 32.28 | 1495 | 37 | 1 |
| | warm 2 | 35.76 | 3.01 | 32.25 | 1495 | 37 | 1 |
| | warm 3 | 36.03 | 3.03 | 32.49 | 1495 | 37 | 1 |
| `HD189_vulcan3` (VULCAN 3 hybrid) | cold | 66.91 | 3.18 | 63.19 | 2820 | 73 | 1 |
| | warm 1 | 64.67 | 3.24 | 60.91 | 2820 | 73 | 1 |
| | warm 2 | 64.51 | 3.26 | 60.74 | 2820 | 73 | 1 |
| | warm 3 | 64.50 | 3.24 | 60.74 | 2820 | 73 | 1 |

| configuration | warm median | warm min | cold | XLA compile (cold - warm) |
|---|---|---|---|---|
| `HD189` VULCAN 2 parity | 35.82 s | 35.76 s | 38.21 s | 2.39 s |
| `HD189_vulcan3` VULCAN 3 hybrid | 64.51 s | 64.50 s | 66.91 s | 2.40 s |

Setup (FastChem equilibrium + network parse + pre-loop) is ~3.1 s and is
unaffected by the XLA cache. The FastChem *binary* build (~10 s) happens once per
install, not per run, and is not in these numbers.

All eight runs are bit-reproducible: identical accepted steps, rejected steps and
`longdy` to six significant figures. That is the compile-cache determinism fix
from `2fdc66b` holding.

**Do not divide these two rows.** They are different numerical schemes reaching
different fixed points with different step counts (1495 central vs 2820 hybrid),
which is exactly the unlike-configuration ratio the plan forbids.

---

## 15. Figure captions

**Figure 1 — `figure_1_zhang_molecular_diffusion`.** Molecular diffusion against
the Zhang, Shia & Yung (2013) analytic diffusive-separation solution, driving the
production VULCAN-JAX kernels on an isothermal Jovian-stratosphere column
(T = 160 K, mu = 2.3, tracer m = 16, nz = 120). *Left:* the central scheme (the
scheme a converged hybrid run ends on) matches the analytic profile to 0.8%
maximum fractional error; first-order upwind is numerically diffusive and
under-separates by 45.7%. Open circles are the upstream VULCAN 2.0 operator
(`op.diffdf` / `op.diffdf_vm`), which VULCAN-JAX reproduces to 0e+00 — the port
is exact. *Right:* why the hybrid needs upwind. Sweeping grid resolution at
K = 0, the central scheme's steady state goes negative above cell Péclet ~2,
while upwind stays positive at every Péclet tested.

**Figure 2 — `figure_2_hd189_hybrid_validation`.** End-to-end HD 189733 b
validation of the hybrid. *A:* `longdy` and `longdydt` versus accepted step for
the hybrid run; the upwind phase is shaded red, the central phase blue, and the
switch fires at step 1500. The run ends "converged" at 2102 accepted steps on
the normal criterion, not the stall fallback. *B:* final mixing-ratio profiles
for six species under central, pure upwind and hybrid. *C:* relative difference
against the pure central run above a declared 1e-12 abundance floor. The hybrid
differs from central by a median 2.3e-06 and a maximum 1.7e-04, while pure
upwind reaches 2.6e+00 — the direct demonstration that the hybrid returns the
central fixed point rather than the upwind one. **No upstream VULCAN 2.0 curve
is shown**: the local copy is not an independent code, and the composition
mismatch of section 6 would dominate any curve from the clean clone.

---

## 16. Remaining questions and blockers

Full list in `unresolved_items.md`. The blocking ones:

1. **The paper should now carry the new parity number.** Section 6 gives a
   real, independent cross-code agreement figure (3.8e-06 median) that the
   project did not previously have. It supersedes any agreement claim measured
   against the local copy, and it requires stating that the composition was
   matched.
2. **The paper's Table 1 and its two `\rev` claims** need revising; 1296 -> 1495
   for the parity configuration, and neither "matches exactly" claim survives.
3. **The extra `NH3 + CH` reaction** has no established provenance. It has been
   present since the first commit and is byte-identical to the local copy's
   network, so it was vendored, not added deliberately. It changes results by
   only ~7e-06 median, so it is not urgent, but it should be reconciled with
   Shami.
4. **Swapping in a different valid network fails** (`IndexError: index 877 out of
   bounds`), which blocks the documented `$VULCAN_JAX_NETWORK` workflow.
