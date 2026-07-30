# Unresolved items

Ranked by how much they would change a number someone relies on.

---

## U1. RESOLVED — the cross-code disagreement was the elemental composition

Kept here because the conclusion is load-bearing for the paper.

A pristine `exoclime/VULCAN` clone (HEAD `8970337`) was built and run. Comparing
its converged HD 189733 b state with VULCAN-JAX's parity run over a 1e-12
abundance floor gave a median relative difference of **2.0e-01**, which is far
too large for a port that agrees to 1e-15..2e-13 per step. Four experiments
isolated the cause:

| test | median vs VULCAN-JAX |
|---|---|
| upstream, upstream defaults | 1.993e-01 |
| upstream, VULCAN-JAX network (chemistry matched) | 1.993e-01 |
| VULCAN-JAX with a 10x tighter convergence criterion | 1.993e-01 |
| **upstream, network AND composition matched** | **3.756e-06** |
| *control:* upstream vs upstream (1 reaction changed) | 7.0e-06 |

Configuration was also verified knob-by-knob: of 122 shared settings, only
`use_live_plot` (a UI flag) differed.

**Cause: `fastchem_vulcan/input/solar_element_abundances.dat`.** VULCAN-JAX ships
Lodders 2019 (Wogan & Tsai 2023) with the rocky elements suppressed to -3.0;
upstream ships Lodders 2009 with them at solar. For C-H-N-O networks the only
changed value is helium (`10.9864` -> `10.9232`), and helium is inert, which is
why it showed as a uniform 11.6% offset in the deep column. Rocky suppression
frees the oxygen upstream sequesters into MgO/SiO2/FeO, giving H2O +25% and
CO2 +42% below 1 bar.

Both choices are deliberate. The rocky suppression was already recorded in
`docs/corrections_to_original_code.md` and Isaac described it to Shami on
2026-07-15. The Lodders 2019 value change was undocumented; it is now recorded.

**What this leaves open:** nothing blocking. Two follow-ups:
- the extra `NH3 + CH -> NH2 + CH2` reaction in VULCAN-JAX's NCHO network
  (present in no upstream branch, byte-identical to the local copy's network,
  present since the first commit) is worth reconciling with Shami, but it moves
  results by only ~7e-06 median;
- `$VULCAN_JAX_NETWORK` cannot actually swap in a different valid network:
  `IndexError: index 877 is out of bounds for axis 0 with size 877`. That blocks
  a documented workflow and should be fixed.

## U2. The paper's Table 1 and two `\rev` claims need revising

`jax_paper/main.tex`:

- Table 1, HD 189733 b, `VULCAN 3.0` column: `35.5 (1296)`. The 1296 came from
  the 13-species `conver_ignore`. The VULCAN 2 parity number is **1495**.
- Caption: *"re-running HD 189733b at the current release gives 1296 accepted
  steps in 37.4 s, matching the tabulated step count exactly."* No longer true,
  and it was never an independent match.
- Footnote †: *"Allowed to converge freely on the same configuration, both codes
  independently accept 1202 steps."* WASP-39b at 1202 does reproduce on the
  VULCAN-JAX side under every `conver_ignore` variant, but the "both codes"
  half was measured against the contaminated local copy and has not been
  re-established against a clean upstream clone.
- Line 249: *"HD 189733b converges in 1296 steps with VULCAN 3.0 and 1396 for
  VULCAN 2.0."* Both numbers need re-deriving. A pristine upstream clone
  converges in **1131** steps, not 1396.

**Not edited this session.** The paper is a separate, unversioned directory and
the right fix depends on U1. Editing the numbers before the network question is
settled would just create a second set of numbers to retract.

---

## U3. Earth is not a usable case

`Earth.yaml` with `conver_ignore: []`, stall off: 20001 accepted steps (cap
20000), `end_case = 3`, `termination_reason = 3`, `longdy = 1.0`, 11750 rejected
steps, and **max |atom_loss| = 2.98e+03**. Atom conservation is violated by more
than three orders of magnitude, so this is not a converged atmosphere and not a
nearly-converged one either. Wall time 1351 s.

The README already says Earth does not converge. It should also say the result is
not physically meaningful. Not retuned: the plan is explicit that Earth carries
upstream's own numerics and must not be tuned just to make it converge.

---

## U4. K2-18b remains classification C (unresolved diagnostic case)

No provenance work was done this session. Still open, all from the original plan:

- map every `remove_list` index with the exact parser and the exact source
  network of the original supplied case, and compare with the current selection;
- confirm the missing bottom-boundary file (`atm/BC_bot_SdepOnly_noSorg.txt`) is
  inert because `use_botflux = false` (the config comment asserts this; it has
  not been verified by running);
- trace S8 / `S8_l_s` through the configuration and decide whether condensation
  was intended.

Do not call it validated. Do not use it as a headline figure.

---

## U5. `K2-18b.yaml` silently gets `batch_max_retries = 64`, not 110

It is one of 25 keys K2-18b omits, so it falls back to the code default
(`outer_loop.py`, `runtime_validation.py`), while the other configs declare 110.
`docs/corrections_to_original_code.md` says "All five shipped configs cite this
anchor" and "update all five configs" — there are now seven. Small, but it is a
real silent divergence in a shipped config.

---

## U6. Adaptive-rtol constants match neither remote

The shipped parity configs carry `adapt_rtol_inc_period: 500` and
`adapt_rtol_inc: 1.5`. Current `vm_branch` `op.py:845,847` uses 1000 and 1.25,
same as master; only the decrease factor differs between the branches
(master 0.75, vm_branch 0.5). The 500 / 1.5 values come from Shami's 2026-07-14
email, which he never committed to the branch.

Currently **inert**: `use_adapt_rtol` is false in every config that sets them.
The VULCAN 3 preset uses the fetched vm_branch values. Left as-is in the parity
configs pending Shami's answer (question 2 in the email draft) rather than
guessing.

Related, and also unresolved: when `use_adapt_rtol` is true, VULCAN-JAX uses the
live `rtol` in the step-acceptance test, whereas upstream binds `rtol` as a
Python default argument in `step_ok`/`step_reject`, which freezes it at the
initial value. Upstream's adaptive rtol therefore only affects step-size
selection, not acceptance. Deliberate on the JAX side and commented, but it is a
behavioural divergence that only shows up when the controller is enabled.

---

## U7. Two shipped network files carry rate defects

Neither is selected by any shipped config, so nothing of ours is affected today.
Recorded rather than edited, because they are vendored upstream data:

- `thermo/SNCHO_photo_network_C3.txt:535` — `CH2CN + H + M` still has the old
  low-pressure rate `1.00E-20` (should be `1.00E-29`, the C1 correction applied
  elsewhere).
- `thermo/SNCHO_DMS_photo_network_Tsai2024.txt:544` — the same reaction has
  `1.00E-10 ... 1.00E-29`, i.e. `k0` and `k_inf` in the opposite order from the
  column header at line 505.

---

## U8. Smaller items

- `src/vulcan_jax/atm/make_spectra_in_nm.py:37` (and the `tools/` copy) still
  contains the epsilon Eridani normalisation bug. The shipped data file is
  correct, but re-running the builder would regenerate the wrong flux. Not fixed
  because the correct normalisation was not independently confirmed this session.
- `y_time_freq` is declared in all shipped configs and read by nothing. It is
  dead upstream too (commented out in `op.py`). Harmless.
- `src/vulcan_jax/atm/stellar_flux/plot_spectra.py` references `vulcan_cfg`
  without importing it: dead shipped code.
- `high_temp_cut` has no test that exercises the driver `Atm.apply_high_temp_cut`
  or the `state.py` gate; the existing tests call the pure re-grid helper
  directly. The one config that enables it (K2-18b) peaks at 2059.92 K, so the
  cut never fires there. A purpose-built config whose profile crosses 3500 K
  would close this.
- `VULCAN-JAX/CLAUDE.md` is 42.4 kB against the ~40 kB threshold noted in
  `docs/vulcan_jax_file_organization.md`. Net change this session was about
  +40 bytes (added the settled rules, trimmed stale history to compensate), so
  it still needs a real consolidation pass.
- The Zhang+2013 benchmark is still not wired into pytest, so its PASS gate is
  not enforced. It had silently rotted (see `changes_made.md` section 4); a
  cheap smoke test would prevent that recurring.
