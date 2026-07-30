# Changes made, file by file

Session of 2026-07-30. Baseline before any edit: `2fdc66b`, suite 242 passed /
5 skipped / 0 failed.

Every change below is either (a) a correction of a claim that was demonstrably
false, (b) a new explicit switch for behaviour that had no off switch, or (c) a
documentation fix that follows from (a) or (b). No physical or numerical setting
was changed to make anything converge faster or to reproduce a step count.

---

## 1. Configuration: `conver_ignore` and the VULCAN 2 / VULCAN 3 split

### `src/vulcan_jax/configs/default.yaml`, `HD189.yaml`, `HD209.yaml`, `W39b.yaml`

**Before**

```yaml
# Master's list (VULCAN-master cfg:149) + HC3N: heavy hydrocarbons on the chem_rhs
# ULP floor that stall convergence.
conver_ignore:
- C6H6
- C2H2
- C6H5
- C2H
- C2H4
- C2H5
- C2H6
- C3H2
- C3H3
- C4H5
- CH2NH
- CH3NH2
- H2CCO
- HC3N
```

**After**

```yaml
conver_ignore: []
```

plus a comment giving the real provenance.

**Why.** The attribution was false. Fetched from the real remotes on 2026-07-30,
`exoclime/VULCAN` master `vulcan_cfg.py:151` and its own
`cfg_examples/vulcan_cfg_HD189.py:148` both ship `[]`, as does `shami-EEG/VULCAN`
master (that file is byte-identical to exoclime's). `shami-EEG/VULCAN` vm_branch
ships `['HC3N']`. The 13-species list exists in **no** upstream repository.
Shami's own email of 2026-07-14 says "Usually, I only have HC3N", and the reply
that day agreed the long list was "just a leftover from tests".

The reason the citation looked checkable is that the list had been written into
the local `../VULCAN-master/` copy, which was then read back as proof. See
section 5.

**Measured effect** (stall fallback explicitly disabled, all physical inputs
identical within a config):

| config | `[]` | `['HC3N']` | 13 + HC3N |
|---|---|---|---|
| HD189 | 1495 steps, longdy 0.09172 | 1495, 0.09172 | 1296, 0.04003 |
| default | 1495, 0.09172 | 1495, 0.09172 | 1296, 0.04003 |
| HD209 | 1206, 0.03013 | 1206, 0.03013 | 1206, 0.03013 |
| W39b | 1202, 0.09900 | 1202, 0.09900 | 1202, 0.09900 |

`[]` and `['HC3N']` are behaviourally **identical** on all four. The 13-species
list only changes HD189/default, and it does so by removing species from the
convergence metric, not by changing the physics. Every run above exits on the
normal convergence criterion (`termination_reason = 1`), never the stall
fallback.

### `src/vulcan_jax/configs/Earth.yaml`

`conver_ignore: ['HC3N']` -> `[]`, `use_conv_stall` -> `false`.

Upstream's own `cfg_examples/vulcan_cfg_Earth.py` declares no `conver_ignore` at
all, and fetched master ships `[]`. **Measured: the two values are identical**
here as well — 20001 accepted steps against the 20000 cap either way,
`termination_reason = 3`, `longdy = 1.0`, max |atom_loss| 2.98e+03. So this is an
upstream-fidelity change with no behavioural cost. Earth is not a usable result
under either value; see `unresolved_items.md` U3.

### `src/vulcan_jax/configs/K2-18b.yaml`

`conver_ignore` left at the collaborator-supplied `['HC3N']` (which is also
vm_branch's value) — this is a VULCAN 3 case, not a parity case, and the plan is
explicit that the collaborator's intent is preserved. Three declarations added:
`use_conv_stall: true` and `conv_stall_window: 200` (previously inherited
implicitly, which after this change would have flipped to the parity default),
and **`batch_max_retries: 110`**. The last is a real fix: K2-18b omitted the key
and was silently taking the code default 64, unlike every other shipped config,
against the C11 anchor in `docs/corrections_to_original_code.md`.

### `src/vulcan_jax/configs/HD189_vulcan3.yaml` (NEW)

The explicit VULCAN 3 preset the source-of-truth policy asks for: HD189's physics
with `shami-EEG/VULCAN` vm_branch's numerics. Each differing knob carries a
comment citing the vm_branch line it came from.

| knob | HD189.yaml (parity) | HD189_vulcan3.yaml | vm_branch source |
|---|---|---|---|
| `conver_ignore` | `[]` | `['HC3N']` | `vulcan_cfg.py:157` |
| `use_vm_mol` | false | true | `vulcan_cfg.py:147` |
| `use_hybrid_vm_mol` | false | true | `vulcan_cfg.py:148` |
| `high_temp_cut` | false | true | `vulcan_cfg.py:149` |
| `count_max` | 10000 | 20000 | `vulcan_cfg.py:121` |
| `mtol_conv` | 1.0e-20 | 1.0e-18 | `vulcan_cfg.py:124` |
| `use_adapt_rtol` | false | true | `vulcan_cfg.py:160` |
| `rtol` | 0.2 | 0.25 | `vulcan_cfg.py:161` |
| `rtol_min` / `rtol_max` | 0.0 / 1.0 | 0.01 / 2.5 | `vulcan_cfg.py:163-164` |
| `adapt_rtol_inc_period` | 500 | 1000 | `op.py:845` |
| `adapt_rtol_inc` | 1.5 | 1.25 | `op.py:847` |
| `use_conv_stall` | false | true | none — JAX-only, flagged as such |

---

## 2. The JAX-only stall fallback now has an off switch

### `src/vulcan_jax/outer_loop.py`

- `_Statics.use_conv_stall` added, read from `cfg` with a `True` code-level
  fallback so a config that omits the key keeps today's behaviour.
- `_convergence_ok` gates the stall predicate on that **static Python bool**, so
  when it is off the predicate folds away at trace time and the run is
  bit-identical to one built without the feature.
- `runner` now records the termination code on exit:
  `final._replace(termination_reason=reason)`. `cond_fn` already evaluated this
  every iteration and threw it away; re-evaluating once on the terminal state is
  O(1) per run.
- `_unpack_state_to_runstate` and `_call_runstate` carry the code onto the
  returned `RunState`; both the RunState and the legacy `(var, atm, para)` exit
  paths now say *how* a converged run converged.

**Why.** `conv_stall_window` has no counterpart in `master` or `vm_branch` —
Shami asked what it was (email item 7) — so a VULCAN 2 parity run must be able to
switch it off, and before this it could not. `conv_stall_window` is a lookback,
not a disable: `runtime_validation` requires it to be `>= 1`, and the predicate
is `count_since_new_min > window`, so a smaller window makes the fallback fire
*sooner*. `0` would have been an anti-disable.

### `src/vulcan_jax/state.py`, `src/vulcan_jax/legacy_io.py`

`ParamInputs.termination_reason` (defaulted to 0 so existing positional callers
keep working), threaded through `params_from_store`, both `para` copy-back
paths, `_Parameters.__init__`, and written into the `.vul` `'parameter'` dict.

**Why.** `end_case` cannot distinguish a normal convergence from a stall
convergence — both are `1` — so a stalled run was indistinguishable in the `.vul`
file, on stdout, and through the Python API. `end_case` is deliberately left
unchanged for backward compatibility (downstream code checks `end_case == 1`);
`termination_reason` is the finer-grained field: 0 running, 1 converged,
2 runtime, 3 step count, 4 stall, 5 non-finite.

### `tests/test_conv_stall_gate.py` (NEW)

Eight tests. The two that matter seed the runner carry into a state where every
stall gate is satisfied and `conv_normal` is deliberately false, then assert:
enabled → stops immediately with `termination_reason == 4`; disabled → the stall
exit is unavailable and the run falls through to the step-count ladder
(`reason == 3`). Also: the fallback cannot fire before its gates, `window: 0` is
rejected rather than meaning "off", parity configs ship it off, the V3 preset
ships it on, and the code reaches the `.vul` file.

---

## 3. The parity audit tool can no longer be fooled

### `tools/audit_master_parity.py`

- New `_check_oracle_is_pristine()`, called first in `audit()`, which **refuses**
  (errors, not warns) if the checkout being used as the oracle contains any
  VULCAN-JAX-only identifier (`conv_stall_window`, `longdy_seen_min`,
  `count_since_new_min`, `wall_clock_max`) or has no `.git`.
- Stale `JAX_ONLY_DEFAULTS["batch_max_retries"]` 64 → 110 (the shipped configs
  say 110).
- Stale comment claiming VULCAN-JAX "ships a leaner convergence-ignore list"
  replaced with the measured facts.

Run against the local sibling it now reports five errors and stops, instead of
returning a clean parity verdict.

### `tests/test_default_master_parity.py`

- `test_audit_master_parity_with_staged_stock_fastchem` previously asserted the
  audit was clean **against the contaminated checkout**. That assertion was the
  circularity in test form. It now skips loudly, naming the reasons, when the
  oracle is not pristine — skipped, not passed.
- New `test_audit_refuses_a_contaminated_oracle` pins the guard itself.

---

## 4. Molecular-diffusion benchmark repaired and wired for the figure

### `benchmarks/zhang2013_moldiff_benchmark.py`

- **Bug fix:** the script was broken and could not run at all —
  `AtmStatic.__new__() missing 1 required positional argument: 'diff_esc_mask'`.
  It builds an `AtmStatic` by hand and had not been updated when the field was
  added. It is not wired into pytest, so nothing caught it. Set to all-False,
  which is correct here (the benchmark's top boundary is zero-flux).
- Added `--outdir`, `--basename`, `--formats`, `--csv` so it can emit the
  PNG + PDF + plotted-CSV set. Default behaviour unchanged.

It now passes its own gate: central (= hybrid converged) 0.8% max error against
the Zhang+2013 analytic solution, pure upwind 45.7%, VULCAN-JAX vs the VULCAN 2.0
operator 0e+00, central negative above cell Péclet ~2, upwind positive
everywhere.

---

## 5. Documentation

### `docs/corrections_to_original_code.md`

New Policy item 2b: "what does upstream do?" is answered by fetching upstream,
never by reading `../VULCAN-master/`, with the file:line inventory of what that
copy actually contains. Conventions updated to say the copy is a numerical
oracle for per-step kernel comparisons only.

### `docs/vulcan_jax_notes.md`

New dated entry recording the provenance work, the contamination inventory, the
controlled measurement table, and the resolution. Two earlier entries (the
2026-07-27 "1296 resolved" entry and the 2026-07-30 six-config table) carry a
**SUPERSEDED** banner pointing at it.

### `README.md`

Configuration table rewritten: corrected step counts, the new preset, and a
"VULCAN 2 parity versus VULCAN 3" section explaining what each flavour means and
warning that a step count is only meaningful together with its configuration.

### `CLAUDE.md` (untracked / gitignored — local only)

The `conver_ignore` rule rewritten from "UNRESOLVED" to the settled decision.
Stall-fallback rule rewritten around `use_conv_stall` and `termination_reason`.
Stale "on by default" claims for `use_vm_mol`/`use_hybrid_vm_mol` corrected.
Stale "the `-n` flag is accepted" claim corrected (the CLI takes only
`--config`/`-c`). Trimmed elsewhere to offset the additions.

### `tests/test_config.py`

Added the new preset to `_EXPECTED_GS` and a `test_expected_gs_covers_every_shipped_config`
that globs `configs/*.yaml`, so a future config cannot silently escape coverage.
