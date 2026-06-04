# Bugs found — VULCAN-JAX validation (2026-06-03)

Bugs surfaced while validating `VULCAN-JAX` against `VULCAN-master`. "Status"
is **fixed** (changed in this pass, test-backed) or **flagged** (left alone —
upstream-only, or risky to change without your call). Full context:
`VALIDATION_REPORT.md`.

## Summary

| # | Bug | Where | Sev | Status |
|---|---|---|---|---|
| 1 | H2S saturation pressure 10× too low (cmHg treated as mmHg) | JAX `atm_setup.py:780` + master `build_atm.py:857` | physics | **fixed (both repos)** |
| 2 | `NH3_l_s` molecular mass = 16.023, should be 17.031 (NH2's mass, off by one H) | JAX + master `thermo/all_compose.txt` | data | **fixed (both repos)** |
| 3 | Master `make_chem_funs.py` crashes under numpy ≥1.24 (`str(bytes)`→`"b'OH'"`) | master `make_chem_funs.py:722/735` | port/env | **worked around** in JAX oracle harness; flagged in master |
| 4 | Stale non-editable `vulcan-jax` wheel shadowed `src/` in the `vulcan` env | conda env | env | **fixed** (editable reinstall) |
| 5 | `audit_master_parity.py` CLI default `--jax-root` stale post-restructuring | JAX `tools/audit_master_parity.py:285` | tooling | **fixed** |
| 6 | `test_diffusion.py` super/sub Jacobian check validated only 1 of 69 species | JAX `tests/test_diffusion.py:145-168` | test | **fixed** |
| 7 | `test_oracle._safe_relerr` ignored mismatched-NaN (contradicted its docstring) | JAX `tests/test_oracle.py:555` | test | **fixed** |
| 8 | Master `sp_sat` S4 branch is unreachable dead code (`S4` not in `sat_sp_list`) | master `build_atm.py:787` | dead-code | **flagged** (JAX is correct) |
| 9 | README numerical-agreement table: 5 stale tolerances + wrong reaction counts | JAX `README.md` | docs | **fixed** |
| 10 | `vulcan_jax.py` referenced everywhere (renamed `vulcan_jax_cli.py`) | JAX `CLAUDE.md`, `FILE_README.md` | docs | **fixed** |
| 11 | ~~FastChem build race~~ — **not a bug** (build is already inside the flock); one transient first-build `make` flake observed | JAX `ini_abun.py:240` | — | **no defect** (re-examined) |

---

## Physics / data bugs

### 1. H2S saturation pressure is 10× too low  — *fixed (both repos)*
- **Where:** `src/vulcan_jax/atm_setup.py:780` (and identically in master `build_atm.py:857`).
- **What:** The Giauque & Blue (1936) Antoine fit for H2S outputs pressure in
  **cmHg**, but the code multiplied by `0.001333` (= 1 mmHg in dyne/cm²) instead
  of `0.01333` (= 1 cmHg in bar). Result: H2S saturation pressure exactly 10×
  too low.
- **Verification:** at the H2S normal boiling point (212.8 K) P must equal 1 atm —
  the formula yields 76.1 cmHg, which with the correct factor is 1.015 bar ≈ 1 atm
  (the bug gave 0.10 bar); at the triple point (187.66 K) the correct factor gives
  0.233 bar (literature), the bug gave 0.0233 bar.
- **Fix:** `0.001333 → 0.01333`; provenance comment restored. Pinned by
  `tests/test_sat_p_h2s_anchor.py`.
- **Impact:** latent — H2S *condensation* is not a wired default runtime path, but
  `compute_sat_p('H2S', …)` is public and was wrong for any consumer.
- **Master:** same error at `build_atm.py:857` — **also fixed** (`0.001333 → 0.01333`)
  so the two repos agree; the HD189 oracle does not use H2S, so the fix is inert there.

### 2. `NH3_l_s` condensate molecular mass is wrong  — *fixed (both repos)*
- **Where:** `src/vulcan_jax/thermo/all_compose.txt` (and master's copy), row `NH3_l_s`.
- **What:** Atom counts are correct (H=3, N=1), but the `mass` column reads
  **16.023** — that is the mass of **NH2** (an apparent copy-paste from the NH2 row).
  The correct NH3 mass is **17.031** (matches the gas-phase `NH3` row in the same file).
- **How found:** least-squares fit of per-element atomic masses across all 250
  species — every element matched IUPAC to <0.01 amu, and `NH3_l_s` was the only
  species whose listed mass was inconsistent with its own atom counts (by −0.98 amu,
  ≈ one H).
- **Impact:** ~6% low. Used as `ms[NH3_l_s]` in mean molecular weight (when
  condensates are included), settling velocity, and molecular diffusion for the
  NH3 condensate. NH3 condensation **is** a supported path, so this affects
  NH3-condensing runs (e.g. cold giants); inert on the default HD189 NCHO network
  (no `NH3_l_s` species, `use_condense=False`).
- **Fix:** `NH3_l_s` mass set to `17.031` in `thermo/all_compose.txt` in **both**
  repos (kept byte-identical so the parity audit stays clean — re-verified PASS).
  Guarded going forward by `tests/test_species_mass_integrity.py`, which
  recomputes every species mass from its atom counts and checks each condensate
  against its gas-phase counterpart.

---

## Original-VULCAN (master) bugs

### 3. `make_chem_funs.py` crashes under modern numpy  — *worked around in JAX*
- **Where:** master `make_chem_funs.py:722` (`np.genfromtxt(..., dtype=None)` with
  no `encoding=`) → numpy ≥1.24 returns `numpy.bytes_`; line 725 `str(b'OH')`
  yields `"b'OH'"`, so `compo_row.index('OH')` raises in `check_conserv()` (line 735).
- **Why it matters:** that crash is *after* `chem_funs.py` is fully written (line
  809 is "the last function that writes"), and master's own `vulcan.py` ignores the
  exit code (`os.system`). But the JAX oracle tests bailed on the non-zero exit, so
  the **end-to-end HD189 parity check silently skipped**.
- **Fix (JAX side):** the oracle now proceeds when `chem_funs.py` imports with a
  valid `(ni, nr)`, not on the exit code (`tests/test_default_master_parity.py`,
  `tests/test_w39b_fastchem_invariant.py`). Both now run and pass.
- **Master fix (if you maintain the sibling):** add `encoding='utf-8'` to the
  `genfromtxt` call, or `.decode()` the bytes in `check_conserv`.

### 8. Master `sp_sat` S4 branch is unreachable dead code  — *flagged*
- **Where:** master `build_atm.py:787` — `sat_sp_list` omits `'S4'`, so the guard
  at line 790 raises `IOError` for `condense_sp=['S4']` before the S4 formula at
  line 832 can run (its `op.py` S4 condensation path is likewise stranded).
- **JAX:** supports S4 correctly (`_SUPPORTED_CONDENSABLES` includes it; formula
  byte-identical). No JAX change needed — JAX is more correct here.

---

## Test / tooling / environment bugs (all fixed)

### 4. Stale non-editable install shadowed the source
The `vulcan` conda env had a regular (non-editable) `vulcan-jax==0.1.7` in
site-packages, so `import vulcan_jax` and the whole test suite ran the **wheel,
not `src/`** (e.g. `photo._UNDERFLOW_DENOM` existed in source but not the wheel →
2 baseline failures). Fixed with `pip install -e . --no-deps`.

### 5. `audit_master_parity.py` CLI default path stale
`main()`'s default `--jax-root` resolved to the repo root, but `vulcan_cfg.py`
now lives under `src/vulcan_jax/` after the flat→package restructuring — so the
documented command failed immediately with "missing JAX root cfg" and ran zero
comparisons. Now defaults to `_paths.PACKAGE_ROOT`.

### 6. `test_diffusion.py` indentation bug
In the super/sub Jacobian-block checks (`tests/test_diffusion.py:145-168`), the
error comparison was dedented out of the `for i in range(ni)` loop, so the tight
1e-10 assertion only ever checked the **last** species (He). Re-indented to check
all 69 (still passes — production block diagonals match master at 0.0 for every
species).

### 7. `test_oracle._safe_relerr` silently passed mismatched NaN
The helper computed `only_one_nan` but never used it; its docstring promised
mismatched-NaN positions are treated as `+inf`. Where one state was NaN and the
other finite (a real divergence) it scored `rel=0`. Implementation now sets
`rel=inf` there, honoring the docstring.

### 11. FastChem build — re-examined, NOT a bug
Originally flagged as a possible build race. On closer reading,
`_ensure_fastchem_binary` is called at `ini_abun.py:240` *inside* the exclusive
`fcntl.flock` that `_load_eq_y` holds (acquired at line 269 before
`_run_fastchem_locked`), so the `make` is already cross-process serialized — two
xdist workers cannot race it. The single transient `make exit 2` seen during the
W39b run was a one-off flaky first-build (no binary existed yet); the binary
persists afterward and the full `-n auto` suite is green. **No code defect; no
change needed.**

---

## Verified correct (no bug)

For the record, these were checked and found correct: `phy_const` (bit-identical
both sides; `hc = h·c`), NASA-9 Gibbs polynomial / `K_eq` sign / branch boundary,
Arrhenius/Lindemann/Troe rate forms, all other saturation formulae
(H2O/NH3/H2SO4/S2/S4/S8/C), `Dzz`/viscosity/settling/Kzz/analytical-T(P), config
& public-API parity, and **all gas-phase species molecular weights** (the
atomic-mass fit matched IUPAC across 250 species; only `NH3_l_s` above was wrong).
