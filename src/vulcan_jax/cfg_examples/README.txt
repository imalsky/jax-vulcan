Reference configurations for VULCAN-JAX. Copy one to ``vulcan_cfg.py`` at
the repository root and run ``python vulcan_jax.py``.

- vulcan_cfg_HD189.py  — HD 189733b hot-Jupiter reference case. Matches
                         the same-named config in ../VULCAN-master/ and is
                         the canonical side-by-side smoke test.
- vulcan_cfg_HD209.py  — HD 209458b (no S species, weaker gravity).
- vulcan_cfg_Earth.py  — Earth troposphere/stratosphere with condensation.
- vulcan_cfg_W39b.py   — WASP-39b paper-match config (Wogan et al.).

The HD189 config exists in both VULCAN-JAX and VULCAN-master with
identical physics-relevant settings; it is the recommended config for
cross-version comparisons. The JAX side declares additional runtime
knobs (adaptive rtol controller, batch_max_retries, step_size_*,
hycean_pin_time, fastchem_newton_*) that master ignores; physics output
is unaffected.
