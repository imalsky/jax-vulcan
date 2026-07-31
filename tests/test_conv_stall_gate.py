"""The JAX-only stalled-convergence fallback: off means off, and the gates hold.

`conv_stall_window` has no counterpart in VULCAN 2.0 `master` or in
`shami-EEG/VULCAN` `vm_branch` (Shami asked what it was, email 2026-07-14), so a
VULCAN 2 parity run must be able to switch it off. Before 2026-07-30 it could
not be: the only knob was the lookback `conv_stall_window`, which
`runtime_validation` requires to be >= 1, and which makes the fallback fire
SOONER as it shrinks. `use_conv_stall` is the explicit off switch.

The tests seed the runner carry directly into a stall-eligible state rather than
integrating to one, so each case is a single `cond_fn` evaluation:

  * every stall gate satisfied,
  * `conv_normal` deliberately FALSE (longdydt is large), so the only way the
    run can stop for a convergence reason is the stall fallback,
  * no runtime/step-count exit available.

Under those conditions the run terminates iff the fallback is enabled, which is
what "disabled means disabled" means operationally.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")

# NO shipped config may enable the fallback. It has no VULCAN counterpart, and
# as of 2026-07-30 no shipped case needs it: every one of them exits on the
# normal convergence criterion or on a budget limit, never on the stall test.
# It stays in the code as an opt-in, and its counters stay as diagnostics
# (vulcan-forward and vulcan-retrieval read them), but nothing ships with a
# convergence decision that VULCAN 2.0 could not also have made.
_PARITY_CONFIGS = ("default", "HD189", "HD209", "W39b")
_V3_CONFIGS = ("HD189_vulcan3",)


def _cfg():
    """HD189 pinned to a small, non-hybrid, non-photo-churn configuration."""
    import vulcan_jax.legacy_io as op

    c = op.default_config()
    # Pure central: hybrid phase 0 never terminates, which would mask the
    # stall exit we are testing.
    c.use_vm_mol = False
    c.use_hybrid_vm_mol = False
    c.use_print_prog = False
    c.use_live_plot = False
    c.use_live_flux = False
    c.count_min = 5
    return c


def _seed_stall_eligible(state, c, *, count_since_new_min=None, longdy=None):
    """Return a carry satisfying every stall gate but NOT `conv_normal`.

    Gates (outer_loop `_convergence_ok` / `_real_terminate`):
      count_since_new_min > conv_stall_window
      longdy_seen_min     < yconv_min
      longdy              < yconv_min
      aflux_change        < flux_cri
      ready: t > trun_min and accept_count > count_min_dyn
    `longdydt` is left large so neither `conv_normal` branch can fire.
    """
    window = int(c.conv_stall_window)
    if count_since_new_min is None:
        count_since_new_min = window + 1
    if longdy is None:
        longdy = 0.5 * float(c.yconv_min)  # below yconv_min, above yconv_cri
    return state._replace(
        t=jnp.float64(10.0 * float(c.trun_min)),
        accept_count=jnp.int32(int(c.count_min) + 5),
        count_min_dyn=jnp.int32(int(c.count_min)),
        count_max_dyn=jnp.int32(int(c.count_min) + 50),
        runtime_dyn=jnp.float64(float(c.runtime)),
        count_since_new_min=jnp.int32(int(count_since_new_min)),
        longdy_seen_min=jnp.float64(0.5 * float(c.yconv_min)),
        longdy=jnp.float64(float(longdy)),
        longdydt=jnp.float64(1.0),
        aflux_change=jnp.float64(0.0),
    )


def _run_from(c, seed_fn):
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.state import RunState

    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=c), cfg=c)
    rs = RunState.with_pre_loop_setup(c)
    state, atm_static = integ.prepare_runstate(rs)
    return integ._runner(seed_fn(state), atm_static)


def test_stall_fires_when_enabled():
    """Enabled + all gates met -> stops immediately, reason 4 (not 1)."""
    c = _cfg()
    c.use_conv_stall = True
    final = _run_from(c, lambda s: _seed_stall_eligible(s, c))
    assert int(final.termination_reason) == 4, (
        "stall fallback did not report its own termination reason; got "
        f"{int(final.termination_reason)}"
    )
    # It stopped on the very first predicate evaluation: no body iteration ran.
    assert int(final.accept_count) == int(c.count_min) + 5


def test_disabled_means_disabled():
    """Disabled + the SAME state -> the stall exit is unavailable.

    The run continues and can only leave via the step-count ladder, so the
    reason must be 3. This is the assertion that `use_conv_stall: false`
    actually removes the behaviour rather than merely making it less likely.
    """
    c = _cfg()
    c.use_conv_stall = False
    final = _run_from(c, lambda s: _seed_stall_eligible(s, c))
    assert int(final.termination_reason) == 3, (
        "with the fallback disabled the run must fall through to the "
        f"step-count exit; got reason {int(final.termination_reason)}"
    )
    assert int(final.accept_count) > int(c.count_min) + 5


@pytest.mark.parametrize(
    "kwargs, why",
    [
        ({"count_since_new_min": 0}, "window not yet elapsed"),
        ({"longdy": 10.0}, "current longdy above yconv_min"),
    ],
)
def test_stall_cannot_fire_before_its_gates(kwargs, why):
    """Enabled but one gate unmet -> must NOT report a stall convergence."""
    c = _cfg()
    c.use_conv_stall = True
    final = _run_from(c, lambda s: _seed_stall_eligible(s, c, **kwargs))
    assert int(final.termination_reason) != 4, f"stall fired despite {why}"


def test_window_is_a_lookback_not_an_off_switch():
    """`conv_stall_window` must be rejected below 1, so it cannot mean 'off'.

    A user reaching for `conv_stall_window: 0` to disable the fallback would
    otherwise get the opposite: the predicate is `count_since_new_min > window`,
    so 0 makes it fire on the first ready step.
    """
    from vulcan_jax.runtime_validation import validate_runtime_config

    c = _cfg()
    c.conv_stall_window = 0
    with pytest.raises(Exception) as excinfo:
        validate_runtime_config(c)
    assert "conv_stall_window" in str(excinfo.value)


def test_parity_configs_ship_the_fallback_off():
    """Shipped VULCAN 2 parity configs must declare `use_conv_stall: false`."""
    from vulcan_jax.config import load_config

    for name in _PARITY_CONFIGS:
        cfg = load_config(name)
        assert getattr(cfg, "use_conv_stall", None) is False, (
            f"{name}.yaml must ship use_conv_stall: false — the fallback has no "
            "VULCAN 2.0 counterpart and must not affect a parity run"
        )


def test_no_shipped_config_enables_the_fallback():
    """Nothing shipped may decide convergence on a non-VULCAN criterion.

    This used to assert the opposite for the VULCAN 3 preset. It was flipped on
    2026-07-30 after measuring that no shipped case needs it: HD189_vulcan3
    converges on the normal criterion at 2820 accepted steps
    (termination_reason 1), so the fallback never fired even when enabled, and
    every other consumer in the workspace was already disabling it by hand
    (vulcan-jwst-tool and three jax_paper adjoint scripts set
    conv_stall_window = 10**9). The mechanism stays available as an opt-in.
    """
    import glob

    import yaml

    enabled = []
    for cfg_path in sorted(glob.glob("src/vulcan_jax/configs/*.yaml")):
        with open(cfg_path) as fh:
            raw = yaml.safe_load(fh)
        if raw.get("use_conv_stall") is True:
            enabled.append(os.path.basename(cfg_path))
    assert not enabled, (
        f"configs enabling the JAX-only stall fallback: {enabled}. No shipped "
        "case needs it, and a run that stops on it is not converged by any "
        "criterion VULCAN 2.0 has. Enable it deliberately per run instead."
    )


def test_every_config_declares_the_fallback_explicitly():
    """An omitted key silently takes the code default, which is True."""
    import glob

    import yaml

    missing = []
    for cfg_path in sorted(glob.glob("src/vulcan_jax/configs/*.yaml")):
        with open(cfg_path) as fh:
            raw = yaml.safe_load(fh)
        if "use_conv_stall" not in raw:
            missing.append(os.path.basename(cfg_path))
    assert not missing, (
        f"configs not declaring use_conv_stall: {missing}. The code default is "
        "True, so an omitted key turns the fallback ON."
    )


def test_termination_reason_reaches_the_vul_parameter_dict():
    """A single-profile result must expose WHY it stopped, not just end_case.

    `end_case` cannot separate a normal convergence from a stall convergence —
    both are 1 — so the finer-grained code has to survive into the saved file.
    """
    from vulcan_jax.state import ParamInputs

    assert "termination_reason" in ParamInputs._fields
    import vulcan_jax.legacy_io as op

    src = Path(op.__file__).read_text()
    assert 'para_save["termination_reason"]' in src, (
        "termination_reason is not written into the .vul 'parameter' dict"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
