"""Every knob `runtime_validation` bound-checks must be declared in every
shipped config.

`_validate_numerical_bounds` used to carry a literal default beside each knob
(`getattr(cfg, "rtol", 0.2)`), a second copy of a number the YAML owns. Those
were removed on 2026-08-14: the validator now bound-checks what a config
DECLARES and skips what it does not, so there is exactly one home for each
value. That skip is only safe while the shipped configs declare everything, and
this test is what makes it safe -- without it, deleting a knob from a YAML would
silently turn its bound check off instead of failing.

Also pins that the literals stay gone.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "src" / "vulcan_jax" / "configs"
VALIDATION = ROOT / "src" / "vulcan_jax" / "runtime_validation.py"

# Conditionally-read knobs: the validator only reaches them when another knob
# turns their subsystem on, so a config that leaves the subsystem off need not
# declare them. Everything else is unconditional.
_CONDITIONAL = {
    "high_temp_cut_K", "high_temp_cut_P",          # only when high_temp_cut
    "fastchem_newton_max_iter", "fastchem_newton_tol",  # only for EQ/const_lowT
    "start_conden_time", "stop_conden_time",       # only when use_condense
}


def _validator_source() -> str:
    return VALIDATION.read_text()


def _checked_knobs() -> set[str]:
    """Knob names `_validate_numerical_bounds` reads, from the source itself."""
    src = _validator_source()
    names = set(re.findall(r'_declared\(cfg,\s*"([A-Za-z_]+)"', src))
    # the two table-driven loops
    for block in re.findall(r"for key, unit in \((.*?)\n    \):", src, re.S):
        names |= set(re.findall(r'\("([a-z_]+)",', block))
    for block in re.findall(
            r'for key in \("count_min".*?\):', src, re.S):
        names |= set(re.findall(r'"([a-z_]+)"', block))
    assert len(names) > 25, f"knob scrape found only {len(names)}; parser broke"
    return names


def _config_files() -> list[Path]:
    files = sorted(CONFIGS.glob("*.yaml"))
    assert files, f"no shipped configs under {CONFIGS}"
    return files


@pytest.mark.parametrize("cfg_path", _config_files(), ids=lambda p: p.stem)
def test_every_shipped_config_declares_every_checked_knob(cfg_path):
    declared = set(yaml.safe_load(cfg_path.read_text()) or {})
    required = _checked_knobs() - _CONDITIONAL
    # dt_max is DERIVED by config.py (runtime * 1e-5), never written in YAML.
    required.discard("dt_max")
    missing = sorted(required - declared)
    assert not missing, (
        f"{cfg_path.name} does not declare {missing}. runtime_validation "
        f"bound-checks these, and since 2026-08-14 it SKIPS a knob a config "
        f"does not declare instead of substituting a literal -- so an "
        f"undeclared knob silently loses its check. Declare it in the YAML."
    )


def test_the_validator_carries_no_literal_config_defaults():
    """No `getattr(cfg, "knob", <number>)` may come back.

    Such a literal is a second copy of a YAML-owned value; outer_loop.py owns
    the runtime back-compat fallback. If the two drift, this module validates a
    number the run never uses.
    """
    tree = ast.parse(_validator_source())
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) == 3):
            continue
        target, name, default = node.args
        if not (isinstance(target, ast.Name) and target.id == "cfg"):
            continue
        if isinstance(default, ast.Constant) and isinstance(
                default.value, (int, float)) and not isinstance(
                default.value, bool):
            offenders.append(
                f"line {node.lineno}: getattr(cfg, "
                f"{getattr(name, 'value', '?')!r}, {default.value!r})")
    assert not offenders, (
        "numeric literal config defaults are back in runtime_validation.py:\n  "
        + "\n  ".join(offenders)
        + "\nUse _declared(cfg, key) and skip an undeclared knob instead.")
