"""Shared test helpers: config pinning and child-process launching.

Neither belongs in `oracle.py`: nothing here touches the upstream checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fast_cfg(**overrides):
    """The process default config, pinned for a fast, quiet, isothermal run.

    Mutates and returns the shared cached `default_config()` object; callers
    rely on the mutation being visible to later `default_config()` readers.

    The diffusion-scheme knobs (`use_vm_mol`, `use_hybrid_vm_mol`) are not
    pinned: test_hybrid_vm_mol tests them, so each caller sets them
    explicitly.
    """
    from vulcan_jax.config import default_config

    cfg = default_config()
    cfg.count_min = 1
    cfg.use_print_prog = False
    cfg.use_photo = False
    cfg.use_ion = False
    cfg.atm_type = "isothermal"
    cfg.Kzz_prof = "Pfunc"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def run_child(child_src: str, *, network: str, label: str, timeout: int = 600):
    """Run `child_src` in a fresh CPU-only interpreter with `network` selected.

    A cold start is required: the network is import-frozen, so it can only be
    chosen through the environment before `vulcan_jax` is first imported.
    Asserts a clean exit and returns the CompletedProcess so the caller can
    make its own assertions about stdout.
    """
    env = {
        **os.environ,
        "JAX_PLATFORM_NAME": "cpu",
        "VULCAN_JAX_NETWORK": network,
    }
    res = subprocess.run(
        [sys.executable, "-c", child_src, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=ROOT,
        check=False,  # the assert below reports stdout/stderr on failure
    )
    assert res.returncode == 0, (
        f"{label} subprocess exited {res.returncode}\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    return res
