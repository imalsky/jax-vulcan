"""Ion charge balance kernel inside the runner.

The body's post-step ion clamp is:

    e_density[z] = -sum_i charge_arr[i] * y[z, i]
    y[:, e_idx]  = e_density

with `charge_arr[i] = compo[i]['e']` for `i` in `var.charge_list`, zero
elsewhere (and zero at `e_idx` itself, so the formula self-consistently
ignores the prior `e` value). Mirrors `op.Ros2.solver`'s post-step hook
(`op.py:3001-3007`).

VULCAN's `compo['e']` convention is *electrons gained*: anions get +1,
cations get -1, and free `e` itself gets +1 (per `thermo/all_compose.txt`).
The clamp formula self-consistently produces a non-negative `e` density
that brings the total electron count to zero (neutral plasma).

This test verifies the math with a tiny synthetic example — the HD189
default config has no charged species, so an end-to-end toggle isn't
possible. Two assertions:

  1. **Formula correctness** — `e[:] = -dot(y, charge_arr)` on a
     hand-constructed (nz=4, ni=5) state with species 1 = anion (+1),
     species 2 = doubly-charged anion (+2), species 3 = cation (-1),
     species 4 = `e` (0 in the array, +1 in the full neutrality sum).

  2. **Charge-neutrality round-trip** — after applying the clamp, the
     net electron count per layer (`sum_i compo_full[i] * y[z, i]` with
     `compo_full[e_idx] = +1`) must be zero.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def _ion_clamp(y, charge_arr, e_idx):
    """Mirror the body's post-step ion clamp (outer_loop.py)."""
    e_density = -jnp.einsum("zi,i->z", y, charge_arr)
    return y.at[:, e_idx].set(e_density)


def main() -> int:
    nz, ni = 4, 5
    e_idx = 4

    rng = np.random.default_rng(0)
    y_np = rng.uniform(1e8, 1e12, size=(nz, ni))
    # Pre-clamp e column gets some garbage value the clamp must overwrite.
    y_np[:, e_idx] = 7.0

    # Charges-as-stored (compo['e']): +1, +2, -1, 0, -1.
    # `charge_arr` zeros the e_idx slot per the build in `_build_statics`.
    charge_arr_np = np.array([0.0, +1.0, +2.0, -1.0, 0.0], dtype=np.float64)

    # ---- 1. Formula correctness ----
    y_clamped = np.asarray(_ion_clamp(jnp.asarray(y_np),
                                      jnp.asarray(charge_arr_np),
                                      e_idx))
    expected_e = -(charge_arr_np[None, :] * y_np).sum(axis=1)
    # einsum and Python sum may differ at last ULP; allow a few eps.
    relerr = np.max(np.abs(y_clamped[:, e_idx] - expected_e)
                    / np.maximum(np.abs(expected_e), 1e-300))
    if relerr > 1e-14:
        print(f"FAIL: e column mismatch (relerr={relerr:.3e}).\n"
              f" got: {y_clamped[:, e_idx]}\n want: {expected_e}")
        return 1
    # Non-e columns must be untouched.
    for j in range(ni):
        if j == e_idx:
            continue
        if not np.array_equal(y_clamped[:, j], y_np[:, j]):
            print(f"FAIL: non-e column {j} was modified")
            return 1
    print("formula  OK (e densities match -dot(y, charge_arr) bit-exactly)")

    # ---- 2. Charge-neutrality round-trip ----
    # The "full" electron-counting sum has +1 for the e slot itself
    # (matches `thermo/all_compose.txt`'s convention). The body's
    # `charge_arr` zeros that slot so the clamp is self-consistent;
    # restoring +1 there should give net electron count = 0 per layer.
    charge_full = charge_arr_np.copy()
    charge_full[e_idx] = +1.0
    net_charge = (charge_full[None, :] * y_clamped).sum(axis=1)
    max_resid = float(np.max(np.abs(net_charge)))
    if max_resid > 1e-3:  # absolute residual; densities are O(1e10)
        print(f"FAIL: net electron count not zero (max |sum| = {max_resid:.3e})")
        return 1
    print(f"neutrality  OK (max |net electron count| = {max_resid:.3e})")

    print()
    print("PASS")
    return 0


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
