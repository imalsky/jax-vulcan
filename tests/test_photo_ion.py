"""Validate the JAX photoionization wrapper against a direct NumPy integral."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import op_jax
import vulcan_cfg


def _manual_jion(aflux, cross, split, dbin1, dbin2):
    left = np.sum(aflux[:, :split] * cross[:split], axis=1) * dbin1
    left -= 0.5 * (aflux[:, 0] * cross[0] + aflux[:, split - 1] * cross[split - 1]) * dbin1
    right = np.sum(aflux[:, split:] * cross[split:], axis=1) * dbin2
    right -= 0.5 * (aflux[:, split] * cross[split] + aflux[:, -1] * cross[-1]) * dbin2
    return left + right


def main() -> int:
    aflux = np.array(
        [
            [1.0, 2.0, 4.0, 8.0, 16.0],
            [0.5, 1.0, 1.5, 2.0, 2.5],
        ],
        dtype=np.float64,
    )
    cross_h2o_1 = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)
    cross_h2o_2 = np.array([0.3, 0.2, 0.1, 0.05, 0.02], dtype=np.float64)
    cross_nh3_1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    split = 3
    dbin1 = 0.1
    dbin2 = 1.5
    f_diurnal = float(vulcan_cfg.f_diurnal)

    var = SimpleNamespace(
        aflux=aflux,
        cross_Jion={
            ("H2O", 1): cross_h2o_1,
            ("H2O", 2): cross_h2o_2,
            ("NH3", 1): cross_nh3_1,
        },
        ion_sp={"H2O", "NH3"},
        ion_branch={"H2O": 2, "NH3": 1},
        ion_rate_index={("H2O", 1): 7, ("H2O", 2): 8, ("NH3", 1): 9},
        sflux_din12_indx=split,
        dbin1=dbin1,
        dbin2=dbin2,
        nbin=aflux.shape[1],
        k={7: np.zeros(aflux.shape[0]), 8: np.zeros(aflux.shape[0]), 9: np.zeros(aflux.shape[0])},
    )

    solver = op_jax.Ros2JAX()
    solver.compute_Jion(var, atm=None)

    expected = {
        ("H2O", 1): _manual_jion(aflux, cross_h2o_1, split, dbin1, dbin2),
        ("H2O", 2): _manual_jion(aflux, cross_h2o_2, split, dbin1, dbin2),
        ("NH3", 1): _manual_jion(aflux, cross_nh3_1, split, dbin1, dbin2),
    }
    expected_total = {
        "H2O": expected[("H2O", 1)] + expected[("H2O", 2)],
        "NH3": expected[("NH3", 1)],
    }

    ok = True
    for key, want in expected.items():
        got = np.asarray(var.Jion_sp[key], dtype=np.float64)
        relerr = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-300))
        print(f"{key}: max relerr = {relerr:.3e}")
        if relerr > 1e-14:
            ok = False
        ridx = var.ion_rate_index[key]
        if not np.allclose(var.k[ridx], want * f_diurnal, rtol=1e-14, atol=0.0):
            print(f"FAIL: k[{ridx}] mismatch for {key}")
            ok = False

    for sp, want in expected_total.items():
        got = np.asarray(var.Jion_sp[(sp, 0)], dtype=np.float64)
        if not np.allclose(got, want, rtol=1e-14, atol=0.0):
            print(f"FAIL: total branch sum mismatch for {sp}")
            ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
