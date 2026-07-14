"""Coverage for `high_temp_cut` (raise P_b to drop ultra-hot deep layers).

Ports vm_branch `build_atm.apply_high_temp_cut`. Two layers:

  - `high_temp_cut_regrid`: pure selection/re-grid logic (no JAX/codegen).
  - end-to-end via `load_TPK` in file mode with a synthetic hot TP profile,
    confirming the reloaded deep column falls below T_max after the cut.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")


def _make_pco(P_b, P_t, nz):
    return np.logspace(np.log10(P_b), np.log10(P_t), nz)


# ---------------------------------------------------------------------------
# Pure selection/re-grid logic
# ---------------------------------------------------------------------------


def test_regrid_raises_Pb_to_first_cool_deep_level():
    from vulcan_jax.atm_setup import high_temp_cut_regrid

    nz = 150
    pco = _make_pco(1e9, 1e-2, nz)  # pco[0] deepest
    # Monotonic: hot at depth, cooler with altitude.
    Tco = np.clip(4200.0 - 350.0 * np.log10(1e9 / pco), 300.0, 4200.0)

    new_pco = high_temp_cut_regrid(pco, Tco, T_max=3500.0, P_min=1e6, P_t=1e-2, nz=nz)
    assert new_pco is not None
    assert new_pco.shape == (nz,)
    assert new_pco[0] < pco[0]  # bottom pressure lowered
    assert new_pco[-1] == pytest.approx(1e-2)  # top unchanged

    # new bottom is the first deep (P>=P_min) level with T<=T_max
    p_cut = int(np.where((pco >= 1e6) & (Tco <= 3500.0))[0][0])
    assert new_pco[0] == pytest.approx(pco[p_cut])


def test_regrid_floors_at_Pmin_when_all_deep_too_hot():
    from vulcan_jax.atm_setup import high_temp_cut_regrid

    nz = 100
    pco = _make_pco(1e9, 1e-2, nz)
    Tco = np.where(pco >= 1e6, 4000.0, 3000.0)  # every deep layer too hot
    new_pco = high_temp_cut_regrid(pco, Tco, T_max=3500.0, P_min=1e6, P_t=1e-2, nz=nz)
    assert new_pco is not None
    assert new_pco[0] == pytest.approx(1e6)  # floored at P_min


def test_regrid_noop_on_cool_column():
    from vulcan_jax.atm_setup import high_temp_cut_regrid

    nz = 80
    pco = _make_pco(1e9, 1e-2, nz)
    Tco = np.full(nz, 1500.0)  # nothing above T_max
    assert high_temp_cut_regrid(pco, Tco, T_max=3500.0, P_min=1e6, P_t=1e-2, nz=nz) is None


def test_regrid_noop_when_no_deep_layers():
    from vulcan_jax.atm_setup import high_temp_cut_regrid

    nz = 80
    pco = _make_pco(1e5, 1e-2, nz)  # P_b below P_min: no eligible deep layers
    Tco = np.full(nz, 4000.0)
    assert high_temp_cut_regrid(pco, Tco, T_max=3500.0, P_min=1e6, P_t=1e-2, nz=nz) is None


# ---------------------------------------------------------------------------
# End-to-end through load_TPK (file mode) — deep column drops below T_max
# ---------------------------------------------------------------------------


def _write_hot_tp_file(path: Path) -> None:
    """Synthetic TP+Kzz file: hot (>3500 K) below ~1 bar, cooling upward."""
    p = np.logspace(9, -2, 60)  # dyne/cm^2, deep -> top
    T = np.clip(4500.0 - 380.0 * np.log10(1e9 / p), 400.0, 4500.0)
    Kzz = np.full_like(p, 1e9)
    with open(path, "w") as f:
        f.write("# synthetic hot atmosphere for high_temp_cut test\n")
        f.write("Pressure\tTemp\tKzz\n")
        for pi, ti, ki in zip(p, T, Kzz):
            f.write("{:.6e}\t{:.6e}\t{:.6e}\n".format(pi, ti, ki))


def test_end_to_end_file_mode_cut(tmp_path):
    from vulcan_jax.atm_setup import compute_pico, high_temp_cut_regrid, load_TPK

    atm_file = tmp_path / "hot_atm.txt"
    _write_hot_tp_file(atm_file)

    P_b, P_t, nz = 1e9, 1e-2, 120
    T_max, P_min = 3500.0, 1e6

    cfg = SimpleNamespace(
        atm_type="file",
        Kzz_prof="file",
        vz_prof="const",
        use_Kzz=True,
        use_vz=False,
        const_Kzz=1e10,
        const_vz=0.0,
        K_max=1e5,
        K_p_lev=0.1,
        Tiso=1234.0,
        P_b=P_b,
        Rp=1.138 * 7.1492e9,
        Mp=2140.0 * (1.138 * 7.1492e9) ** 2 / 6.67430e-8,  # -> g=G*Mp/Rp^2 = 2140
        para_anaTP=[120.0, 1500.0, 0.1, 0.02, 1.0, 1.0],
        atm_file=str(atm_file),
        vul_ini="output/",
    )

    pco = _make_pco(P_b, P_t, nz)
    pico = np.asarray(compute_pico(pco))
    out = load_TPK(cfg, pco, pico=pico)
    Tco0 = np.asarray(out["Tco"])
    assert np.any(Tco0[pco >= P_min] > T_max)  # deep column is genuinely too hot

    new_pco = high_temp_cut_regrid(pco, Tco0, T_max=T_max, P_min=P_min, P_t=P_t, nz=nz)
    assert new_pco is not None
    assert new_pco.shape == (nz,)
    assert new_pco[0] < pco[0]

    new_pico = np.asarray(compute_pico(new_pco))
    out2 = load_TPK(cfg, new_pco, pico=new_pico)
    Tco1 = np.asarray(out2["Tco"])
    # After the cut, the deep column (P>=P_min) no longer exceeds T_max.
    assert np.all(Tco1[new_pco >= P_min] <= T_max + 1e-6)
