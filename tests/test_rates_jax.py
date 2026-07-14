"""rates_jax (differentiable T -> k_arr) must match the NumPy rates.build_rate_array
bit-closely, and be finite under forward-mode AD w.r.t. T.

Guards the temperature-differentiability port: the runner's k_arr is frozen
host-side, so a T gradient rebuilds it via rates_jax; if this drifts from the
NumPy build, T-profile sensitivities are silently wrong.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def main() -> int:
    from vulcan_jax.config import default_config
    vulcan_cfg = default_config()
    from vulcan_jax.state import RunState, legacy_view
    from vulcan_jax import network as net_mod, rates as rates_np, rates_jax
    from vulcan_jax.gibbs import load_nasa9
    from vulcan_jax._paths import resolve_data_path

    vulcan_cfg.use_live_plot = vulcan_cfg.use_live_flux = vulcan_cfg.use_print_prog = (
        False
    )
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    _, atm, _ = legacy_view(rs)
    network = net_mod.parse_network(str(resolve_data_path(vulcan_cfg.network)))

    T = np.asarray(atm.Tco, dtype=np.float64)
    M = np.asarray(atm.M, dtype=np.float64)
    thermo_dir = resolve_data_path(vulcan_cfg.network).parent
    if not (thermo_dir / "NASA9").exists():
        thermo_dir = resolve_data_path("thermo")
    nasa9, _ = load_nasa9(network.species, thermo_dir)
    remove_list = getattr(vulcan_cfg, "remove_list", None)

    k_np = rates_np.build_rate_array(vulcan_cfg, network, atm, nasa9)
    k_jx = np.asarray(
        rates_jax.build_rate_array(
            network, jnp.asarray(T), jnp.asarray(M), nasa9, remove_list=remove_list
        )
    )

    assert k_np.shape == k_jx.shape, (k_np.shape, k_jx.shape)
    both = np.abs(k_np) > 0
    assert np.count_nonzero(k_jx) == np.count_nonzero(k_np)
    rel = np.abs(k_jx[both] - k_np[both]) / np.abs(k_np[both])
    assert rel.max() < 1e-9, f"rates_jax vs NumPy max rel err {rel.max():.3e}"

    # Forward-mode AD w.r.t. a uniform T shift must be finite (this is the whole
    # point: T -> k on the graph).
    def k_of_T(Tj):
        return rates_jax.build_rate_array(
            network, Tj, jnp.asarray(M), nasa9, remove_list=remove_list
        )

    _, dk = jax.jvp(k_of_T, (jnp.asarray(T),), (jnp.ones_like(jnp.asarray(T)),))
    assert bool(jnp.all(jnp.isfinite(dk))), "non-finite jvp of rates_jax wrt T"

    # Low-T caps: rates_jax.apply_lowT_caps must match rates.apply_lowT_caps on a
    # cool T where the caps fire (the default network carries the three cap rxns).
    T_cool = np.full_like(T, 250.0)
    kf_np = rates_np.compute_forward_k(network, T_cool, M)
    kf_np_capped = rates_np.apply_lowT_caps(network, kf_np, T_cool, M)
    # The caps must actually fire at 250 K (CH3/C2H4 thresholds). Magnitudes are
    # ~1e-30, far below np.allclose's default atol, so compare on touched rows.
    changed_rows = np.nonzero(np.any(kf_np_capped != kf_np, axis=1))[0]
    assert changed_rows.size >= 1, "lowT caps did not fire at 250 K"
    kf_jx_capped = np.asarray(
        rates_jax.apply_lowT_caps(
            network,
            rates_jax.compute_forward_k(network, jnp.asarray(T_cool), jnp.asarray(M)),
            jnp.asarray(T_cool),
            jnp.asarray(M),
        )
    )
    for i in changed_rows:
        rel = np.abs(kf_jx_capped[i] - kf_np_capped[i]) / np.maximum(
            np.abs(kf_np_capped[i]), 1e-300
        )
        assert rel.max() < 1e-9, f"lowT-cap parity row {i}: {rel.max():.3e}"

    # jvp wrt T through the capped build stays finite.
    def k_capped_of_T(Tj):
        return rates_jax.build_rate_array(
            network,
            Tj,
            jnp.asarray(M),
            nasa9,
            remove_list=remove_list,
            use_lowT_caps=True,
        )

    _, dkc = jax.jvp(
        k_capped_of_T, (jnp.asarray(T_cool),), (jnp.ones_like(jnp.asarray(T_cool)),)
    )
    assert bool(jnp.all(jnp.isfinite(dkc))), "non-finite jvp of capped rates_jax wrt T"

    # Arrhenius rate-coefficient overrides: passing the network's own `a` matches
    # the default build, and a jvp w.r.t. `a` is finite (rate-uncertainty path).
    a0 = jnp.asarray(np.asarray(network.a, dtype=np.float64))
    k_ovr = np.asarray(
        rates_jax.compute_forward_k(network, jnp.asarray(T), jnp.asarray(M), a=a0)
    )
    k_def = np.asarray(
        rates_jax.compute_forward_k(network, jnp.asarray(T), jnp.asarray(M))
    )
    onz = np.abs(k_def) > 0
    assert np.abs(k_ovr[onz] - k_def[onz]).max() / np.abs(k_def[onz]).max() < 1e-12

    def k_of_a(av):
        return rates_jax.build_rate_array(
            network,
            jnp.asarray(T),
            jnp.asarray(M),
            nasa9,
            remove_list=remove_list,
            rate_coeffs={"a": av},
        )

    _, dka = jax.jvp(k_of_a, (a0,), (a0,))
    assert bool(jnp.all(jnp.isfinite(dka))), (
        "non-finite jvp of rates_jax wrt Arrhenius a"
    )
    return 0


def test_main():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
