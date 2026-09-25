"""`rates_jax.build_rate_array` must stay differentiable.

The rate build is one implementation shared by setup and gradients (parity
against master is `test_rates.py` / `test_gibbs.py`, and against the legacy
chain `test_read_rate.py`). What is only testable here is that the tangents
survive: a non-finite jvp w.r.t. T or w.r.t. an Arrhenius coefficient would
silently wreck every T-profile / rate-uncertainty sensitivity.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def main() -> int:
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    from vulcan_jax.state import RunState, legacy_view
    from vulcan_jax import network as net_mod, rates_jax
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

    # Forward-mode AD w.r.t. a uniform T shift must be finite (this is the whole
    # point: T -> k on the graph).
    def k_of_T(Tj):
        return rates_jax.build_rate_array(
            network, Tj, jnp.asarray(M), nasa9, remove_list=remove_list
        )

    _, dk = jax.jvp(k_of_T, (jnp.asarray(T),), (jnp.ones_like(jnp.asarray(T)),))
    assert bool(jnp.all(jnp.isfinite(dk))), "non-finite jvp of rates_jax wrt T"

    # The same, on a cool column where the three Moses+2005 caps fire (the cap
    # branch is a `where`, so its tangent has to be checked separately). The
    # cap formulas themselves are pinned in test_read_rate.py.
    T_cool = np.full_like(T, 250.0)

    def k_capped_of_T(Tj):
        return rates_jax.build_rate_array(
            network,
            Tj,
            jnp.asarray(M),
            nasa9,
            remove_list=remove_list,
            use_lowT_caps=True,
        )

    kf = rates_jax.compute_forward_k(
        network, jnp.asarray(T_cool), jnp.asarray(M)
    )
    kf_capped = rates_jax.apply_lowT_caps(
        network, kf, jnp.asarray(T_cool), jnp.asarray(M)
    )
    changed = np.nonzero(np.any(np.asarray(kf_capped) != np.asarray(kf), axis=1))[0]
    assert changed.size >= 1, "lowT caps did not fire at 250 K"

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


@pytest.mark.parametrize(
    "T, Pr", [(1000.0, 0.1), (1500.0, 0.1), (1000.0, 1.0), (1500.0, 10.0)]
)
def test_troe_oh_ch3_is_visscher_moses_eq14(T, Pr):
    """The OH+CH3+M row follows Visscher & Moses 2011 eqs 13-14 (log10 Troe
    width, C20) with the eq 24-26 fits. The only guard against a transcription
    error: the oracle comparison carries the same declared correction on its
    side."""
    from vulcan_jax import rates_jax

    k0 = 1.932e3 * T**-9.88 * np.exp(-7544.0 / T) + 5.109e-11 * T**-6.25 * np.exp(
        -1433.0 / T
    )
    kinf = 1.031e-10 * T**-0.018 * np.exp(16.74 / T)
    Fc = 0.1855 * np.exp(-T / 155.8) + 0.8145 * np.exp(-T / 1675.0) + np.exp(-4531.0 / T)
    beta = 1.0 / (1.0 + (np.log10(Pr) / (0.75 - 1.27 * np.log10(Fc))) ** 2)
    want = k0 / (1.0 + Pr) * 10.0 ** (beta * np.log10(Fc))
    Tz, M = np.array([T]), np.array([Pr * kinf / k0])
    got = float(rates_jax._troe_OH_CH3(jnp.asarray(Tz), jnp.asarray(M))[0])
    assert abs(got - want) <= 1e-12 * want


def test_lindemann_with_zero_k_inf_is_the_zero_rate_limit():
    """`k_inf == 0` on a falloff row is the zero-rate limit: the row reads
    exactly 0, with a finite tangent, not the guard value's low-pressure
    `arr / (1 + arr*M)`. A mis-shaped M raises instead of broadcasting the
    bottom layer's density over the column."""
    from vulcan_jax import network as net_mod
    from vulcan_jax import rates_jax
    from vulcan_jax._paths import resolve_data_path
    from vulcan_jax.config import default_config

    net = net_mod.parse_network(str(resolve_data_path(default_config().network)))
    falloff = np.logical_and(
        np.asarray(net.has_kinf, dtype=bool),
        np.logical_not(np.asarray(net.is_special, dtype=bool)),
    )
    rows = np.flatnonzero(falloff)
    T = jnp.linspace(500.0, 2500.0, 6)
    M = jnp.full(6, 1e18)
    k_ref = rates_jax.compute_forward_k(net, T, M)
    assert bool(jnp.any(k_ref[rows] > 0.0)), "no live falloff row to test"

    a_inf = jnp.asarray(net.a_inf, dtype=jnp.float64).at[rows].set(0.0)
    k, dk = jax.jvp(
        lambda a: rates_jax.compute_forward_k(net, T, M, a_inf=a),
        (a_inf,), (jnp.ones_like(a_inf),),
    )
    assert bool(jnp.all(k[rows] == 0.0))
    assert bool(jnp.all(jnp.isfinite(k))) and bool(jnp.all(jnp.isfinite(dk)))

    with pytest.raises(ValueError, match="same shape"):
        rates_jax.compute_forward_k(net, T, M[:1])


if __name__ == "__main__":
    raise SystemExit(main())
