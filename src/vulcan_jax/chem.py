"""Vectorised JAX chemistry RHS and Jacobian.

Convention:
  y[nz, ni]    number densities (cm^-3)
  M[nz]        total third-body density (cm^-3)
  k[nr+1, nz]  rate constants per reaction per layer (1-based)

Rate per reaction per layer:
    rate[i, z] = k[i, z] * Π_slot (y[reactant_idx[i,slot], z] ** stoich[i,slot])
    if is_three_body[i]: rate[i, z] *= M[z]

Per-species dy/dt is then a `segment_sum` of (production - loss) contributions.

Production uses `make_chem_funs.build_chem_rhs(net)` (SymPy-faithful per-
reaction codegen, master-bit-faithful term order). The vectorised
`segment_sum` form below is preserved as `chem_rhs_segment_sum` for
test references — `chem_jac_per_layer = jax.jacrev(chem_rhs_per_layer_segment_sum)`
is still the oracle for the analytical Jacobian. The analytical Jacobian
itself uses the same `y_r ** stoich` formulation because the leave-one-
out derivative form is naturally written that way and the Jacobian has
no cancellation amplifier (machine-precision agreement vs master).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .network import Network


jax.config.update("jax_enable_x64", True)

# Reaction-axis chunk size for the analytical-Jacobian scatter. The
# per-reaction contribution tensor is (chunk, 2*max_terms, max_terms) per
# layer; under vmap (nz layers x batch lanes) the un-chunked (nr+1, 6, 3)
# form peaked at ~60 GiB for batch 512 x nz 150 on the SNCHO network
# (878 reactions) and OOM'd a 96 GB GH200. lax.scan over 128-reaction
# blocks bounds the transient at ~chunk/nr of that (~7x smaller) while
# the carried accumulator is only (ni+1)^2 floats. Not a tuning knob:
# changing it permutes float summation order (~1e-16 per cell), which is
# within the documented step-level reproducibility envelope but would
# still churn step counts, so it stays a code constant.
_JAC_CHUNK_REACTIONS = 128


class NetworkArrays:
    """Network stoichiometry packed for JAX.

    Registered as a custom pytree with `ni`/`nr` as static aux_data so
    jit/vmap don't retrace per network and `num_segments` stays concrete.
    """

    __slots__ = (
        "ni",
        "nr",
        "reactant_idx",
        "product_idx",
        "reactant_stoich",
        "product_stoich",
        "is_three_body",
    )

    def __init__(
        self,
        ni,
        nr,
        reactant_idx,
        product_idx,
        reactant_stoich,
        product_stoich,
        is_three_body,
    ):
        self.ni = int(ni)
        self.nr = int(nr)
        self.reactant_idx = reactant_idx
        self.product_idx = product_idx
        self.reactant_stoich = reactant_stoich
        self.product_stoich = product_stoich
        self.is_three_body = is_three_body


def _network_arrays_flatten(net):
    children = (
        net.reactant_idx,
        net.product_idx,
        net.reactant_stoich,
        net.product_stoich,
        net.is_three_body,
    )
    aux = (net.ni, net.nr)
    return children, aux


def _network_arrays_unflatten(aux, children):
    ni, nr = aux
    r_idx, p_idx, r_st, p_st, is3 = children
    return NetworkArrays(ni, nr, r_idx, p_idx, r_st, p_st, is3)


jtu.register_pytree_node(
    NetworkArrays, _network_arrays_flatten, _network_arrays_unflatten
)


def to_jax(net: Network) -> NetworkArrays:
    """Pack a Network's relevant arrays into jnp form for the chemistry RHS."""
    return NetworkArrays(
        ni=net.ni,
        nr=net.nr,
        reactant_idx=jnp.asarray(net.reactant_idx, dtype=jnp.int64),
        product_idx=jnp.asarray(net.product_idx, dtype=jnp.int64),
        reactant_stoich=jnp.asarray(net.reactant_stoich, dtype=jnp.float64),
        product_stoich=jnp.asarray(net.product_stoich, dtype=jnp.float64),
        is_three_body=jnp.asarray(net.is_three_body, dtype=jnp.bool_),
    )


def chem_rhs_per_layer_segment_sum(
    y: jnp.ndarray,  # [ni]
    M: float | jnp.ndarray,
    k: jnp.ndarray,  # [nr+1]
    net: NetworkArrays,
) -> jnp.ndarray:
    """Vectorised RHS via masked y**stoich + segment_sum. Test reference only.

    Production uses `make_chem_funs.build_chem_rhs(net)` (codegen). This
    kernel is preserved for vmap-consistency tests, the jacrev oracle,
    and benchmarks against the codegen path.
    """
    # Pad y so reactant_idx==ni is a no-op multiplier (the padding stoich
    # is also 0, so this is double-safe).
    yp = jnp.concatenate([y, jnp.ones((1,), dtype=y.dtype)])

    y_r = yp[net.reactant_idx]  # [nr+1, max_terms]
    # `where` guards against 0**0 producing a NaN gradient on padded slots.
    factor = jnp.where(net.reactant_stoich > 0, y_r**net.reactant_stoich, 1.0)
    prod_r = jnp.prod(factor, axis=1)  # [nr+1]
    rate = k * prod_r
    rate = jnp.where(net.is_three_body, rate * M, rate)

    flat_r_idx = net.reactant_idx.reshape(-1)
    flat_p_idx = net.product_idx.reshape(-1)
    flat_r_st = net.reactant_stoich.reshape(-1)
    flat_p_st = net.product_stoich.reshape(-1)
    rate_repeat = jnp.repeat(rate, net.reactant_idx.shape[1])

    # num_segments = ni+1: the last segment collects padding contributions
    # (reactant_idx==ni); we drop it on slice.
    loss = jax.ops.segment_sum(
        flat_r_st * rate_repeat,
        flat_r_idx,
        num_segments=net.ni + 1,
        indices_are_sorted=False,
    )[: net.ni]
    prod = jax.ops.segment_sum(
        flat_p_st * rate_repeat,
        flat_p_idx,
        num_segments=net.ni + 1,
        indices_are_sorted=False,
    )[: net.ni]
    return prod - loss


chem_rhs_segment_sum = jax.vmap(
    chem_rhs_per_layer_segment_sum,
    in_axes=(0, 0, 1, None),
)


# jacrev beats jacfwd here ("scatter at the end" pattern). Kept as the test
# oracle for `chem_jac_analytical`; production uses the analytical form
# (~36× faster on the SNCHO network). Bound to the segment_sum reference
# kernel — the Jacobian has no cancellation amplifier so the floor that
# motivated the codegen RHS does not apply here.
chem_jac_per_layer = jax.jacrev(chem_rhs_per_layer_segment_sum, argnums=0)
chem_jac = jax.vmap(
    chem_jac_per_layer,
    in_axes=(0, 0, 1, None),
)


def chem_jac_analytical_per_layer(
    y: jnp.ndarray,  # [ni]
    M: float | jnp.ndarray,
    k: jnp.ndarray,  # [nr+1]
    net: NetworkArrays,
) -> jnp.ndarray:
    """Stoichiometry-driven chemistry Jacobian for one layer. Returns [ni, ni].

    Builds J[i, j] = Σ_r sign_i * stoich_i * (∂rate[r]/∂y_j) directly from
    the network tables, skipping `jacrev`'s ni reverse-mode passes.
    """
    yp = jnp.concatenate([y, jnp.ones((1,), dtype=y.dtype)])

    y_r = yp[net.reactant_idx]  # [nr+1, max_terms]
    factor_l = jnp.where(net.reactant_stoich > 0, y_r**net.reactant_stoich, 1.0)

    # Leave-one-out reactant product: Π_{l != j} factor_l[r, l]. Loop over
    # the static `max_terms` axis (typically 3 for VULCAN).
    max_terms = net.reactant_idx.shape[1]
    slot_arange = jnp.arange(max_terms)
    leave_out_cols = []
    for j in range(max_terms):
        f_excl = jnp.where(slot_arange == j, 1.0, factor_l)
        leave_out_cols.append(jnp.prod(f_excl, axis=1))
    leave_out = jnp.stack(leave_out_cols, axis=1)

    # The y^(stoich-1) form sidesteps the (stoich/y)*rate divide when y == 0.
    # Forward-mode AD safety: a stoich-1 reactant gives exponent 0, and the
    # outer `where` *selects* that branch, so `y_r ** 0` (primal 1.0) leaks a
    # NaN jvp at y_r == 0 (d/dy y^0 = 0 * y^-1 = 0*inf). Clipped cells make
    # y_r == 0 routine mid-run, so never raise to the 0 power: stoich==1 is a
    # constant 1, stoich>=2 a real power with exponent >= 1 (finite jvp at 0).
    # Primal is bit-identical to y_r**(stoich-1); masks fold at trace time
    # (reactant_stoich is static), so there is no runtime cost.
    safe_exp = jnp.where(net.reactant_stoich > 1, net.reactant_stoich - 1, 1)
    pow_minus_one = jnp.where(
        net.reactant_stoich >= 2,
        y_r**safe_exp,
        jnp.where(net.reactant_stoich == 1, 1.0, 0.0),
    )
    drate_dy = net.reactant_stoich * pow_minus_one * leave_out * k[:, None]
    drate_dy = jnp.where(net.is_three_body[:, None], drate_dy * M, drate_dy)

    # Stack reactants (sign -1) and products (sign +1) into one "output" axis;
    # contrib[r, s_i, s_j] = out_stoich_signed[r, s_i] * drate_dy[r, s_j].
    ni = net.ni
    out_idx = jnp.concatenate([net.reactant_idx, net.product_idx], axis=1)
    out_stoich_signed = jnp.concatenate(
        [-net.reactant_stoich, net.product_stoich], axis=1
    )

    # Scatter in `lax.scan` chunks over the reaction axis instead of one flat
    # (nr+1, 2*max_terms, max_terms) contrib tensor: under vmap every lane's
    # transient is live at once, so the flat form is the batch-512 OOM driver
    # (see _JAC_CHUNK_REACTIONS). Pad rows scatter exact zeros (stoich and
    # drate_dy pads are 0) into the stripped padding row/col (index ni).
    n_rows = out_idx.shape[0]
    chunk = min(_JAC_CHUNK_REACTIONS, n_rows)
    n_chunks = -(-n_rows // chunk)
    pad = n_chunks * chunk - n_rows

    r_idx_c = jnp.pad(net.reactant_idx, ((0, pad), (0, 0)), constant_values=ni)
    out_idx_c = jnp.pad(out_idx, ((0, pad), (0, 0)), constant_values=ni)
    out_st_c = jnp.pad(out_stoich_signed, ((0, pad), (0, 0)))
    drate_c = jnp.pad(drate_dy, ((0, pad), (0, 0)))

    r_idx_c = r_idx_c.reshape(n_chunks, chunk, max_terms)
    out_idx_c = out_idx_c.reshape(n_chunks, chunk, 2 * max_terms)
    out_st_c = out_st_c.reshape(n_chunks, chunk, 2 * max_terms)
    drate_c = drate_c.reshape(n_chunks, chunk, max_terms)

    def _scatter_chunk(J_acc, xs):
        o_idx, o_st, r_idx, dr = xs
        contrib = o_st[:, :, None] * dr[:, None, :]
        row = jnp.broadcast_to(o_idx[:, :, None], contrib.shape)
        col = jnp.broadcast_to(r_idx[:, None, :], contrib.shape)
        keys = row.reshape(-1) * (ni + 1) + col.reshape(-1)
        J_acc = J_acc + jax.ops.segment_sum(
            contrib.reshape(-1),
            keys,
            num_segments=(ni + 1) * (ni + 1),
            indices_are_sorted=False,
        )
        return J_acc, None

    # unroll=1 keeps XLA from fusing the chunks back into one flat scatter,
    # which would silently reintroduce the un-bounded transient.
    J_flat, _ = jax.lax.scan(
        _scatter_chunk,
        jnp.zeros((ni + 1) * (ni + 1), dtype=drate_dy.dtype),
        (out_idx_c, out_st_c, r_idx_c, drate_c),
        unroll=1,
    )
    return J_flat.reshape(ni + 1, ni + 1)[:ni, :ni]


chem_jac_analytical = jax.vmap(
    chem_jac_analytical_per_layer,
    in_axes=(0, 0, 1, None),
)


def chem_rhs_numpy(
    y: np.ndarray, M: np.ndarray, k: np.ndarray, net: Network
) -> np.ndarray:
    """NumPy reference RHS, master-faithful term order.

    Used as the rtol=1e-13 oracle in `tests/test_chem_rhs_codegen.py`.
    Mirrors `make_chem_funs.emit_chem_rhs_source`'s emission rules:
      - per-reaction rate uses stoich-replicated `*` (not `**stoich`),
        terminal `*M` when three-body
      - per-reaction `v_pair = v[i] - v[i+1]` (k[i+1]==0 zeros out the
        second term for unpaired photo/conden/radiative/ion reactions)
      - per-species accumulator walks i=1, 3, 5, ..., products-then-
        reactants per reaction, repeated by stoich (one `+= v_pair` per
        stoich count, not `+= st * v_pair`)
    The NumPy `*` is left-associative and float64 throughout; this gives
    bit-identical emission order to master's chemdf body.
    """
    nz, ni = y.shape
    PAD = ni
    nr = net.nr
    M_arr = np.asarray(M, dtype=np.float64)

    v = np.zeros((nr + 1, nz), dtype=np.float64)
    for i in range(1, nr + 1):
        rate = np.asarray(k[i], dtype=np.float64).copy()
        for kslot in range(net.reactant_idx.shape[1]):
            sp = int(net.reactant_idx[i, kslot])
            st = int(net.reactant_stoich[i, kslot])
            if st == 0 or sp == PAD:
                continue
            for _ in range(st):
                rate = rate * y[:, sp]
        if bool(net.is_three_body[i]):
            rate = rate * M_arr
        v[i] = rate

    dydt = np.zeros_like(y)
    for i in range(1, nr + 1, 2):
        v_pair = v[i] - v[i + 1]
        for kslot in range(net.product_idx.shape[1]):
            sp = int(net.product_idx[i, kslot])
            st = int(net.product_stoich[i, kslot])
            if st == 0 or sp == PAD:
                continue
            for _ in range(st):
                dydt[:, sp] = dydt[:, sp] + v_pair
        for kslot in range(net.reactant_idx.shape[1]):
            sp = int(net.reactant_idx[i, kslot])
            st = int(net.reactant_stoich[i, kslot])
            if st == 0 or sp == PAD:
                continue
            for _ in range(st):
                dydt[:, sp] = dydt[:, sp] - v_pair
    return dydt
