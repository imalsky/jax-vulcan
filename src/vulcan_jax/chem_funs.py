"""JAX-native chem_funs: parses the network once per process at import and
exposes ni, nr, spec_list, the parsed network (_NETWORK, _NET_JAX) and the
codegen chemistry RHS (chem_rhs_codegen)."""

from __future__ import annotations

import jax

from .config import default_config
from . import network as _network
from . import chem as _chem
from . import make_chem_funs as _make_chem_funs
from ._paths import resolve_data_path

jax.config.update("jax_enable_x64", True)

_CFG = default_config()


_NETWORK = _network.parse_network(str(resolve_data_path(_CFG.network)))
_NET_JAX = _chem.to_jax(_NETWORK)

# Build (or reload from cache) the SymPy-faithful per-reaction RHS.
# Memoised in `_make_chem_funs._BUILD_CACHE`, so the warmup call in
# `state._build_pre_loop_runstate` returns the same `Callable`.
_CHEM_RHS_CODEGEN = _make_chem_funs.build_chem_rhs(_NETWORK)

ni: int = _NETWORK.ni
nr: int = _NETWORK.nr
spec_list: list[str] = list(_NETWORK.species)

# The production RHS (the integrator binds it).
chem_rhs_codegen = _CHEM_RHS_CODEGEN
