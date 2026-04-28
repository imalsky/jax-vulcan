"""Per-species composition / mass tables (Phase 21).

Extracted from `build_atm.py`. Reads `vulcan_cfg.com_file` once at import
time and exposes:

- `compo`        — numpy structured array; one row per species, columns
                   are the element names (int counts) plus `mass` (float).
                   Lookup pattern: `compo[compo_row.index(sp)][atom]`.
- `compo_row`    — list of species names, indexes rows of `compo`.
- `species`      — list of species names from `chem_funs.spec_list`.
- `compo_array`  — `(ni, n_atoms)` JAX float64 array of stoichiometric
                   coefficients in `atom_list` order. Used by the JAX-
                   native `compute_atom_ini` so it can do a single
                   `jnp.einsum("ai,zi->za", ...)` instead of a
                   per-species Python loop.
- `atom_list`    — tuple of element names from the structured array
                   header (excluding `species` and `mass`).
"""

import numpy as np
import jax.numpy as jnp

import vulcan_cfg
import chem_funs

species = chem_funs.spec_list

with open(vulcan_cfg.com_file, "r") as _f:
    _columns = _f.readline()
    _num_ele = len(_columns.split()) - 2  # minus "species" + "mass"
_type_list = ["int"] * _num_ele
_type_list.insert(0, "U20")
_type_list.append("float")
compo = np.genfromtxt(vulcan_cfg.com_file, names=True, dtype=_type_list)
compo_row = list(compo["species"])

atom_list: tuple[str, ...] = tuple(
    name for name in compo.dtype.names if name not in ("species", "mass")
)

# Pre-computed (ni, n_atoms) coefficient table for vectorised aggregation.
# Order: rows = chem_funs.spec_list, cols = atom_list.
_ni = chem_funs.ni
_compo_array_np = np.zeros((_ni, len(atom_list)), dtype=np.float64)
for _i, _sp in enumerate(species):
    _row = compo[compo_row.index(_sp)]
    for _j, _atom in enumerate(atom_list):
        _compo_array_np[_i, _j] = float(_row[_atom])
compo_array = jnp.asarray(_compo_array_np)
