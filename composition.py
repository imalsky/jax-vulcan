"""Per-species composition / mass tables, loaded once from `vulcan_cfg.com_file`."""

import numpy as np
import jax.numpy as jnp

import vulcan_cfg
import chem_funs

species = chem_funs.spec_list

with open(vulcan_cfg.com_file, "r") as _f:
    _columns = _f.readline()
    _num_ele = len(_columns.split()) - 2
_type_list = ["int"] * _num_ele
_type_list.insert(0, "U20")
_type_list.append("float")
compo = np.genfromtxt(vulcan_cfg.com_file, names=True, dtype=_type_list)
compo_row = list(compo["species"])

atom_list: tuple[str, ...] = tuple(
    name for name in compo.dtype.names if name not in ("species", "mass")
)

_ni = chem_funs.ni
_compo_array_np = np.zeros((_ni, len(atom_list)), dtype=np.float64)
for _i, _sp in enumerate(species):
    _row = compo[compo_row.index(_sp)]
    for _j, _atom in enumerate(atom_list):
        _compo_array_np[_i, _j] = float(_row[_atom])
compo_array = jnp.asarray(_compo_array_np)
