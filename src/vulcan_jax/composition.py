"""Per-species composition / mass tables, loaded once from the config `com_file`."""

import numpy as np
import jax.numpy as jnp

from .config import default_config
from . import chem_funs
from ._paths import resolve_data_path

_CFG = default_config()
species = chem_funs.spec_list

# Resolved path of the composition table actually loaded at import. The table
# is import-frozen (ini_abun imports these arrays directly), so a later
# make_config(com_file=...) cannot change it; state._assert_com_file_matches_import
# fails fast on a mismatch instead of silently using this default.
COM_FILE_PATH = resolve_data_path(_CFG.com_file)

with open(COM_FILE_PATH, "r") as _f:
    _columns = _f.readline()
    _num_ele = len(_columns.split()) - 2
_type_list = ["int"] * _num_ele
_type_list.insert(0, "U20")
_type_list.append("float")
compo = np.genfromtxt(
    resolve_data_path(_CFG.com_file), names=True, dtype=_type_list
)
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
