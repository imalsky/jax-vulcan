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
compo = np.genfromtxt(resolve_data_path(_CFG.com_file), names=True, dtype=_type_list)
compo_row = list(compo["species"])

atom_list: tuple[str, ...] = tuple(
    name for name in compo.dtype.names if name not in ("species", "mass")
)

# Upstream ships six species twice; three of those pairs disagree, so the row
# picked decides the mass. Both codes take the first (`list.index`), which is
# what these values are — keeping them is a parity choice, not an endorsement
# (HCS's 45.178 is 0.099 amu above its own elemental sum). Any OTHER
# disagreeing duplicate is new data corruption and must refuse rather than
# silently depend on file order.
_PARITY_DUPLICATES = {"C2H4O", "CH3NO2", "HCS"}


_seen: dict[str, int] = {}
_bad = [
    f"{sp} (rows {_seen[sp] + 1} and {i + 1})"
    for i, sp in enumerate(compo_row)
    if _seen.setdefault(sp, i) != i
    and sp not in _PARITY_DUPLICATES
    and tuple(compo[_seen[sp]]) != tuple(compo[i])
]
if _bad:
    raise ValueError(
        f"{COM_FILE_PATH}: duplicated species with disagreeing rows: "
        f"{', '.join(_bad)}. First row wins, so the physics would depend on "
        "file order. Fix the table, or add it to _PARITY_DUPLICATES if it is a "
        "known upstream bug kept deliberately for parity."
    )

_ni = chem_funs.ni
_compo_array_np = np.zeros((_ni, len(atom_list)), dtype=np.float64)
for _i, _sp in enumerate(species):
    if _sp not in compo_row:
        raise ValueError(
            f"species {_sp!r} is used by the loaded network but has no row in "
            f"{COM_FILE_PATH}, so its atom counts and mass are unknown and "
            "elemental bookkeeping, mean molecular weight, and molecular "
            "diffusion would all be wrong."
        )
    _row = compo[compo_row.index(_sp)]
    for _j, _atom in enumerate(atom_list):
        _compo_array_np[_i, _j] = float(_row[_atom])
compo_array = jnp.asarray(_compo_array_np)
