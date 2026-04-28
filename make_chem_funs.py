"""No-op shim for VULCAN-master's SymPy chem_funs codegen step.

VULCAN-JAX's `chem_funs.py` is hand-written and JAX-native — it re-exports
`ni` / `nr` / `spec_list` / `Gibbs` / `chemdf` / `re_dict` / `re_wM_dict` from
`network.py` + `gibbs.py` + `chem.py` and there is no SymPy-based code
generator to invoke. This script exists only so the upstream
`python make_chem_funs.py` invocation pattern still exits cleanly.
"""

import sys

print("VULCAN-JAX uses JAX-native chem_funs; no codegen needed.")
sys.exit(0)
