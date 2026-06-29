"""One-off data-prep: concatenate per-species NASA9 thermo files into one.

Reads each `<species>.txt` in `spec_list` from the current directory and
writes `issi_nasa9.txt`, with each species' block prefixed by its name and
separated by blank lines.
"""

# species list
spec_list = [
    "H",
    "H2O",
    "OH",
    "H2",
    "O",
    "CH",
    "C",
    "CH2",
    "CH3",
    "CH4",
    "C2",
    "C2H2",
    "C2H",
    "C2H3",
    "C2H4",
    "C2H5",
    "C2H6",
    "CO",
    "CO2",
    "CH2OH",
    "H2CO",
    "HCO",
    "CH3O",
    "CH3OH",
    "CH3CO",
    "O2",
    "H2CCO",
    "HCCO",
    "He",
]

ost = ""
for sp in spec_list:
    with open(sp + ".txt") as f:
        ost += sp + "\n"
        lines = f.read()
        ost += lines + "\n" + "\n"

with open("issi_nasa9.txt", "w") as fout:  # This removes the file contents
    fout.write(ost)
