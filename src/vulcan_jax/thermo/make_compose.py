# ruff: noqa
import numpy as np

com_file = "nomass_all_compose.txt"


compo = np.genfromtxt(com_file, names=True, dtype=None)
compo_row = list(compo["species"])

ost = "species			H	O	C	He	N	S	P	Na	K	Si 	mass      \n"
with open(com_file) as f:
    for line in f.readlines():
        if not line.startswith("species") and line.strip():
            col = line.split()
            indx = compo_row.index(col[0])

            # Mass from per-element atom counts looked up by COLUMN NAME, so the
            # result is independent of the column order in nomass_all_compose.txt.
            # (Positional compo[indx][1..10] silently produced wrong masses if the
            # header was reordered -- same bug class as the FastChem element-order
            # regression.)
            mass = (
                compo[indx]["H"] * 1.008
                + compo[indx]["O"] * 16.0
                + compo[indx]["C"] * 12.011
                + compo[indx]["He"] * 4.003
                + compo[indx]["N"] * 14.007
                + compo[indx]["S"] * 32.065
                + compo[indx]["P"] * 30.974
                + compo[indx]["Na"] * 22.990
                + compo[indx]["K"] * 39.10
                + compo[indx]["Si"] * 28.085
            )

            line = line[:-1]
            line = line.rstrip() + "\t" + str(mass) + "\n"
            ost += line


with open("all_compose.txt", "w+") as f:
    f.write(ost)
