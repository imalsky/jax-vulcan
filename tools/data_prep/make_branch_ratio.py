# -*- coding: utf-8 -*-
"""One-off data-prep: build a photodissociation branching-ratio CSV.

For the molecule set by the `molecule` / `num_br` constants below, reads its
PhiDRates cross-section file, normalizes the per-channel rates by their sum at
each wavelength, and writes `<molecule>/<molecule>_branch.csv`. Configured for
SO2 (2 branches: SO+O, S+O2).
"""

import numpy as np

molecule = "SO2"
# number of branches
num_br = 2

# 907    [ SO2 -> SO + O                      ]              SO2     1
# 909    [ SO2 -> S + O2                      ]              SO2     2

outstr = (
    "# Branching ratios for "
    + molecule
    + " -> (1)SO + O  (2)S+ O2  nm  from PhiDRates\n"
)
outstr += "# lambda "
for i in range(1, num_br + 1):
    outstr += "{:>12}".format(", br_ratio_" + str(i))
outstr += "\n"

# Lambda  Total   SO/O(ch1)   S/O2(ch2)

phid = np.genfromtxt(
    "../../../../../../Bern/python/photo_cross_data/phidrates_data/"
    + molecule
    + ".txt",
    dtype=float,
    skip_header=1,
    names=["lambda", "tot", "1", "2"],
    usecols=[0, 1, 2, 3],
)

for n, lmd in enumerate(phid["lambda"]):
    if lmd / 10 > 0:
        dis_tot = phid["1"][n] + phid["2"][n]

        if dis_tot > 0:
            outstr += (
                "{:<9.3e}".format(lmd / 10.0)
                + ","
                + "{:>9.3f}".format(float(phid["1"][n]) / dis_tot)
                + ","
                + "{:>9.3f}".format(float(phid["2"][n]) / dis_tot)
                + "\n"
            )


with open(molecule + "/" + molecule + "_branch.csv", "w") as of:
    outstr = outstr[:-1]
    of.write(outstr)
print(molecule + " done")
