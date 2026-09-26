"""One-off data-prep: tabulate a Rayleigh scattering cross section vs wavelength.

For the species selected by the `sp` constant below, evaluates the cross
section over `lmd_array` (0.1-800 nm) and writes `<sp>_rayleigh.txt`. The
refractive-index and King-factor dispersion formulae follow Daniel's note.
"""

import numpy as np

# 1./(lmd/NM_PER_CM) converts the wavelength (nm) to the wavenumber (cm^-1) of
# Daniel's note.
NM_PER_CM = 1.0e7
# Reference number density (cm^-3) of the N2, CO2 and He refractive indices;
# the values are upstream's (thermo/photo_cross/rayleigh/make_rayleigh.py).
N_REF_STP = 2.546899e19


def cross(lmd, n_ref, nr, K):
    """Rayleigh scattering cross section (cm^2) from Daniel's note, scaled by n_ref^2.

    lmd: wavelength (nm in the code; cm in the note).
    n_ref: reference number density the refractive index is tabulated at
        (not the actual number density).
    nr: refractive index evaluated at lmd.
    K: King correction factor evaluated at lmd.
    """
    cross_ns = (
        24
        * np.pi**3
        / (n_ref**2 * (lmd / NM_PER_CM) ** 4.0)
        * ((nr**2 - 1.0) / (nr**2 + 2.0)) ** 2
        * K
    )

    return cross_ns


lmd_array = np.arange(0.1, 800.1, 0.1)
n_indx, King, n_ref = {}, {}, {}
n_indx["H2"] = lambda lmd: 1.358e-4 * (1.0 + 7.52e-3 * (lmd / 1.0e3) ** (-2)) + 1.0
King["H2"] = lambda lmd: 1.0
n_ref["H2"] = 2.65163e19

n_indx["O2"] = lambda lmd: (
    (20564.8 + 2.480899e13 / (4.09e9 - (1.0 / (lmd / NM_PER_CM)) ** 2.0)) * 1.0e-8 + 1.0
)
n_ref["O2"] = 2.68678e19
King["O2"] = lambda lmd: (
    1.09
    + 1.385e-11 * (1.0 / (lmd / NM_PER_CM)) ** 2.0
    + 1.448e-20 * (1.0 / (lmd / NM_PER_CM)) ** 4.0
)

n_indx["N2"] = lambda lmd: (
    (5677.465 + 3.1881874e14 / (1.44e10 - (1.0 / (lmd / NM_PER_CM)) ** 2.0)) * 1.0e-8 + 1.0
)
n_ref["N2"] = N_REF_STP
King["N2"] = lambda lmd: 1.034 + 3.17e-12 * (1.0 / (lmd / NM_PER_CM))

n_indx["CO2"] = lambda lmd: (
    (
        5799.25 / (128908.9**2 - (1.0 / (lmd / NM_PER_CM)) ** 2)
        + 120.05 / (89223.8**2 - (1.0 / (lmd / NM_PER_CM)) ** 2)
        + 5.3334 / (75037.5**2 - (1.0 / (lmd / NM_PER_CM)) ** 2)
        + 4.3244 / (67837.7**2 - (1.0 / (lmd / NM_PER_CM)) ** 2)
        + 0.1218145e-6 / (2418.136**2 - (1.0 / (lmd / NM_PER_CM)) ** 2)
    )
    * 1.1427e3
    + 1.0
)
n_ref["CO2"] = N_REF_STP
King["CO2"] = lambda lmd: 1.1364 + 25.3e-12 * (1.0 / (lmd / NM_PER_CM)) ** 2.0

n_indx["He"] = lambda lmd: (
    (2283.0 + (1.8102e13) / (1.5342e10 - (1.0 / (lmd / NM_PER_CM)) ** 2.0)) * 1.0e-8 + 1
)
n_ref["He"] = N_REF_STP
King["He"] = lambda lmd: 1.0


# Choose the species
sp = "He"

ost = "#lambda (nm)  cross(cm^2)\n"

for lm in lmd_array:
    ost += (
        "{0:<14s}".format(str(lm))
        + "{:.3E}".format(cross(lm, n_ref[sp], n_indx[sp](lm), King[sp](lm)))
        + "\n"
    )

ost = ost[:-1]
with open(sp + "_rayleigh.txt", "w") as f:
    f.write(ost)
