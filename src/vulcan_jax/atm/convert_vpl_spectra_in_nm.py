# ruff: noqa
"""One-off data-prep: convert the VPL solar spectrum to surface flux in nm.

Reads `vpl_sun_original.txt` (wavelength in micron, flux in W/cm^2/micron at
1 AU), converts to wavelength in nm and flux in ergs/cm^2/s/nm at the solar
surface (scaled by (au/r_sun)^2), and writes `VPL_solar.txt`.
"""

import numpy as np
import scipy

### PHYS CONSTANTS
# Planck constant times the light speed
hc = 1.98644568e-9  # erg.nm
au = 1.4959787e13  # cm
r_sun = 6.957e10  # cm
###

# VPL SUN: W/cm^2/micron
new_str = (
    '# solar flux at the "surface of Sun" from VPL\n# WL(nm)\t Flux(ergs/cm**2/s/nm)\n'
)

with open("vpl_sun_original.txt") as f:
    for line in f.readlines():
        if not line.startswith("#") and line.split():
            li = line.split()
            new_str += (
                "{:<12}".format(float(li[0]) * 1.0e3)
                + "\t"
                + "{:.2E}".format(float(li[1]) * 1.0e4 * (au / r_sun) ** 2.0)
                + "\n"
            )


with open("VPL_solar.txt", "w+") as f:
    f.write(new_str)
