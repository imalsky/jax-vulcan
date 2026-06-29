# ruff: noqa
"""One-off data-prep: convert a MUSCLES SED to stellar-surface flux in nm.

Reads the GJ1214 MUSCLES broadband FITS SED, converts wavelength
(Angstrom -> nm) and flux (erg/s/cm^2/Angstrom -> ergs/cm^2/s/nm), scales the
observed flux back to the stellar surface using GJ1214's distance (47.5 ly)
and radius (0.2064 r_sun), and writes `sflux-GJ1214.txt`. The distance/radius
are hardcoded inline (see the per-star reference block below).
"""

import numpy as np
import scipy
from astropy.io import fits

au = 1.4959787e13  # cm
r_sun = 6.957e10  # cm

# Epsilon Eridani is 10.475 light years away and with 0.735 solar radius
# GJ876 is 15.2 light years away and has 0.3761 solar radius
# GJ551 (proxima cen) is 4.246 light years away and has 0.1542 solar radius
# GJ436 is 31.8 light years away and has 0.42 solar radius
# GJ1214 is 47.5 light years away and has 0.2064 solar radius


hdulist = fits.open(
    "hlsp_muscles_multi_multi_gj1214_broadband_v22_adapt-const-res-sed.fits"
)
print(hdulist.info())
spec = fits.getdata(
    "hlsp_muscles_multi_multi_gj1214_broadband_v22_adapt-const-res-sed.fits", 1
)

# WAVELENGTH : midpoint of the wavelength bin in Angstroms
# WAVELENGTH0: left (blue) edge of the wavelength bin in Angstroms
# WAVELENGTH1: right (red) edge of the wavelength bin in Angstroms
# FLUX : average flux density in the wavelength bin in erg s-1 cm-2 Angstroms-1
# need to convert to ergs/cm**2/s/nm

new_str = "# WL(nm)\t Flux(ergs/cm**2/s/nm)\n"

for n, wl in enumerate(spec["WAVELENGTH"]):
    new_str += (
        "{:<12}".format(wl * 0.1)
        + "{:>12.2E}".format(
            float(
                spec["FLUX"][n] * 10.0 * (47.5 * (63241 * au) / (r_sun * 0.2064)) ** 2
            )
        )
        + "\n"
    )


with open("sflux-GJ1214.txt", "w+") as f:
    f.write(new_str)
