#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import numpy as np 
import astropy
from astropy.cosmology import Planck18 as cosmo_planck
from astropy.cosmology import FlatLambdaCDM

cosmo_generic = FlatLambdaCDM(H0=70, Om0=0.3)

REDSHIFT = 0.2666
# REDSHIFT = 0.0512

planck18_dist = cosmo_planck.luminosity_distance(REDSHIFT)
generic_dist = cosmo_generic.luminosity_distance(REDSHIFT)

print(f"Planck 18 distance: {planck18_dist}")
print(f"Generic distance: {generic_dist}")