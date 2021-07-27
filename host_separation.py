#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

infile_coords = os.path.join("data", "coordinates.csv")

df = pd.read_csv(infile_coords)

def get_coords(band):
	coords = SkyCoord(df.query(f"band == '{band}'")["RA"].values[0], df.query(f"band == '{band}'")["Dec"].values[0], unit="deg")
	return coords

host_coords = "17 09 06.8532135309 +26 51 20.499469058"
IC200530A_coords = ""

coords_host = SkyCoord(host_coords, unit=(u.hourangle, u.deg))
neutrino_coords = ["255.37", "26.61"]
coords_IC200530A = SkyCoord(neutrino_coords[0], neutrino_coords[1], unit="deg")
coords_g = get_coords("g")
coords_r = get_coords("r")
coords_i = get_coords("i")

g_sep = coords_host.separation(coords_g).arcsec
r_sep = coords_host.separation(coords_r).arcsec
i_sep = coords_host.separation(coords_i).arcsec

g_neutrinosep = coords_IC200530A.separation(coords_g).deg

print(f"Tywin location in ZTF g-band: {coords_g}")

print(f"Median offset from Host (Gaia) in ZTF g-band: {g_sep:.3f} arcsec")
print(f"Median offset from Host (Gaia) in ZTF r-band: {r_sep:.3f} arcsec")
print(f"Median offset from Host (Gaia) in ZTF i-band: {i_sep:.3f} arcsec")
print(f"Median offset from IC200530A (GCN) in ZTF g-band: {g_neutrinosep:.3f} deg")