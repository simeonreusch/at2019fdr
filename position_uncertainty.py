#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from nuztf.ampel_api import ampel_api_name

ztf_name = "ZTF19aatubsj"

query_res = ampel_api_name(ztf_name=ztf_name, with_history=True)

gaia_pos = SkyCoord(
    ra=257.278556517 * u.degree, dec=26.8556949467 * u.degree, frame="icrs"
)
tywin_positions = []
distances = []

for entry in query_res[0]["prv_candidates"]:
    if "ra" in entry.keys():
        pos = SkyCoord(
            ra=entry["ra"] * u.degree, dec=entry["dec"] * u.degree, frame="icrs"
        )
        tywin_positions.append(pos)


for pos in tywin_positions:
    dist = pos.separation(gaia_pos).arcsec
    distances.append(dist)

mean_distance = np.mean(distances)
median_distance = np.median(distances)
std = np.std(distances)

print(f"The mean distance to Gaia is {mean_distance:.3f}")
print(f"The median distance to Gaia is {median_distance:.3f}")
print(f"The standard deviation is {std:.3f}")
