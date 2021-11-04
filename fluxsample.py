#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os
import numpy as np
import pandas as pd
import astropy
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo_planck
from astropy.cosmology import FlatLambdaCDM

from nuztf.ampel_api import ampel_api_name

infile = os.path.join("data", "fluxsample.csv")
outfile = os.path.join("data", "fluxsample_extended.csv")


# df = pd.read_csv(infile)
# df = df.drop(columns=["Unnamed: 0"])

# peak_g_mags = []


# for i, ztfid in enumerate(df.ztfid.values):
#     query_res = ampel_api_name(ztf_name=ztfid, with_history=True)
#     print(ztfid)
#     g_mags = []

#     if len(query_res) > 0:

#         for j, cand in enumerate(query_res[0]["prv_candidates"]):
#             if "fid" in cand.keys():
#                 if cand["fid"] == 1:
#                     if "magpsf" in cand.keys():
#                         g_mags.append(cand["magpsf"])
#                     else:
#                         g_mags.append(999)
#                 else:
#                     g_mags.append(999)

#         peak_g_mag = np.min(g_mags)
#         peak_g_mags.append(peak_g_mag)

#     else:
#         peak_g_mags.append(99999999)

# df["peak_g_mag"] = peak_g_mags

# df.to_csv(outfile)

df = pd.read_csv(outfile)

# df = df.query("type != 'Nuclear'")

peak_g_fluxes = np.power(10, (-(df["peak_g_mag"].values) / 2.5))

df["peak_g_flux"] = peak_g_fluxes

summed_flux = np.sum(df["peak_g_flux"].values)

bran_flux = df.query("ztfid == 'ZTF19aapreis'")["peak_g_flux"].values[0]

tywin_flux = df.query("ztfid == 'ZTF19aatubsj'")["peak_g_flux"].values[0]


print(
    f"Bran contributes {(bran_flux/summed_flux)*100:.1f}% of population peak g-band flux."
)
print(
    f"Tywin contributes {(tywin_flux/summed_flux)*100:.1f}% of population peak g-band flux."
)
