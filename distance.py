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

cosmo_generic = FlatLambdaCDM(H0=70, Om0=0.3)

# REDSHIFT = 0.2666
# # REDSHIFT = 0.051

# planck18_dist = cosmo_planck.luminosity_distance(REDSHIFT)
# generic_dist = cosmo_generic.luminosity_distance(REDSHIFT)

# print(f"Planck 18 distance: {planck18_dist}")
# print(f"Generic distance: {generic_dist}")

infile = os.path.join("data", "tde_sample.csv")

df = pd.read_csv(infile)


log_bolo_lumis = df["L_bb_log_erg_s"].values
bolo_lumis = [10 ** lumi for lumi in log_bolo_lumis]

distances = []
bolo_fluxes = []

for i, z in enumerate(df["z"].values):
    generic_dist = cosmo_generic.luminosity_distance(z)
    distance_cm = generic_dist.to(u.cm).value
    distances.append(distance_cm)
    bolo_flux = bolo_lumis[i] / (4 * np.pi * distance_cm ** 2)
    bolo_fluxes.append(bolo_flux)

df["bolo_flux"] = bolo_fluxes

df_sorted = df.sort_values(by=["bolo_flux"], ascending=False)

summed_flux = np.sum(df["bolo_flux"].values)
tywin_flux = df.query("IAU == 'AT2019fdr'")["bolo_flux"].values[0]
bran_flux = df.query("IAU == 'AT2019dsg'")["bolo_flux"].values[0]

print(df_sorted)

print("--------------------------------------------------------")
print(f"Summed bolometric flux = {summed_flux:.2e} erg/s/cm**2")
print(f"Bran contributes {(bran_flux/summed_flux)*100:.1f}% of population flux.")
print(f"Tywin contributes {(tywin_flux/summed_flux)*100:.1f}% of population flux.")
