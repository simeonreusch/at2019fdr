#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os
import numpy as np
import astropy.units as u
import pandas as pd
from modelSED import utilities
from astropy.time import Time
from astropy.cosmology import FlatLambdaCDM


filter_wl = utilities.load_info_json("filter_wl")

infile = os.path.join("data", "lightcurves", "full_lightcurve.csv")

TELESCOPE = "Swift"
BAND = "UVW1"
# TELESCOPE = "P48"
# BAND = "ZTF_g"

REDSHIFT = 0.267

df = (
    pd.read_csv(infile)
    .query(f"band == '{BAND}' and telescope == '{TELESCOPE}'")
    .reset_index()
    .drop(columns=["Unnamed: 0", "index"])
)

df["flux_density"] = utilities.abmag_to_flux(df.mag)
df["flux_density_err"] = utilities.abmag_err_to_flux_err(df.mag, df.mag_err)

fluxes = []
flux_errs = []
for row in df.iterrows():
    instrband = row[1]["telescope"] + "+" + row[1]["band"]
    flux, flux_err = utilities.flux_density_to_flux(
        filter_wl[instrband], row[1].flux_density, row[1].flux_density_err
    )
    fluxes.append(flux)
    flux_errs.append(flux_err)

df["flux"] = fluxes
df["flux_err"] = flux_errs

i = np.argmax(df.flux.values)

peak_obsmjd = df["obsmjd"].values[i]
peakmag = df["mag"].values[i]
peak_fluxdensity = df["flux_density"].values[i]
peakflux = df["flux"].values[i]
t = Time(peak_obsmjd, format="mjd")
peak_date = t.iso

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
d = cosmo.luminosity_distance(REDSHIFT)
d = d.to(u.cm).value

peaklumi = peakflux * 4 * np.pi * d ** 2


print(f"Telescope = {TELESCOPE} / band = {BAND}")
print(f"peak obsmjd = {peak_obsmjd:.2f}")
print(f"peak date = {peak_date}")
print(f"peak mag = {peakmag:.2f}")
print(f"peak flux density = {peak_fluxdensity:.2e} erg/s/cm**2/Hz")
print(f"peak flux = {peakflux:.2e} erg/s/cm**2")
print(f"peak luminosity = {peaklumi:.2e} erg/s")

# fluxes = []
# flux_errs = []

# for row in df.iterrows():
# 	mag = row[1]["mag"]
# 	mag_err = row[1]["mag_err"]

# 	flux = utilities.abmag_to_flux(mag)
# 	flux_err = utilities.abmag_err_to_flux_err(mag, mag_err)

# 	fluxes.append(flux)
# 	flux_errs.append(flux_err)

# df["flux"] = fluxes
# df["flux_err"] = flux_errs

# i = np.argmax(fluxes)

# nu_hz = 3.988e15

# nu_fnu = fluxes[i] * nu_hz

# print(f"obsmjd = {df.obsmjd.values[i]}")
# print(f"flux = {fluxes[i]}")
# print(f"nu fnu = {nu_fnu}")

# nufnu = utilities.flux_density_to_flux(filter_wl["P48+ZTF_g"], flux, flux_err)

# print(nufnu)

# print
