#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo

from modelSED import utilities, sncosmo_spectral_v13

XRTCOLUMN = "flux0310_pi_-2"
nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)
filter_wl = utilities.load_info_json("filter_wl")
cmap = utilities.load_info_json("cmap")
filterlabel = utilities.load_info_json("filterlabel")

REDSHIFT = 0.267
FONTSIZE = 14
FONTSIZE_LEGEND = 14
ANNOTATION_FONTSIZE = 14
FONTSIZE_TICKMARKS = 12
DPI = 400
PLOTDIR = "plots"
DATA_DIR = os.path.join("data", f"lightcurves")
WISE_OBS_FILE = os.path.join(DATA_DIR, "wise.csv")
WISE_OBS_W1_FILE = os.path.join(DATA_DIR, "Tywin-W1-pkfit-lc.txt")
WISE_OBS_W2_FILE = os.path.join(DATA_DIR, "Tywin-W2-pkfit-lc.txt")


df_w1 = pd.read_csv(WISE_OBS_W1_FILE, sep="\t")
df_w2 = pd.read_csv(WISE_OBS_W2_FILE, sep="\t")
df_w1["telescope"] = "WISE"
df_w2["telescope"] = "WISE"
df_w1["band"] = "W1"
df_w2["band"] = "W2"
df_wise = pd.concat([df_w1, df_w2], ignore_index=True, sort=False)

keys = df_wise.keys()
newkeys = {key: key.strip(" ") for key in keys}
df_wise = df_wise.rename(columns=newkeys)
wise_prepeak = df_wise.query("mjd <= 58600")

wise_mean_absmags = {}
for band in wise_prepeak.band.unique():
    df_temp = wise_prepeak.query(f"band == '{band}'")
    wise_mean_absmags.update({band: df_temp.mag_vega.median()})

abmags_transient = []
abmag_errs_transient = []

wise_transient = df_wise.query("mjd > 58600")

for i, x in wise_transient.iterrows():
    band = x["band"]
    vegamag_observed = x["mag_vega"]
    vegamag_err_observed = x["e_mag"]

    abmag_observed = utilities.wise_vega_to_ab(vegamag_observed, band)
    fluxerr_obs = utilities.abmag_err_to_flux_err(
        abmag_observed, vegamag_err_observed, magzp=None, magzp_err=None
    )

    vegamag_host = wise_mean_absmags[band]
    abmag_host = utilities.wise_vega_to_ab(vegamag_host, band)

    flux_host = utilities.abmag_to_flux(abmag_host)
    flux_observed = utilities.abmag_to_flux(abmag_observed)
    flux_diff = flux_observed - flux_host

    abmag_transient = utilities.flux_to_abmag(flux_diff)
    abmag_err_transient = utilities.flux_err_to_abmag_err(flux_diff, fluxerr_obs)

    abmags_transient.append(abmag_transient)
    abmag_errs_transient.append(abmag_err_transient)

wise_transient.drop(
    columns=["mag_vega", "e_mag", "diff_flux_mJy", "e_diff_flux_mJy", "year"],
    inplace=True,
)
wise_transient.rename(columns={"mjd": "obsmjd"}, inplace=True)

wise_transient["mag"] = abmags_transient
wise_transient["mag_err"] = abmag_errs_transient
wise_transient["alert"] = True

outfile = os.path.join(DATA_DIR, "wise_subtracted.csv")
wise_transient.to_csv(outfile)
