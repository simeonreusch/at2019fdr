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
SWIFT_OBS_FILE = os.path.join(DATA_DIR, "Swift_newcal.csv")


infile = os.path.join("data", "host_model", "Tywin_parasfh_spectrum_with_WISE.dat")
host_spectrum = pd.read_table(
    infile, names=["wl", "flux", "abmag"], sep="\s+", comment="#"
)
swift_observed = pd.read_csv(SWIFT_OBS_FILE)
host_spectrum["abmag"] = utilities.flux_to_abmag(flux_nu=host_spectrum.flux, flux_nu_zp=0)
swift_observed.drop(columns=["jd"], inplace=True)


# Now we get magnitudes using bandpasses
# First we need to construct a proper spectrum
spectrum = sncosmo_spectral_v13.Spectrum(
    wave=host_spectrum.wl.values,
    flux=host_spectrum.flux.values * 3.631e-20,
    unit=utilities.FNU,
)

abmag_subtracted = []
abmag_err_subtracted = []

for i, x in swift_observed.iterrows():
    band = x["band"]
    bandtelescope = "Swift+" + band
    abmag_host = utilities.magnitude_in_band(band=bandtelescope, spectrum=spectrum)
    abmag_observed = x["abmag"]
    abmag_err_observed = x["abmag_err"]
    fluxmaggie_host = utilities.abmag_to_flux(abmag_host, magzp=0)
    fluxmaggie_err_host = 0.05 * fluxmaggie_host
    fluxmaggie_observed = utilities.abmag_to_flux(abmag_observed, magzp=0)
    fluxmaggie_err_observed = utilities.abmag_err_to_flux_err(
        abmag_observed,
        abmag_err_observed,
        magzp=0,
        magzp_err=0,
        )

    fluxmaggie_diff = fluxmaggie_observed - fluxmaggie_host
    fluxmaggie_err_diff = fluxmaggie_err_observed + fluxmaggie_err_host


    abmag_after_subtraction = utilities.flux_to_abmag(fluxmaggie_diff, 0)
    abmag_err_after_subtraction = utilities.flux_err_to_abmag_err(fluxmaggie_diff, fluxmaggie_err_diff)

    print("---------------------------------------------------------")
    print(f"abmag host: {abmag_host:.2f}")
    print(
        f"abmag before sub: {abmag_observed:.2f} // abmag after sub: {abmag_after_subtraction:.2f}"
    )

    abmag_subtracted.append(abmag_after_subtraction)
    abmag_err_subtracted.append(abmag_err_after_subtraction)


swift_subtracted = swift_observed.drop(columns=["abmag", "abmag_err"])

swift_subtracted["mag"] = abmag_subtracted
swift_subtracted["mag_err"] = abmag_err_subtracted

swift_subtracted.to_csv(os.path.join(DATA_DIR, "swift_subtracted_synthetic.csv"))
