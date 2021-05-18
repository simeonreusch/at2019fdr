#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os
from modelSED.sed import SED

redshift = 0.2666

nbins = 60

fittype = "blackbody"
fitglobal = False
fitlocal = True

bands = [
    "P48+ZTF_g",
    "P48+ZTF_r",
    "P48+ZTF_i",
    "Swift+UVM2",
    "Swift+UVW1",
    "Swift+UVW2",
    # "P200+Ks",
    # "P200+J",
    # "P200+H",
    ### "Swift+UVM1"
]

path_to_lightcurve = os.path.join("data", "lightcurves", "full_lightcurve.csv")

sed = SED(
    redshift=redshift,
    fittype=fittype,
    nbins=nbins,
    path_to_lightcurve=path_to_lightcurve,
)


if fitglobal:
    sed.fit_global(bands=bands, plot=False, min_datapoints=6)
sed.load_global_fitparams()

bands = [
    "P48+ZTF_g",
    "P48+ZTF_r",
    "P48+ZTF_i",
    "Swift+UVM2",
    "Swift+UVW1",
    "Swift+UVW2",
    "P200+Ks",
    "P200+J",
    "P200+H",
    "WISE+W1",
    "WISE+W2"
    ### "Swift+UVM1"
]

if fitlocal:
    if fittype == "powerlaw":
        sed.fit_bins(
            alpha=sed.fitparams_global["alpha"],
            alpha_err=sed.fitparams_global["alpha_err"],
            bands=bands,
            min_bands_per_bin=2,
            verbose=False,
        )
    else:
        sed.fit_bins(
            extinction_av=sed.fitparams_global["extinction_av"],
            extinction_av_err=sed.fitparams_global["extinction_av_err"],
            extinction_rv=sed.fitparams_global["extinction_rv"],
            extinction_rv_err=sed.fitparams_global["extinction_rv_err"],
            bands=bands,
            min_bands_per_bin=2,
            # neccessary_bands=["Swift+UVM2"],
            verbose=False,
            # fit_algorithm="emcee",
        )
sed.load_fitparams()
sed.plot_lightcurve(bands=bands, nufnu=True)
if fittype == "blackbody":
    sed.plot_temperature()
sed.plot_luminosity()
