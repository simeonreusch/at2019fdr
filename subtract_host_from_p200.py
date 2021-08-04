#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, time
import astropy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.utils.console import ProgressBar
#from astropy.time import Time
#from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.io import fits
from astropy.io.fits import getdata
from astropy.utils.console import ProgressBar
from matplotlib.colors import LogNorm, Normalize
import subprocess
from os import listdir
from os.path import isfile, join
from pathlib import Path
from modelSED import utilities, sncosmo_spectral_v13


HOST_MODEL_DIR = os.path.join("/", "Users", "simeon", "tywin", "data", "host_model")
LIGHTCURVE_DIR = os.path.join("/", "Users", "simeon", "tywin", "data", "lightcurves")
infile_from_epoch_1 = os.path.join(LIGHTCURVE_DIR, "galfit_result_twocomp_fromepoch1.csv")
infile_from_epoch_4 = os.path.join(LIGHTCURVE_DIR, "galfit_result_twocomp_fromepoch4.csv")

df = pd.DataFrame()
df_1 = pd.read_csv(infile_from_epoch_1)
df_4 = pd.read_csv(infile_from_epoch_4)
df["obsmjd"] = df_1["obsmjd"]

for band in ["H", "J", "Ks"]:
    for obj in ["host", "psf"]:
        df_1[f"{band}_abmag_{obj}"] = utilities.p200_vega_to_ab(vegamag=df_1[f"{band}_vegamag_{obj}"].values, band=f"P200+{band}")
        df_4[f"{band}_abmag_{obj}"] = utilities.p200_vega_to_ab(vegamag=df_4[f"{band}_vegamag_{obj}"].values, band=f"P200+{band}")
        df_1[f"{band}_flux_{obj}"] = utilities.abmag_to_flux(df_1[f"{band}_abmag_{obj}"].values, magzp=0)
        df_4[f"{band}_flux_{obj}"] = utilities.abmag_to_flux(df_4[f"{band}_abmag_{obj}"].values, magzp=0)
    df_1[f"{band}_flux_comb"] = df_1[f"{band}_flux_host"] + df_1[f"{band}_flux_psf"]
    df_4[f"{band}_flux_comb"] = df_4[f"{band}_flux_host"] + df_1[f"{band}_flux_psf"]
    df_1[f"{band}_abmag_comb"] = utilities.flux_to_abmag(df_1[f"{band}_flux_comb"].values, flux_nu_zp=0)
    df_4[f"{band}_abmag_comb"] = utilities.flux_to_abmag(df_4[f"{band}_flux_comb"].values, flux_nu_zp=0)

    df[f"{band}_abmag_1"] = df_1[f"{band}_abmag_comb"]
    df[f"{band}_abmag_4"] = df_4[f"{band}_abmag_comb"]
    df[f"{band}_flux_1"] = df_1[f"{band}_flux_comb"]
    df[f"{band}_flux_4"] = df_4[f"{band}_flux_comb"]

    col1 = df.loc[: , f"{band}_abmag_1":f"{band}_abmag_4"]
    col2 = df.loc[: , f"{band}_flux_1":f"{band}_flux_4"]
    df[f"{band}_abmag_mean"] = col1.mean(axis=1)
    df[f"{band}_flux_mean"] = col2.mean(axis=1)
    df[f"{band}_abmag_err"] = np.abs(df[f"{band}_abmag_1"] - df[f"{band}_abmag_4"])
    df[f"{band}_flux_err"] = utilities.abmag_err_to_flux_err(
            df[f"{band}_abmag_mean"], df[f"{band}_abmag_err"], magzp=0, magzp_err=0
        )

    infile_host_spectrum = os.path.join(HOST_MODEL_DIR, "Tywin_parasfh_wduste_spectrum_NEW.dat")

    host_spectrum = pd.read_table(
        infile_host_spectrum,
        names=["wl", "flux", "abmag"],
        sep="\s+",
        comment="#",
    )

    host_spectrum["mag"] = utilities.flux_to_abmag(
        flux_nu=host_spectrum.flux,
        flux_nu_zp=0,
    )

    spectrum = sncosmo_spectral_v13.Spectrum(
        wave=host_spectrum.wl.values,
        flux=host_spectrum.flux.values * 3.631e-20,
        unit=utilities.FNU,
    )
    abmag_host_synthetic = utilities.magnitude_in_band(
            band=f"P200+{band}",
            spectrum=spectrum
    )
    flux_host_synthetic = utilities.abmag_to_flux(
        abmag_host_synthetic,
        magzp=0,
    )

    df[f"{band}_abmag_synthetic"] = abmag_host_synthetic
    df[f"{band}_flux_synthetic"] = flux_host_synthetic
    df[f"{band}_flux_synthetic_err"] = flux_host_synthetic * 0.05

    df[f"{band}_flux_transient_err"] = df[f"{band}_flux_err"] + df[f"{band}_flux_synthetic_err"]

    df[f"{band}_flux_transient"] = df[f"{band}_flux_mean"] - flux_host_synthetic
    df[f"{band}_abmag_transient"] = utilities.flux_to_abmag(
        flux_nu=df[f"{band}_flux_transient"],
        flux_nu_zp=0,
    )

    df[f"{band}_abmag_err_transient"] = utilities.flux_err_to_abmag_err(
        df[f"{band}_flux_transient"],
        df[f"{band}_flux_transient_err"]
    )


abmags = []
abmag_errs = []
bands = []
obsmjds = []

for band in ["J", "H", "Ks"]:
    for i, obsmjd in enumerate(df.obsmjd.values):
        abmag = df.query(f"obsmjd == {obsmjd}")[f"{band}_abmag_transient"].values[0]
        abmag_err = df.query(f"obsmjd == {obsmjd}")[f"{band}_abmag_err_transient"].values[0]
        abmags.append(abmag)
        abmag_errs.append(abmag_err)
        bands.append(band)
        obsmjds.append(obsmjd)


output_df = pd.DataFrame(columns=["band", "telescope", "obsmjd", "mag", "mag_err", "alert"])

output_df["band"] = bands
output_df["telescope"] = "P200"
output_df["obsmjd"] = obsmjds
output_df["mag"] = abmags
output_df["mag_err"] = abmag_errs
output_df["alert"] = True

# lightcurve_infile = os.path.join(LIGHTCURVE_DIR, "full_lightcurve.csv")
# lc = pd.read_csv(lightcurve_infile)
# lc.drop(columns=["Unnamed: 0.1"], inplace=True)

# lc_comb = pd.concat([output_df, lc]).reset_index(drop=True)
# lc_comb.drop(columns=["Unnamed: 0"], inplace=True)
outfile = os.path.join(LIGHTCURVE_DIR, "p200_subtracted_synthetic.csv")

output_df = output_df.dropna()
output_df.to_csv(outfile)




