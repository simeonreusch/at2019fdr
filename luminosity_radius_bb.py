#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, argparse, time, json
import numpy as np
import pandas as pd
import matplotlib
import astropy.units as u
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import json
from extinction import ccm89, apply, remove, calzetti00
from modelSED import utilities, fit, sncosmo_spectral_v13
from modelSED.utilities import broken_powerlaw_spectrum, FNU, flux_nu_to_lambda
from astropy.modeling.models import BlackBody
from astropy.cosmology import FlatLambdaCDM
from lmfit import Model, Parameters, Minimizer, report_fit, minimize
import lmfit
from astropy.cosmology import Planck15 as cosmo

REDSHIFT = 0.2666
FNU = u.erg / (u.cm ** 2 * u.s * u.Hz)
FLAM = u.erg / (u.cm ** 2 * u.s * u.AA)


def luminosity_radius(temp, scale, redshift):
    d = cosmo.luminosity_distance(redshift)
    d = d.to(u.m)

    radius_m = np.sqrt(d ** 2 * scale_nu1.value) / np.sqrt(np.pi)
    radius_cm = np.sqrt(d ** 2 * scale_nu1.value) * (100 * u.cm) / u.m / np.sqrt(np.pi)

    temp = temp * u.K

    luminosity_watt = const.sigma_sb * (temp) ** 4 * 4 * np.pi * (radius_m ** 2)
    luminosity_erg = luminosity_watt.to(u.erg / u.s)

    return luminosity_erg, radius_cm


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# infile = os.path.join("fit", "chisq", "calzetti00", "bb_final.csv")
infile = os.path.join("fit", "chisq", "calzetti00_3.1_0.4502", "bb_final.csv")

df = pd.read_csv(infile).drop(columns=["Unnamed: 0"])

# corrcoeffs_oldavrv_calzetti = {0: {"temp1scale1": 0.965, "temp2scale2": 0.992}, 1: {"temp1scale1": 0.965, "temp2scale2": 0.940}, 2: {"temp1scale1": 0.995, "temp2scale2": 0.98}} #calzetti old

corrcoeffs = {
    0: {"temp1scale1": 0.967, "temp2scale2": 0.987},
    1: {"temp1scale1": 0.972, "temp2scale2": 0.946},
    2: {"temp1scale1": 0.975, "temp2scale2": 0.98},
}  # calzetti, rv=3.1, av=0.4502

for epoch in [0]:
    df_temp = df.query("epoch == @epoch")

    penalty = 1

    if epoch == 0:
        penalty = np.sqrt(4.21)

    temp1 = df_temp["temp1"].values[0]
    temp1_p = df_temp["temp1+"].values[0] * penalty
    temp1_m = df_temp["temp1-"].values[0] * penalty
    temp2 = df_temp["temp2"].values[0]
    temp2_p = df_temp["temp2+"].values[0]  # * penalty
    temp2_m = df_temp["temp2-"].values[0]  # * penalty
    scale1 = df_temp["scale1"].values[0]
    scale1_p = df_temp["scale1+"].values[0] * penalty
    scale1_m = df_temp["scale1-"].values[0] * penalty
    scale2 = df_temp["scale2"].values[0]
    scale2_p = df_temp["scale2+"].values[0] * penalty
    scale2_m = df_temp["scale2-"].values[0] * penalty

    scale_nu1 = 1 / scale1 * FNU / u.sr
    scale_nu2 = 1 / scale2 * FNU / u.sr

    # lumi1, radius1 = luminosity_radius(temp=temp1, scale=scale_nu1, redshift=REDSHIFT)
    # lumi2, radius2 = luminosity_radius(temp=temp2, scale=scale_nu2, redshift=REDSHIFT)

    (
        lumi1,
        lumi1_err_upper,
        radius1,
        radius1_err_upper,
    ) = utilities.calculate_bolometric_luminosity(
        temperature=temp1,
        scale=scale1,
        redshift=REDSHIFT,
        temperature_err=temp1_p,
        scale_err=scale1_p,
        cosmo="generic",
        corrcoeff=corrcoeffs[epoch]["temp1scale1"],
    )

    (
        lumi1,
        lumi1_err_lower,
        radius1,
        radius1_err_lower,
    ) = utilities.calculate_bolometric_luminosity(
        temperature=temp1,
        scale=scale1,
        redshift=REDSHIFT,
        temperature_err=temp1_m,
        scale_err=scale1_m,
        cosmo="generic",
        corrcoeff=corrcoeffs[epoch]["temp1scale1"],
    )

    (
        lumi2,
        lumi2_err_upper,
        radius2,
        radius2_err_upper,
    ) = utilities.calculate_bolometric_luminosity(
        temperature=temp2,
        scale=scale2,
        redshift=REDSHIFT,
        temperature_err=temp2_p,
        scale_err=scale2_p,
        cosmo="generic",
        corrcoeff=corrcoeffs[epoch]["temp1scale1"],
    )

    (
        lumi2,
        lumi2_err_lower,
        radius2,
        radius2_err_lower,
    ) = utilities.calculate_bolometric_luminosity(
        temperature=temp2,
        scale=scale2,
        redshift=REDSHIFT,
        temperature_err=temp2_m,
        scale_err=scale2_m,
        cosmo="generic",
        corrcoeff=corrcoeffs[epoch]["temp2scale2"],
    )

    lumi1_err_upper = lumi1_err_upper * 1.3
    lumi1_err_lower = lumi1_err_lower * 1.3

    lumi2_err_upper = lumi2_err_upper * 1.3
    lumi2_err_lower = lumi2_err_lower * 1.3

    comb_lumi = lumi1 + lumi2
    comb_lumi_err_upper = np.sqrt(lumi1_err_upper ** 2 + lumi2_err_upper ** 2)
    comb_lumi_err_lower = np.sqrt(lumi1_err_lower ** 2 + lumi2_err_lower ** 2)

    print("-----------------------------------------------------------------")
    print(f"EPOCH {epoch}")
    print("-----------------------------------------------------------------")
    print(f"Temp1: {temp1:.0f} + {temp1_p:.0f} - {temp1_m:.0f}")
    print(
        f"Radius 1: {radius1:.2e} + {radius1_err_upper:.2e} - {radius1_err_lower:.2e}"
    )
    print(f"Luminosity 1: {lumi1:.3e} + {lumi1_err_upper:.3e} - {lumi1_err_lower:.3e}")
    print("------")
    print(f"Temp2: {temp2:.0f} + {temp2_p:.0f} - {temp2_m:.0f}")
    print(
        f"Radius 2: {radius2:.2e} + {radius2_err_upper:.2e} - {radius2_err_lower:.2e}"
    )
    print(f"Luminosity 2: {lumi2:.3e} + {lumi2_err_upper:.3e} - {lumi2_err_lower:.3e}")
    print("------")
    # print(f"Combined Luminosity: {comb_lumi:.3e} + {comb_lumi_err_upper:.3e} - {comb_lumi_err_lower:.3e}")
    print("-----------------------------------------------------------------")


# for epoch in
