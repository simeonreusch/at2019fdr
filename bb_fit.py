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
from modelSED.utilities import (
    broken_powerlaw_spectrum,
    FNU,
    flux_nu_to_lambda,
    calculate_bolometric_luminosity,
)
from astropy.modeling.models import BlackBody
from astropy.cosmology import FlatLambdaCDM

from lmfit import Model, Parameters, Minimizer, report_fit, minimize
import lmfit
from dust_model import plot_results_brute

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

REFIT = True

filter_wl = utilities.load_info_json("filter_wl")


def get_wavelengths_and_frequencies():
    wavelengths = np.arange(1000, 50000, 100) * u.AA
    frequencies = const.c.value / (wavelengths.value * 1e-10) * u.Hz
    return wavelengths, frequencies


def radius_to_scale(redshift, radius_cm):
    d = cosmo.luminosity_distance(REDSHIFT)
    d_m = d.to(u.m)
    d_m = d_m.value

    radius_m = radius_cm / 100

    s = d_m ** 2 / (np.pi * radius_m ** 2)
    return s


FNU = u.erg / (u.cm ** 2 * u.s * u.Hz)
FLAM = u.erg / (u.cm ** 2 * u.s * u.AA)

MJD_INTERVALS = [[58700, 58720], [59006, 59130], [59220, 59271]]


REDSHIFT = 0.2666
FONTSIZE = 12
FONTSIZE_LABEL = 13
FONTSIZE_LEGEND = 5
ANNOTATION_FONTSIZE = 8
FONTSIZE_TICKMARKS = 9
FIG_WIDTH = 6
DPI = 400


INSTRUMENT_DATA_DIR = "instrument_data"
PLOTDIR = os.path.join("plots", "double_blackbody")
FITDIR = os.path.join("fit", "double_blackbody")


def load_info_json(filename: str):
    with open(os.path.join(INSTRUMENT_DATA_DIR, filename + ".json")) as json_file:
        outfile = json.load(json_file)
    return outfile


def double_blackbody_minimizer(params, x, data=None, data_err=None, **kwargs):

    if "extinction_av" in params:
        extinction_av = params["extinction_av"]
    else:
        extinction_av = GLOBAL_AV

    if "extinction_rv" in params:
        extinction_rv = params["extinction_rv"]
    else:
        extinction_rv = GLOBAL_RV

    wl_filter = {v: k for k, v in filter_wl.items()}

    temp1 = params["temp1"]
    scale1 = params["scale1"]

    temp2 = params["temp2"]
    scale2 = params["scale2"]

    scale1 = 1 / scale1 * FNU / u.sr
    scale2 = 1 / scale2 * FNU / u.sr

    bb_nu_1 = BlackBody(temperature=temp1 * u.K, scale=scale1)
    bb_nu_2 = BlackBody(temperature=temp2 * u.K, scale=scale2)

    wl_generic, _ = get_wavelengths_and_frequencies()

    flux_nu_bb_1 = bb_nu_1(wl_generic) * u.sr
    flux_nu_bb_2 = bb_nu_2(wl_generic) * u.sr

    flux_lambda_bb_1 = flux_nu_to_lambda(flux_nu_bb_1, wl_generic)
    flux_lambda_bb_2 = flux_nu_to_lambda(flux_nu_bb_2, wl_generic)

    flux_lambda_bb_reddened_1 = apply(
        calzetti00(wl_generic, extinction_av, extinction_rv),
        np.asarray(flux_lambda_bb_1),
    )
    flux_lambda_bb_reddened_2 = apply(
        calzetti00(wl_generic, extinction_av, extinction_rv),
        np.asarray(flux_lambda_bb_2),
    )

    flux_nu_bb_reddened_1 = utilities.flux_lambda_to_nu(
        flux_lambda_bb_reddened_1, wl_generic
    )
    flux_nu_bb_reddened_2 = utilities.flux_lambda_to_nu(
        flux_lambda_bb_reddened_2, wl_generic
    )

    spectrum_bb_reddened_1 = sncosmo_spectral_v13.Spectrum(
        wave=wl_generic, flux=flux_nu_bb_reddened_1, unit=FNU
    )
    spectrum_bb_reddened_2 = sncosmo_spectral_v13.Spectrum(
        wave=wl_generic, flux=flux_nu_bb_reddened_2, unit=FNU
    )

    spectrum_bb_reddened_1.z = 0
    spectrum_bb_reddened_2.z = 0

    spectrum_bb_reddened_redshifted_1 = spectrum_bb_reddened_1.redshifted_to(
        REDSHIFT, cosmo=cosmo
    )
    spectrum_bb_reddened_redshifted_2 = spectrum_bb_reddened_2.redshifted_to(
        REDSHIFT, cosmo=cosmo
    )

    combined_flux = (
        spectrum_bb_reddened_redshifted_1.flux + spectrum_bb_reddened_redshifted_2.flux
    )

    combined_spectrum = sncosmo_spectral_v13.Spectrum(
        wave=spectrum_bb_reddened_redshifted_1.wave, flux=combined_flux, unit=FNU
    )

    model_flux = []

    for i, wl in enumerate(x):
        bandname = wl_filter[round(wl.value, 1)]

        abmag = utilities.magnitude_in_band(bandname, combined_spectrum)
        flux = utilities.abmag_to_flux(abmag)

        model_flux.append(flux)

    residual = (model_flux - data) / data_err

    chisq = np.sum(residual ** 2)

    # print(f"chisq: {chisq:.5f}")

    return residual


def main(epoch, s2, t2):
    if epoch == 2:
        H_CORRECTION_I_BAND = 1.32003
    else:
        H_CORRECTION_I_BAND = 1.12495345056821688

    LIGHTCURVE_INFILE = os.path.join("data", "lightcurves", "full_lightcurve.csv")

    df = pd.read_csv(LIGHTCURVE_INFILE).drop(columns=["Unnamed: 0"])
    df = (
        df.query(
            f"obsmjd > {MJD_INTERVALS[epoch][0]} and obsmjd <= {MJD_INTERVALS[epoch][1]}"
        )
        .reset_index()
        .drop(columns=["index", "alert"])
    )
    df = df.query("band != 'B' and band != 'V'")

    # df = df.query("telescope == 'Swift' or (obsmjd > 58709 and obsmjd<58710.233)")

    if epoch == 0:
        # df = df.query("telescope != 'Swift' or (obsmjd > 58712 and obsmjd<58713)")
        # df = df.query("telescope != 'P48' or (obsmjd > 58708 and obsmjd<58711)")
        df = df.query("band != 'ZTF_g' or mag < 18.12")
        df = df.query("band != 'ZTF_r' or mag < 18.009")
        df = df.query("band != 'ZTF_i' or mag < 18.009")

    df = df.query("telescope != 'WISE'")  # " and band != 'J'")
    # df = df.query("band != 'J'")

    flux = []
    flux_err = []
    fnus = []
    fnu_errs = []
    wls = []

    for i, row in df.iterrows():
        telescope = row[1]
        band = row[0]
        mag = row[3]
        magerr = row[4]
        wl = filter_wl[telescope + "+" + band]
        wls.append(wl)
        nu = const.c.to("Angstrom/s").value / (wl)
        f = utilities.abmag_to_flux(mag)

        if band == "ZTF_i":
            f = f / H_CORRECTION_I_BAND

        ferr = utilities.abmag_err_to_flux_err(mag, magerr)
        fnu = f * nu
        fnu_err = ferr * nu
        flux.append(f)
        flux_err.append(ferr)

    df["wl"] = wls
    df["flux"] = flux
    df["flux_err"] = flux_err

    wls = []
    mean_flux = []
    mean_flux_errs = []

    for band in df.band.unique():
        _df = df.query("band == @band")
        wls.append(np.mean(_df.wl.values))
        mean_f = np.mean(_df.flux.values)
        mean_f_err = np.sqrt(np.sum(_df.flux_err.values ** 2)) / np.sqrt(len(_df))
        mean_flux.append(mean_f)
        mean_flux_errs.append(mean_f_err)

    df_mean = pd.DataFrame()
    df_mean["wl"] = wls
    df_mean["flux"] = mean_flux
    df_mean["flux_err"] = mean_flux_errs
    df_mean = df_mean.sort_values("wl", ascending=False)

    params = Parameters()
    params.add("extinction_av", value=0.4502, min=0, max=2, vary=False)
    params.add("extinction_rv", value=3.1, min=2.5, max=4.5, vary=False)
    # epoch1
    params.add("temp1", value=11731, min=7000, max=50000, vary=True)
    # epoch2
    # params.add("temp1", value=10230, min=7000, max=50000, vary=False)
    # epoch1
    params.add("scale1", value=2.2832e23, min=1e20, max=1e27, vary=True)
    # epoch2
    # params.add("scale1", value=3.1101e23, min=1e20, max=1e27, vary=False)
    # params.add("temp2", value=1650, min=500, max=2300, vary=True)
    params.add("temp2", value=t2, min=t2 - 1000, max=t2 + 1000, vary=False)
    params.add("scale2", value=s2, min=1e18, max=8e21, vary=False)

    x = df_mean.wl.values * u.AA
    data = df_mean.flux.values
    data_err = df_mean.flux_err.values

    minimizer_fcn = double_blackbody_minimizer

    minimizer = Minimizer(
        userfcn=minimizer_fcn,
        params=params,
        fcn_args=(x, data, data_err),
        fcn_kws=None,
        calc_covar=True,
    )

    method = "lm"

    if REFIT:
        out = minimizer.minimize(method=method)

        # print("----------")
        # print(f"chisqr: {out.chisqr}")
        # print(f"reduced chisq: {out.redchi}")
        # print("----------")
        # print(report_fit(out.params))

        temp1 = out.params["temp1"].value
        temp2 = out.params["temp2"].value
        scale1 = out.params["scale1"].value
        scale2 = out.params["scale2"].value
        extinction_av = out.params["extinction_av"]
        extinction_rv = out.params["extinction_rv"]

        with open(
            os.path.join("fit", "double_blackbody_3.1", f"epoch{epoch}.json"), "w"
        ) as f:
            out.params.dump(f)

    else:
        p = Parameters()
        with open(
            os.path.join("fit", "double_blackbody_3.1", f"epoch{epoch}.json"), "r"
        ) as f:
            params = p.load(f)
        temp1 = params["temp1"].value
        temp2 = params["temp2"].value
        scale1 = params["scale1"].value
        scale2 = params["scale2"].value
        extinction_av = params["extinction_av"]
        extinction_rv = params["extinction_rv"]

    scale_nu1 = 1 / scale1 * FNU / u.sr
    scale_nu2 = 1 / scale2 * FNU / u.sr

    bb_nu1 = BlackBody(temperature=temp1 * u.K, scale=scale_nu1)
    bb_nu2 = BlackBody(temperature=temp2 * u.K, scale=scale_nu2)

    wavelengths, frequencies = get_wavelengths_and_frequencies()

    flux_nu_bb1 = bb_nu1(wavelengths) * u.sr
    flux_nu_bb2 = bb_nu2(wavelengths) * u.sr

    bb_spec1 = sncosmo_spectral_v13.Spectrum(
        wave=wavelengths, flux=flux_nu_bb1, unit=FNU
    )

    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    d = cosmo.luminosity_distance(REDSHIFT)
    d_m = d.to(u.m)
    d_cm = d.to(u.cm)
    boloflux1 = bb_nu1.bolometric_flux
    boloflux2 = bb_nu2.bolometric_flux
    lumi_from_boloflux1 = boloflux1 * (4 * np.pi * ((d_cm) ** 2))
    lumi_from_boloflux2 = boloflux2 * (4 * np.pi * ((d_cm) ** 2))
    radius_cm1 = (
        np.sqrt(d_m ** 2 * scale_nu1.value) * (100 * u.cm) / u.m / np.sqrt(np.pi)
    )
    radius_cm2 = (
        np.sqrt(d_m ** 2 * scale_nu2.value) * (100 * u.cm) / u.m / np.sqrt(np.pi)
    )

    # print(f"lumi_from_boloflux 1: {lumi_from_boloflux1:.2e}")
    # print(f"lumi_from_boloflux 2: {lumi_from_boloflux2:.2e}")
    # print(f"lumi_from_boloflux comb: {lumi_from_boloflux1+lumi_from_boloflux2:.2e}")
    # print(f"radius 1: {radius_cm1:.4e}")
    # print(f"radius 2: {radius_cm2:.4e}")
    # print("-------------------------")

    # d = cosmo.luminosity_distance(REDSHIFT)
    # d = d.to(u.m)

    # radius1_m = np.sqrt(d ** 2 * scale_nu1.value) / np.sqrt(np.pi)
    # radius1_cm = np.sqrt(d ** 2 * scale_nu1.value) * (100 * u.cm) / u.m / np.sqrt(np.pi)

    # temperature1 = temp1 * u.K

    # luminosity1_watt = const.sigma_sb * (temperature1) ** 4 * 4 * np.pi * (radius1_m ** 2)
    # luminosity1 = luminosity1_watt.to(u.erg / u.s)

    # flux_lambda_bb1 = flux_nu_to_lambda(flux_nu_bb1, wavelengths)
    # flux_lambda_bb2 = flux_nu_to_lambda(flux_nu_bb2, wavelengths)

    # flux_lambda_bb_reddened1 = apply(
    #     calzetti00(np.asarray(wavelengths), extinction_av, extinction_rv),
    #     np.asarray(flux_lambda_bb1),
    # )
    # flux_lambda_bb_reddened2 = apply(
    #     calzetti00(np.asarray(wavelengths), extinction_av, extinction_rv),
    #     np.asarray(flux_lambda_bb2),
    # )

    # flux_nu_bb_reddened1 = utilities.flux_lambda_to_nu(
    #     flux_lambda_bb_reddened1, wavelengths
    # )
    # flux_nu_bb_reddened2 = utilities.flux_lambda_to_nu(
    #     flux_lambda_bb_reddened2, wavelengths
    # )

    # spectrum_bb_reddened1 = sncosmo_spectral_v13.Spectrum(
    #     wave=wavelengths, flux=flux_nu_bb_reddened1, unit=FNU
    # )
    # spectrum_bb_reddened2 = sncosmo_spectral_v13.Spectrum(
    #     wave=wavelengths, flux=flux_nu_bb_reddened2, unit=FNU
    # )

    # spectrum_bb_reddened1.z = 0
    # spectrum_bb_reddened2.z = 0

    # spectrum_bb_reddened_redshifted1 = spectrum_bb_reddened1.redshifted_to(
    #     REDSHIFT, cosmo=cosmo
    # )
    # spectrum_bb_reddened_redshifted2 = spectrum_bb_reddened2.redshifted_to(
    #     REDSHIFT, cosmo=cosmo
    # )

    # combined = spectrum_bb_reddened_redshifted1.flux + spectrum_bb_reddened_redshifted2.flux
    # combined_abmag = utilities.flux_to_abmag(combined)

    # combined_spectrum = sncosmo_spectral_v13.Spectrum(
    #     wave=spectrum_bb_reddened_redshifted1.wave, flux=combined, unit=FNU
    # )

    # wl_filter = {v: k for k, v in filter_wl.items()}

    # model_flux = []

    # for i, wl in enumerate(df_mean["wl"]):
    #     bandname = wl_filter[round(wl, 1)]

    #     abmag = utilities.magnitude_in_band(bandname, combined_spectrum)
    #     model_f = utilities.abmag_to_flux(abmag)
    #     model_flux.append(model_f)

    # plt.figure(figsize=(FIG_WIDTH, 1 / 1.414 * FIG_WIDTH), dpi=DPI)
    # ax1 = plt.subplot(111)
    # ax1.set_ylim([1e-28, 1e-26])
    # ax1.set_xlim([50000, 1000])
    # ax1.plot(
    #     spectrum_bb_reddened_redshifted1.wave,
    #     spectrum_bb_reddened_redshifted1.flux,
    #     label="1 redd redsh",
    # )
    # ax1.plot(
    #     spectrum_bb_reddened_redshifted2.wave,
    #     spectrum_bb_reddened_redshifted2.flux,
    #     label="2 redd redsh",
    # )
    # ax1.plot(spectrum_bb_reddened_redshifted2.wave, combined, label="comb")
    # ax1.set_yscale("log")
    # ax1.set_xscale("log")
    # plt.legend()
    # ax1.errorbar(x=df_mean["wl"], y=df_mean["flux"], yerr=df_mean["flux_err"], fmt=".")
    # ax1.scatter(df_mean["wl"], model_flux, color="black")
    # plt.savefig(f"test_{epoch}.png")

    # plt.figure(figsize=(FIG_WIDTH, 1 / 1.414 * FIG_WIDTH), dpi=DPI)
    # ax1 = plt.subplot(111)
    # # ax1.set_ylim([1e-28, 1e-26])
    # # ax1.set_xlim([50000, 1000])
    # nu1 = const.c.to("Angstrom/s").value / (spectrum_bb_reddened_redshifted1.wave)
    # ax1.plot(
    #     spectrum_bb_reddened_redshifted1.wave,
    #     spectrum_bb_reddened_redshifted1.flux * nu1,
    #     label="1 redd redsh",
    # )

    # nu2 = const.c.to("Angstrom/s").value / (spectrum_bb_reddened_redshifted2.wave)

    # ax1.plot(
    #     spectrum_bb_reddened_redshifted2.wave,
    #     spectrum_bb_reddened_redshifted2.flux * nu2,
    #     label="2 redd redsh",
    # )
    # ax1.plot(spectrum_bb_reddened_redshifted2.wave, combined * nu2, label="comb")

    # ax1.set_yscale("log")
    # ax1.set_xscale("log")
    # ax1.set_ylim([1e-13, 1e-11])
    # ax1.set_xlim([50000, 1000])
    # plt.legend()

    # nu3 = const.c.to("Angstrom/s").value / (df_mean["wl"].values)
    # ax1.errorbar(
    #     x=df_mean["wl"], y=df_mean["flux"] * nu3, yerr=df_mean["flux_err"] * nu3, fmt="."
    # )

    # # ax1.scatter(df_mean["wl"], model_flux, color="black")
    # plt.savefig(f"test_{epoch}_nufnu.png")

    return out.chisqr, lumi_from_boloflux2.value


# # CODE FOR EPOCH 1


# t_vals = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1762, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
# r_vals = [2.4854e17, 1.2e17, 1.5e17, 1.7e17, 1.9e17, 2.1e17, 2.3e17, 2.5e17, 2.7e17, 2.9e17, 3.1e17, 3.3e17, 3.5e17, 3.7e17, 3.9e17, 4.1e17, 4.3e17]

# t_vals = np.linspace(700, 3000, num=15)
# r_vals = np.linspace(1e17, 8e17, num=15)
# total_length = len(t_vals) * len(r_vals)

# for t in t_vals:
#     for r in r_vals:
#         s = radius_to_scale(redshift=REDSHIFT, radius_cm=r)
#         chisq, lumi = main(epoch=1, s2=s, t2=t)
#         temp = {i: {"temp": t, "radius": r, "chisq": chisq, "lumi": lumi}}
#         results.update(temp)
#         i += 1
#         print(f"{i} of {total_length}")

# df = pd.DataFrame.from_dict(results, orient="index")
# df.to_csv("epoch1free_combvals.csv")

# # print(df)

# df = pd.read_csv("epoch1free_combvals.csv").drop(columns="Unnamed: 0")
# df = df.sort_values(by=["temp", "radius"])

# x = df.temp.unique()
# y = df.radius.unique()
# z = np.empty([len(x), len(y)])


# for i, t in enumerate(x):
#     df_temp = df.query("temp == @t")
#     ch = df_temp.chisq.values
#     for j, c in enumerate(ch):
#         z[i, j] = c


# plt.figure(figsize=(6, 1/ 1.618 * 6), dpi=DPI)
# ax1 = plt.subplot(111)
# ax1.set_xlabel("Radius (cm)")
# ax1.set_ylabel("Temperature (K)")
# levels=[15, 20, 25, 30, 35, 40, 50, 100, 200, 1000, 5000, 10000, 100000]
# ax1.contourf(y, x, z, levels=levels, norm=matplotlib.colors.LogNorm())
# plt.savefig("test_epoch1free.png")


## CODE FOR EPOCH 2

epoch = 1

results = {}
i = 0

# t = 1505
# t = 1560
# r = 2.2e17
# s = radius_to_scale(redshift=REDSHIFT, radius_cm=r)

# chisq, lumi = main(epoch=epoch, s2=s, t2=t)
# quit()

# min_chisq = [1.394639, 19.738749, 0.322777]
# min_chisq = [1.394639, 3.2731, 0.322777] # no WISE


t_vals = np.linspace(700, 3000, num=40)
r_vals = np.linspace(1e17, 8e17, num=40)
total_length = len(t_vals) * len(r_vals)


# for t in t_vals:
#     for r in r_vals:
#         s = radius_to_scale(redshift=REDSHIFT, radius_cm=r)
#         chisq, lumi = main(epoch=epoch, s2=s, t2=t)
#         temp = {i: {"temp": t, "radius": r, "chisq": chisq, "lumi": lumi}}
#         results.update(temp)
#         i += 1
#         print(f"{i} of {total_length}")

# df = pd.DataFrame.from_dict(results, orient="index")
# df.to_csv(f"epoch{epoch}_nowise_free_combvals.csv")
# # df.to_csv(f"epoch{epoch}_noJ_free_combvals.csv")
# #print(df)

# quit()


plt.figure(figsize=(4.5, 4), dpi=DPI)
ax1 = plt.subplot(111)
fontsize = 14

plt.xscale("log")
ax1.set_xlabel("Radius (cm)", fontsize=fontsize)
ax1.set_ylabel("Temperature (K)", fontsize=fontsize)

ax1.tick_params(axis="x", labelsize=fontsize - 3, which="both")
ax1.tick_params(axis="y", labelsize=fontsize - 3)

labels = ["epoch 1", "epoch 2", "epoch 3"]

delta_chisq = 4.605
clrs = ["deeppink", "red", "darkred"]


for epoch in [0, 1, 2]:

    if epoch == 1:
        df = pd.read_csv(f"epoch1_nowise_free_combvals.csv").drop(columns="Unnamed: 0")
        # df = pd.read_csv(f"epoch1free_combvals.csv").drop(columns="Unnamed: 0")
    else:
        df = pd.read_csv(f"epoch{epoch}free_combvals.csv").drop(columns="Unnamed: 0")

    # df = pd.read_csv(f"epoch{epoch}free_combvals.csv").drop(columns="Unnamed: 0")

    minchisq = min(df.chisq.values)
    # print(f"EPOCH = {epoch}")
    # print(minchisq)
    # print(df.query("chisq == @minchisq"))

    df = df.sort_values(by=["temp", "radius"])

    x = df.temp.unique()
    y = df.radius.unique()
    z = np.empty([len(x), len(y)])

    for i, t in enumerate(x):
        df_temp = df.query("temp == @t")
        ch = df_temp.chisq.values
        for j, c in enumerate(ch):
            z[i, j] = c

    levels = [minchisq, minchisq + delta_chisq]

    ax1.contour(
        y,
        x,
        z,
        norm=matplotlib.colors.LogNorm(),
        levels=levels,
        colors=("blue", clrs[epoch]),
    )
    ax1.plot([3e17], [2000], label=labels[epoch], color=clrs[epoch])

    ax1.axhline(1850, color="black", ls="dotted", alpha=0.2)

    ax1.text(
        4e17,
        1700,
        "1850 K",
        # fontsize=BIG_FONTSIZE - 2,
        color="black",
        alpha=0.2,
        fontsize=fontsize,
    )
ax1.set_xlim((1e17, 6e17))

plt.legend(fontsize=fontsize)
plt.tight_layout()
# outfile = os.path.join("plots", "bb_contourplots", "contour_all.pdf")
# outfile = os.path.join("plots", "bb_contourplots", "contour_all_no_WISE_no_J.pdf")
# outfile = os.path.join("plots", "bb_contourplots", "contour_all_no_J.pdf")
outfile = os.path.join("plots", "bb_contourplots", "contour_all_no_WISE.pdf")


plt.savefig(outfile)
# plt.savefig(f"test_all_free_no_WISE_J_inepoch1.pdf")
