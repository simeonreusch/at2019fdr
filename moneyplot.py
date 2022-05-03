#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, time, argparse, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import constants as const
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from modelSED import utilities, sncosmo_spectral_v13
from modelSED.utilities import FNU
from matplotlib.patches import ConnectionPatch, Polygon
import matplotlib
import matplotlib.colors as mcolors
from lmfit import Parameters

pd.options.mode.chained_assignment = None

flabel_sel = "filterlabel"

cmap = utilities.load_info_json("cmap")
filterlabel = utilities.load_info_json(flabel_sel)
filter_wl = utilities.load_info_json("filter_wl")
wl_filter = {v: k for k, v in filter_wl.items()}

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Times New Roman"

MJD_INTERVALS = [[58700, 58720], [59006, 59130], [59220, 59271]]
markers = {"WISE": "p", "P200": "s", "P48": ".", "Swift": "D"}

markers_bandwise = {
    "WISE+W1": "p",
    "WISE+W2": "h",
    "P200+Ks": "s",
    "P200+H": "d",
    "P200+J": "d",
    "P48+ZTF_i": "s",
    "P48+ZTF_r": "^",
    "P48+ZTF_g": "o",
    "Swift+U": "o",
    # "Swift+UVW1": "$\u2734$",
    # "Swift+UVW1": "$\u2665$",#heart
    "Swift+UVW1": "$\u2665$",
    "Swift+UVM2": "D",
    "Swift+UVW2": "v",
}

markersizes_bandwise = {
    "WISE+W1": 6,
    "WISE+W2": 6,
    "P200+Ks": 4,
    "P200+H": 5,
    "P200+J": 7,
    "P48+ZTF_i": 5,
    "P48+ZTF_r": 5,
    "P48+ZTF_g": 5,
    "Swift+U": 5,
    "Swift+UVW1": 7,
    "Swift+UVM2": 5,
    "Swift+UVW2": 5,
}

markerfill_bandwise = {
    "WISE+W1": None,
    "WISE+W2": None,
    "P200+Ks": None,
    "P200+H": None,
    "P200+J": "none",
    "P48+ZTF_i": "none",
    "P48+ZTF_r": "none",
    "P48+ZTF_g": None,
    "Swift+U": "none",
    "Swift+UVW1": "none",
    "Swift+UVM2": "none",
    "Swift+UVW2": "none",
}


def nu_to_ev(nu):
    """ """
    energy = const.h.value * const.c.value / (utilities.nu_to_lambda(nu) * 1e-10)
    ev = energy / 1.602e-19
    return ev


def ev_to_nu(ev):
    """ """
    lam = const.h.value * const.c.value / (ev * 1e-10)
    nu = utilities.lambda_to_nu(lam)
    return nu


def convert_mJy_to_abmag(df):
    """ """
    fnu = df["fnu_mJy"] / 1000 * 1e-23
    fnu_err = df["fnu_mJy_err"] / 1000 * 1e-23
    df["mag"] = utilities.flux_to_abmag(fnu)
    print(df.mag)
    df["mag_err"] = utilities.flux_err_to_abmag_err(fnu, fnu_err)
    df.drop(columns=["fnu_mJy", "fnu_mJy_err"], inplace=True)
    return df


def create_sed(ax, epoch):

    mjd_interval = MJD_INTERVALS[epoch]

    df_cut = df.query(
        f"obsmjd > {mjd_interval[0]} and obsmjd < {mjd_interval[1]} and telescope != 'P200_sextractor'"
    )
    df_cut["telescopeband"] = df_cut["telescope"] + "+" + df["band"]

    p = Parameters()
    with open(
        os.path.join("fit", "double_blackbody_3.1", f"epoch{epoch}.json"), "r"
    ) as f:
        params = p.load(f)

    fitted_spectrum_1, bolo_flux_1 = utilities.blackbody_spectrum(
        temperature=params["temp1"],
        scale=params["scale1"],
        extinction_av=GLOBAL_AV,
        extinction_rv=GLOBAL_RV,
        redshift=REDSHIFT,
        get_bolometric_flux=True,
    )

    fitted_spectrum_2, bolo_flux_2 = utilities.blackbody_spectrum(
        temperature=params["temp2"],
        scale=params["scale2"],
        extinction_av=GLOBAL_AV,
        extinction_rv=GLOBAL_RV,
        redshift=REDSHIFT,
        get_bolometric_flux=True,
    )

    combined_flux = fitted_spectrum_1.flux + fitted_spectrum_2.flux

    # # # Calculate luminosity
    luminosity_1, radius1 = utilities.calculate_bolometric_luminosity(
        temperature=params["temp1"],
        scale=params["scale1"],
        redshift=REDSHIFT,
        temperature_err=None,
        scale_err=None,
    )
    luminosity_2, radius2 = utilities.calculate_bolometric_luminosity(
        temperature=params["temp2"],
        scale=params["scale2"],
        redshift=REDSHIFT,
        temperature_err=None,
        scale_err=None,
    )
    total_luminosity = luminosity_1 + luminosity_2
    print(f"{luminosity_1:.2e}")
    print(f"{luminosity_2:.2e}")
    print(f"{total_luminosity:.2e}")

    combined_spectrum = sncosmo_spectral_v13.Spectrum(
        wave=fitted_spectrum_1.wave, flux=combined_flux, unit=FNU
    )

    ax.text(
        0.58e14,
        1.67e-12,
        # rf"L = {total_luminosity.value:.1e} erg/s",
        "$L_{\mathrm{O+UV}}$ = " + f"{luminosity_1.value:.1e} erg s$^{{-1}}$",
        bbox=bbox,
        fontsize=SMALL_FONTSIZE,
    )

    ax.plot(
        utilities.lambda_to_nu(fitted_spectrum_1.wave),
        fitted_spectrum_1.flux * utilities.lambda_to_nu(fitted_spectrum_1.wave),
        color="tab:blue",
        linestyle="dotted",
        lw=1,
    )

    ax.plot(
        utilities.lambda_to_nu(fitted_spectrum_2.wave),
        fitted_spectrum_2.flux * utilities.lambda_to_nu(fitted_spectrum_2.wave),
        color="tab:red",
        linestyle="dashed",
        linewidth=1,
    )

    ax.plot(
        utilities.lambda_to_nu(combined_spectrum.wave),
        combined_spectrum.flux * utilities.lambda_to_nu(combined_spectrum.wave),
        color="black",
        lw=1,
    )

    telescopebands = [
        "WISE+W2",
        "WISE+W1",
        "P200+Ks",
        "P200+H",
        "P200+J",
        "P48+ZTF_i",
        "P48+ZTF_r",
        "P48+ZTF_g",
        "Swift+U",
        "Swift+UVW1",
        "Swift+UVM2",
        "Swift+UVW2",
    ]

    for telescopeband in telescopebands:

        df_red = df_cut.query(f"telescopeband == '{telescopeband}'")
        mag = np.mean(df_red.mag.values)
        mag_err = np.mean(df_red.mag_err.values)
        telescope = telescopeband.split("+")[0]
        band = telescopeband.split("+")[1]
        flux = utilities.abmag_to_flux(mag)
        flux_err = utilities.abmag_err_to_flux_err(mag, mag_err)

        if telescopeband == "P48+ZTF_i":
            flux = flux / H_CORRECTION_I_BAND

        nu = utilities.lambda_to_nu(filter_wl[telescopeband])

        markersizes = {"WISE": 5 + 1, "P200": 4 + 1, "P48": 8 + 3, "Swift": 4 + 1}

        ax.errorbar(
            x=nu,
            xerr=None,
            y=flux * nu,
            yerr=flux_err * nu,
            color=cmap[telescopeband],
            mec=cmap[telescopeband],
            mfc=cmap_rgba[telescopeband],
            label=filterlabel[telescopeband],
            marker=markers_bandwise[telescopeband],
            ms=markersizes_bandwise[telescopeband],
            linestyle=" ",
            elinewidth=1,
        )

    ax.errorbar(
        1e14,
        0,
        1e-14,
        label="SRG/eROSITA",
        fmt="D",
        markersize=markersizes["P200"],
        color="darkcyan",
    )

    df_erosita_ulims = pd.read_csv(os.path.join(LC_DIR, "erosita_ulims.csv"))
    y = df_erosita_ulims.flux
    yerr = y / 5

    lc_ax1.errorbar(
        x=df_erosita_ulims.obsmjd,
        xerr=df_erosita_ulims.obsmjd - df_erosita_ulims.obsmjd_start,
        y=df_erosita_ulims.flux,
        yerr=yerr,
        uplims=True,
        fmt="D",
        ms=3,
        color="darkcyan",
    )

    if flabel_sel == "filterlabel":
        # legendpos = (2.634, 1.58)
        legendpos = (2.5, 1.58)
        fontsize = 8.6
    else:
        legendpos = (2.5, 1.58)
        fontsize = 8

    if flabel_sel == "filterlabel":
        ncol = 7
    else:
        ncol = 6

    if epoch == 1:
        ax.legend(
            ncol=ncol,
            bbox_to_anchor=legendpos,
            fancybox=True,
            shadow=False,
            fontsize=fontsize,
            edgecolor="gray",
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=BIG_FONTSIZE - 2)

    lumi = lambda flux: flux * 4 * np.pi * d ** 2
    flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
    ax2 = ax.secondary_yaxis("right", functions=(lumi, flux))

    if epoch != 2:
        ax2.set_ticks([])

    ax.grid(which="both", alpha=0.15)


if __name__ == "__main__":

    REDSHIFT = 0.267
    FIG_WIDTH = 8
    BIG_FONTSIZE = 14
    SMALL_FONTSIZE = 8
    DPI = 400
    GOLDEN_RATIO = 1 / 1.618
    GLOBAL_AV = 0.4502
    GLOBAL_RV = 3.1

    H_CORRECTION_I_BAND = 1.0495345056821688

    CURRENT_FILE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "data"))
    PLOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "plots"))
    SPECTRA_DIR = os.path.join(DATA_DIR, "spectra")
    LC_DIR = os.path.join(DATA_DIR, "lightcurves")
    FITDIR = os.path.join("fit", "double_blackbody")
    DUSTDIR = os.path.join("fit", "dust_model")

    infile_lightcurve = os.path.join(LC_DIR, "full_lightcurve.csv")
    infile_dustmodel = os.path.join(DUSTDIR, "dust_model.json")

    df = pd.read_csv(infile_lightcurve)
    df["flux_density"] = utilities.abmag_to_flux(df.mag)
    df["flux_density_err"] = utilities.abmag_err_to_flux_err(df.mag, df.mag_err)

    cmap_rgba = {}

    for entry in cmap:
        rgba = {entry: mcolors.to_rgba(cmap[entry])}
        cmap_rgba.update(rgba)

    for entry in markerfill_bandwise:
        if markerfill_bandwise[entry] == "none":
            temp = list(cmap_rgba[entry])
            temp[-1] = 0.4
            cmap_rgba.update({entry: tuple(temp)})

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

    df_ztf_g = df.query("telescope == 'P48' and band == 'ZTF_g'")
    df_wise_w1 = df.query("telescope == 'WISE' and band == 'W1'")
    df_wise_w2 = df.query("telescope == 'WISE' and band == 'W2'")
    df_p200 = df.query("telescope == 'P200' and band == 'Ks'")

    with open(infile_dustmodel) as f:
        dustmodel_dict = json.load(f)

    fig = plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH * GOLDEN_RATIO))

    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.81, right=0.9)

    lc_ax1 = fig.add_subplot(2, 3, (4, 6))
    sed1 = fig.add_subplot(2, 3, 1)
    sed2 = fig.add_subplot(2, 3, 2)
    sed3 = fig.add_subplot(2, 3, 3)

    sed_xlims = [5e13, 2e15]
    # sed_ylims = [7e-14, 3e-12]
    sed_ylims = [9e-14, 2.3e-12]

    def set_scales(ax):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(sed_xlims)
        ax.set_ylim(sed_ylims)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    for sed in [sed1, sed2, sed3]:
        set_scales(sed)

    sed2.axes.yaxis.set_ticks([])
    sed3.axes.yaxis.set_ticks([])

    lc_ax1.set_yscale("log")
    lc_ylim = [1.5e-14, 2e-12]
    lc_ax1.set_ylim(lc_ylim)

    lc_ax1.errorbar(
        x=df_ztf_g.obsmjd,
        y=df_ztf_g.flux,
        yerr=df_ztf_g.flux_err,
        color=cmap["P48+ZTF_g"],
        marker=".",
        linestyle=" ",
        elinewidth=1,
        label=filterlabel["P48+ZTF_g"],
    )

    lc_ax1.errorbar(
        x=df_wise_w1.obsmjd,
        y=df_wise_w1.flux,
        yerr=df_wise_w1.flux_err,
        color=cmap["WISE+W1"],
        marker=markers["WISE"],
        markersize=7,
        linestyle=" ",
        label=filterlabel["WISE+W1"],
    )

    if flabel_sel == "filterlabel_with_wl":
        label = "eROSITA (0.2-2 keV)"
    else:
        label = "SRG eROSITA"

    lc_ax1.errorbar(
        x=59283.685482,
        y=6.2e-14,
        yerr=[[2.7e-14], [2.1e-14]],
        fmt="D",
        ms=6,
        color="darkcyan",
        label=label,
    )

    lc_ax1.errorbar(
        x=df_p200.obsmjd,
        y=df_p200.flux,
        yerr=df_p200.flux_err,
        color=cmap["P200+Ks"],
        marker=markers["P200"],
        markersize=3,
        linestyle=" ",
        label=filterlabel["P200+Ks"],
    )

    lc_ax1.plot(
        dustmodel_dict["mjds"],
        dustmodel_dict["convolution"],
        color="black",
        ls="dashdot",
        lw=1.0,
    )

    # lc_ax1.set_xlim([58570, 59460])
    lc_ax1.set_xlim([58570, 59500])

    d = cosmo.luminosity_distance(REDSHIFT)
    d = d.to(u.cm).value
    lumi = lambda flux: flux * 4 * np.pi * d ** 2
    flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
    lc_ax2 = lc_ax1.secondary_yaxis("right", functions=(lumi, flux))
    lc_ax2.tick_params(axis="y", which="major")
    lc_ax1.set_xlabel("Date (MJD)", fontsize=BIG_FONTSIZE - 2)
    lc_ax1.set_ylabel(r"$\nu$ F$_\nu$ (erg s$^{-1}$ cm$^{-2}$)", fontsize=BIG_FONTSIZE)
    # lc_ax1.set_ylabel(r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=BIG_FONTSIZE)
    lc_ax2.set_ylabel(r"$\nu$ L$_\nu$ (erg s$^{-1}$)", fontsize=BIG_FONTSIZE)
    # lc_ax2.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}]", fontsize=BIG_FONTSIZE))
    lc_ax1.grid(which="both", b=True, axis="both", alpha=0.2)
    t_neutrino = Time("2020-05-30T07:54:29.43", format="isot", scale="utc")
    lc_ax1.axvline(
        t_neutrino.mjd,
        linestyle="dotted",
        label="IC200530A",
        color="tab:red",
        zorder=50,
    )
    bbox = dict(boxstyle="round", fc="w", ec="gray")
    lc_ax1.text(
        t_neutrino.mjd - 99,
        1.25e-13,
        "Neutrino",
        # rotation="vertical",
        # bbox=bbox,
        fontsize=BIG_FONTSIZE - 2,
        color="tab:red",
    )

    bbox = dict(boxstyle="round", fc="w", ec="gray")
    lc_ax1.text(
        # 59310,
        # 4e-13,
        58680,
        1.25e-13,
        "Dust echo",
        # rotation="vertical",
        # bbox=bbox,
        fontsize=BIG_FONTSIZE - 2,
        color="black",
    )

    loc_upper = (0.05, 0.65)
    loc_lower = (0.15, 0.009)

    for interval in MJD_INTERVALS:
        lc_ax1.axvspan(interval[0], interval[1], alpha=0.2, color="gray")

    create_sed(sed1, 0)
    create_sed(sed2, 1)
    create_sed(sed3, 2)

    for i, sed in enumerate([sed1, sed2, sed3]):

        array = np.asarray([])

        con1 = ConnectionPatch(
            xyA=(sed_xlims[0], sed_ylims[0]),
            coordsA=sed.transData,
            xyB=(MJD_INTERVALS[i][0], lc_ylim[1]),
            coordsB=lc_ax1.transData,
            color="gray",
            alpha=0.3,
        )
        con2 = ConnectionPatch(
            xyA=(sed_xlims[1], sed_ylims[0]),
            coordsA=sed.transData,
            xyB=(MJD_INTERVALS[i][1], lc_ylim[1]),
            coordsB=lc_ax1.transData,
            color="gray",
            alpha=0.3,
        )

        for con in [con1, con2]:
            fig.add_artist(con)

        line2 = con2.get_path().vertices
        line1 = con1.get_path().vertices

        line1_new = np.asarray([line1[0], line1[2]])
        line2_new = np.asarray([line2[0], line2[2]])
        coords1 = np.asarray([line1[0], line1[2], line2[0]])
        coords2 = np.asarray([line2[0], line2[2], line1[2]])

        polygon1 = plt.Polygon(coords1, ec=None, fc="gray", clip_on=False, alpha=0.2)
        polygon2 = plt.Polygon(coords2, ec=None, fc="gray", clip_on=False, alpha=0.2)

        for polygon in [polygon1, polygon2]:
            fig.add_artist(polygon)

    outfile_pdf = os.path.join(PLOT_DIR, "lightcurve_and_sed.pdf")
    fig.savefig(outfile_pdf)
