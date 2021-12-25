#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, time, argparse
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import constants as const
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from modelSED import utilities

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)


XRTCOLUMN = "flux0310_pi_-2"
ANNOTATION_FONTSIZE = 14
FIG_WIDTH = 8
BIG_FONTSIZE = 14
SMALL_FONTSIZE = 9
DPI = 400


REDSHIFT_TYWIN = 0.267
REDSHIFT_BRAN = 0.0512


def angstrom_to_ev(angstrom):
    energy = const.h.value * const.c.value / (angstrom * 1e-10)
    ev = energy / 1.602e-19
    return ev


def plot_lightcurve(df_bran, df_tywin, fluxplot=False):

    df = df_bran
    plt.figure(figsize=(0.8 * FIG_WIDTH, 0.5658 * FIG_WIDTH), dpi=DPI)
    filter_wl = utilities.load_info_json("filter_wl")
    ax1 = plt.subplot(111)

    plt.yscale("log")

    colors = ["blue", "red"]
    i = 0
    redshifts = [REDSHIFT_BRAN, REDSHIFT_TYWIN]

    for df in [df_bran, df_tywin]:
        for instrband in cmap:
            telescope, band = instrband.split("+")
            if instrband not in BANDS_TO_EXCLUDE:
                lc = df.query(f"telescope == '{telescope}' and band == '{band}'")
                d = cosmo.luminosity_distance(redshifts[i])
                d = d.to(u.cm).value
                flux_density = utilities.abmag_to_flux(lc.mag)
                flux_density_err = utilities.abmag_err_to_flux_err(lc.mag, lc.mag_err)
                flux, flux_err = utilities.flux_density_to_flux(
                    filter_wl[instrband], flux_density, flux_density_err
                )
                lumi = flux * 4 * np.pi * d ** 2
                lumi_err = flux_err * 4 * np.pi * d ** 2
                # y = flux
                # yerr = flux_err
                y = lumi
                yerr = lumi_err
                ax1.errorbar(
                    x=lc.obsmjd,
                    y=y,
                    yerr=yerr,
                    color=colors[i],
                    marker=".",
                    linestyle=" ",
                    label=filterlabel[instrband],
                    markersize=10,
                )
        i += 1

    ax1.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=BIG_FONTSIZE + 2)

    ax1.tick_params(axis="both", which="major", labelsize=BIG_FONTSIZE)
    ax1.set_xlabel("Date [MJD]", fontsize=BIG_FONTSIZE + 2)
    ax1.grid(b=True, which="major", axis="both", alpha=0.33)

    ax1.set_xlim(58500, 59200)
    ax1.set_ylim(2.7e41, 4.5e44)

    t_neutrino_tywin = Time("2020-05-30T07:54:29.43", format="isot", scale="utc")
    t_neutrino_bran = Time("2019-10-01T07:54:29.43", format="isot", scale="utc")
    times = [t_neutrino_bran.mjd, t_neutrino_tywin.mjd]
    ax1.axvline(
        t_neutrino_bran.mjd, linestyle=":", label="IC200530A", color="blue", linewidth=2
    )
    ax1.axvline(
        t_neutrino_tywin.mjd, linestyle=":", label="IC200530A", color="red", linewidth=2
    )

    bbox_bran = dict(boxstyle="round", fc="1", color="blue", pad=0.1)
    bbox_tywin = dict(boxstyle="round", fc="1", color="red", pad=0.1)
    bboxes = [bbox_bran, bbox_tywin]
    neutrinos = ["IceCube-191001A", "IceCube-200530A"]
    pos = [-160, -90]
    i = 0
    for time in times:
        ax1.annotate(
            neutrinos[i],
            (pos[i] + time, 3.25e41),
            fontsize=ANNOTATION_FONTSIZE + 1,
            bbox=bboxes[i],
            color=colors[i],
        )
        i += 1

    # ax1.legend(fontsize=SMALL_FONTSIZE, ncol=2, framealpha=1)
    sns.despine(top=False, right=False)
    plt.tight_layout()

    ax1.text(
        0.02,
        0.26,
        "AT2019dsg",
        transform=ax1.transAxes,
        fontsize=ANNOTATION_FONTSIZE + 1,
        bbox=bbox_bran,
        color="blue",
    )
    ax1.text(
        0.02,
        0.945,
        "AT2019fdr",
        transform=ax1.transAxes,
        fontsize=ANNOTATION_FONTSIZE + 1,
        bbox=bbox_tywin,
        color="red",
    )

    outpath = "bran_tywin_flux.png"
    plt.savefig(os.path.join(PLOT_DIR, outpath))

    # percent_forced = (
    #     100
    #     / (counts_alert_photometry + counts_forced_photometry)
    #     * counts_forced_photometry
    # )
    # print(f"{percent_forced:.2f}% of ZTF datapoints are from forced photometry")
    plt.tight_layout()
    plt.close()


if __name__ == "__main__":

    FLUXPLOT = True

    CURRENT_FILE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "data"))
    PLOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "plots"))
    SPECTRA_DIR = os.path.join(DATA_DIR, "spectra")
    LC_DIR = os.path.join(DATA_DIR, "lightcurves")

    paths = [DATA_DIR, PLOT_DIR, SPECTRA_DIR, LC_DIR]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    BANDS_TO_EXCLUDE = [
        "P200+J",
        "P200+H",
        "P200+Ks",
        "Swift+UVW1",
        "Swift+UVW2",
        "Swift+U",
        "Swift+V",
        "Swift+B",
        "P48+ZTF_g",
        "P48+ZTF_i",
    ]

    infile_ztf_forced_tywin = os.path.join(LC_DIR, "ZTF19aatubsj_SNT_5.0.csv")
    infile_ztf_alert_tywin = os.path.join(LC_DIR, "ZTF19aatubsj_alert.csv")
    infile_ztf_forced_bran = os.path.join(LC_DIR, "ZTF19aapreis_SNT_5.0.csv")
    infile_ztf_alert_bran = os.path.join(LC_DIR, "ZTF19aapreis_alert.csv")

    df_ztf_forced_tywin = pd.read_csv(infile_ztf_forced_tywin)
    df_ztf_alert_tywin = pd.read_csv(infile_ztf_alert_tywin)
    df_ztf_forced_bran = pd.read_csv(infile_ztf_forced_bran)
    df_ztf_alert_bran = pd.read_csv(infile_ztf_alert_bran)

    def clean_df_forced(df):
        df = df[["obsmjd", "mag", "mag_err", "filter"]]
        df.rename(columns={"filter": "band"}, inplace=True)
        df["telescope"] = "P48"
        df["alert"] = False
        df.query("mag < 99", inplace=True)
        return df

    def clean_df_alert(df):
        df = df[["obsmjd", "filter_id", "mag", "mag_err"]]
        df["telescope"] = "P48"
        df["alert"] = True
        df.replace({"filter_id": {1: "ZTF_g", 2: "ZTF_r", 3: "ZTF_i"}}, inplace=True)
        df.rename(columns={"filter_id": "band"}, inplace=True)
        return df

    df_tywin_forced = clean_df_forced(df_ztf_forced_tywin)
    df_bran_forced = clean_df_forced(df_ztf_forced_bran)

    df_tywin_alert = clean_df_alert(df_ztf_alert_tywin)
    df_bran_alert = clean_df_alert(df_ztf_alert_bran)

    df_tywin = pd.concat([df_tywin_forced, df_tywin_alert], ignore_index=True)
    df_bran = pd.concat([df_bran_forced, df_bran_alert], ignore_index=True)
    cmap = utilities.load_info_json("cmap")
    filterlabel = utilities.load_info_json("filterlabel")

    plot_lightcurve(df_bran, df_tywin, fluxplot=FLUXPLOT)
