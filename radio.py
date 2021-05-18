#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, time, argparse
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import constants as const
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from modelSED import utilities
import matplotlib.font_manager


def plot_radio(df):

    nice_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Times New Roman",
    }
    matplotlib.rcParams.update(nice_fonts)

    plt.figure(figsize=(FIG_WIDTH, 1 / 1.414 * FIG_WIDTH), dpi=DPI)

    ax1 = plt.subplot(111)

    filter_wl = utilities.load_info_json("filter_wl")

    plt.xscale("log")
    plt.yscale("log")

    # ujansky = lambda flux: flux * 1e29
    # flux = lambda ujansky: ujansky / 1e29
    # ax2 = ax1.secondary_yaxis("right", functions=(ujansky, flux))
    # ax2.set_ylabel(r"F$_\nu$ [$\mu$ Jy]", fontsize=BIG_FONTSIZE)

    d = cosmo.luminosity_distance(REDSHIFT)
    d = d.to(u.cm).value
    lumi = lambda flux: flux * 4 * np.pi * d ** 2
    flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
    ax2 = ax1.secondary_yaxis("right", functions=(lumi, flux))
    ax2.tick_params(axis="y", which="major", labelsize=BIG_FONTSIZE)
    ax2.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=BIG_FONTSIZE)

    ax1.set_xlabel("Frequency [Hz]", fontsize=BIG_FONTSIZE)
    ax3 = ax1.secondary_xaxis("top", functions=(utilities.nu_to_ev, utilities.ev_to_nu))
    ax3.set_xlabel("Energy [eV]", fontsize=BIG_FONTSIZE)
    # ax1.set_ylabel(
    #     r"F$_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$] ", fontsize=BIG_FONTSIZE
    # )
    ax1.set_ylabel(r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$] ", fontsize=BIG_FONTSIZE)

    colors = {0: "#42b3a5", 1: "#4083ac"}
    formats = {0: "d", 1: "s"}
    dates = {0: ["July 3, 2020", None, None], 1: ["September 9, 2020", None, None]}
    df_temp = df_vla

    i = 0
    if len(df_temp) > 0:
        for obsmjd in df_temp.obsmjd.unique():
            k = 0
            for band in ["3GHz", "6GHz", "10GHz"]:
                lc_temp = df_temp.query(f"band == '{band}' and obsmjd == '{obsmjd}'")
                if len(lc_temp) > 0:
                    wl = filter_wl[f"VLA+{band}"]
                    nu = utilities.lambda_to_nu(wl)
                    # nu_ghz = nu / 1e9
                    flux_jansky = lc_temp["fluxJy"]
                    flux_jansky_err = lc_temp["fluxerrJy"]
                    flux_density = flux_jansky / 1e23
                    flux_density_err = flux_jansky_err / 1e23
                    flux = flux_density * nu
                    flux_err = flux_density_err * nu
                    ax1.errorbar(
                        x=nu,
                        y=flux,
                        yerr=flux_err,
                        markersize=8,
                        fmt=formats[i],
                        color=colors[i],
                        label=dates[i][k],
                        capsize=8,
                        fillstyle="none",
                    )
                    k += 1
            i += 1

    # Plot limit of 0.15 mJy
    ax1.errorbar(
        x=3e9,
        xerr=0.1e9,  # [np.asarray([2.99]), np.asarray([3.01])],
        y=150 / 1e29 * 3e9,
        yerr=flux / 30,
        uplims=True,
        fmt=" ",
        color="tab:red",
        label="Archival sensitivity limit",
    )

    # bbox1 = dict(boxstyle="round", fc="1", color="#42b3a5")
    # bbox2 = dict(boxstyle="round", fc="1", color="#4083ac")

    # ax1.text(
    #     0.03,
    #     0.78,
    #     "July 3, 2020",
    #     transform=ax1.transAxes,
    #     fontsize=BIG_FONTSIZE,
    #     bbox=bbox1,
    #     color="#42b3a5",
    # )
    # ax1.text(
    #     0.4,
    #     0.28,
    #     "September 9, 2020",
    #     transform=ax1.transAxes,
    #     fontsize=BIG_FONTSIZE,
    #     bbox=bbox2,
    #     color="#4083ac",
    # )

    ax1.legend(fontsize=BIG_FONTSIZE, ncol=1, framealpha=1)  # , loc="lower right")
    plt.grid(which="both", alpha=0.15)
    plt.tight_layout()
    outpath = f"radio.png"

    plt.savefig(os.path.join(PLOT_DIR, outpath))
    plt.close()


if __name__ == "__main__":

    FLUXPLOT = True

    REDSHIFT = 0.267
    FIG_WIDTH = 6
    BIG_FONTSIZE = 12
    SMALL_FONTSIZE = 12
    DPI = 400

    BANDS_TO_EXCLUDE = {}

    CURRENT_FILE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "data"))
    PLOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "plots"))
    LC_DIR = os.path.abspath(os.path.join(DATA_DIR, "lightcurves"))

    paths = [DATA_DIR, PLOT_DIR]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    cmap = utilities.load_info_json("cmap")
    filterlabel = utilities.load_info_json("filterlabel")

    infile_vla = os.path.join(LC_DIR, "vla.csv")

    df_vla = pd.read_csv(infile_vla)

    plot_radio(df=df_vla)
