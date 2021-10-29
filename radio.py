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

    plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH - 1), dpi=DPI)

    ax1 = plt.subplot(111)

    filter_wl = utilities.load_info_json("filter_wl")

    plt.xscale("log")
    plt.yscale("log")

    ax1.set_xlim([1e0, 2e1])

    # ujansky = lambda flux: flux * 1e29
    # flux = lambda ujansky: ujansky / 1e29
    # ax2 = ax1.secondary_yaxis("right", functions=(ujansky, flux))
    # ax2.set_ylabel(r"F$_\nu$ [$\mu$ Jy]", fontsize=BIG_FONTSIZE)

    # d = cosmo.luminosity_distance(REDSHIFT)
    # d = d.to(u.cm).value
    # flux_to_lumi = lambda flux: flux * 4 * np.pi * d ** 2
    # lumi_to_flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
    # ax2 = ax1.secondary_yaxis("right", functions=(flux_to_lumi, lumi_to_flux))
    # ax2.tick_params(axis="y", which="major", labelsize=BIG_FONTSIZE)
    # ax2.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=BIG_FONTSIZE)

    ax1.set_xlabel("Frequency [GHz]", fontsize=BIG_FONTSIZE)

    # ax3 = ax1.secondary_xaxis("top", functions=(utilities.nu_to_ev, utilities.ev_to_nu))
    # ax3.set_xlabel("Energy [eV]", fontsize=BIG_FONTSIZE)

    # ax1.set_ylabel(r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$] ", fontsize=BIG_FONTSIZE)
    ax1.set_ylabel(r"Flux [mJy]", fontsize=BIG_FONTSIZE)
    ax1.tick_params(axis="both", which="both", labelsize=BIG_FONTSIZE)

    config = {
        # 59033: {"c": "#42b3a5", "f": "d", "date": "2020-07-03"},
        # 59105: {"c": "#4083ac", "f": "s", "date": "2020-09-13"},
        # 59160: {"c": "tab:red", "f": "p", "date": "2020-11-07"},
        59033: {"c": "#24a885", "f": "d", "date": "2020-07-03"},
        59105: {"c": "#197aa1", "f": "s", "date": "2020-09-13"},
        59160: {"c": "#46469f", "f": "p", "date": "2020-11-07"},
    }

    for obsmjd in df.obsmjd.unique():
        temp = df.query("obsmjd == @obsmjd")
        ax1.errorbar(
            x=temp["band"],
            y=temp["flux_muJy"] / 1e3,
            yerr=temp["fluxerr_muJy"] / 1e3,
            markersize=8,
            color=config[obsmjd]["c"],
            label=config[obsmjd]["date"],
            fmt=config[obsmjd]["f"],
            ls="solid",
        )

    # Plot limit of 0.15 mJy
    y = 0.15
    yerr = y / 10

    ax1.errorbar(
        x=3,
        xerr=0.4,
        y=y,
        yerr=yerr,
        uplims=True,
        fmt=" ",
        color="tab:red",
        label="Archival limit",
    )
    ax1.grid(color="gray", alpha=0.1, axis="both", which="both")

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

    ax1.legend(
        fontsize=BIG_FONTSIZE - 0.5, ncol=1, framealpha=1
    )  # , loc="lower right")
    # plt.grid(which="both", alpha=0.15)
    plt.tight_layout()
    outpath = f"radio.pdf"

    plt.savefig(os.path.join(PLOT_DIR, outpath))
    plt.close()


if __name__ == "__main__":

    FLUXPLOT = True

    REDSHIFT = 0.267
    FIG_WIDTH = 5
    BIG_FONTSIZE = 14
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
