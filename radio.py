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

    ax1.set_xlabel("Frequency (GHz)", fontsize=BIG_FONTSIZE)

    ax1.set_ylabel(r"Flux (mJy)", fontsize=BIG_FONTSIZE)
    ax1.tick_params(axis="both", which="both", labelsize=BIG_FONTSIZE)

    config = {
        59033: {
            "c": "#24a885",
            "mfc": "#24a885",
            "f": "d",
            "date": "2020-07-03 (59033)",
            "ls": "solid",
        },
        59105: {
            "c": "#197aa1",
            "mfc": "#197aa1",
            "f": "s",
            "date": "2020-09-13 (59105)",
            "ls": "solid",
        },
        59160: {
            "c": "#46469f",
            "mfc": "white",
            "f": "o",
            "date": "2020-11-07 (59160)",
            "ls": "dashed",
        },
    }

    # print(df.drop(labels=[0,1]))

    for obsmjd in [59160]:
        temp = df.query("obsmjd == @obsmjd")
        ax1.errorbar(
            x=temp["band"],
            y=temp["flux_muJy"] / 1e3,
            yerr=temp["fluxerr_muJy"] / 1e3,
            markersize=8,
            color=config[obsmjd]["c"],
            # label=config[obsmjd]["date"],
            fmt=config[obsmjd]["f"],
            ls="dashed",
            mfc=config[obsmjd]["mfc"],
        )

    for obsmjd in df.obsmjd.unique():
        temp = df.query("obsmjd == @obsmjd")
        if obsmjd == 59160:
            temp = temp.drop(labels=[8, 9])
        ax1.errorbar(
            x=temp["band"],
            y=temp["flux_muJy"] / 1e3,
            yerr=temp["fluxerr_muJy"] / 1e3,
            markersize=8,
            color=config[obsmjd]["c"],
            label=config[obsmjd]["date"],
            fmt=config[obsmjd]["f"],
            ls="solid",
            # mfc="solid",
        )

    # Plot limit of 0.15 mJy
    y = 0.32
    yerr = y / 10

    ax1.errorbar(
        x=3,
        xerr=0.4,
        y=y,
        yerr=yerr,
        uplims=True,
        fmt=" ",
        color="tab:red",
        label="Archival limit\n2017-11-25 (58082)",
    )
    ax1.grid(color="gray", alpha=0.1, axis="both", which="both")

    ax1.legend(fontsize=BIG_FONTSIZE - 1, ncol=1, framealpha=1)  # , loc="lower right")
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
