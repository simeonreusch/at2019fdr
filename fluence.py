#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

import matplotlib

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)

GENERIC_COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)

REDSHIFT = 0.267
IC200530A_ENERGY = 4.915
FIG_WIDTH = 6
BIG_FONTSIZE = 14
SMALL_FONTSIZE = 8
GOLDEN_RATIO = 1.618
DPI = 400

if __name__ == "__main__":
    CURRENT_FILE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "data", "effective_area"))
    PLOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "plots"))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    infile_jetted = os.path.join(DATA_DIR, "fl_jetted_tywin_dbb_v4.csv")
    infile_corona = os.path.join(DATA_DIR, "kohta_corona.csv")
    infile_wind = os.path.join(DATA_DIR, "kohta_wind2.csv")

    jetted = pd.read_csv(infile_jetted)
    corona = pd.read_csv(infile_corona)
    wind = pd.read_csv(infile_wind)

    plt.figure(dpi=DPI, figsize=(5, 4))

    ax1 = plt.subplot(111)

    corona["a"] = corona["a"] / 1e9
    corona["h"] = corona["h"] * 1e9
    corona_x = np.log10(corona["a"] / 1.267)
    corona_y = np.log10(corona["a"] * corona["a"] * corona["h"] * 1.474 / 3)

    wind["a"] = wind["a"] / 1e9
    wind["h"] = wind["h"] * 1e9
    wind_x = np.log10(wind["a"] / 1.267)
    wind_y = np.log10(wind["a"] * wind["a"] * wind["h"] * 55.38 / 3)

    df1 = pd.DataFrame()
    df1["log10E_GeV"] = wind_x
    df1["fluence_GeV"] = wind_y

    df2 = pd.DataFrame()
    df2["log10E_GeV"] = corona_x
    df2["fluence_GeV"] = corona_y

    df3 = pd.DataFrame()
    df3["log10E_GeV"] = jetted["log10_e_nu"]
    df3["fluence_GeV"] = jetted["log10_F"]

    df1.to_csv(os.path.join(DATA_DIR, "wind_new.csv"))
    df2.to_csv(os.path.join(DATA_DIR, "corona.csv"))
    df3.to_csv(os.path.join(DATA_DIR, "jet_new.csv"))

    ax1.plot(
        jetted["log10_e_nu"],
        jetted["log10_F"],
        label=r"Relativistic jet",
        linestyle="dashdot",
    )

    ax1.plot(corona_x, corona_y, color="red", label="Disk-corona")

    ax1.plot(
        wind_x, wind_y, color="green", label="Sub-relativistic wind", linestyle="dashed"
    )

    ax1.set_xlim([2, 9])
    ax1.set_ylim([-5.5, 0])

    ax1.set_xlabel(r"log$_{10}~E_{\nu}$ (GeV)", fontsize=BIG_FONTSIZE)
    ax1.set_ylabel(
        r"log$_{10}~E_{\nu}^2 ~\mathcal{F}_\mu$ (GeV/cm$^2$)", fontsize=BIG_FONTSIZE
    )

    # ax1.arrow(IC200530A_ENERGY, -3.5, 0, 0.7, width=0.01, color="black")
    ax1.text(IC200530A_ENERGY - 1.05, -1, r"$E_{\nu,~\rm obs}$", fontsize=BIG_FONTSIZE)
    ax1.axvline(IC200530A_ENERGY, linestyle="dotted", color="black")
    ax1.tick_params(axis="both", labelsize=BIG_FONTSIZE)

    plt.legend(fontsize=BIG_FONTSIZE - 2)
    plt.tight_layout()

    outfile = os.path.join(PLOT_DIR, "fluence.pdf")
    plt.savefig(outfile)
