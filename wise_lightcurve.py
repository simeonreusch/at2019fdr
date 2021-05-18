#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, time, argparse
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
import matplotlib

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)


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


def plot_lightcurve(df, fluxplot=False):
    """ """
    plt.figure(dpi=DPI, figsize=(FIG_WIDTH, 5))
    filter_wl = utilities.load_info_json("filter_wl")
    ax1 = plt.subplot(111)

    if fluxplot:
        plt.yscale("log")

    for instrband in cmap:
        telescope, band = instrband.split("+")
        if instrband not in BANDS_TO_EXCLUDE:
            lc = df.query(f"telescope == '{telescope}' and band == '{band}'")
            if not fluxplot:
                y = lc.mag
                yerr = lc.mag_err
            else:
                d = cosmo.luminosity_distance(REDSHIFT)
                d = d.to(u.cm).value
                lumi = lambda flux: flux * 4 * np.pi * d ** 2
                flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
                ax2 = ax1.secondary_yaxis("right", functions=(lumi, flux))
                ax2.tick_params(axis="y", which="major", labelsize=BIG_FONTSIZE)

                flux_density = utilities.abmag_to_flux(lc.mag)
                flux_density_err = utilities.abmag_err_to_flux_err(lc.mag, lc.mag_err)
                flux, flux_err = utilities.flux_density_to_flux(
                    filter_wl[instrband], flux_density, flux_density_err
                )
                y = flux
                yerr = flux_err
            ax1.errorbar(
                x=lc.obsmjd,
                y=y,
                yerr=yerr,
                color=cmap[instrband],
                marker=".",
                linestyle=" ",
                label=filterlabel[instrband],
            )

    ax1.set_ylabel(r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=BIG_FONTSIZE + 1)
    ax2.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=BIG_FONTSIZE + 1)

    ax1.tick_params(axis="both", which="major", labelsize=BIG_FONTSIZE)
    ax1.set_xlabel("Date [MJD]", fontsize=BIG_FONTSIZE + 1)
    ax1.grid(which="both", b=True, axis="both", alpha=0.2)

    t_neutrino = Time("2020-05-30T07:54:29.43", format="isot", scale="utc")

    t_tywinstart = Time("58600", format="mjd")

    ax1.axvline(t_neutrino.mjd, linestyle=":", label="IC200530A")
    ax1.axvline(
        t_tywinstart.mjd, linestyle=":", label="Tywin starts here", color="black"
    )

    ax1.legend(fontsize=SMALL_FONTSIZE - 1, ncol=2, framealpha=1)

    mjd_intervals = [
        [t_tywinstart.mjd - (i * 365), t_tywinstart.mjd - ((i - 1) * 365)]
        for i in range(1, 6)
    ]

    for i, interval in enumerate(mjd_intervals[::-1]):
        start = Time(interval[0], format="mjd").iso[:10]
        end = Time(interval[1], format="mjd").iso[:10]

        print(f"Interval {i}: Start={start} End={end}")

    colors = ["lightgray", "gray"]

    for i, interval in enumerate(mjd_intervals):
        ax1.axvspan(interval[0], interval[1], alpha=0.4, color=colors[i % 2])

    sns.despine(top=False, right=False)
    plt.tight_layout()

    outpath = "lightcurve_wise.png"
    plt.savefig(os.path.join(PLOT_DIR, outpath))
    plt.close()


if __name__ == "__main__":

    FLUXPLOT = True

    REDSHIFT = 0.267
    FIG_WIDTH = 8
    BIG_FONTSIZE = 14
    SMALL_FONTSIZE = 12
    DPI = 400

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
        "Swift+UVM2",
        "Swift+U",
        "Swift+V",
        "Swift+B",
        "P48+ZTF_g",
        "P48+ZTF_r",
        "P48+ZTF_i",
        # "WISE+W1",
        # "WISE+W2",
    ]

    # infile_wise = os.path.join(LC_DIR, "wise_subtracted_prepeak.csv")
    infile_wise = os.path.join(LC_DIR, "wise.csv")
    df_wise = pd.read_csv(infile_wise)

    df = df_wise

    cmap = utilities.load_info_json("cmap")
    filterlabel = utilities.load_info_json("filterlabel")

    plot_lightcurve(df=df, fluxplot=FLUXPLOT)
