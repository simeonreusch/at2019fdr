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

XRT_COLUMN = "flux0310_bb_25eV"
nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)

# MJD_INTERVALS = [[58700, 58720], [59023, 59043], [59110, 59130], [59220,59265]]
MJD_INTERVALS = [[58700, 58720], [59006, 59130], [59220, 59271]]


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


def plot_lightcurve(df, fluxplot=False):
    """ """
    plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH/GOLDEN_RATIO))
    # plt.figure(dpi=DPI, figsize=(FIG_WIDTH, 2.5))
    filter_wl = utilities.load_info_json("filter_wl")
    ax1 = plt.subplot(111)

    ax1.set_xlim([58580, 59480])

    if fluxplot:
        plt.yscale("log")
        ax1.set_ylim([4e-14, 2e-12])

    for instrband in cmap:
        telescope, band = instrband.split("+")
        if telescope == "P200_sextractor":
            fmt = "*"
        else:
            fmt = "."
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
                marker=fmt,
                linestyle=" ",
                label=filterlabel[instrband],
            )
    if fluxplot:
        y = df_swift_xrt[XRT_COLUMN]
        yerr = df_swift_xrt[XRT_COLUMN] / 10
        ax1.errorbar(
            x=df_swift_xrt.obsmjd,
            y=y,
            yerr=yerr,
            uplims=True,
            fmt=fmt,
            color="darkviolet",
            label="Swift XRT",
        )

        y = df_fermi["flux"]
        yerr = df_fermi["flux"] / 10
        ax1.errorbar(
            x=df_fermi.obsmjd,
            xerr=df_fermi.obsmjd - df_fermi.obsmjd_start,
            y=y,
            yerr=yerr,
            uplims=True,
            fmt=" ",
            color="turquoise",
            label="Fermi LAT",
        )

    if not fluxplot:
        ax1.invert_yaxis()
        ax1.set_ylabel(r"Apparent Magnitude [AB]", fontsize=BIG_FONTSIZE)
    else:
        ax1.set_ylabel(
            r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=BIG_FONTSIZE + 1
        )
        ax2.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=BIG_FONTSIZE + 1)

    ax1.tick_params(axis="both", which="major", labelsize=BIG_FONTSIZE)
    ax1.set_xlabel("Date [MJD]", fontsize=BIG_FONTSIZE + 1)
    ax1.grid(which="both", b=True, axis="both", alpha=0.2)
    if not fluxplot:
        ax1.set_ylim([22, 16])
        absmag = lambda mag: mag - cosmo.distmod(REDSHIFT).value
        mag = lambda absmag: absmag + cosmo.distmod(REDSHIFT).value
        ax2 = ax1.secondary_yaxis("right", functions=(absmag, mag))
        ax2.tick_params(axis="both", which="major", labelsize=BIG_FONTSIZE)
        ax2.set_ylabel(rf"Absolute Magnitude (z={REDSHIFT:.3f})", fontsize=BIG_FONTSIZE)

    t_neutrino = Time("2020-05-30T07:54:29.43", format="isot", scale="utc")

    ax1.axvline(t_neutrino.mjd, linestyle=":", label="IC200530A")

    loc_upper = (0.05, 0.65)
    loc_lower = (0.09, 0.009)

    ax1.legend(fontsize=SMALL_FONTSIZE , ncol=2, framealpha=1, loc=loc_lower)

    for interval in MJD_INTERVALS:
        ax1.axvspan(interval[0], interval[1], alpha=0.2, color="gray")

    sns.despine(top=False, right=False)
    plt.tight_layout()
    if not fluxplot:
        outfile_png = "lightcurve_mag.png"
        outfile_pdf = "lightcurve_mag.pdf"
    else:
        outfile_png = "lightcurve_flux.png"
        outfile_pdf = "lightcurve_flux.pdf"

    outfile_png = os.path.join(PLOT_DIR, outfile_png)
    outfile_pdf = os.path.join(PLOT_DIR, outfile_pdf)
    plt.savefig(outfile_png)
    plt.savefig(outfile_pdf)

    percent_forced = (
        100
        / (counts_alert_photometry + counts_forced_photometry)
        * counts_forced_photometry
    )
    print(f"{percent_forced:.2f}% of ZTF datapoints are from forced photometry")
    plt.close()


def plot_sed(mjd_bounds, title="sed_peak", log=False):
    plt.figure(figsize=(0.8 * FIG_WIDTH, 0.8 * 1 / 1.414 * FIG_WIDTH), dpi=DPI)

    ax1 = plt.subplot(111)

    filter_wl = utilities.load_info_json("filter_wl")

    plt.yscale("log")
    plt.xscale("log")

    # ax1.set_ylim([2e-18, 2e-11])
    # ax1.set_xlim([1e-6, 4e12])
    ax1.set_ylim([1e-14, 2e-12])
    ax1.set_xlim([1e-1, 1e2])

    d = cosmo.luminosity_distance(REDSHIFT)
    d = d.to(u.cm).value
    lumi = lambda flux: flux * 4 * np.pi * d ** 2
    flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
    ax2 = ax1.secondary_yaxis("right", functions=(lumi, flux))
    ax3 = ax1.secondary_xaxis("top", functions=(nu_to_ev, ev_to_nu))

    ax1.set_xlabel("Energy [eV]", fontsize=SMALL_FONTSIZE)
    ax1.set_ylabel(
        r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=SMALL_FONTSIZE + 2
    )
    ax2.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=SMALL_FONTSIZE + 2)
    ax3.set_xlabel(r"Frequency [Hz]", fontsize=SMALL_FONTSIZE + 2)

    for instrband in cmap:
        telescope, band = instrband.split("+")
        if instrband not in BANDS_TO_EXCLUDE:

            lc = df.query(
                f"telescope == '{telescope}' and band == '{band}' and obsmjd >= {mjd_bounds[0]} and obsmjd <= {mjd_bounds[1]}"
            )
            mag_median = np.median(lc.mag)
            mag_err_median = np.median(lc.mag_err)

            flux_density = utilities.abmag_to_flux(mag_median)
            flux_density_err = utilities.abmag_err_to_flux_err(
                np.median(lc.mag), np.median(mag_err_median)
            )

            flux, flux_err = utilities.flux_density_to_flux(
                filter_wl[instrband], flux_density, flux_density_err
            )

            y = flux
            yerr = flux_err
            wl_angstrom = filter_wl[instrband]
            nu = utilities.lambda_to_nu(wl_angstrom)
            ev = nu_to_ev(nu)
            if telescope == "P200_sextractor":
                fmt = "*"
            else:
                fmt = "."
            ax1.errorbar(
                x=ev,
                y=y,
                yerr=yerr,
                color=cmap[instrband],
                marker=fmt,
                markersize=10,
                linestyle=" ",
                label=filterlabel[instrband],
            )

    lc = df_swift_xrt.query(f"obsmjd >= {mjd_bounds[0]} and obsmjd <= {mjd_bounds[1]}")
    flux = np.median(lc[XRT_COLUMN].values)
    x = nu_to_ev(utilities.lambda_to_nu(filter_wl["Swift+XRT"]))
    y = flux

    xerr = [np.asarray([x - 0.1e3]), np.asarray([x + 10e3])]
    ax1.errorbar(
        x=x,
        xerr=xerr,
        y=flux,
        yerr=flux / 3,
        uplims=True,
        fmt=" ",
        color="darkviolet",
        label="Swift XRT",
    )

    lc = df_fermi.query(
        f"obsmjd_start <= {mjd_bounds[0]} and obsmjd_end >= {mjd_bounds[1]}"
    )
    flux = np.median(lc["flux"].values)
    x = nu_to_ev(utilities.lambda_to_nu(filter_wl["Fermi+LAT"]))
    y = flux

    xerr = [np.asarray([x - 0.1e9]), np.asarray([x + 700e9])]
    ax1.errorbar(
        x=x,
        xerr=xerr,
        y=y,
        yerr=flux / 3,
        uplims=True,
        fmt=" ",
        color="turquoise",
        label="Fermi LAT",
    )

    # colors = {"10GHz": "magenta", "3GHz": "maroon", "6GHz": "peru"}
    # df_temp = df_vla.query(f"obsmjd >= {mjd_bounds[0]} and obsmjd <= {mjd_bounds[1]}")

    # if len(df_temp) > 0:
    #     for band in ["3GHz", "6GHz", "10GHz"]:
    #         lc_temp = df_temp.query(f"band == '{band}'")
    #         if len(lc_temp) > 0:
    #             wl = filter_wl[f"VLA+{band}"]
    #             flux_density = lc_temp["flux"]
    #             flux_density_err = lc_temp["fluxerr"]
    #             flux, flux_err = utilities.flux_density_to_flux(
    #                 wl, flux_density, flux_density_err
    #             )
    #             x = nu_to_ev(utilities.lambda_to_nu(wl))
    #             y = flux
    #             yerr = flux_err
    #             ax1.errorbar(
    #                 x=x,
    #                 y=y,
    #                 yerr=yerr,
    #                 markersize=10,
    #                 fmt=".",
    #                 color=colors[band],
    #                 label="VLA " + band,
    #             )

    ax1.tick_params(axis="both", which="major", labelsize=SMALL_FONTSIZE)
    ax2.tick_params(axis="both", which="major", labelsize=SMALL_FONTSIZE)
    ax3.tick_params(axis="both", which="major", labelsize=SMALL_FONTSIZE)
    ax1.legend(fontsize=SMALL_FONTSIZE, ncol=1, framealpha=1, loc="lower right")
    plt.grid(which="major", alpha=0.15)
    plt.tight_layout()
    outpath = f"{title}.png"
    plt.savefig(os.path.join(PLOT_DIR, outpath))
    plt.close()


if __name__ == "__main__":

    FLUXPLOT = True

    REDSHIFT = 0.267
    FIG_WIDTH = 8
    BIG_FONTSIZE = 14
    SMALL_FONTSIZE = 8
    GOLDEN_RATIO = 1.618
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
        "P200_sextractor+J",
        "P200_sextractor+H",
        "P200_sextractor+Ks",
        # "Swift+UVW1",
        # "Swift+UVW2",
        # "Swift+U",
        "Swift+V",
        "Swift+B",
    ]

    # BANDS_TO_EXCLUDE = [
    #     "P200_sextractor+J",
    #     "P200_sextractor+H",
    #     "P200_sextractor+Ks",
    #     "Swift+UVW1",
    #     "Swift+UVW2",
    #     "Swift+U",
    #     "Swift+V",
    #     "Swift+B",
    #     "WISE+W1",
    #     "WISE+W2",
    #     "P48+ZTF_i",
    #     "P48+ZTF_r",
    #     "Swift+UVM2",
    #     "P200+J",
    #     "P200+Ks",
    #     "P200+H"
    # ]

    infile_swift = os.path.join(LC_DIR, "swift_subtracted_synthetic.csv")
    infile_p200 = os.path.join(LC_DIR, "p200_subtracted_synthetic.csv")
    infile_ztf_forced = os.path.join(LC_DIR, "ZTF19aatubsj_SNT_5.0.csv")
    infile_ztf_alert = os.path.join(LC_DIR, "ZTF19aatubsj_alert.csv")
    infile_swift_xrt = os.path.join(LC_DIR, "swift_xrt_ulims.csv")
    infile_fermi = os.path.join(LC_DIR, "fermi_ulims.csv")
    infile_vla = os.path.join(LC_DIR, "vla.csv")
    infile_wise = os.path.join(LC_DIR, "wise_subtracted_baseline.csv")

    outfile_lightcurve = os.path.join(LC_DIR, "full_lightcurve.csv")

    df_swift = pd.read_csv(infile_swift)
    df_p200 = pd.read_csv(infile_p200)
    df_ztf_forced = pd.read_csv(infile_ztf_forced)
    df_ztf_alert = pd.read_csv(infile_ztf_alert)
    df_swift_xrt = pd.read_csv(infile_swift_xrt)
    df_fermi = pd.read_csv(infile_fermi)
    df_vla = pd.read_csv(infile_vla)
    df_wise = pd.read_csv(infile_wise)


    df_ztf_forced = df_ztf_forced[["obsmjd", "mag", "mag_err", "filter"]]
    df_ztf_forced.rename(columns={"filter": "band"}, inplace=True)
    df_ztf_forced["telescope"] = "P48"
    df_ztf_forced["alert"] = False
    df_ztf_forced.query("mag < 99", inplace=True)
    counts_forced_photometry = len(df_ztf_forced)

    df_ztf_alert = df_ztf_alert[["obsmjd", "filter_id", "mag", "mag_err"]]
    df_ztf_alert["telescope"] = "P48"
    df_ztf_alert["alert"] = True
    df_ztf_alert.replace(
        {"filter_id": {1: "ZTF_g", 2: "ZTF_r", 3: "ZTF_i"}}, inplace=True
    )
    df_ztf_alert.rename(columns={"filter_id": "band"}, inplace=True)
    counts_alert_photometry = len(df_ztf_alert)

    df = pd.concat(
        [
            df_p200,
            df_swift,
            df_ztf_forced,
            df_ztf_alert,
            df_wise,
        ],
        ignore_index=True,
    )

    df.reset_index(inplace=True, drop=True)
    df.drop(columns=["Unnamed: 0"], inplace=True)

    cmap = utilities.load_info_json("cmap")
    filterlabel = utilities.load_info_json("filterlabel")

    plot_lightcurve(df=df)
    plot_lightcurve(df=df, fluxplot=True)

    # titles = ["sed_1_peak", "sed_2_P200_epoch1", "sed_3_P200_epoch2", "sed_4_P200_epoch3"]
    titles = ["sed_1_peak", "sed_2_P200_epoch1", "sed_3_P200_epoch2"]
    for i, interval in enumerate(MJD_INTERVALS):
        plot_sed(interval, title=titles[i])

    # Save lightcurve for further use

    df.to_csv(outfile_lightcurve)
