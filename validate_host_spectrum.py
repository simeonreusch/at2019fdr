#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.time import Time

from modelSED import utilities, sncosmo_spectral_v13

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)


def twomass_vega_to_ab(band, mag):
    diff = {"P200+J": 0.91, "P200+H": 1.39, "P200+Ks": 1.85}
    return mag + diff[band]


def ukirt_vega_to_ab(band, mag):
    diff = {"UKIRT+J": 0.938}
    return mag + diff[band]


def wise_vega_to_ab(band, mag):
    diff = {"W1": 2.683, "W2": 3.319, "W3": 5.242, "W4": 6.604}
    return mag + diff[band]


VERBOSE = False

REDSHIFT = 0.267
FONTSIZE = 14
FONTSIZE_LEGEND = 8
ANNOTATION_FONTSIZE = 10
FONTSIZE_TICKMARKS = 12
DPI = 400
PLOTDIR = "plots"

colors = {
    "P200": "tab:orange",
    "WISE": "tab:purple",
    "PS1": "lime",
    "SDSS": "fuchsia",
    "hostmodel": "tab:blue",
    "UKIRT": "tab:red",
    "P200_ref": "black",
}

DATES = ["2020_07_01", "2020_09_29", "2021_02_04", "2021_05_28"]
DATE_TO_ABBR_MJD = {"2020_07_01": 59031, "2020_09_29": 59121, "2021_02_04": 59249, "2021_05_28": 59362}
DATES_ISO = [date.replace("_", "-") + "T00:00:00" for date in DATES]
DATES_MJD = [Time(date_iso, format="isot", scale="utc").mjd for date_iso in DATES_ISO]

for i, DATE in enumerate(DATES):

    if VERBOSE:
        print(f"\n{DATE}\n")

    HOST_MODEL_DIR = os.path.join("data", "host_model")

    P200_SEXTRACTOR_FILE = os.path.join("data", "P200_unsubtracted.csv")
    P200_GALFIT_FILE_REF_EPOCH1 = os.path.join(
        "/", "Users", "simeon", "galfit", "galfit_result_twocomp_fromepoch1.csv"
    )
    P200_GALFIT_FILE_REF_EPOCH4 = os.path.join(
        "/", "Users", "simeon", "galfit", "galfit_result_twocomp_fromepoch4.csv"
    )
    # infile = os.path.join(HOST_MODEL_DIR, "Tywin_parasfh_spectrum_with_WISE.dat")
    infile = os.path.join(HOST_MODEL_DIR, "Tywin_parasfh_wduste_spectrum_NEW.dat")

    host_spectrum = pd.read_table(
        infile, names=["wl", "flux", "abmag"], sep="\s+", comment="#"
    )

    host_spectrum["mag"] = utilities.flux_to_abmag(
        flux_nu=host_spectrum.flux, flux_nu_zp=0
    )

    filter_wl = utilities.load_info_json("filter_wl")
    cmap = utilities.load_info_json("cmap")
    filterlabel = utilities.load_info_json("filterlabel")

    all_bands = ["P200+J", "P200+H", "P200+Ks", "P48+ZTF_g", "P48+ZTF_r", "P48+ZTF_i"]

    # Now we get magnitudes using bandpasses
    # First we need to construct a proper spectrum
    spectrum = sncosmo_spectral_v13.Spectrum(
        wave=host_spectrum.wl.values,
        flux=host_spectrum.flux.values * 3.631e-20,
        unit=utilities.FNU,
    )

    bandpassfiles = utilities.load_info_json("bandpassfiles")

    # Now get PS1 data
    # infile = os.path.join(HOST_MODEL_DIR, "PS1_host_detections.csv")
    infile = os.path.join(HOST_MODEL_DIR, "PS1_forced_host.csv")
    df = pd.read_csv(infile)
    ps1_abmag = []
    ps1_abmag_err = []
    p200_bands = ["J", "H", "Ks"]
    # p200_bands = ["H"]
    ps1_bands = ["g", "r", "i", "z", "y"]
    sdss_bands = ["u", "g", "r", "i", "z"]
    wise_bands = ["W1", "W2"]
    ukirt_bands = ["J"]
    ukirt_filterids = {"Z": 1, "Y": 2, "J": 3, "H": 4, "K": 5, "H2": 6, "B": 7}

    ps1_filterids = {"g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

    for band in ps1_bands:
        # df_temp = df.query(f"filterID == {ps1_filterids[band]}")
        # psf_flux_median = np.median(df_temp["apFlux"].values *1e-23)
        # psf_flux_err_median = np.median(df_temp["apFluxErr"].values *1e-23)
        # abmag = utilities.flux_to_abmag(psf_flux_median)
        # abmag_err = utilities.flux_err_to_abmag_err(psf_flux_median, psf_flux_err_median)
        psf_flux = df[f"{band}FKronFlux"].values[0] * 1e-23
        psf_flux_err = df[f"{band}FKronFluxErr"].values[0] * 1e-23
        abmag = utilities.flux_to_abmag(psf_flux)
        abmag_err = utilities.flux_err_to_abmag_err(psf_flux, psf_flux_err)
        ps1_abmag.append(abmag)
        ps1_abmag_err.append(abmag_err)

    df_ps1 = pd.DataFrame(columns=["telescope", "band", "abmag", "abmag_err"])
    df_ps1["telescope"] = ["PS1" for i in range(5)]
    df_ps1["band"] = ps1_bands
    df_ps1["abmag"] = ps1_abmag
    df_ps1["abmag_err"] = ps1_abmag_err

    # Now we get the P200 data
    # df_p200_sextractor = pd.read_csv(P200_SEXTRACTOR_FILE, index_col="obsmjd")

    # And the P200 galfit datafram
    df_p200_galfit_ref_epoch1 = pd.read_csv(
        P200_GALFIT_FILE_REF_EPOCH1, index_col="obsmjd"
    )
    df_p200_galfit_ref_epoch4 = pd.read_csv(
        P200_GALFIT_FILE_REF_EPOCH4, index_col="obsmjd"
    )

    # Now we get the SDSS data
    infile = os.path.join(HOST_MODEL_DIR, "SDSS_prepeak.csv")
    df_sdss = pd.read_csv(infile)

    # Now we get the UKIDSS data
    infile = os.path.join(HOST_MODEL_DIR, "UKIRT_prepeak.csv")
    df_ukirt = pd.read_csv(infile)

    # Aaaand finally, the WISE data
    infile = os.path.join(HOST_MODEL_DIR, "WISE_sjoert_prepeak.csv")
    df_wise = pd.read_csv(infile)

    abmag = [
        wise_vega_to_ab(df_wise["band"].values[i], mag)
        for i, mag in enumerate(df_wise["mag_vega"].values)
    ]

    df_wise["mag_ab"] = abmag
    wise_w1_median = np.median(df_wise.query("band == 'W1'")["mag_ab"].values)
    wise_w2_median = np.median(df_wise.query("band == 'W2'")["mag_ab"].values)
    wise_w1_median_vega = np.median(df_wise.query("band == 'W1'")["mag_vega"].values)
    wise_w2_median_vega = np.median(df_wise.query("band == 'W2'")["mag_vega"].values)

    wise_w1_mean = np.average(
        df_wise.query("band == 'W1'")["mag_ab"].values,
        weights=1 / df_wise.query("band == 'W1'")["e_mag"].values,
    )
    wise_w2_mean = np.average(
        df_wise.query("band == 'W2'")["mag_ab"].values,
        weights=1 / df_wise.query("band == 'W2'")["e_mag"].values,
    )

    wise_values = {"WISE+W1": wise_w1_mean, "WISE+W2": wise_w2_mean}

    # Now we plot this
    fig, ax1 = plt.subplots(1, 1, figsize=[6, 6 / 1.414], dpi=DPI)

    ax1.plot(
        utilities.lambda_to_nu(host_spectrum.wl),
        # utilities.abmag_to_flux(host_spectrum.mag),
        host_spectrum.mag,
        label="modeled Host spectrum",
        color="tab:blue",
    )

    for band in p200_bands:
        date_mjd_abbr = DATE_TO_ABBR_MJD[DATE]

        # abmag_sextractor_both = df_p200_sextractor.query(f"obsmjd == {date_mjd_abbr}")[
            # f"{band}_mag_iso_AB"
        # ].values[0]

        vegamag_host_1 = df_p200_galfit_ref_epoch1.loc[DATES_MJD[i]][
            f"{band}_vegamag_host"
        ]
        vegamag_host_4 = df_p200_galfit_ref_epoch4.loc[DATES_MJD[i]][
            f"{band}_vegamag_host"
        ]
        vegamag_psf_1 = df_p200_galfit_ref_epoch1.loc[DATES_MJD[i]][
            f"{band}_vegamag_psf"
        ]
        vegamag_psf_4 = df_p200_galfit_ref_epoch4.loc[DATES_MJD[i]][
            f"{band}_vegamag_psf"
        ]

        abmag_host_1 = twomass_vega_to_ab("P200+" + band, vegamag_host_1)
        abmag_host_4 = twomass_vega_to_ab("P200+" + band, vegamag_host_4)
        abmag_psf_1 = twomass_vega_to_ab("P200+" + band, vegamag_psf_1)
        abmag_psf_4 = twomass_vega_to_ab("P200+" + band, vegamag_psf_4)

        flux_jy_host_1 = utilities.abmag_to_flux(abmag_host_1, 0)
        flux_jy_host_4 = utilities.abmag_to_flux(abmag_host_4, 0)
        flux_jy_psf_1 = utilities.abmag_to_flux(abmag_psf_1, 0)
        flux_jy_psf_4 = utilities.abmag_to_flux(abmag_psf_4, 0)

        flux_jy_host_and_psf_1 = flux_jy_host_1 + flux_jy_psf_1
        flux_jy_host_and_psf_4 = flux_jy_host_4 + flux_jy_psf_4
        # flux_jy_host_and_psf_sextractor = utilities.abmag_to_flux(
        #     abmag_sextractor_both, 0
        # )
        flux_jy_host_and_psf_average = (
            flux_jy_host_and_psf_1 + flux_jy_host_and_psf_4
        ) / 2

        if VERBOSE:
            print(f"flux_jy_host_and_psf_1: {flux_jy_host_and_psf_1}")
            print(f"flux_jy_host_and_psf_4: {flux_jy_host_and_psf_4}")

        abmag_host_and_psf_1 = utilities.flux_to_abmag(flux_jy_host_and_psf_1, 0)
        abmag_host_and_psf_4 = utilities.flux_to_abmag(flux_jy_host_and_psf_4, 0)
        abmag_host_and_psf_average = utilities.flux_to_abmag(
            flux_jy_host_and_psf_average, 0
        )
        if VERBOSE:
            print(f"abmag_host_and_psf_1: {abmag_host_and_psf_1}")
            print(f"abmag_host_and_psf_4: {abmag_host_and_psf_4}")

        abmag_host_and_psf_err = abs(abmag_host_and_psf_1 - abmag_host_and_psf_4)

        if VERBOSE:
            print(f"{band}: abmag_host_and_psf_average: {abmag_host_and_psf_average}")
            print(f"{band}: abmag_host_and_psf_err: {abmag_host_and_psf_err}")
        flux_host_and_psf_err = utilities.abmag_err_to_flux_err(
            abmag_host_and_psf_average, abmag_host_and_psf_err, magzp=0, magzp_err=0
        )
        if VERBOSE:
            print(f"flux_jy_host_and_psf_average: {flux_jy_host_and_psf_average}")
            print(f"flux_host_and_psf_err: {flux_host_and_psf_err}")

        abmag_host_synthetic = utilities.magnitude_in_band(
            band=f"P200+{band}", spectrum=spectrum
        )
        flux_jy_host_synthetic = utilities.abmag_to_flux(abmag_host_synthetic, magzp=0)
        flux_jy_transient = flux_jy_host_and_psf_average - flux_jy_host_synthetic
        # flux_jy_transient_sextractor = (
        #     flux_jy_host_and_psf_sextractor - flux_jy_host_synthetic
        # )
        if VERBOSE:
            print(f"flux_jy_host_synthetic: {flux_jy_host_synthetic}")
            print(f"flux_jy_transient: {flux_jy_transient}")

        # print("#################################")
        # print(abmag_host_and_psf_average)
        # print(abmag_sextractor_both)
        # print(flux_jy_transient)
        # print(flux_jy_transient_sextractor)
        # print("#################################")

        abmag_transient = utilities.flux_to_abmag(flux_jy_transient, 0)
        abmag_err_transient = utilities.flux_err_to_abmag_err(
            flux_jy_transient, flux_host_and_psf_err
        )
        # abmag_transient_sextractor = utilities.flux_to_abmag(
        #     flux_jy_transient_sextractor, 0
        # )

        if VERBOSE:
            print(f"abmag_transient: {abmag_transient}")
            print(f"abmag_err_transient: {abmag_err_transient}")

        print(
            f"mjd: {DATES_MJD[i]} / band: {band} / abmag_avg: {abmag_transient:.2f} +/- {abmag_err_transient:.2f}"
        )
        # print(f"abmag_sextractor: {abmag_transient_sextractor:.2f}")

        ax1.errorbar(
            utilities.lambda_to_nu(filter_wl["P200+" + band]),
            abmag_host_and_psf_average,
            abmag_host_and_psf_err,
            marker=".",
            color=colors["P200"],
        )
        ax1.errorbar(
            utilities.lambda_to_nu(filter_wl["P200+" + band]),
            abmag_transient,
            abmag_err_transient,
            marker=".",
            color="black",
        )

        # ax1.scatter(
        #     utilities.lambda_to_nu(filter_wl["P200+" + band]),
        #     abmag_sextractor_both,
        #     marker="*",
        #     color=colors["P200"],
        # )
        # ax1.scatter(
        #     utilities.lambda_to_nu(filter_wl["P200+" + band]),
        #     abmag_transient_sextractor,
        #     marker="*",
        #     color="black",
        # )

    # for band in p200_bands:
    #     abmag_host2 = df_p200_galfit.loc[DATES_MJD[i]][f"{band}_abmag_host2"]
    #     ax1.scatter(
    #         utilities.lambda_to_nu(filter_wl["P200+"+band]),
    #         abmag_host2,
    #         marker="s",
    #         color="black"
    #     )

    for band in ps1_bands:
        bandlong = "PS1+" + band
        ax1.errorbar(
            utilities.lambda_to_nu(filter_wl["PS1+" + band]),
            df_ps1.query(f"band == '{band}'")["abmag"].values[0],
            df_ps1.query(f"band == '{band}'")["abmag_err"].values[0],
            marker=".",
            label=f"{bandlong} archival mean forced".replace("+", " "),
            # color=cmap[bandlong]
            color=colors["PS1"],
        )

    for band in sdss_bands:
        ax1.errorbar(
            utilities.lambda_to_nu(filter_wl["SDSS+" + band]),
            df_sdss.query(f"band == '{band}'")["mag_ab"].values[0],
            df_sdss.query(f"band == '{band}'")["e_mag"].values[0],
            marker=".",
            label=f"{bandlong} archival mean forced".replace("+", " "),
            color=colors["SDSS"],
        )

    for band in ukirt_bands:
        ukirt_mag_types = ["petroMag", "kronMag", "isoMag"]
        vegamag = df_ukirt.query(f"filterID == {ukirt_filterids[band]}")[
            ukirt_mag_types[1]
        ].values[0]
        abmag = ukirt_vega_to_ab("UKIRT+" + band, vegamag)
        ax1.errorbar(
            utilities.lambda_to_nu(filter_wl["UKIRT+" + band]),
            abmag,
            # df_ukirt.query(f"filterID == {ukirt_filterids[band]}")[ukirt_mag_types[2]+"Err"].values[0],
            marker=".",
            color=colors["UKIRT"],
        )

    for band in wise_bands:
        bandlong = "WISE+" + band
        ax1.scatter(
            utilities.lambda_to_nu(filter_wl[bandlong]),
            wise_values[bandlong],
            marker="d",
            s=13,
            color=colors["WISE"],
        )

    print("------------------------------------------------------")

    # ax1.set_xlim(1.17e14, 10e14)
    ax1.set_xlim(6e13, 10e14)
    ax1.set_ylim(20.8, 14.5)
    ax2 = ax1.secondary_xaxis(
        "top", functions=(utilities.nu_to_lambda, utilities.lambda_to_nu)
    )
    ax1.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE)
    ax2.set_xlabel(r"Wavelength [\AA]", fontsize=FONTSIZE)
    ax1.set_ylabel(r"Magnitude [AB]", fontsize=FONTSIZE)

    labels = [
        "WISE archival",
        "UKIRT archival",
        "SDSS archival",
        "PS1 archival",
        "P200 transient+host",
        "P200 transient only",
    ]
    telescope = ["WISE", "UKIRT", "SDSS", "PS1", "P200", "P200_ref"]
    color = ["tab:purple", "tab:red", "fuchsia", "lime", "tab:orange", "black"]
    # pos = [(6.8e13, 18.3), (1.4e14, 18.5), (5e14, 18.4), (3.4e14, 19.8)]
    xpos = 4.5e14
    ypos = 15
    delta = 0.6
    pos = [(xpos, ypos + delta * i) for i in range(6)]

    for i, label in enumerate(labels):
        plt.annotate(
            label,
            pos[i],
            bbox=dict(boxstyle="round", fc="w", ec=colors[telescope[i]]),
            color=colors[telescope[i]],
            fontsize=ANNOTATION_FONTSIZE,
        )

    plt.xscale("log")
    ax1.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKMARKS)

    if not os.path.exists(PLOTDIR):
        os.makedirs(PLOTDIR)

    outfile = os.path.join(PLOTDIR, f"tywin_host_spectrum_validation_{DATE}.png")
    plt.grid(which="both", alpha=0.15)
    # plt.legend(fontsize=FONTSIZE_LEGEND, loc=1)
    plt.tight_layout()
    fig.savefig(outfile)
