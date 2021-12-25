#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, argparse, time, json
import numpy as np
import pandas as pd
import matplotlib
from astropy import units as u
import matplotlib.pyplot as plt
import json
from modelSED import utilities, fit, sncosmo_spectral_v13
from modelSED.utilities import broken_powerlaw_spectrum, FNU
from astropy.cosmology import Planck15 as cosmo

# from modelSED.fit import powerlaw_minimizer
from lmfit import Model, Parameters, Minimizer, report_fit, minimize

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)

LIGHTCURVE_INFILE = os.path.join("data", "lightcurves", "full_lightcurve.csv")

# MJD_INTERVALS = [[58700, 58720], [59006, 59032], [59110, 59130], [59220,59265]]
MJD_INTERVALS = [[58700, 58720], [59006, 59130], [59220, 59271]]

# FITPARAMS_FILE = os.path.join("data", "fitparams_late_epoch.json")
MJD = 59005.12846212167
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

## EXTINCTION FROM EPOCH 1
# GLOBAL_AV = 1.48477495
# GLOBAL_RV = 3.93929588

## EXTINCTION FROM EPOCH 0
GLOBAL_AV = 0.3643711523794127
GLOBAL_RV = 4.2694173002543225

FITMETHOD = "lm"

REFIT = True
FIT = 3
INTERVALS = [0]
EXTINCTIONFIT_INTERVAL = 4


for directory in [PLOTDIR, FITDIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_info_json(filename: str):
    with open(os.path.join(INSTRUMENT_DATA_DIR, filename + ".json")) as json_file:
        outfile = json.load(json_file)
    return outfile


def double_blackbody_minimizer(params, x, data=None, data_err=None, **kwargs):
    """ """
    filter_wl = utilities.load_info_json("filter_wl")

    wl_filter = {v: k for k, v in filter_wl.items()}

    temp1 = params["temp1"]
    scale1 = params["scale1"]
    if FIT == 3:
        temp2 = params["temp2"]
        scale2 = params["scale2"]
    if "extinction_av" in params:
        extinction_av = params["extinction_av"]
    elif INTERVAL != EXTINCTIONFIT_INTERVAL:
        extinction_av = GLOBAL_AV
    else:
        extinction_av = None

    if "extinction_rv" in params:
        extinction_rv = params["extinction_rv"]
    elif INTERVAL != EXTINCTIONFIT_INTERVAL:
        extinction_rv = GLOBAL_RV
    else:
        extinction_rv = None

    redshift = REDSHIFT

    spectrum1 = utilities.blackbody_spectrum(
        temperature=temp1,
        scale=scale1,
        extinction_av=extinction_av,
        extinction_rv=extinction_rv,
        redshift=redshift,
    )

    if FIT == 3:
        spectrum2 = utilities.blackbody_spectrum(
            temperature=temp2,
            scale=scale2,
            extinction_av=extinction_av,
            extinction_rv=extinction_rv,
            redshift=redshift,
        )

    ab_model_list = []
    flux_list = []

    flux1 = spectrum1.flux
    if FIT == 3:
        flux2 = spectrum2.flux
    else:
        flux2 = 0

    fluxcomb = flux1 + flux2
    spectrum = sncosmo_spectral_v13.Spectrum(
        wave=spectrum1.wave, flux=fluxcomb, unit=FNU
    )

    for i in x:
        ab_model = utilities.magnitude_in_band(wl_filter[i], spectrum)
        flux = utilities.abmag_to_flux(ab_model)
        ab_model_list.append(ab_model)
        flux_list.append(flux)

    if "flux" in kwargs.keys():
        if data:
            return np.asarray(flux_list) - np.asarray(data)
        else:
            return flux_list

    if data and not data_err:
        residual = np.asarray(ab_model_list) - np.asarray(data)
        print(residual)
        return residual
    elif data_err:
        residual = (np.asarray(ab_model_list) - np.asarray(data)) / np.asarray(data_err)
        print(residual)
        print(np.mean(abs(residual)))
        print("-------------------------------------------")
        return residual
    else:
        return ab_model_list


# BANDS_TO_EXCLUDE = ["P200+J", "P48+ZTF_g", "P48+ZTF_r", "P48+ZTF_i", "Swift+B", "Swift+V"]
# BANDS_TO_EXCLUDE = ["P200+J"]
BANDS_TO_EXCLUDE = [
    "P200_sextractor+J",
    "P200_sextractor+H",
    "P200_sextractor+Ks",
    "Swift+B",
    "Swift+U",
    "Swift+V",
]
BANDS_TO_FIT_BB_1 = ["P48+ZTF_g", "P48+ZTF_r", "P48+ZTF_i", "Swift+UVM2"]
BANDS_TO_FIT_BB_2 = ["P200+J", "P200+H", "P200+Ks", "WISE+W1", "WISE+W2"]

for INTERVAL in INTERVALS:
    FITFILENAMES = {
        1: os.path.join(FITDIR, f"{INTERVAL}_fitparams_optical_uv.json"),
        2: os.path.join(FITDIR, f"{INTERVAL}_fitparams_infrared.json"),
        3: os.path.join(FITDIR, f"{INTERVAL}_fitparams_all.json"),
    }

    magnitudes = {}

    df = pd.read_csv(LIGHTCURVE_INFILE)

    df_cut = df.query(
        f"obsmjd > {MJD_INTERVALS[INTERVAL][0]} and obsmjd < {MJD_INTERVALS[INTERVAL][1]}"
    )

    df_cut["telescope_band"] = df_cut.telescope + "+" + df_cut.band

    for tband in df_cut["telescope_band"].unique():
        if tband not in BANDS_TO_EXCLUDE:
            _df = df_cut.query(f"telescope_band == @tband")
            magnitudes.update(
                {tband: [np.mean(_df.mag.values), np.mean(_df.mag_err.values)]}
            )

    columns = ["instrument", "band", "mag", "mag_err"]
    df = pd.DataFrame(columns=columns)
    instrument = []
    band = []
    mag = []
    mag_err = []
    for index, entry in enumerate(magnitudes):
        mag.append(magnitudes[entry][0])
        mag_err.append(magnitudes[entry][1])
        instrument.append(entry.split("+")[0])
        band.append(entry.split("+")[1])

    df["instrument"] = instrument
    df["band"] = band
    df["mag"] = mag
    df["mag_err"] = mag_err
    df["flux"] = utilities.abmag_to_flux(df.mag)
    df["flux_err"] = utilities.abmag_err_to_flux_err(df.mag, df.mag_err)

    filter_wl = load_info_json("filter_wl")
    cmap = load_info_json("cmap")
    filterlabel = load_info_json("filterlabel")

    # Now fit the sum of two spectra
    mags = []
    mag_errs = []
    wls = []

    df["instrumentband"] = df["instrument"] + "+" + df["band"]

    if FIT == 1 or FIT == 2:
        df_fit = df.query(f"instrumentband in @BANDS_TO_FIT_BB_{FIT}")
    else:
        df_fit = df

    for index, row in df_fit.iterrows():
        mags.append(row["mag"])
        mag_errs.append(row["mag_err"])
        instrumentband = row["instrument"] + "+" + row["band"]
        wls.append(filter_wl[instrumentband])

    params = Parameters()
    params.add("temp1", value=14000, min=100, max=150000)
    params.add("scale1", value=1e23, min=1e18, max=1e27)

    if FIT == 3:
        params.add("temp2", value=1400, min=100, max=150000)
        params.add("scale2", value=1e20, min=1e18, max=1e27)
    if (FIT == 1 or FIT == 3) and INTERVAL == EXTINCTIONFIT_INTERVAL:
        params.add("extinction_av", value=0.1, min=0.000000001, max=4)
        params.add("extinction_rv", value=3.1, min=1, max=5)

    x = wls
    data = mags
    data_err = mag_errs

    minimizer_fcn = double_blackbody_minimizer

    if REFIT:
        minimizer = Minimizer(
            minimizer_fcn, params, fcn_args=(x, data, data_err), fcn_kws=None
        )
        out = minimizer.minimize(method=FITMETHOD)
        print(report_fit(out.params))

        temp1 = out.params["temp1"].value
        scale1 = out.params["scale1"].value

        if "extinction_av" in out.params.keys():
            extinction_av = out.params["extinction_av"].value
        else:
            extinction_av = None
        if "extinction_rv" in out.params.keys():
            extinction_rv = out.params["extinction_rv"].value
        else:
            extinction_rv = None
        if "temp2" in out.params.keys():
            temp2 = out.params["temp2"].value
        else:
            temp2 = None
        if "scale2" in out.params.keys():
            scale2 = out.params["scale2"].value
        else:
            scale2 = None

        fitresult = {
            "temp1": temp1,
            "scale1": scale1,
            "temp2": temp2,
            "scale2": scale2,
            "extinction_av": extinction_av,
            "extinction_rv": extinction_rv,
        }

        with open(FITFILENAMES[FIT], "w") as outfile:
            json.dump(fitresult, outfile)

    else:
        with open(FITFILENAMES[FIT]) as infile:
            fitresult = json.load(infile)

    wavelengths, frequencies = utilities.get_wavelengths_and_frequencies()

    if INTERVAL == EXTINCTIONFIT_INTERVAL:
        extinction_av = fitresult["extinction_av"]
        extinction_rv = fitresult["extinction_rv"]
    else:
        extinction_av = GLOBAL_AV
        extinction_rv = GLOBAL_RV

    fitted_spectrum_1, bolo_flux_1 = utilities.blackbody_spectrum(
        temperature=fitresult["temp1"],
        scale=fitresult["scale1"],
        extinction_av=extinction_av,
        extinction_rv=extinction_rv,
        redshift=REDSHIFT,
        get_bolometric_flux=True,
    )
    unextincted_spectrum_1, bolo_flux_1_unext = utilities.blackbody_spectrum(
        temperature=fitresult["temp1"],
        scale=fitresult["scale1"],
        extinction_av=None,
        extinction_rv=None,
        redshift=None,
        get_bolometric_flux=True,
    )

    if FIT == 3:
        fitted_spectrum_2, bolo_flux_2 = utilities.blackbody_spectrum(
            temperature=fitresult["temp2"],
            scale=fitresult["scale2"],
            extinction_av=extinction_av,
            extinction_rv=extinction_rv,
            redshift=REDSHIFT,
            get_bolometric_flux=True,
        )
        unextincted_spectrum_2, bolo_flux_2_unext = utilities.blackbody_spectrum(
            temperature=fitresult["temp2"],
            scale=fitresult["scale2"],
            extinction_av=None,
            extinction_rv=None,
            redshift=None,
            get_bolometric_flux=True,
        )

        combined_flux = fitted_spectrum_1.flux + fitted_spectrum_2.flux

        combined_spectrum = sncosmo_spectral_v13.Spectrum(
            wave=fitted_spectrum_1.wave, flux=combined_flux, unit=FNU
        )

    # # # Calculate luminosity
    luminosity_1, _, radius1, _ = utilities.calculate_bolometric_luminosity(
        temperature=fitresult["temp1"],
        scale=fitresult["scale1"],
        redshift=REDSHIFT,
        temperature_err=None,
        scale_err=None,
    )
    luminosity_2, _, radius2, _ = utilities.calculate_bolometric_luminosity(
        temperature=fitresult["temp2"],
        scale=fitresult["scale2"],
        redshift=REDSHIFT,
        temperature_err=None,
        scale_err=None,
    )
    total_luminosity = luminosity_1 + luminosity_2

    print("--------------------------------")
    print(f"temp optical/UV: {fitresult['temp1']:.0f} K")
    print(f"temp infrared: {fitresult['temp2']:.0f} K")
    print(f"luminosity optical/UV = {luminosity_1:.2e}")
    print(f"luminosity infrared = {luminosity_2:.2e}")
    print(f"total luminosity = {total_luminosity:.2e}")
    print(f"radius optical/UV = {radius1:.2e}")
    print(f"radius infrared = {radius2:.2e}")
    print("--------------------------------")

    # Now we plot
    ###
    plotmag = False
    ###

    plt.figure(figsize=(FIG_WIDTH, 1 / 1.414 * FIG_WIDTH), dpi=DPI)
    ax1 = plt.subplot(111)
    plt.xscale("log")

    if not plotmag:
        ax1.set_ylabel(
            r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=FONTSIZE_LABEL
        )
        ax1.set_xlabel("Frequency [Hz] (source frame)", fontsize=FONTSIZE_LABEL)
        ax1.set_xlim([5e13, 2e15])
        ax1.set_ylim([9e-14, 1e-11])
        # ax1.set_ylim([9e-16, 1e-11])
        plt.yscale("log")
        for band in df.band:
            df_red = df.query(f"band == '{band}'")
            key = (df_red.instrument.values + "+" + df_red.band.values)[0]
            nu = utilities.lambda_to_nu(filter_wl[key])

            ax1.errorbar(
                nu * (1 + REDSHIFT),
                df_red.flux.values * nu * (1 + REDSHIFT),
                df_red.flux_err.values * nu * (1 + REDSHIFT),
                color=cmap[key],
                label=filterlabel[key],
                fmt=".",
                markersize=10,
            )
        nu = utilities.lambda_to_nu(fitted_spectrum_1.wave)

        # OPTICAL / UV
        ax1.plot(
            utilities.lambda_to_nu(fitted_spectrum_1.wave) * (1 + REDSHIFT),
            fitted_spectrum_1.flux
            * utilities.lambda_to_nu(fitted_spectrum_1.wave)
            * (1 + REDSHIFT),
            color="tab:blue",
            linestyle="dotted",
            label=f"1 extincted",
        )
        ax1.plot(
            utilities.lambda_to_nu(unextincted_spectrum_1.wave),
            unextincted_spectrum_1.flux
            * utilities.lambda_to_nu(unextincted_spectrum_1.wave),
            color="tab:blue",
            linestyle="dotted",
            linewidth=0.6,
            label=f"1 unextincted",
        )

        if FIT == 3:
            ax1.plot(
                utilities.lambda_to_nu(fitted_spectrum_2.wave) * (1 + REDSHIFT),
                fitted_spectrum_2.flux
                * utilities.lambda_to_nu(fitted_spectrum_2.wave)
                * (1 + REDSHIFT),
                color="tab:red",
                linestyle="dotted",
                label=f"2 extincted",
            )
            ax1.plot(
                utilities.lambda_to_nu(unextincted_spectrum_2.wave),
                unextincted_spectrum_2.flux
                * utilities.lambda_to_nu(unextincted_spectrum_2.wave),
                color="tab:red",
                linestyle="dotted",
                linewidth=0.6,
                label=f"2 unextincted",
            )
            ax1.plot(
                utilities.lambda_to_nu(combined_spectrum.wave) * (1 + REDSHIFT),
                combined_spectrum.flux
                * utilities.lambda_to_nu(combined_spectrum.wave)
                * (1 + REDSHIFT),
                color="black",
                # linestyle="dotted",
                label=rf"combined spectrum",
            )

        ax2 = ax1.secondary_xaxis(
            "top", functions=(utilities.nu_to_ev, utilities.ev_to_nu)
        )
        ax2.set_xlabel(r"Energy [eV]", fontsize=FONTSIZE_LABEL)
        plt.grid(which="both", alpha=0.15)

        d = cosmo.luminosity_distance(REDSHIFT)
        d = d.to(u.cm).value
        lumi = lambda flux: flux * 4 * np.pi * d ** 2
        flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
        ax3 = ax1.secondary_yaxis("right", functions=(lumi, flux))
        ax3.tick_params(axis="y", which="major", labelsize=FONTSIZE_TICKMARKS)
        ax3.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=FONTSIZE_LABEL)

    else:
        ax1.set_ylabel("Magnitude [AB]", fontsize=FONTSIZE_LABEL)
        ax1.set_xlabel(r"Wavelength $[\AA]$", fontsize=FONTSIZE_LABEL)
        ax1.invert_yaxis()
        ax1.set_ylim([21, 15])

        for band in df.band:
            df_red = df.query(f"band == '{band}'")
            key = (df_red.instrument.values + "+" + df_red.band.values)[0]
            ax1.errorbar(
                filter_wl[key],
                df_red.mag.values,
                df_red.mag_err.values,
                color=cmap[key],
                fmt=".",
                label=filterlabel[key],
                markersize=10,
            )
        ax1.plot(
            fitted_spectrum_1.wave,
            utilities.flux_to_abmag(fitted_spectrum_1.flux),
            color="darkcyan",
            linestyle="dotted",
            label="spectrum 1",
        )

        # ax1.plot(
        #     fitted_spectrum_2.wave,
        #     utilities.flux_to_abmag(fitted_spectrum_2.flux),
        #     color="yellowgreen",
        #     linestyle="dotted",
        #     label="spectrum 2",
        # )
        # ax1.plot(
        #     fitted_total_spectrum.wave,
        #     utilities.flux_to_abmag(fitted_total_spectrum.flux),
        #     color="purple",
        #     # linestyle="dotted",
        #     label="sum of fitted spectra",
        # )
        ax2 = ax1.secondary_xaxis(
            "top", functions=(utilities.nu_to_lambda, utilities.lambda_to_nu)
        )
        ax2.set_xlabel("Frequency [Hz] (source frame)", fontsize=FONTSIZE_LABEL)

    ax1.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKMARKS)
    ax2.tick_params(axis="y", which="major", labelsize=FONTSIZE_TICKMARKS)

    bbox = dict(boxstyle="round", fc="w", ec="gray")
    annotation = f"lumin. opt/UV: {luminosity_1:.2e}\nlumin. IR: {luminosity_2:.2e}\nlumin. total: {total_luminosity:.2e}"
    ax1.text(
        0.35,
        0.9,
        annotation,
        transform=ax1.transAxes,
        bbox=bbox,
        fontsize=FONTSIZE_LEGEND,
    )

    if not os.path.exists(PLOTDIR):
        os.makedirs(PLOTDIR)

    if plotmag:
        outfile = os.path.join(PLOTDIR, f"double_bb_mag_{INTERVAL}_sourceframe.png")
    else:
        outfile = os.path.join(PLOTDIR, f"double_bb_nufnu_{INTERVAL}_sourceframe.png")

    loc = {0: "upper left", 1: "upper right", 2: "upper right"}

    plt.legend(fontsize=FONTSIZE_LEGEND, loc=loc[INTERVAL])
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
