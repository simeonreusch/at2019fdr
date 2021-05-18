#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, argparse, time, json
import numpy as np
import pandas as pd
import matplotlib
from astropy import units as u
import matplotlib.pyplot as plt
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

MJD_INTERVALS = [[58700, 58720], [59006, 59032], [59110, 59130], [59220, 59265]]

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
PLOTDIR = os.path.join("plots", "broken_powerlaw")

if not os.path.exists(PLOTDIR):
    os.makedirs(PLOTDIR)


def load_info_json(filename: str):
    with open(os.path.join(INSTRUMENT_DATA_DIR, filename + ".json")) as json_file:
        outfile = json.load(json_file)
    return outfile


def broken_powerlaw_minimizer(params, x, data=None, data_err=None, **kwargs):
    """ """
    filter_wl = utilities.load_info_json("filter_wl")
    wl_filter = {v: k for k, v in filter_wl.items()}

    if "alpha1" in kwargs:
        alpha1 = kwargs["alpha1"]
    else:
        alpha1 = params["alpha1"]

    if "alpha2" in kwargs:
        alpha2 = kwargs["alpha2"]
    else:
        alpha2 = params["alpha2"]

    if "scale1" in params.keys():
        scale1 = params["scale1"]
    else:
        scale1 = None

    if "scale2" in params.keys():
        scale2 = params["scale2"]
    else:
        scale2 = None

    if "redshift" in params.keys():
        redshift = params["redshift"]
    else:
        redshift = None

    wavelengths, frequencies = utilities.get_wavelengths_and_frequencies()
    logflux_nu1 = np.log(frequencies.value) * alpha1 + scale1
    logflux_nu2 = np.log(frequencies.value) * alpha2 + scale2

    flux_nu1 = np.exp(logflux_nu1)
    flux_nu2 = np.exp(logflux_nu2)

    flux_nu = flux_nu1 + flux_nu2

    spectrum = sncosmo_spectral_v13.Spectrum(wave=wavelengths, flux=flux_nu, unit=FNU)

    ab_model_list = []
    flux_list = []

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
        return residual
    else:
        return ab_model_list


# BANDS_TO_EXCLUDE = ["P200+J", "P48+ZTF_g", "P48+ZTF_r", "P48+ZTF_i", "Swift+B", "Swift+V"]
# BANDS_TO_EXCLUDE = ["P200+J"]
BANDS_TO_EXCLUDE = []

for INTERVAL in [0, 1, 2, 3]:

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

    for index, row in df.iterrows():
        mags.append(row["mag"])
        mag_errs.append(row["mag_err"])
        instrumentband = row["instrument"] + "+" + row["band"]
        wls.append(filter_wl[instrumentband])

    params = Parameters()
    params.add("alpha1", value=-0.67, min=-10, max=0.01)
    params.add("alpha2", value=-5.74, min=-10, max=0.01)
    params.add("scale1", value=-40, min=-1e2, max=400)
    params.add("scale2", value=127, min=-1e2, max=400)
    # params.add("alpha1", value=-0.67, min=-100, max=100)
    # params.add("alpha2", value=-20, min=-100, max=100)
    # params.add("scale1", value=-40, min=-1e2, max=400)
    # params.add("scale2", value=127, min=-1e2, max=400)
    x = wls
    data = mags
    data_err = mag_errs

    minimizer_fcn = broken_powerlaw_minimizer

    minimizer = Minimizer(
        minimizer_fcn, params, fcn_args=(x, data, data_err), fcn_kws=None
    )
    out = minimizer.minimize()
    print(report_fit(out.params))

    alpha1 = out.params["alpha1"].value
    alpha2 = out.params["alpha2"].value
    scale1 = out.params["scale1"].value
    scale2 = out.params["scale2"].value

    wavelengths, frequencies = utilities.get_wavelengths_and_frequencies()
    logflux_nu1 = np.log(frequencies.value) * alpha1 + scale1
    logflux_nu2 = np.log(frequencies.value) * alpha2 + scale2
    flux_nu1 = np.exp(logflux_nu1)
    flux_nu2 = np.exp(logflux_nu2)
    flux_nu = flux_nu1 + flux_nu2
    fitted_total_spectrum = sncosmo_spectral_v13.Spectrum(
        wave=wavelengths, flux=flux_nu, unit=FNU
    )
    fitted_spectrum_1 = sncosmo_spectral_v13.Spectrum(
        wave=wavelengths, flux=flux_nu1, unit=FNU
    )
    fitted_spectrum_2 = sncosmo_spectral_v13.Spectrum(
        wave=wavelengths, flux=flux_nu2, unit=FNU
    )

    # # Calculate luminosity
    total_luminosity = utilities.calculate_luminosity(
        fitted_total_spectrum,
        wl_min=utilities.lambda_to_nu(1e14),
        wl_max=utilities.lambda_to_nu(2e15),
        redshift=REDSHIFT,
    )
    print(f"total luminosity = {total_luminosity:.2e}")

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
        ax1.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE_LABEL)
        ax1.set_xlim([1e14, 2e15])
        ax1.set_ylim([9e-14, 1e-11])
        # ax1.set_ylim([9e-16, 1e-11])
        plt.yscale("log")
        for band in df.band:
            df_red = df.query(f"band == '{band}'")
            key = (df_red.instrument.values + "+" + df_red.band.values)[0]
            nu = utilities.lambda_to_nu(filter_wl[key])

            ax1.errorbar(
                nu,
                df_red.flux.values * nu,
                df_red.flux_err.values * nu,
                color=cmap[key],
                label=filterlabel[key],
                fmt=".",
                markersize=10,
            )
        nu = utilities.lambda_to_nu(fitted_spectrum_1.wave)
        ax1.plot(
            nu,
            fitted_spectrum_1.flux * nu,
            color="tab:blue",
            linestyle="dotted",
            label=rf"spectrum 1",
        )
        nu = utilities.lambda_to_nu(fitted_spectrum_2.wave)
        ax1.plot(
            nu,
            fitted_spectrum_2.flux * nu,
            color="tab:red",
            linestyle="dotted",
            label=rf"spectrum 2",
        )
        nu = utilities.lambda_to_nu(fitted_total_spectrum.wave)
        ax1.plot(
            nu,
            fitted_total_spectrum.flux * nu,
            color="purple",
            # linestyle="dotted",
            label="sum of fitted spectra",
            # label=rf"$\alpha$ optical/UV={fitparams['alpha1']:.2f}",
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
        ax1.plot(
            fitted_spectrum_2.wave,
            utilities.flux_to_abmag(fitted_spectrum_2.flux),
            color="yellowgreen",
            linestyle="dotted",
            label="spectrum 2",
        )
        ax1.plot(
            fitted_total_spectrum.wave,
            utilities.flux_to_abmag(fitted_total_spectrum.flux),
            color="purple",
            # linestyle="dotted",
            label="sum of fitted spectra",
        )
        ax2 = ax1.secondary_xaxis(
            "top", functions=(utilities.nu_to_lambda, utilities.lambda_to_nu)
        )
        ax2.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE_LABEL)

    ax1.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKMARKS)
    ax2.tick_params(axis="y", which="major", labelsize=FONTSIZE_TICKMARKS)

    bbox = dict(boxstyle="round", fc="none", ec="black")

    if not os.path.exists(PLOTDIR):
        os.makedirs(PLOTDIR)

    if plotmag:
        outfile = os.path.join(PLOTDIR, f"broken_mag_{INTERVAL}.png")
    else:
        outfile = os.path.join(PLOTDIR, f"broken_nufnu_{INTERVAL}.png")
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
