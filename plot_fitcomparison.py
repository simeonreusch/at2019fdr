#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, argparse, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from modelSED import utilities, fit, sncosmo_spectral_v13
from modelSED.utilities import broken_powerlaw_spectrum, FNU

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)

# from modelSED.fit import powerlaw_minimizer
from lmfit import Model, Parameters, Minimizer, report_fit, minimize

P200_MAGNITUDE_FILE = os.path.join(
    "data", "P200_NIR_observations_2020_07_01", "subtracted_magnitudes.json"
)
ZTF_SWIFT_MAGNITUDE_FILE = os.path.join("data", "ztf_swift_mags_late_epoch.json")
FITPARAMS_FILE_POWERLAW = os.path.join("data", "fitparams_late_epoch.json")
FITPARAMS_FILE_BB = os.path.join("fit", "blackbody.json")
MJD = 59005.12846212167
REDSHIFT = 0.2666
FONTSIZE = 14
FONTSIZE_LEGEND = 11
ANNOTATION_FONTSIZE = 11
FONTSIZE_TICKMARKS = 11
FIG_WIDTH = 6
DPI = 400
INSTRUMENT_DATA_DIR = "instrument_data"
PLOTDIR = "plots"
FIT_DATA_DIR = "fit"


def load_info_json(filename: str):
    with open(os.path.join(INSTRUMENT_DATA_DIR, filename + ".json")) as json_file:
        outfile = json.load(json_file)
    return outfile


def broken_powerlaw_minimizer(params, x, data=None, **kwargs):
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

    if data:
        print(np.asarray(ab_model_list) - np.asarray(data))
        return np.asarray(ab_model_list) - np.asarray(data)
    else:
        return ab_model_list


with open(P200_MAGNITUDE_FILE, "r") as read_file:
    magnitudes_p200 = json.load(read_file)

with open(ZTF_SWIFT_MAGNITUDE_FILE, "r") as read_file:
    magnitudes_ztf_swift = json.load(read_file)

magnitudes = {}

magnitudes.update(magnitudes_p200)
magnitudes.update(magnitudes_ztf_swift)

with open(FITPARAMS_FILE_POWERLAW, "r") as read_file:
    fitparams_powerlaw = json.load(read_file)

with open(FITPARAMS_FILE_BB, "r") as read_file:
    fitparams_bb = json.load(read_file)

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


fitted_optical_spectrum = utilities.powerlaw_spectrum(
    alpha=fitparams_powerlaw["alpha"],
    scale=fitparams_powerlaw["scale"],
    redshift=None,
    extinction_av=None,
    extinction_rv=None,
)

# bb_temp = 18216.905876650806
# bb_scale = 1.2604073367936622e23
# bb_extinction_av = 1.6896748383322513
# bb_extinction_rv = 3.9999999999873292

mjds_fitted = [fitparams_bb[entry]["mjd"] for entry in fitparams_bb]
mjds_dist = [abs(entry - MJD) for entry in mjds_fitted]
mjd = mjds_fitted[np.argmin(mjds_dist)]

for entry in fitparams_bb:
    if fitparams_bb[entry]["mjd"] == mjd:
        bb_temp = fitparams_bb[entry]["temperature"]
        bb_scale = fitparams_bb[entry]["scale"]
        bb_extinction_av = fitparams_bb[entry]["extinction_av"]
        bb_extinction_rv = fitparams_bb[entry]["extinction_rv"]


bb_spectrum = utilities.blackbody_spectrum(
    temperature=bb_temp,
    scale=bb_scale,
    redshift=REDSHIFT,
    extinction_av=bb_extinction_av,
    extinction_rv=bb_extinction_rv,
)
powerlaw_scale = 7.936243137091513e-16
powerlaw_alpha = -0.8184778778364405
powerlaw_spectrum = utilities.powerlaw_spectrum(
    alpha=powerlaw_alpha, scale=powerlaw_scale
)

ir_bb_specrum = utilities.blackbody_spectrum(
    temperature=1160,
    scale=bb_scale / 25000,
    redshift=0,
)

# Now we plot
###
plotmag = False
###


plt.figure(figsize=(FIG_WIDTH, 1 / 1.414 * FIG_WIDTH), dpi=DPI)
ax1 = plt.subplot(111)
plt.xscale("log")


texts = ["powerlaw fit", "blackbody fit"]
position = [(1.1e14, 3.8e-13), ((2.4e14, 1.8e-13))]
colors = ["tab:red", "tab:blue"]
for i, text in enumerate(texts):
    plt.annotate(
        text,
        position[i],
        bbox=dict(boxstyle="round", fc="w", ec=colors[i]),
        color=colors[i],
        fontsize=ANNOTATION_FONTSIZE,
    )

if not plotmag:
    ax1.set_ylabel(r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=FONTSIZE)
    ax1.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE)
    ax1.set_xlim([1e14, 2e15])
    ax1.set_ylim([1e-13, 1e-11])
    plt.yscale("log")
    for band in df.band:
        df_red = df.query(f"band == '{band}'")
        key = (df_red.instrument.values + "+" + df_red.band.values)[0]
        flux_density = df_red.flux.values
        flux_density_err = df_red.flux_err.values
        flux, flux_err = utilities.flux_density_to_flux(
            filter_wl[key], flux_density, flux_density_err
        )
        ax1.errorbar(
            utilities.lambda_to_nu(filter_wl[key]),
            flux,
            flux_err,
            color=cmap[key],
            label=filterlabel[key],
            fmt=".",
            markersize=10,
        )
    flux_density_bb = bb_spectrum.flux
    flux_density_powerlaw = powerlaw_spectrum.flux
    flux_density_ir = ir_bb_specrum.flux

    wl_bb_lambda = bb_spectrum.wave
    wl_powerlaw_lambda = powerlaw_spectrum.wave
    wl_ir_lambda = ir_bb_specrum.wave

    wl_bb_nu = utilities.lambda_to_nu(wl_bb_lambda)
    wl_powerlaw_nu = utilities.lambda_to_nu(wl_powerlaw_lambda)
    wl_ir_nu = utilities.lambda_to_nu(wl_ir_lambda)

    flux_bb = utilities.flux_density_to_flux(
        wl_bb_lambda,
        flux_density_bb,
    )
    flux_powerlaw = utilities.flux_density_to_flux(
        wl_powerlaw_lambda,
        flux_density_powerlaw,
    )
    flux_ir = utilities.flux_density_to_flux(
        wl_ir_lambda,
        flux_density_ir,
    )

    ax1.plot(
        wl_powerlaw_nu,
        flux_powerlaw,
        color="tab:red",
        # label=rf"powerlaw fit spectrum",
    )
    ax1.plot(
        wl_bb_nu,
        flux_bb,
        color="tab:blue",
        # label=rf"BB optical spectrum",
    )
    # ax1.plot(
    #     wl_ir_nu,
    #     flux_ir,
    #     color="tab:green",
    #     # label=rf"IR BB dust spectrum",
    # )
    # ax1.plot(
    #     wl_ir_nu,
    #     flux_ir+flux_bb,
    #     color="tab:green",
    #     # label=rf"IR BB dust + Optical BB spectrum",
    # )

    ax2 = ax1.secondary_xaxis("top", functions=(utilities.nu_to_ev, utilities.ev_to_nu))
    ax2.set_xlabel("Energy [eV]", fontsize=FONTSIZE)

    d = cosmo.luminosity_distance(REDSHIFT)
    d = d.to(u.cm).value
    lumi = lambda flux: flux * 4 * np.pi * d ** 2
    flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
    ax3 = ax1.secondary_yaxis("right", functions=(lumi, flux))
    ax3.tick_params(axis="y", which="major", labelsize=FONTSIZE_TICKMARKS)
    ax3.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=FONTSIZE)

else:
    ax1.set_ylabel("Magnitude [AB]", fontsize=FONTSIZE)
    ax1.set_xlabel(r"Wavelength [Å]", fontsize=FONTSIZE)
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
        bb_spectrum.wave,
        utilities.flux_to_abmag(bb_spectrum.flux),
        color="tab:blue",
        # linestyle="dotted",
        label="blackbody fit spectrum",
    )
    ax1.plot(
        powerlaw_spectrum.wave,
        utilities.flux_to_abmag(powerlaw_spectrum.flux),
        color="tab:red",
        # linestyle="dotted",
        label="powerlaw fit spectrum",
    )
    ax2 = ax1.secondary_xaxis(
        "top", functions=(utilities.nu_to_lambda, utilities.lambda_to_nu)
    )
    ax2.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE)

ax1.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKMARKS)
ax2.tick_params(axis="y", which="major", labelsize=FONTSIZE_TICKMARKS)

bbox = dict(boxstyle="round", fc="none", ec="black")

plt.grid(which="both", alpha=0.15)

if not os.path.exists(PLOTDIR):
    os.makedirs(PLOTDIR)

if plotmag:
    outfile = os.path.join(PLOTDIR, "fitcomparison_mag.png")
else:
    outfile = os.path.join(PLOTDIR, "fitcomparison_flux.png")
plt.legend(fontsize=FONTSIZE_LEGEND)
plt.tight_layout()
plt.savefig(outfile)
plt.close()
