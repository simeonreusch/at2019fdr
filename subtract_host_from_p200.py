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

from modelSED import utilities, sncosmo_spectral_v13

XRTCOLUMN = "flux0310_pi_-2"
nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)


def vega_to_ab(band, mag):
    diff = {"P200+J": 0.91, "P200+H": 1.39, "P200+Ks": 1.85}
    print(mag)
    print(mag + diff[band])
    return mag + diff[band]


DATE = "2020_07_01"
# DATE = "2020_09_29"
REDSHIFT = 0.267
FONTSIZE = 14
FONTSIZE_LEGEND = 14
ANNOTATION_FONTSIZE = 14
FONTSIZE_TICKMARKS = 12
DPI = 400
PLOTDIR = "plots"
DATA_DIR = os.path.join("data", f"P200_NIR_observations_{DATE}")


P200_OBS_FILE = os.path.join(DATA_DIR, "unsubtracted_magnitudes.json")

infile = os.path.join("data", "host_model", "Tywin_parasfh_spectrum_with_WISE.dat")

host_spectrum = pd.read_table(
    infile, names=["wl", "flux", "abmag"], sep="\s+", comment="#"
)

with open(P200_OBS_FILE, "r") as read_file:
    p200_observed = json.load(read_file)

host_spectrum["mag"] = utilities.flux_to_abmag(flux_nu=host_spectrum.flux, flux_nu_zp=0)


filter_wl = utilities.load_info_json("filter_wl")
cmap = utilities.load_info_json("cmap")
filterlabel = utilities.load_info_json("filterlabel")

p200_bands = ["P200+J", "P200+H", "P200+Ks"]

all_bands = ["P200+J", "P200+H", "P200+Ks", "P48+ZTF_g", "P48+ZTF_r", "P48+ZTF_i"]


# Now we get magnitudes using bandpasses
# First we need to construct a proper spectrum
spectrum = sncosmo_spectral_v13.Spectrum(
    wave=host_spectrum.wl.values,
    flux=host_spectrum.flux.values * 3.631e-20,
    unit=utilities.FNU,
)

p200_host = {}
all_host = {}

for band in all_bands:
    abmag = utilities.magnitude_in_band(band=band, spectrum=spectrum)
    all_host.update({band: abmag})

for band in p200_bands:
    abmag = utilities.magnitude_in_band(band=band, spectrum=spectrum)
    p200_host.update({band: abmag})

# Now we calculate the fluxes for observation and host-model, subtract them
# and calculate the subtracted magnitude
p200_subtracted = {}

for band in p200_host:
    fluxmaggie_host = utilities.abmag_to_flux(p200_host[band], magzp=0)
    fluxmaggie_observed = utilities.abmag_to_flux(p200_observed[band][0], magzp=0)
    fluxmaggie_observed = utilities.abmag_to_flux(
        vega_to_ab(band, p200_observed[band][0]), magzp=0
    )
    fluxmaggie_diff = fluxmaggie_observed - fluxmaggie_host
    print(f"{band} host mag: {utilities.flux_to_abmag(fluxmaggie_host, 0)}")
    print(f"{band} obs mag: {utilities.flux_to_abmag(fluxmaggie_observed, 0)}")
    abmag_after_subtraction = utilities.flux_to_abmag(fluxmaggie_diff, 0)
    p200_subtracted.update({band: abmag_after_subtraction})

bandpassfiles = utilities.load_info_json("bandpassfiles")


# Now we plot this
fig, ax1 = plt.subplots(1, 1, figsize=[6, 6 / 1.414], dpi=DPI)
# fig.suptitle("AT2019fdr host model spectrum, NIR observations")
ax1.plot(
    utilities.lambda_to_nu(host_spectrum.wl),
    utilities.abmag_to_flux(host_spectrum.mag),
    label="modeled Host spectrum",
    color="tab:blue",
)

# ax1.invert_yaxis()
# ax1.set_xlim(1.4e3, 2e5)
# ax1.set_ylim(27.5, 14)
ax1.set_xlim(1.17e14, 3e14)
ax1.set_ylim(1e-27, 4e-26)
ax1.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE)
ax1.set_ylabel(r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=FONTSIZE)

for band in p200_bands:
    wl = filter_wl[band]
    abmag = p200_observed[band][0]
    abmag_err = p200_observed[band][1]
    ax1.errorbar(
        x=utilities.lambda_to_nu(wl),
        y=utilities.abmag_to_flux(abmag),
        yerr=utilities.abmag_err_to_flux_err(abmag, abmag_err),
        color=cmap[band],
        # linewidths=1,
        marker="s",
        label=f"{filterlabel[band]} observed",
        # edgecolors="black",
        # size=42,
    )

    wl = filter_wl[band]
    abmag_sub = p200_subtracted[band]
    abmag_sub_err = p200_observed[band][1]
    ax1.errorbar(
        x=utilities.lambda_to_nu(wl),
        y=utilities.abmag_to_flux(abmag_sub),
        yerr=utilities.abmag_err_to_flux_err(abmag_sub, abmag_sub_err),
        color=cmap[band],
        label=f"{filterlabel[band]} subtracted",
        marker="o",
        # linewidths=1,
        # edgecolors="black",
        # size=42,
    )

    bbox = dict(boxstyle="round", fc="1", color=cmap[band])

    ax1.annotate(
        filterlabel[band],
        (1.04 * utilities.lambda_to_nu(wl), utilities.abmag_to_flux(abmag)),
        fontsize=ANNOTATION_FONTSIZE,
        bbox=bbox,
        color=cmap[band],
    )

bbox = dict(boxstyle="round", fc="1", color="tab:blue")
ax1.text(
    0.08,
    0.18,
    "Synthetic host spectrum",
    transform=ax1.transAxes,
    fontsize=ANNOTATION_FONTSIZE,
    bbox=bbox,
    color="tab:blue",
)

plt.xscale("log")
plt.yscale("log")

d = cosmo.luminosity_distance(REDSHIFT)
d = d.to(u.cm).value
lumi = lambda flux: flux * 4 * np.pi * d ** 2
flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
ax2 = ax1.secondary_yaxis("right", functions=(lumi, flux))
ax2.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=FONTSIZE)
ax3 = ax1.secondary_xaxis("top", functions=(utilities.nu_to_ev, utilities.ev_to_nu))
ax3.set_xlabel(r"Energy [eV]", fontsize=FONTSIZE)
ax1.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKMARKS)
ax2.tick_params(axis="y", which="major", labelsize=FONTSIZE_TICKMARKS)

if not os.path.exists(PLOTDIR):
    os.makedirs(PLOTDIR)

outfile = os.path.join(PLOTDIR, "tywin_host_spectrum.png")
plt.grid(which="both", alpha=0.15)
plt.tight_layout()
fig.savefig(outfile)

for band in p200_subtracted:
    p200_subtracted.update({band: [p200_subtracted[band], p200_observed[band][1]]})

OUTFILE = os.path.join(DATA_DIR, "subtracted_magnitudes.json")
with open(OUTFILE, "w") as outfile:
    json.dump(p200_subtracted, outfile)

print(f"Subtracted magnitudes are:")

for key in p200_subtracted:
    print(f"{key}: {p200_subtracted[key][0]:.5f} +/- {p200_subtracted[key][1]:.5f}")
