#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time
from astropy.io import fits
import astropy.units as u
from datetime import date
from scipy.interpolate import UnivariateSpline
from modelSED import sncosmo_spectral_v13, utilities
import sncosmo

FNU = u.erg / (u.cm ** 2 * u.s * u.Hz)
FLAM = u.erg / (u.cm ** 2 * u.s * u.AA)


def nu_to_lambda(fluxnu, wav):
    return np.asarray(fluxnu) * 2.99792458e18 / np.asarray(wav) ** 2 * FLAM


def lambda_to_nu(fluxlambda, wav):
    return np.asarray(fluxlambda) * 3.33564095e-19 * np.asarray(wav) ** 2 * FNU


data_folder = "data"
plot_folder = "plots"
spectra_folder = os.path.join(data_folder, "spectra")
lc_folder = os.path.join(data_folder, "lightcurves")

path_not = os.path.join(spectra_folder, "ZTF19aatubsj_20200430_NOT_v1.ascii")
path_tns = os.path.join(spectra_folder, "tns_2019fdr_2019-06-08.33_MDM-2.4_OSMOS.flm")

redshift = 1 + 0.2666

spectrum_not = pd.read_table(
    path_not, names=["wl", "flux_lambda"], sep="\s+", comment="#"
)
spectrum_tns = pd.read_table(
    path_tns, names=["wl", "flux_lambda", "fluxerr_lambda"], sep="\s+", comment="#"
)

# Convert to f-nu
spectrum_not_nu = lambda_to_nu(spectrum_not["flux_lambda"], spectrum_not["wl"])
spectrum_not["flux_nu"] = spectrum_not_nu.value
spectrum_tns_nu = lambda_to_nu(spectrum_tns["flux_lambda"], spectrum_tns["wl"])
spectrum_tns["flux_nu"] = spectrum_tns_nu.value


mask = spectrum_not["flux_nu"] > 0.0
spectrum_not["flux_nu"][~mask] = 0.00
mask = spectrum_tns["flux_nu"] > 0.0
spectrum_tns["flux_nu"][~mask] = 0.00


smooth = 6
f = np.array(list(spectrum_not["flux_nu"]))
sf = np.zeros(len(f) - smooth)
swl = np.zeros(len(f) - smooth)

f_tns = np.array(list(spectrum_tns["flux_nu"]))
sf_tns = np.zeros(len(f_tns) - smooth)
swl_tns = np.zeros(len(f_tns) - smooth)

for i in range(smooth):
    sf += np.array(list(f)[i : -smooth + i])
    swl += np.array(list(spectrum_not["wl"])[i : -smooth + i])
    sf_tns += np.array(list(f_tns)[i : -smooth + i])
    swl_tns += np.array(list(spectrum_tns["wl"])[i : -smooth + i])

sf /= float(smooth)
swl /= float(smooth)
sf_tns /= float(smooth)
swl_tns /= float(smooth)

fig_width = 10
fig_height = 6
big_fontsize = 12
annotation_fontsize = 9

plt.figure(figsize=(fig_width, fig_height), dpi=300)

ax1 = plt.subplot(111)
cols = ["C1", "C7", "k", "k"]


discovery_date = date(2019, 5, 13)
mdm_date = date(2019, 6, 8)
not_date = date(2020, 4, 30)
delta_mdm = mdm_date - discovery_date
delta_not = not_date - discovery_date
days_mdm = delta_mdm.days
days_not = delta_not.days


# plt.plot(spectrum_not["wl"], spectrum_not["flux_nu"]/np.mean(spectrum_not["flux_nu"]), linewidth=0.5, color="C0", alpha=0.3)
# plot1,  = plt.plot(swl, sf/np.mean(sf), color="black", label=f"NOT (+{days_not} days)")
plt.plot(
    spectrum_tns["wl"], spectrum_tns["flux_nu"], linewidth=0.5, color="C0", alpha=0.3
)
(plot2,) = plt.plot(swl_tns, sf_tns / np.mean(sf_tns), color="black")

lower_border = 8150
upper_border = 8400

mask1 = swl > lower_border
mask2 = swl < upper_border
mask = np.logical_and(mask1, mask2)
mask3 = swl_tns > lower_border
mask4 = swl_tns < upper_border
mask_tns = np.logical_and(mask3, mask4)

# spline1 = UnivariateSpline(swl, sf/np.mean(sf))
# spline1.set_smoothing_factor(5)
# spline2 = UnivariateSpline(swl[~mask], sf[~mask]/np.mean(sf[~mask]))
# spline2.set_smoothing_factor(100)
spline3 = UnivariateSpline(swl_tns, sf_tns / np.mean(sf_tns))
spline3.set_smoothing_factor(1)
spline4 = UnivariateSpline(
    swl_tns[~mask_tns], sf_tns[~mask_tns] / np.mean(sf_tns[~mask_tns])
)
spline4.set_smoothing_factor(50)

wl_spline = np.linspace(np.min(swl_tns) - 1000, np.max(swl_tns) + 500, num=1500)

padded_spectrum = []
for index, wl in enumerate(wl_spline):
    if wl <= np.min(swl_tns):
        padded_spectrum.append(spline4(wl) - 0.02)
    elif wl > np.min(swl_tns) and wl <= np.max(swl_tns):
        padded_spectrum.append(spline3(wl))
    else:
        padded_spectrum.append(spline4(wl) - 0.02)

wl_spline_new = np.linspace(np.min(swl_tns) - 1000, np.max(swl_tns) + 500, num=1500)
# plot1, = plt.plot(wl_spline, spline1(wl_spline), color="r")
# plot1, = plt.plot(wl_spline, spline2(wl_spline)-0.1, color="g")
(plot1,) = plt.plot(wl_spline, padded_spectrum, color="r")
(plot1,) = plt.plot(wl_spline_new, spline4(wl_spline_new) - 0.02, color="g")


flux_tns_with_h = sncosmo_spectral_v13.Spectrum(
    wave=wl_spline, flux=padded_spectrum, unit=FNU
)
flux_tns_without_h = sncosmo_spectral_v13.Spectrum(
    wave=wl_spline, flux=spline4(wl_spline_new) - 0.02, unit=FNU
)
print(flux_tns_with_h)
print(flux_tns_without_h)

abmag_tns_with_h_g = utilities.magnitude_in_band("P48+ZTF_g", flux_tns_with_h)
abmag_tns_without_h_g = utilities.magnitude_in_band("P48+ZTF_g", flux_tns_without_h)

flux_with_h = utilities.abmag_to_flux(abmag_tns_with_h_g)
flux_without_h = utilities.abmag_to_flux(abmag_tns_without_h_g)

print(flux_with_h / flux_without_h)
# ab = sncosmo.get_magsystem("ab")

# bp_g = sncosmo_spectral_v13.read_bandpass("bandpasses/csv/g_mod.csv")
# bp_r = sncosmo_spectral_v13.read_bandpass("bandpasses/csv/r_mod.csv")
# bp_i = sncosmo_spectral_v13.read_bandpass("bandpasses/csv/i_mod.csv")
# zp_flux_g = ab.zpbandflux("ztfg")
# zp_flux_r = ab.zpbandflux("ztfr")
# zp_flux_i = ab.zpbandflux("ztfi")


# bandflux_g_tns_with_h = flux_tns_with_h.bandflux(bp_g) / zp_flux_g
# bandflux_r_tns_with_h = flux_tns_with_h.bandflux(bp_r) / zp_flux_r
# bandflux_i_tns_with_h = flux_tns_with_h.bandflux(bp_i) / zp_flux_i

# bandflux_g_tns_without_h = flux_tns_without_h.bandflux(bp_g) / zp_flux_g
# bandflux_r_tns_without_h = flux_tns_without_h.bandflux(bp_r) / zp_flux_r
# bandflux_i_tns_without_h = flux_tns_without_h.bandflux(bp_i) / zp_flux_i


# ratio_g_tns = bandflux_g_tns_with_h / bandflux_g_tns_without_h
# ratio_r_tns = bandflux_r_tns_with_h / bandflux_r_tns_without_h
# ratio_i_tns = bandflux_i_tns_with_h / bandflux_i_tns_without_h

# print(ratio_g_tns)
# print(ratio_r_tns)
# print(ratio_i_tns)


# ztf_filters = {"g": [4086, 5521], "r": [5600, 7316], "i": [7027, 8883]}
# ztf_colors = {"g": "green", "r": "red", "i": "orange"}

# plt.ylabel(r"$F_{\nu}$", fontsize=big_fontsize)

# for band in ["g", "r", "i"]:
#     ax1.axvspan(
#         ztf_filters[band][0],
#         ztf_filters[band][1],
#         alpha=0.3,
#         color=ztf_colors[band],
#     )

# rslim = ax1.get_xlim()
# ax1.set_xlabel(r"Observed Wavelength [$\rm \AA$]", fontsize=big_fontsize)
# ax1.tick_params(axis="both", which="major", labelsize=big_fontsize)
# plt.tight_layout()

# filename = os.path.join(plot_folder, "h_modeling.png")

# plt.savefig(filename)
