#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.time import Time
from astropy.coordinates import SkyCoord
from matplotlib.ticker import ScalarFormatter
from modelSED import utilities
from ztffps import connectors
from astroquery.ned import Ned
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter

from moneyplot import MJD_INTERVALS

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
mpl.rcParams.update(nice_fonts)
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = "\\usepackage{amsmath}"

VERBOSE = True

GLOBAL_AV = 0.3643711523794127
GLOBAL_RV = 4.2694173002543225
REDSHIFT = 0.267
FIG_WIDTH = 8
BIG_FONTSIZE = 14
SMALL_FONTSIZE = 8
DPI = 400
GOLDEN_RATIO = 1/1.618
MJD_START = 58600
MJD_END = 59416

PLOTDIR = "plots"
FITDIR = os.path.join("fit", "double_blackbody")
# FITDIR = os.path.join("fit", "double_blackbody")
DATADIR = os.path.join("data", "lightcurves")

total_luminosities = []
total_luminosities_err = []
infrared_luminosities = []
infrared_luminosities_err = []
optical_luminosities = []
optical_luminosities_err = []
infrared_temps = []
infrared_temps_err = []
optical_temps = []
optical_temps_err = []
infrared_radii = []
infrared_radii_err = []
optical_radii = []
optical_radii_err = []

for epoch in [0,1,2]:

    infile = os.path.join(FITDIR, f"{epoch}_fitparams_all.json")

    with open(infile) as json_file:
        params = json.load(json_file)

    optical_temp = params["temp1"]
    optical_temp_err = params["temp1_err"]
    infrared_temp = params["temp2"]
    infrared_temp_err = params["temp2_err"]
    optical_scale = params["scale1"]
    optical_scale_err = params["scale1_err"]
    infrared_scale = params["scale2"]
    infrared_scale_err = params["scale2_err"]

    optical_temps.append(optical_temp)
    optical_temps_err.append(optical_temp_err)

    infrared_temps.append(infrared_temp)
    infrared_temps_err.append(infrared_temp_err)

    optical_luminosity, optical_luminosity_err, optical_radius, optical_radius_err = utilities.calculate_bolometric_luminosity(
        temperature=optical_temp,
        temperature_err=optical_temp_err,
        scale=optical_scale,
        scale_err=optical_scale_err,
        redshift=REDSHIFT,
        cosmo="generic",
    )
    infrared_luminosity, infrared_luminosity_err, infrared_radius, infrared_radius_err = utilities.calculate_bolometric_luminosity(
        temperature=infrared_temp,
        temperature_err=infrared_temp_err,
        scale=infrared_scale,
        scale_err=infrared_scale_err,
        redshift=REDSHIFT,
        cosmo="generic",
    )

    comb_lumi = optical_luminosity + infrared_luminosity
    comb_lumi_err = np.sqrt(optical_luminosity_err**2 + infrared_luminosity_err**2)

    total_luminosities.append(comb_lumi.value)
    total_luminosities_err.append(comb_lumi_err.value)

    optical_luminosities.append(optical_luminosity.value)
    optical_luminosities_err.append(optical_luminosity_err.value)

    infrared_luminosities.append(infrared_luminosity.value)
    infrared_luminosities_err.append(infrared_luminosity_err.value)

    optical_radii.append(optical_radius.value)
    optical_radii_err.append(optical_radius_err.value)

    infrared_radii.append(infrared_radius.value)
    infrared_radii_err.append(infrared_radius_err.value)

    if VERBOSE:

        print(f"opt. temp.: {optical_temp:.0f} +/- {optical_temp_err:.0f} K")
        print(f"infrared temp.: {infrared_temp:.0f} +/- {infrared_temp_err:.0f}K")


        print(f"opt. radius: {optical_radius:.1e} +/- {optical_radius_err:.1e}")
        print(f"infrared radius: {infrared_radius:.1e} +/- {infrared_radius_err:.1e}")

        print(f"opt. lumi: {optical_luminosity:.1e} +/- {optical_luminosity_err:.1e}")
        print(f"infrared lumi: {infrared_luminosity:.1e} +/- {infrared_luminosity_err:.1e}")
        print(f"comb. lumi: {comb_lumi:.3e} +/- {comb_lumi_err:.3e}")

        print("--------------")

obsmjd_lumi = [np.average(entry) for entry in MJD_INTERVALS]

df_params = pd.DataFrame(index=obsmjd_lumi)
df_params["optical_temp"] = optical_temps
df_params["optical_temp_err"] = optical_temps_err
df_params["infrared_temp"] = infrared_temps
df_params["infrared_temp_err"] = infrared_temps_err
df_params["optical_radius"] = optical_radii
df_params["optical_radius_err"] = optical_radii_err
df_params["infrared_radius"] = infrared_radii
df_params["infrared_radius_err"] = infrared_radii_err
df_params["optical_luminosity"] = optical_luminosities
df_params["optical_luminosity_err"] = optical_luminosities_err
df_params["infrared_luminosity"] = infrared_luminosities
df_params["infrared_luminosity_err"] = infrared_luminosities_err
df_params["total_luminosity"] = total_luminosities
df_params["total_luminosity_err"] = total_luminosities_err

df_params.to_csv("fit_lumi_radii.csv")



obsmjd_lumi = np.insert(obsmjd_lumi, [0,3], [MJD_START, MJD_END])
total_luminosities = np.insert(total_luminosities, [0,3], [0, 0])


lc_infile = os.path.join(DATADIR, "full_lightcurve_final.csv")
lc = pd.read_csv(lc_infile)
lc_g = lc.query("telescope == 'P48' and band == 'ZTF_g'")
lc_g = lc_g.sort_values(by=["obsmjd"])
lc_g["flux"] = utilities.abmag_to_flux(lc_g["mag"])

# # interpolation1 = interp1d(lc_g["obsmjd"], lc_g["mag"])
# # interpolation2 = UnivariateSpline(lc_g["obsmjd"], lc_g["mag"], k=5)
# # interpolation3 = savgol_filter((lc_g["obsmjd"], lc_g["mag"]), 101, 5)

fig = plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH*GOLDEN_RATIO))
ax1 = plt.subplot(111)

ax2 = ax1.twinx()
# ax2.set_ylim([21.5, 18.])
ax1.plot(obsmjd_lumi, total_luminosities)
ax1.set_ylabel("Luminosity [erg/s]")
ax2.set_ylabel(r"Spectral flux density [erg/s/cm$^2$/Hz]")
ax1.set_xlabel("Date [MJD]")
# ax2.scatter(lc_g["obsmjd"], lc_g["mag"]*(-1)+22, c="g", s=2)
ax2.scatter(lc_g["obsmjd"], lc_g["flux"], c="g", s=2)
# # ax2.plot(lc_g["obsmjd"], interpolation3[1], c="black")
outfile = os.path.join(PLOTDIR, "integrated_luminosity.pdf")
plt.savefig(outfile)



integrated_bolo_lumi = np.trapz(total_luminosities, x=obsmjd_lumi) * u.erg / u.s * u.day
integrated_bolo_lumi = integrated_bolo_lumi.to(u.erg)

days = 59245.5-58710.0 
sec = days*24*60*60

approx = total_luminosities[1] * sec * 0.75

print(f"Integrated bolometric luminosity: {integrated_bolo_lumi:.1e}")
print(f"Approximate total luminosity: {approx:.1e}")




