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
from modelSED.utilities import broken_powerlaw_spectrum, FNU, FLAM
from astropy.cosmology import Planck18 as cosmo
from astropy import constants as c
from astropy.modeling.models import BlackBody

REDSHIFT = 0.2666
GLOBAL_AV = 0.3643711523794127
GLOBAL_RV = 4.2694173002543225
 
# epoch 0 optical/UV:
temp1 = 13578.089306104634
scale1 = 1.3236379281006812e23


spectrum_unext_noz, boloflux_unext_noz = utilities.blackbody_spectrum(
    temperature=temp1,
    scale=scale1,
    extinction_av=None,
    extinction_rv=None,
    redshift=None,
    get_bolometric_flux=True,
)

wavelengths, frequencies = utilities.get_wavelengths_and_frequencies()
scale_lambda = 1 * FLAM / u.sr
scale_nu = 1 * FNU / u.sr


bb_nu = BlackBody(temperature=temp1 * u.K, scale=scale1 ** 2 * FNU / u.sr)
flux_nu = bb_nu(wavelengths) * u.sr
boloflux = bb_nu.bolometric_flux
spectrum = sncosmo_spectral_v13.Spectrum(wave=wavelengths, flux=flux_nu, unit=FNU)


spectrum_unext_z, boloflux_unext_z = utilities.blackbody_spectrum(
    temperature=temp1,
    scale=scale1,
    extinction_av=None,
    extinction_rv=None,
    redshift=REDSHIFT,
    get_bolometric_flux=True,
)
# spectrum_ext_z, boloflux_ext_z = utilities.blackbody_spectrum(
#     temperature=temp1,
#     scale=scale1,
#     extinction_av=GLOBAL_AV,
#     extinction_rv=GLOBAL_RV,
#     redshift=REDSHIFT,
#     get_bolometric_flux=True,
# )
# spectrum_ext_noz, boloflux_ext_noz = utilities.blackbody_spectrum(
#     temperature=temp1,
#     scale=scale1,
#     extinction_av=GLOBAL_AV,
#     extinction_rv=GLOBAL_RV,
#     redshift=None,
#     get_bolometric_flux=True,
# )
# lumi = utilities.calculate_bolometric_luminosity(
#         temperature=temp1,
#         scale=scale1,
#         redshift=REDSHIFT,
#         temperature_err= None,
#         scale_err= None,
#     )
# lumitest = utilities.calculate_luminosity(
#     spectrum_unext_noz,
#     wl_min = 0,
#     wl_max = 100000000000000000000000,
#     redshift = REDSHIFT
# )
# print(lumitest)
# print(lumi)

plt.figure(figsize=(6, 1 / 1.414 * 6), dpi=300)
ax1 = plt.subplot(111)
plt.xscale("log")
ax1.set_ylabel(r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=10)
ax1.set_xlabel("Frequency [Hz]", fontsize=10)
ax1.set_xlim([5e13, 2e15])
ax1.set_ylim([9e-14, 1e-11])
plt.yscale("log")


SCALE = 1.3236379281006812e23
# SCALE = SCALE * 0.35
wavelengths, frequencies = utilities.get_wavelengths_and_frequencies()
bb_nu = BlackBody(temperature=temp1 * u.K, scale=1 / SCALE * FNU / u.sr)
boloflux = bb_nu.bolometric_flux

flux_nu = bb_nu(wavelengths)
nu = utilities.lambda_to_nu(wavelengths)
nu_fnu = flux_nu * nu

d = cosmo.luminosity_distance(REDSHIFT)
d = d.to(u.cm)

int_flux = np.abs(np.trapz(flux_nu, nu))
int_lumi = int_flux * 4 * np.pi * d ** 2
# print("\n")
# print("selfdone:")
# print(print(f"integrated flux: {int_flux:.2e}"))
# print(f"integrated luminosity: {int_lumi:.2e}")
# print("------------")

spectrum = sncosmo_spectral_v13.Spectrum(wave=wavelengths, flux=flux_nu, unit=FNU)
lumitest = utilities.calculate_luminosity(
    spectrum, wl_min=0, wl_max=100000000000000000000000, redshift=REDSHIFT
)
print(f"integrated luminosity: {lumitest:.2e}")
print("-----------")


d = cosmo.luminosity_distance(REDSHIFT)
d_m = d.to(u.m)
d_cm = d.to(u.cm)
d_pc = d.to(u.pc)
temperature = temp1 * u.K
print(f"Temperature Tywin: {temperature:.0f}")

radius_cm = np.sqrt(d_cm ** 2 / SCALE) / np.sqrt(np.pi)
radius_m = np.sqrt(d_m ** 2 / SCALE) / np.sqrt(np.pi)


print(f"radius Tywin in cm: {radius_cm:.2e}")
print(f"radius Tywin in m: {radius_m:.2e}")

luminosity_watt = c.sigma_sb * (temperature) ** 4 * 4 * np.pi * (radius_m ** 2)
luminosity = luminosity_watt.to(u.erg / u.s)

print(f"luminosity: {luminosity:.2e}")


lumi_from_boloflux = boloflux * (4 * np.pi * ((d_cm) ** 2))
lumi_from_boloflux_gw = lumi_from_boloflux.to(u.m ** 2 * u.kilogram / u.s ** 3) / 1e9

print(f"bolometric flux: {boloflux:.2e}")
print(f"{lumi_from_boloflux:.2e}")
print(f"{lumi_from_boloflux_gw.value:.2e} GWatt")


plt.figure(figsize=(6, 1 / 1.414 * 6), dpi=300)
ax1 = plt.subplot(111)
plt.xscale("log")
ax1.set_ylabel(r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=10)
ax1.set_xlabel("Frequency [Hz]", fontsize=10)
ax1.set_xlim([3e13, 3e15])
ax1.set_ylim([1e-13, 1e-9])
plt.yscale("log")
ax1.plot(nu, nu_fnu)
d = cosmo.luminosity_distance(REDSHIFT)
d = d.to(u.cm).value
lumi = lambda flux: flux * 4 * np.pi * d ** 2
flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
ax2 = ax1.secondary_xaxis("top", functions=(utilities.nu_to_ev, utilities.ev_to_nu))
ax2.set_xlabel(r"Energy [eV]", fontsize=8)
plt.grid(which="both", alpha=0.15)
plt.legend()
plt.savefig("test.png")


# ax1.plot(
#     nu,
#     nu_fnu,
#     color="tab:blue",
#     linestyle="dotted",
#     label=rf"spectrum unextincted / not redshifted",
# )
# ax1.plot(
#     nu,
#     spectrum_unext_z.flux * nu,
#     color="tab:blue",
#     linestyle="dotted",
#     label=rf"spectrum unextincted / redshifted",
# )
# ax1.plot(
#     nu,
#     spectrum_ext_noz.flux * nu,
#     color="tab:red",
#     label=rf"spectrum extincted / not redshifted",
# )
# ax1.plot(
#     nu,
#     spectrum_ext_z.flux * nu,
#     color="tab:red",
#     label=rf"spectrum extincted / redshifted",
# )
# d = cosmo.luminosity_distance(REDSHIFT)
# d = d.to(u.cm).value
# lumi = lambda flux: flux * 4 * np.pi * d ** 2
# flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
# ax3 = ax1.secondary_yaxis("right", functions=(lumi, flux))
# ax3.tick_params(axis="y", which="major", labelsize=10)
# ax3.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=10)
# ax2 = ax1.secondary_xaxis("top", functions=(utilities.nu_to_ev, utilities.ev_to_nu))
# ax2.set_xlabel(r"Energy [eV]", fontsize=8)
# plt.grid(which="both", alpha=0.15)
# plt.legend()
# plt.savefig("test.png")

# int_flux = np.trapz(spectrum_unext_noz.flux, x=None, dx=1.0, axis=-1)
# int_lumi = int_flux * 4 * np.pi * d ** 2
# # print()
# print(int_flux)
# print(int_lumi)
