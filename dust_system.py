#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os
from astropy import units as u
from astropy import constants as const
import pandas as pd
import matplotlib as mpl
import numpy as np
from modelSED import utilities
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, convolve
from scipy.interpolate import splev, splrep
from lmfit import Model, Parameters, Minimizer, report_fit, minimize
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
mpl.rcParams.update(nice_fonts)
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]  # for \text command

FIT = False
PLOT = True

DPI = 400
FIG_WIDTH = 6
GOLDEN_RATIO = 1/1.618
ANNOTATION_FONTSIZE = 12
AXIS_FONTSIZE = 14
REDSHIFT = 0.2666

MJD_OPT_PEAK = 58709.84
MJD_IR_PEAK = 59074.01


infile_fitdf = "fit_lumi_radii.csv"

PLOT_DIR = "plots"

df_fit = pd.read_csv(infile_fitdf)

opt_ir_delay_day = (MJD_IR_PEAK - MJD_OPT_PEAK)*u.day

opt_ir_delay_s = opt_ir_delay_day.to(u.s)

light_travel_distance = opt_ir_delay_s*const.c

light_travel_distance = light_travel_distance.to(u.cm)


fitted_max_optical_radius = np.max(df_fit["optical_radius"].values)
fitted_max_optical_radius = fitted_max_optical_radius * u.cm

fitted_max_ir_radius = np.max(df_fit["infrared_radius"].values)
fitted_max_ir_radius = fitted_max_ir_radius * u.cm



def equation_12(T=1850, R=0.1):
    """ 
    T in Kelvin
    Radius in parsec
    """
    L_abs = 5 * 10**44 * (R/0.1)**2 * (T/1850)**5.8 * u.erg / u.s

    return L_abs




infile_lc = os.path.join("data", "lightcurves", "full_lightcurve_final.csv")
lc = pd.read_csv(infile_lc)
lc_g = lc.query("telescope == 'P48' and band == 'ZTF_g'")
lc_g = lc_g.sort_values(by=["obsmjd"])

lc_w1 = lc.query("telescope == 'WISE' and band == 'W1'")
lc_w2 = lc.query("telescope == 'WISE' and band == 'W2'")
wl_w1 = 33156.56
wl_w2 = 46028.00
wl_g = 4722.74

obsmjd_w1 = lc_w1.obsmjd.values
obsmjd_w1 = np.array([58710, 58911, 59074])
obsmjd_w2 = obsmjd_w1
obsmjd_g = lc_g.obsmjd.values

mag_w1 = lc_w1.mag.values
mag_w2 = lc_w2.mag.values
mag_g = lc_g.mag.values

mag_err_w1 = lc_w1.mag_err.values
mag_err_w2 = lc_w2.mag_err.values
mag_err_g = lc_g.mag_err.values

flux_w1 = utilities.abmag_to_flux(mag_w1)
flux_w2 = utilities.abmag_to_flux(mag_w2)
flux_g = utilities.abmag_to_flux(mag_g)

flux_err_w1 = utilities.abmag_err_to_flux_err(mag_w1, mag_err_w1)
flux_err_w2 = utilities.abmag_err_to_flux_err(mag_w2, mag_err_w2)
flux_err_g = utilities.abmag_err_to_flux_err(mag_g, mag_err_g)

nu_fnu_w1, nu_fnu_err_w1 = utilities.flux_density_to_flux(wl_w1, flux_w1, flux_err_w1)
nu_fnu_w2, nu_fnu_err_w2 = utilities.flux_density_to_flux(wl_w2, flux_w2, flux_err_w2)
nu_fnu_g, nu_fnu_err_g = utilities.flux_density_to_flux(wl_g, flux_g, flux_err_g)

# We fit the optical lightcurve with a spline
spline_g = splrep(obsmjd_g, nu_fnu_g, s=4e-25)


mjds = np.arange(58000, 59801, 1)
spline_eval_g = splev(mjds, spline_g)
spline_final = []

for i, mjd in enumerate(mjds):
    if mjd < 58600 or mjd > 59450:
        spline_final.append(0)
    else:
        spline_final.append(spline_eval_g[i])

# Now we create a box function

def minimizer_function(params, x, data=None, data_err=None, **kwargs):
    delay = params["delay"]
    amplitude = params["amplitude"]

    mjds = np.arange(58000, 59801, 1)

    _spline_g = kwargs["spline_g"]


    _boxfunc = []
    for i, mjd in enumerate(mjds):
        if mjd < (MJD_OPT_PEAK) or mjd > (MJD_OPT_PEAK + (2*delay)):
            _boxfunc.append(0)
        else:
            _boxfunc.append(1)

    _convolution = convolve(_spline_g, _boxfunc, mode="same") / sum(_boxfunc)*amplitude

    spline_conv = splrep(mjds, _convolution, s=1e-30)

    # spline_eval_conv = splev(mjds, spline_conv)
    # print(spline_eval_conv)
    # quit()

    residuals = []
    for i, flux in enumerate(data):
        mjd = x[i]
        j = np.where(mjds == mjd)
        flux_err = data_err[i]

        fitval = splev(mjd+delay, spline_conv) 

        delta = fitval - flux

        res = delta/flux_err

        residuals.append(res)


    residuals = np.array(residuals)
    print(residuals)
    print(np.mean(abs(residuals)))


    return residuals


minimizer_fcn = minimizer_function

params = Parameters()
params.add("delay", min=160, max=180)#, value=140)#, min=50, max=350)
params.add("amplitude", min=1.0, max=1.4, value=1.0)#, min=0.5, max=2.)

x = obsmjd_w1
data = nu_fnu_w1
data_err = nu_fnu_err_w1


minimizer = Minimizer(
    userfcn=minimizer_fcn,
    params=params,
    fcn_args=(x, data, data_err),
    fcn_kws={"spline_g": spline_final},
    calc_covar=False,
)

if FIT:
    FITMETHOD = "nelder"
    res = minimizer.minimize(method=FITMETHOD)#, Ns=30, workers=1)
    print(report_fit(res))

    # print(report_fit(res.params, min_correl=0.01))

    delay = res.params["delay"].value
    amplitude = res.params["amplitude"].value

else:
    delay = 171.08
    amplitude = 1.194
    delay = 183
    amplitude = 1.3

dust_distance_model = (delay * u.day * const.c).to(u.cm)

if PLOT:

    boxfunc = []
    for i, mjd in enumerate(mjds):
        if mjd < (MJD_OPT_PEAK) or mjd > (MJD_OPT_PEAK + (2*delay)):
            boxfunc.append(0)
        else:
            boxfunc.append(1)

    # We calculate the convolution
    convolution = convolve(spline_final, boxfunc, mode="same") / sum(boxfunc)*amplitude

    # spline_conv = splrep(mjds, convolution, s=1e-30)

    # spline_eval_conv = splev(mjds, spline_conv)


    # And now we plot


    fig = plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH*GOLDEN_RATIO))
    ax = fig.add_subplot(1,1,1)
    ax2 = ax.twinx()
    ax.set_xlim([58000-MJD_OPT_PEAK+500, 59800-MJD_OPT_PEAK])
    ax.set_yscale("log")
    ax.set_ylim([1e-14, 1e-11])
    ax.set_xlabel("Days since peak")
    ax.set_ylabel(r"$\nu F_{\nu}$ [erg/s/cm$^2$]")
    ax2.set_ylabel("Transfer function")
    ax.errorbar(obsmjd_g-MJD_OPT_PEAK, nu_fnu_g, nu_fnu_err_g, color="tab:green", alpha=0.5, label="ZTF g-band", fmt=".")
    ax.errorbar(obsmjd_w1-MJD_OPT_PEAK, nu_fnu_w1, nu_fnu_err_w1, color="tab:blue", label="WISE W1", fmt=".")

    # ax.errorbar(obsmjd_w2-MJD_OPT_PEAK, nu_fnu_w2, nu_fnu_err_w2, color="tab:red", label="WISE W2", fmt=".")


    ax.plot(mjds-MJD_OPT_PEAK, spline_final, c="green")

    ax.plot(mjds-MJD_OPT_PEAK+delay, convolution, c="tab:blue", alpha=1)
    # ax.plot(mjds-MJD_OPT_PEAK+delay, spline_eval_conv, c="black")

    ax2.plot(mjds-MJD_OPT_PEAK+delay, boxfunc, ls="dashed", c="black", alpha=0.3, label="transfer function")
    # ax.plot(mjds-MJD_OPT_PEAK, convolution, c="red")

    ax.legend(loc=2)
    ax2.legend()
    outpath_png = os.path.join(PLOT_DIR, "dust_modeling.png")
    outpath_pdf = os.path.join(PLOT_DIR, "dust_modeling.pdf")
    fig.savefig(outpath_png)
    fig.savefig(outpath_pdf)

delay = delay * u.d

print("\n")
print(f"--- TIME DELAYS ----")
print(f"time delay between optical and IR peak: {opt_ir_delay_day:.0f}")
print(f"time delay inferred from Sjoert's model (boxfunc/2): {delay:.0f}")
print("\n")
print("----- DUST DISTANCE -----")
print(f"inferred from light travel time: {light_travel_distance:.2e}")
print(f"inferred from BB fit: {fitted_max_ir_radius:.2e}")
print(f"inferred from Sjoert's model: {dust_distance_model:.2e}")


dist_sjoertmethod = delay.to(u.s) * const.c
R_Tywin_sjoertmethod = dist_sjoertmethod.to(u.pc).value

T_Tywin = 1750

L_abs = equation_12(T=T_Tywin, R=R_Tywin_sjoertmethod)

log10L_abs = np.log10(L_abs.value)
print("\n")
print("----- ENERGETICS (ALL FROM SJOERT'S MODEL) -------")
print(f"R used for following calculations: {R_Tywin_sjoertmethod:.3f} pc")
print(f"T used for following calculations: {T_Tywin} K (from BB fit)")
print("\n")
print(f"L_abs (paper eq. 12) = {L_abs:.2e}")
print(f"log L_abs = {log10L_abs:.2f}")

# Now we integrate this over the optical lightcurve
L_abs = L_abs.value

time = spline_g[0] - min(spline_g[0])
time = [(t * u.day).to(u.s).value for t in time]

max_of_g = max(spline_g[1])

d = cosmo.luminosity_distance(REDSHIFT)
d = d.to(u.cm).value
lumi = max_of_g * 4 * np.pi * d ** 2

new_spline = spline_g[1] / max_of_g * L_abs

E_abs = np.trapz(y=new_spline, x=time)

E_abs = E_abs * u.erg
log10E_abs = np.log10(E_abs.value)

print(f"E_abs (from optical lightcurve normalized to L_abs) = {E_abs:.2e}")
print(f"log E_abs = {log10E_abs:.2f}")


time = mjds - min(mjds)
time = [(t * u.day).to(u.s).value for t in time]

spline_ir = spline_final / max_of_g * L_abs

E_dust = np.trapz(y=spline_ir, x=time)
E_dust = E_dust * u.erg

log10E_dust = np.log10(E_dust.value)
print(f"E_dust (from fitted IR lightcurve normalized to L_abs) = {E_dust:.2e}")
print(f"log E_dust = {log10E_dust:.2f}")

f = E_dust/E_abs

print(f"Covering factor (E_dust/E_abs) = {f:.2f}")



