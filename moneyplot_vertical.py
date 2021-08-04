#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, time, argparse, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import constants as const
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from modelSED import utilities, sncosmo_spectral_v13
from modelSED.utilities import FNU
from matplotlib.patches import ConnectionPatch, Polygon
import matplotlib


pd.options.mode.chained_assignment = None

cmap = utilities.load_info_json("cmap")
filterlabel = utilities.load_info_json("filterlabel")
filter_wl = utilities.load_info_json("filter_wl")
wl_filter = {v: k for k, v in filter_wl.items()}

XRT_COLUMN = "flux0310_bb_25eV"
nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)

MIR = [250000, 25000]
NIR = [25000, 7800]
OPT = [7800, 3800]
UV = [3800, 100]


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

def create_sed(ax, epoch):

    mjd_interval = MJD_INTERVALS[epoch]

    df_cut = df.query(
        f"obsmjd > {mjd_interval[0]} and obsmjd < {mjd_interval[1]} and telescope != 'P200_sextractor'"
    )
    df_cut["telescopeband"] = df_cut["telescope"] + "+" + df["band"]

    fitparams_infile = os.path.join(FITDIR, f"{epoch}_fitparams_all.json")
    with open(fitparams_infile) as infile:
        params = json.load(infile)

    fitted_spectrum_1, bolo_flux_1 = utilities.blackbody_spectrum(
        temperature=params["temp1"],
        scale=params["scale1"],
        extinction_av=GLOBAL_AV,
        extinction_rv=GLOBAL_RV,
        redshift=REDSHIFT,
        get_bolometric_flux=True,
    )

    fitted_spectrum_1_lower, bolo_flux_1_lower = utilities.blackbody_spectrum(
        temperature=params["temp1"]-params["temp1_err"],
        scale=params["scale1"]+params["scale1_err"],
        extinction_av=GLOBAL_AV,
        extinction_rv=GLOBAL_RV,
        redshift=REDSHIFT,
        get_bolometric_flux=True,
    )
    fitted_spectrum_1_upper, bolo_flux_1_upper = utilities.blackbody_spectrum(
        temperature=params["temp1"]+params["temp1_err"],
        scale=params["scale1"]-params["scale1_err"],
        extinction_av=GLOBAL_AV,
        extinction_rv=GLOBAL_RV,
        redshift=REDSHIFT,
        get_bolometric_flux=True,
    )


    fitted_spectrum_2, bolo_flux_2 = utilities.blackbody_spectrum(
        temperature=params["temp2"],
        scale=params["scale2"],
        extinction_av=GLOBAL_AV,
        extinction_rv=GLOBAL_RV,
        redshift=REDSHIFT,
        get_bolometric_flux=True,
    )
    fitted_spectrum_2_lower, bolo_flux_2_lower = utilities.blackbody_spectrum(
        temperature=params["temp2"]-params["temp2_err"],
        scale=params["scale2"]+params["scale2_err"],
        extinction_av=GLOBAL_AV,
        extinction_rv=GLOBAL_RV,
        redshift=REDSHIFT,
        get_bolometric_flux=True,
    )
    fitted_spectrum_2_upper, bolo_flux_2_upper = utilities.blackbody_spectrum(
        temperature=params["temp2"]+params["temp2_err"],
        scale=params["scale2"]-params["scale2_err"],
        extinction_av=GLOBAL_AV,
        extinction_rv=GLOBAL_RV,
        redshift=REDSHIFT,
        get_bolometric_flux=True,
    )

    combined_flux = fitted_spectrum_1.flux + fitted_spectrum_2.flux

    # # # Calculate luminosity
    luminosity_1, _, radius1, _ = utilities.calculate_bolometric_luminosity(
        temperature=params["temp1"],
        scale=params["scale1"],
        redshift=REDSHIFT,
        temperature_err=None,
        scale_err=None,
    )
    luminosity_2, _, radius2, _ = utilities.calculate_bolometric_luminosity(
        temperature=params["temp2"],
        scale=params["scale2"],
        redshift=REDSHIFT,
        temperature_err=None,
        scale_err=None,
    )
    total_luminosity = luminosity_1 + luminosity_2
    print(f"{luminosity_1:.2e}")
    print(f"{luminosity_2:.2e}")
    print(f"{total_luminosity:.2e}")

    combined_spectrum = sncosmo_spectral_v13.Spectrum(
        wave=fitted_spectrum_1.wave, flux=combined_flux, unit=FNU
    )

    ax.text(
        2.9e14,
        1.85e-12,
        f"L = {total_luminosity:.1e}",
        bbox=bbox,
        fontsize=SMALL_FONTSIZE,
    )

    ax.plot(
        utilities.lambda_to_nu(fitted_spectrum_1.wave),
        fitted_spectrum_1.flux * utilities.lambda_to_nu(fitted_spectrum_1.wave),
        color="tab:blue",
        linestyle="dotted",
    )

    ax.plot(
        utilities.lambda_to_nu(fitted_spectrum_2.wave),
        fitted_spectrum_2.flux * utilities.lambda_to_nu(fitted_spectrum_2.wave),
        color="tab:red",
        linestyle="dotted",
    )

    ax.plot(
        utilities.lambda_to_nu(combined_spectrum.wave),
        combined_spectrum.flux * utilities.lambda_to_nu(combined_spectrum.wave),
        color="black",
    )


    # ax.fill_between(x=utilities.lambda_to_nu(fitted_spectrum_1.wave), y2=fitted_spectrum_1_lower.flux * utilities.lambda_to_nu(fitted_spectrum_1_lower.wave), y1=fitted_spectrum_1_upper.flux * utilities.lambda_to_nu(fitted_spectrum_1_upper.wave), alpha=0.3, facecolor="tab:blue")

    # ax.fill_between(x=utilities.lambda_to_nu(fitted_spectrum_2.wave), y2=fitted_spectrum_2_lower.flux * utilities.lambda_to_nu(fitted_spectrum_2_lower.wave), y1=fitted_spectrum_2_upper.flux * utilities.lambda_to_nu(fitted_spectrum_2_upper.wave), alpha=0.3, facecolor="tab:red")



    telescopebands = ['WISE+W2', 'WISE+W1', 'P200+Ks', 'P200+H', 'P200+J', 'P48+ZTF_i', 'P48+ZTF_r', 'P48+ZTF_g', 'Swift+U', 'Swift+UVW1', 'Swift+UVM2','Swift+UVW2']

    for telescopeband in telescopebands:

        df_red = df_cut.query(f"telescopeband == '{telescopeband}'")
        mag = np.mean(df_red.mag.values)
        mag_err = np.mean(df_red.mag_err.values)
        telescope = telescopeband.split("+")[0]
        band = telescopeband.split("+")[1]
        flux = utilities.abmag_to_flux(mag)
        flux_err = utilities.abmag_err_to_flux_err(mag, mag_err)

        if telescopeband == "P48+ZTF_i":
            flux = flux/H_CORRECTION_I_BAND

        nu = utilities.lambda_to_nu(filter_wl[telescopeband])
        
        markers = {"WISE": "p", "P200": "s", "P48": ".", "Swift": "D"}
        markersizes = {"WISE": 5, "P200": 4, "P48": 8, "Swift": 4}

        ax.errorbar(
            nu,
            flux * nu,
            flux_err * nu,
            color=cmap[telescopeband],
            label=filterlabel[telescopeband],
            fmt=markers[telescope],
            markersize=markersizes[telescope],
        )

    if epoch == 1:
        ax.legend(ncol=6,bbox_to_anchor=(1.2, 2.82),fancybox=True, shadow=False, fontsize=9, edgecolor="black")
    if epoch == 3:
        ax.set_xlabel("Frequency [Hz]", fontsize=BIG_FONTSIZE-2)
    # ax2 = ax.secondary_xaxis(
    #     "top", functions=(utilities.nu_to_ev, utilities.ev_to_nu)
    # )
    # ax2.set_xlabel(r"Energy [eV]")


    ax.axvspan(utilities.lambda_to_nu(MIR[0]), utilities.lambda_to_nu(MIR[1]), color="red", alpha=0.1, lw=0)
    ax.axvspan(utilities.lambda_to_nu(NIR[0]), utilities.lambda_to_nu(NIR[1]), color="orange", alpha=0.1, lw=0)
    ax.axvspan(utilities.lambda_to_nu(OPT[0]), utilities.lambda_to_nu(OPT[1]), color="green", alpha=0.1, lw=0)
    ax.axvspan(utilities.lambda_to_nu(UV[0]), utilities.lambda_to_nu(UV[1]), color="violet", alpha=0.1, lw=0)


    ax.grid(which="both", alpha=0.15)

    return ax
    

if __name__ == "__main__":

    REDSHIFT = 0.267
    FIG_WIDTH = 7.5
    BIG_FONTSIZE = 14
    SMALL_FONTSIZE = 8
    DPI = 400
    GOLDEN_RATIO = 1/1.618
    GLOBAL_AV = 0.3643711523794127
    GLOBAL_RV = 4.2694173002543225

    H_CORRECTION_I_BAND = 1.0495345056821688


    CURRENT_FILE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "data"))
    PLOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "plots"))
    SPECTRA_DIR = os.path.join(DATA_DIR, "spectra")
    LC_DIR = os.path.join(DATA_DIR, "lightcurves")
    FITDIR = os.path.join("fit", "double_blackbody")


    infile_lightcurve = os.path.join(LC_DIR, "full_lightcurve.csv")

    df = pd.read_csv(infile_lightcurve)
    df_ztf_g = df.query("telescope == 'P48' and band == 'ZTF_g'")
    instrband = "P48+ZTF_g"



    fig = plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH*0.58))


    plt.subplots_adjust(bottom = 0.12, left = 0.12, top = 0.86, right = 0.9)
    # lc_ax1 = fig.add_subplot(3,4,(1,11))
    # sed1 = fig.add_subplot(3,4,4)
    # sed2 = fig.add_subplot(3,4,8)
    # sed3 = fig.add_subplot(3,4,12)

    # lc_ax1 = fig.add_subplot(3,2,(1,5))
    # sed1 = fig.add_subplot(3,2,2)
    # sed2 = fig.add_subplot(3,2,4)
    # sed3 = fig.add_subplot(3,2,6)

    lc_ax1 = fig.add_subplot(3,5,(1,13))
    sed1 = fig.add_subplot(3,5,(4,5))
    sed2 = fig.add_subplot(3,5,(9,10))
    sed3 = fig.add_subplot(3,5,(14,15))



    sed_xlims = [4.5e13, 2e15]
    sed_ylims = [1e-14, 2e-12]

    def set_scales(ax):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(sed_xlims)
        ax.set_ylim(sed_ylims)
        # ax.xaxis.tick_top()
        # ax.xaxis.set_label_position('top')
        ax.yaxis.tick_right()
    
    for sed in [sed1, sed2, sed3]:
        set_scales(sed)

    # sed1.set_ylabel(r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]"))
    # sed2.axes.yaxis.set_ticks([])
    # sed3.axes.yaxis.set_ticks([])

    for sed in [sed1, sed2]:
        sed.axes.xaxis.set_ticks([])

    sed3.set_xlabel("Frequency [Hz]", fontsize=BIG_FONTSIZE-2)

    sed2.yaxis.set_label_position('right')
    sed2.set_ylabel(r"$\nu$ F$_\nu$ [erg / s / cm$^2$]", fontsize=BIG_FONTSIZE)

    # sed1.legend(ncol=6,bbox_to_anchor=(1.2, 2.68),fancybox=True, shadow=False, fontsize=9, edgecolor="gray")

    lc_ax1.set_yscale("log")
    lc_ylim = [5e-14, 2e-12]
    lc_ax1.set_ylim(lc_ylim)
    flux_density = utilities.abmag_to_flux(df_ztf_g.mag)
    flux_density_err = utilities.abmag_err_to_flux_err(df_ztf_g.mag, df_ztf_g.mag_err)
    flux, flux_err = utilities.flux_density_to_flux(
        filter_wl[instrband], flux_density, flux_density_err
    )

    lc_ax1.errorbar(
        x=df_ztf_g.obsmjd,
        y=flux,
        yerr=flux_err,
        color=cmap[instrband],
        marker=".",
        linestyle=" ",
        label=filterlabel[instrband],
    )
    d = cosmo.luminosity_distance(REDSHIFT)
    d = d.to(u.cm).value
    lumi = lambda flux: flux * 4 * np.pi * d ** 2
    flux = lambda lumi: lumi / (4 * np.pi * d ** 2)
    # lc_ax2 = lc_ax1.secondary_yaxis("right", functions=(lumi, flux))
    # lc_ax2.tick_params(axis="y", which="major")
    lc_ax1.set_xlabel("Date [MJD]", fontsize=BIG_FONTSIZE-2)
    # lc_ax1.set_ylabel(
    #     r"$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=BIG_FONTSIZE
    # )
    lc_ax1.set_ylabel(
        r"$\nu$ F$_\nu$ [erg / s / cm$^2$]", fontsize=BIG_FONTSIZE
    )
    # lc_ax2.set_ylabel(r"$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=BIG_FONTSIZE)
    # lc_ax2.set_ylabel(r"$\nu$ L$_\nu$ [erg / s]", fontsize=BIG_FONTSIZE))
    lc_ax1.grid(which="both", b=True, axis="both", alpha=0.2)
    t_neutrino = Time("2020-05-30T07:54:29.43", format="isot", scale="utc")
    lc_ax1.axvline(
        t_neutrino.mjd,
        linestyle=":",
        label="IC200530A",
        color="tab:red",
        zorder=50,
    )
    bbox = dict(boxstyle="round", fc="w", ec="black")
    lc_ax1.text(
        t_neutrino.mjd-180,
        1.3e-13,
        "Neutrino",
        # rotation="vertical",
        # bbox=bbox,
        fontsize=BIG_FONTSIZE-2,
        color="tab:red",
    )


    loc = [t_neutrino.mjd-302, t_neutrino.mjd+55, t_neutrino.mjd+234]

    bbox = dict(boxstyle="circle", fc="#e5e5e5", ec="black")
    for i in range(1,4):
        lc_ax1.text(
            loc[i-1],
            1.8e-12,
            i,
            # rotation="vertical",
            bbox=bbox,
            fontsize=BIG_FONTSIZE-1,
            color="black",
        )

    loc_upper = (0.05, 0.65)
    loc_lower = (0.15, 0.009)

    for interval in MJD_INTERVALS:
        lc_ax1.axvspan(interval[0], interval[1], alpha=0.2, color="gray", edgecolor="black")

    bbox = dict(boxstyle="round", fc="w", ec="black")
    ax1 = create_sed(sed1, 0)
    ax2 = create_sed(sed2, 1)
    ax3 = create_sed(sed3, 2)

    bbox = dict(boxstyle="circle", fc="#e5e5e5", ec="black")
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.text(
            4.2e13,
            2.7e-13,
            i+1,
            # rotation="vertical",
            bbox=bbox,
            fontsize=BIG_FONTSIZE-1,
            color="black",
        )



    params = {"MIR": [6e13, 7.0e-13, "red"], "NIR": [1.8e14,7.0e-13,"orange"], "Opt": [4.8e14, 7.0e-13, "green"], "UV": [1.15e15, 7.0e-13, "violet"]}

    bbox = dict(boxstyle="round", fc="w", ec="black")
    for wl_range in ["MIR", "NIR", "Opt", "UV"]:
        bbox = dict(boxstyle="round", fc="white", ec="white")
        ax3.text(  
            params[wl_range][0],
            params[wl_range][1],
            wl_range,
            bbox=bbox,
            color=params[wl_range][2],
            alpha=1,
            fontsize=SMALL_FONTSIZE,
        )    
    
    for i, sed in enumerate([sed1, sed2, sed3]):

        array = np.asarray([])

        con1 = ConnectionPatch(
            xyA=(sed_xlims[0], sed_ylims[0]),
            coordsA=sed.transData, 
            xyB=(MJD_INTERVALS[i][0], lc_ylim[1]), 
            coordsB=lc_ax1.transData,
            color = "gray",
            alpha = 0.3,
        )
        con2 = ConnectionPatch(
            xyA=(sed_xlims[1], sed_ylims[0]),
            coordsA=sed.transData, 
            xyB=(MJD_INTERVALS[i][1], lc_ylim[1]), 
            coordsB=lc_ax1.transData,
            color = "gray",
            alpha = 0.3,
        )
    
        # for con in [con1, con2]:
        #     fig.add_artist(con)

        # line2=con2.get_path().vertices
        # line1=con1.get_path().vertices

        # line1_new = np.asarray([line1[0], line1[2]])
        # line2_new = np.asarray([line2[0], line2[2]])
        # coords1 = np.asarray([line1[0], line1[2], line2[0]])
        # coords2 = np.asarray([line2[0], line2[2], line1[2]])

        
        # polygon1 = plt.Polygon(coords1,ec=None,fc='gray', clip_on=False, alpha=0.2)
        # polygon2 = plt.Polygon(coords2,ec=None,fc='gray', clip_on=False, alpha=0.2)

        # for polygon in [polygon1, polygon2]:
        #     fig.add_artist(polygon)

    # plt.tight_layout()
    outfile_png = os.path.join(PLOT_DIR, "lightcurve_and_sed_vertical.png")
    outfile_pdf = os.path.join(PLOT_DIR, "lightcurve_and_sed_vertical.pdf")
    fig.savefig(outfile_png)
    fig.savefig(outfile_pdf)



