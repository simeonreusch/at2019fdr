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


# style = {
#         "pgf.rcfonts":False,
#         "pgf.texsystem": "pdflatex",
#         "text.usetex": True,
#         "font.family": "sans-serif"
#         }
# #set
# mpl.rcParams.update(style)

# mpl.rcParams["text.usetex"] = True
# mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

RELOAD_AVRO_FROM_AMPEL = False

DPI = 300
PLOTDIR = "plots"
DATADIR = "data"
PLOTDIR_DISTNR = os.path.join("plots", "distnr")
if not os.path.exists(PLOTDIR_DISTNR):
    os.makedirs(PLOTDIR_DISTNR)

FIG_WIDTH = 8
BIG_FONTSIZE = 14
SMALL_FONTSIZE = 8
DPI = 400
GOLDEN_RATIO = 1 / 1.618

infile_tdes_baratheons = os.path.join(DATADIR, "tdes_and_baratheons.csv")

INTERVALS = [1.75]
ZTF_HZ = {
    1: utilities.lambda_to_nu(4722.74),
    2: utilities.lambda_to_nu(6339.61),
    3: utilities.lambda_to_nu(7886.13),
}
ZTF_HZ_LETTER = {
    "g": utilities.lambda_to_nu(4722.74),
    "r": utilities.lambda_to_nu(6339.61),
    "i": utilities.lambda_to_nu(7886.13),
}


def get_freq(filterletter):
    return ZTF_HZ_LETTER[filterletter]


def create_subplot(ax, max_distnr):

    BTS_SAMPLE_INFILE = os.path.join(DATADIR, "bts_sample.csv")

    df = pd.read_csv(BTS_SAMPLE_INFILE)
    df_tywin = df.query("ZTFID == 'ZTF19aatubsj'").reset_index()
    df_bran = df.query("ZTFID == 'ZTF19aapreis'").reset_index()

    tywin_duration = float(df_tywin["duration"].values[0])
    tywin_peakmag = float(df_tywin["peakmag"].values[0])
    bran_duration = float(df_bran["duration"].values[0])
    bran_peakabsmag = float(df_bran["peakabs"].values[0])
    bran_peakmag = float(df_bran["peakmag"].values[0])

    tywin_lumi = (
        0.75
        * utilities.abmag_to_flux(tywin_peakmag)
        * tywin_duration
        * 86400
        * ZTF_HZ[2]
    )
    bran_lumi = (
        0.75 * utilities.abmag_to_flux(bran_peakmag) * bran_duration * 86400 * ZTF_HZ[1]
    )

    SPECIAL_OBJECTS = {
        "tywin": {
            "df": df_tywin,
            "duration": tywin_duration,
            "peakabsmag": -22.59,
            "peakmag": tywin_peakmag,
            "lumi": tywin_lumi,
            "distnr": 0.19,
            "loc": (160, 17.9),
            # "loc": (110, 2.3e-5),
            "label": "AT2019fdr",
        },
        "bran": {
            "df": df_bran,
            "duration": bran_duration,
            "peakabsmag": bran_peakabsmag,
            "peakmag": bran_peakmag,
            "lumi": bran_lumi,
            "distnr": 0.25,
            "loc": (50, 17.6),
            # "loc": (86, 0.97e-5),
            "label": "AT2019dsg",
        },
        "lancel": {
            "duration": 258,
            "peakabsmag": -19.215971997908706,
            "peakmag": 16.7645092010498,
            "lumi": bran_lumi,
            "distnr": 0.2851865142583845,
            "loc": (135, 16.57),
            # "loc": (86, 0.97e-5),
            "label": "AT2019aalc",
        },
        # "asassn": {
        #     "duration": 60,
        #     "peakabsmag": -23.5,
        #     "distnr": 0.18,
        #     "loc": (17,-23.8),
        #     "label": "ASASSN-15lh"
        # }
    }

    flux_tywin = utilities.abmag_to_flux(SPECIAL_OBJECTS["tywin"]["peakabsmag"])
    flux_bran = utilities.abmag_to_flux(SPECIAL_OBJECTS["bran"]["peakabsmag"])

    # flux_asassn15lh = utilities.abmag_to_flux(SPECIAL_OBJECTS["asassn"]["peakabsmag"])
    abmag_tywin = utilities.flux_to_abmag(flux_tywin / 2)
    abmag_bran = utilities.flux_to_abmag(flux_bran / 2)
    # abmag_asassn15lh = utilities.flux_to_abmag(flux_asassn15lh/2)
    # print(f"Half of peak magnitude Tywin: {abmag_tywin:.3f}")
    # print(f"Half of peak magnitude Bran: {abmag_bran:.3f}")
    # print(f"Half of peak magnitude ASASSN-15lh: {abmag_asassn15lh:.3f}")

    ANNOTATION_FONTSIZE = 12
    AXIS_FONTSIZE = 14

    SN = [
        "SN",
        "SN Ia",
        "SN II",
        "SN IIn",
        "SN Ia-91T",
        "SN IIP",
        "SN Ic",
        "SN Iax",
        "SN Ia-pec",
        "SN Ib/c",
        "SN Ic-BL",
        "SN Ibn",
    ]
    IA = ["SN Ia", "SN Ia-91T", "SN Iax"]
    CCSN = ["SN II", "SN IIn", "SN IIP", "SN Ic", "SN Ib/c", "SN Ic-BL", "SN Ibn"]
    SLSN = ["SLSN-II", "SLSN-I"]
    TDE = ["TDE"]
    NOVA = ["Nova", "nova"]

    ZTF_IDS_BARATHEONS = ["ZTF19aabbnzo"]

    df_tdes_baratheons = pd.read_csv(infile_tdes_baratheons)
    durations_tdes_baratheons = []
    peak_absmag_tdes_baratheons = []
    peak_mag_tdes_baratheons = []
    peakfilter_tdes_baratheons = []
    distnr_tdes_baratheons = []

    # print(df_baratheons.name.values)
    # for ztfid in df_tdes_baratheons.name.values:

    #     ampel = connectors.AmpelInfo([ztfid], nprocess=16, logger=None)

    #     for i in range(len(ampel.queryresult)):

    #         ztfid = ampel.queryresult[i][0]
    #         obsjds = np.asarray(ampel.queryresult[i][4])
    #         mags = np.asarray(ampel.queryresult[i][5])
    #         fids = np.asarray(ampel.queryresult[i][8])
    #         distnr = np.median(ampel.queryresult[i][9])

    #         if ztfid == "ZTF19aaiqmgl":
    #             obsjds_mask = np.where(obsjds < 58756 + 2400000.5)
    #             obsjds = obsjds[obsjds_mask]
    #             mags = mags[obsjds_mask]
    #             fids = fids[obsjds_mask]

    #         peakmag = np.min(mags)
    #         peakfilter = fids[np.argmin(mags)]

    #         z = df_tdes_baratheons.query(f"name == '{ztfid}'")["redshift"].values[0]

    #         absmags = [utilities.mag_to_absmag(mag, z) for mag in mags]

    #         abs_fluxes = [utilities.abmag_to_flux(absmag) for absmag in absmags]

    #         half_absfluxes = [flux/2 for flux in abs_fluxes]

    #         peak_absmag = np.min(absmags)
    #         peak_absmag_index = np.argmin(absmags)
    #         peak_mag = np.min(mags)
    #         peak_absmag_filter = int(fids[peak_absmag_index])
    #         half_peakflux = half_absfluxes[peak_absmag_index]

    #         in_range = []

    #         for i, flux in enumerate(abs_fluxes):
    #             if flux >= half_peakflux: #and fids[i] == peak_absmag_filter:
    #                 in_range.append(i)

    #         min_jd = obsjds[min(in_range)]
    #         max_jd = obsjds[max(in_range)]

    #         duration = max_jd - min_jd
    #         durations_tdes_baratheons.append(duration)
    #         peak_absmag_tdes_baratheons.append(peak_absmag)
    #         peak_mag_tdes_baratheons.append(peakmag)
    #         peakfilter_tdes_baratheons.append(peakfilter)
    #         distnr_tdes_baratheons.append(distnr)

    #         # time integrated flux (est)

    # df_tdes_baratheons["duration"] = durations_tdes_baratheons
    # df_tdes_baratheons["peakabs"] = peak_absmag_tdes_baratheons
    # df_tdes_baratheons["peakmag"] = peak_mag_tdes_baratheons
    # df_tdes_baratheons["peakfilter"] = peakfilter_tdes_baratheons
    # df_tdes_baratheons["distnr"] = distnr_tdes_baratheons

    # df_tdes_baratheons.query("name not in ['ZTF19aatubsj', 'ZTF18aabtxvd', 'ZTF18aahqkbt', 'ZTF18acpdvos', 'ZTF19aapreis']", inplace=True)

    # freq = []
    # for entry in df_tdes_baratheons["peakfilter"].values:
    #     freq.append(ZTF_HZ[entry])

    # lumi = 0.75 * utilities.abmag_to_flux(df_tdes_baratheons["peakmag"].values) * df_tdes_baratheons["duration"].values * 86400 * freq

    # df_tdes_baratheons["lumi"] = lumi

    # outfile = os.path.join("data", "tdes_and_baratheons.csv")
    # df_tdes_baratheons.sort_values(by="lumi").to_csv(outfile)

    # quit()

    infile = os.path.join(DATADIR, "tdes_and_baratheons.csv")
    df_tdes_baratheons = pd.read_csv(infile)

    def delete_incomplete_datapoints(df):
        """ """
        df = df[[(x[0] != ">") for x in df["duration"]]]
        df = df[[(len(x) > 1) for x in df["peakabs"]]]
        df = df.astype(
            {"duration": "float32", "peakabs": "float32", "peakmag": "float32"}
        )
        df = df.reset_index()
        return df

    if RELOAD_AVRO_FROM_AMPEL:
        df = pd.read_csv(BTS_SAMPLE_INFILE)

        # First we tidy up
        df = delete_incomplete_datapoints(df)

        print(f"total number of objects: {len(df)}")

        # Now we do an AMPEL-query to obtain host-distance
        ztfids = df["ZTFID"].values
        ampel = connectors.AmpelInfo(ztfids, nprocess=16, logger=None)

        print(f"\nqueryresults: {len(ampel.queryresult)}")

        distnrs = {}

        for i in range(len(ampel.queryresult)):
            ztfid = ampel.queryresult[i][0]
            distnr = np.median(ampel.queryresult[i][9])
            distnrs.update({ztfid: distnr})

        distnr_list = []
        for ztfid in ztfids:
            distnr_list.append(distnrs[ztfid])

        df["distnr"] = distnr_list
        outfile = os.path.join(DATADIR, "bts_sample_dist.csv")
        df.to_csv(outfile)

    def get_transparency(df_obj, df_tot):
        """ """
        if len(df_obj) > 0:
            alpha = np.interp(
                df_obj["peakmag"].values,
                (df_tot["peakmag"].values.min(), df_tot["peakmag"].values.max()),
                (+1, +0.05),
            )
        else:
            alpha = 0
        return alpha

    infile = os.path.join(DATADIR, "bts_sample_dist.csv")
    df_nodistcut = pd.read_csv(infile)
    len_before_cut = len(df_nodistcut)

    df = df_nodistcut.query(f"distnr <= {max_distnr}").reset_index()

    print(
        f"{len(df)} objects of {len_before_cut} survive the distnr-cut (distnr <= {max_distnr:.2f})"
    )

    freq = []

    for peakfilt in df["peakfilt"].values:
        freq.append(ZTF_HZ_LETTER[peakfilt])

    df["lumi"] = (
        utilities.abmag_to_flux(df["peakmag"].values)
        * df["duration"].values
        * 86400
        * freq
    )

    # Now we group
    df_sn = df.query("type in @SN").reset_index()
    df_ccsn = df.query("type in @CCSN").reset_index()
    df_slsn = df.query("type in @SLSN").reset_index()
    df_ia = df.query("type in @IA").reset_index()
    df_tde_bts = df.query("type in @TDE and ZTFID != 'ZTF19aapreis'").reset_index()
    df_tde = df_tdes_baratheons.query("type == 'TDE'")
    df_nova = df.query("type in @NOVA").reset_index()

    durations_new = []
    names = []
    for name in df_tde.name:
        query = df_tde_bts.query(f"ZTFID == '{name}'")
        if len(query) == 1:
            durations_new.append(query["duration"].values[0])
        else:
            durations_new.append(
                df_tde.query(f"name == '{name}'")["duration"].values[0]
            )
        names.append(name)

    df_tde["duration"] = durations_new

    df_tde = df_tde.query(f"distnr < {max_distnr}")

    df_baratheons = df_tdes_baratheons.query(
        f"type == 'Baratheon' and distnr < {max_distnr}"
    )

    if max_distnr > 0.2:
        label_baratheons = f"TDE-\nlike ({len(df_baratheons)+2})"
        label_tdes = f"TDE ({len(df_tde)+1})"
    else:
        label_baratheons = f"TDE-\nlike ({len(df_baratheons)+1})"
        label_tdes = f"TDE ({len(df_tde)})"

    plotparams = {
        "Baratheons": {
            "df": df_baratheons,
            "c": "magenta",
            "m": "X",
            "l": label_baratheons,
            "s": 20,
            "a": 1.0,
            "zorder": 7,
        },
        "TDE": {
            "df": df_tde,
            "c": "tab:orange",
            "m": "D",
            "l": label_tdes,
            "s": 14,
            "a": 1,
            "zorder": 6,
        },
        "SLSNe": {
            "df": df_slsn,
            "c": "tab:red",
            "m": "P",
            "l": f"SLSNe ({len(df_slsn)})",
            "s": 14,
            "a": 0.8,
            "zorder": 5,
        },
        "SNe Ia": {
            "df": df_ia,
            "c": "tab:green",
            "m": "p",
            "l": f"SNe Ia\n({len(df_ia)})",
            "s": 14,
            "a": 0.1,
            "zorder": 4,
        },
        "CCSNe": {
            "df": df_ccsn,
            "c": "tab:blue",
            "m": "s",
            "l": f"CCSNe ({len(df_ccsn)})",
            "s": 14,
            "a": 0.3,
            "zorder": 3,
        },
        "Novae": {
            "df": df_nova,
            "c": "brown",
            "m": "o",
            "l": f"Novae ({len(df_nova)})",
            "s": 14,
            "a": 1,
            "zorder": 2,
        },
    }

    # ax.set_xlabel(
    #     r"Rest-frame duration ($F \geq \frac{F_{\text{peak}}}{2}$) [days]",
    #     fontsize=AXIS_FONTSIZE-1,
    # )

    # ax1.set_ylabel("Peak absolute magnitude", fontsize=AXIS_FONTSIZE)
    # ax1.set_ylabel(r"Observed optical fluence [erg/cm$^2$]", fontsize=AXIS_FONTSIZE-1))
    # ax.set_ylabel(r"Time integrated optical flux (approx.) [erg/cm$^2$]", fontsize=AXIS_FONTSIZE-1)
    # ax.set_ylabel(r"Observed peak magnitude", fontsize=AXIS_FONTSIZE-1)
    ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax1.set_ylim([-5, -25])
    # ax.set_xlim([5, 600])
    # ax1.set_yticks([-5, -10, -15, -20, -25])

    # fmt = mpl.ticker.StrMethodFormatter("{x}")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    # ax.xaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis="x")
    # ax.xaxis.get_major_formatter()._usetex = False

    for i, name in enumerate(plotparams.keys()):
        param = plotparams[name]
        ax.scatter(
            param["df"]["duration"].values,
            param["df"]["peakmag"].values,
            # param["df"]["peakmag"].values,
            label=param["l"],
            marker=param["m"],
            s=param["s"],
            c=param["c"],
            zorder=param["zorder"],
            alpha=param["a"],
            linewidths=0,
        )
        # ax1.scatter(param["df"]["duration"].values, param["df"]["peakabs"].values, label=param["l"], marker=param["m"], s=7, edgecolors=param["c"], c="None", alpha=param["a"])

    for obj in SPECIAL_OBJECTS:
        if SPECIAL_OBJECTS[obj]["distnr"] <= max_distnr:
            ax.scatter(
                SPECIAL_OBJECTS[obj]["duration"],
                SPECIAL_OBJECTS[obj]["peakmag"],
                color="black",
                marker="*",
                alpha=1,
            )
            ax.annotate(
                SPECIAL_OBJECTS[obj]["label"],
                SPECIAL_OBJECTS[obj]["loc"],
                color="black",
                fontsize=ANNOTATION_FONTSIZE - 4,
            )
    # plt.annotate(
    #     f"max distnr: {max_distnr:.3f} arcsec",
    #     (6, -9),
    #     color="black",
    #     fontsize=ANNOTATION_FONTSIZE,
    # )

    # plt.annotate(
    #     f"max distnr: {max_distnr:.3f} arcsec",
    #     (6, 3e-24),
    #     color="black",
    #     fontsize=ANNOTATION_FONTSIZE,
    # )

    legend = ax.legend(fontsize=ANNOTATION_FONTSIZE - 5, loc=1, ncol=2)
    for lh in legend.legendHandles:
        lh.set_alpha(1)

    return ax


if __name__ == "__main__":

    FIG_WIDTH = 10
    GOLDEN_RATIO = 1 / 1.618
    ANNOTATION_FONTSIZE = 12
    AXIS_FONTSIZE = 14

    fig = plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH / 2.8))
    plt.subplots_adjust()  # bottom = 0.1, left = 0.1, top = 0.81, right = 0.9)
    # lc_ax1 = fig.add_subplot(1,3)
    sed1 = fig.add_subplot(1, 3, 1)
    sed2 = fig.add_subplot(1, 3, 2)
    sed3 = fig.add_subplot(1, 3, 3)

    distances = [10000, 0.4, 0.2]

    ax1 = create_subplot(sed1, distances[0])
    ax2 = create_subplot(sed2, distances[1])
    ax3 = create_subplot(sed3, distances[2])

    ax1.set_title("No cut")
    ax2.set_title(f"Max host distance: {distances[1]} arcsec")
    ax3.set_title(f"Max host distance: {distances[2]} arcsec")

    for ax in [ax1, ax2, ax3]:
        ax.set_ylim([20, 13])
        ax.set_xlim([5, 450])

    ax1.set_ylabel(r"Observed peak magnitude", fontsize=AXIS_FONTSIZE - 1)

    # ax2.set_xlabel(
    #     r"Rest-frame duration ($F \geq \frac{F_{\text{peak}}}{2}$) [days]",
    #     fontsize=AXIS_FONTSIZE - 1,
    # )
    ax2.set_xlabel(
        r"Rest-frame duration ($F \geq \frac{F_{peak}}{2}$) [days]",
        fontsize=AXIS_FONTSIZE - 1,
    )

    plt.tight_layout()

    outfile_png = os.path.join(PLOTDIR, "population_multipanel.png")
    outfile_pdf = os.path.join(PLOTDIR, "population_multipanel.pdf")
    fig.savefig(outfile_png)
    fig.savefig(outfile_pdf)
