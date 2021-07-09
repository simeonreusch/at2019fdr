#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.time import Time
from astropy.coordinates import SkyCoord
from matplotlib.ticker import ScalarFormatter
from modelSED import utilities
from ztffps import connectors
from astroquery.ned import Ned

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
mpl.rcParams.update(nice_fonts)
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]  # for \text command

RELOAD_AVRO_FROM_AMPEL = False

DPI = 300
PLOTDIR = "plots"
DATADIR = "data"
PLOTDIR_DISTNR = os.path.join("plots", "distnr")
if not os.path.exists(PLOTDIR_DISTNR):
    os.makedirs(PLOTDIR_DISTNR)

# c = SkyCoord("22:02:15.4263", "−61:39:34.910", unit=(u.hourangle, u.deg))
# result_table = Ned.query_region(c, radius=0.1 * u.deg)
# res_df = result_table.to_pandas()
infile_tdes_baratheons = os.path.join(DATADIR, "tdes_and_baratheons.csv")

INTERVALS = [1.75]
# INTERVALS = np.linspace(0,1.75,875)
STEPS = len(INTERVALS)

for global_i, MAX_DISTNR in enumerate(INTERVALS):

    BTS_SAMPLE_INFILE = os.path.join(DATADIR, "bts_sample.csv")

    df = pd.read_csv(BTS_SAMPLE_INFILE)
    df_tywin = df.query("ZTFID == 'ZTF19aatubsj'").reset_index()
    df_bran = df.query("ZTFID == 'ZTF19aapreis'").reset_index()

    tywin_duration = float(df_tywin["duration"].values[0])
    tywin_peakmag = float(df_tywin["peakmag"].values[0])
    bran_duration = float(df_bran["duration"].values[0])
    bran_peakabsmag = float(df_bran["peakabs"].values[0])
    bran_peakmag = float(df_bran["peakmag"].values[0])
    tywin_lumi = utilities.abmag_to_flux(tywin_peakmag) * tywin_duration
    bran_lumi = utilities.abmag_to_flux(bran_peakmag) * bran_duration

    SPECIAL_OBJECTS = {
        "tywin": {
            "df": df_tywin,
            "duration": tywin_duration,
            "peakabsmag": -22.59,
            "peakmag": tywin_peakmag,
            "lumi": tywin_lumi,
            "distnr": 0.19,
            # "loc": (122, -23.0),
            "loc": (110, 7.5e-25),
            "label": "AT2019fdr",
        },
        "bran": {
            "df": df_bran,
            "duration": bran_duration,
            "peakabsmag": bran_peakabsmag,
            "peakmag": bran_peakmag,
            "lumi": bran_lumi,
            "distnr": 0.25,
            # "loc": (100, -18.0),
            "loc": (86, 2.3e-25),
            "label": "AT2019dsg",
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

    # print(df_baratheons.name.values)
    for ztfid in df_tdes_baratheons.name.values:

        ampel = connectors.AmpelInfo([ztfid], nprocess=16, logger=None)

        for i in range(len(ampel.queryresult)):

            ztfid = ampel.queryresult[i][0]
            obsjdsd = ampel.queryresult[i][4]
            mags = ampel.queryresult[i][5]

            fids = ampel.queryresult[i][8]
            distnr = np.median(ampel.queryresult[i][9])
            
            peakmag = np.min(mags)
            peakfilter = fids[np.argmin(mags)]

            z = df_tdes_baratheons.query(f"name == '{ztfid}'")["redshift"].values[0]

            absmags = [utilities.mag_to_absmag(mag, z) for mag in mags]

            abs_fluxes = [utilities.abmag_to_flux(absmag) for absmag in absmags]

            half_absfluxes = [flux/2 for flux in abs_fluxes]

            peak_absmag = np.min(absmags)
            peak_absmag_index = np.argmin(absmags)
            peak_mag = np.min(mags)
            peak_absmag_filter = int(fids[peak_absmag_index])
            half_peakflux = half_absfluxes[peak_absmag_index]

            in_range = []

            for i, flux in enumerate(abs_fluxes):
                if flux >= half_peakflux: #and fids[i] == peak_absmag_filter:
                    in_range.append(i)

            min_jd = obsjdsd[min(in_range)]
            max_jd = obsjdsd[max(in_range)]

            duration = max_jd - min_jd
            durations_tdes_baratheons.append(duration)
            peak_absmag_tdes_baratheons.append(peak_absmag)
            peak_mag_tdes_baratheons.append(peakmag)


    df_tdes_baratheons["duration"] = durations_tdes_baratheons
    df_tdes_baratheons["peakabs"] = peak_absmag_tdes_baratheons
    df_tdes_baratheons["peakmag"] = peak_mag_tdes_baratheons

    df_tdes_baratheons.query("name not in ['ZTF19aatubsj', 'ZTF18aabtxvd', 'ZTF18aahqkbt', 'ZTF18acpdvos', 'ZTF19aapreis']", inplace=True)

    lumi = utilities.abmag_to_flux(df_tdes_baratheons["peakmag"].values) * df_tdes_baratheons["duration"].values
    df_tdes_baratheons["lumi"] = lumi

    df_tdes_baratheons.sort_values(by="lumi").to_csv("test.csv")
    # df_tdes_baratheons.query("type == 'Baratheon'").to_csv("test.csv")

    def delete_incomplete_datapoints(df):
        """ """
        df = df[[(x[0] != ">") for x in df["duration"]]]
        df = df[[(len(x) > 1) for x in df["peakabs"]]]
        df = df.astype({"duration": "float32", "peakabs": "float32", "peakmag": "float32"})
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

    df = df_nodistcut.query(f"distnr <= {MAX_DISTNR}").reset_index()

    print(
        f"{len(df)} objects of {len_before_cut} survive the distnr-cut (distnr <= {MAX_DISTNR:.2f})"
    )

    df["lumi"] = utilities.abmag_to_flux(df["peakmag"].values) * df["duration"].values

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
            durations_new.append(df_tde.query(f"name == '{name}'")["duration"].values[0])
        names.append(name)


    df_tde["duration"] = durations_new


    plotparams = {
        "Baratheons": {
            "df": df_tdes_baratheons.query("type == 'Baratheon'"),
            "c": "magenta",
            "m": "X",
            "l": "NLSy1\nflares",
            "s": 10,
            "a": 1.0,
            "zorder": 7,
        },
        "TDE": {
            "df": df_tde,
            "c": "tab:orange",
            "m": "D",
            "l": "TDE",
            "s": 7,
            "a": 1,
            "zorder": 6,
        },
        "SLSNe": {
            "df": df_slsn,
            "c": "tab:red",
            "m": "P",
            "l": "SLSNe",
            "s": 7,
            "a": 0.8,
            "zorder": 5,
        },
        "SNe Ia": {
            "df": df_ia,
            "c": "tab:green",
            "m": "p",
            "l": "SNe Ia",
            "s": 7,
            "a": 0.1,
            "zorder": 4,
        },
        "CCSNe": {
            "df": df_ccsn,
            "c": "tab:blue",
            "m": "s",
            "l": "CCSNe",
            "s": 7,
            "a": 0.2,
            "zorder": 3,
        },
        "Novae": {
            "df": df_nova,
            "c": "brown",
            "m": "o",
            "l": "Novae",
            "s": 7,
            "a": 1,
            "zorder": 2,
        },
    }

    fig, ax1 = plt.subplots(1, 1, figsize=[4.5, 4.5], dpi=DPI)
    ax1.set_xlabel(
        r"Rest-frame duration ($F > 0.5~F_{\text{peak}}$) [days]",
        fontsize=AXIS_FONTSIZE,
    )
    # ax1.set_ylabel("Peak absolute magnitude", fontsize=AXIS_FONTSIZE)
    ax1.set_ylabel(r"Observed g/r peak flux $\times$ duration", fontsize=AXIS_FONTSIZE)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    # ax1.set_ylim([-5, -25])
    ax1.set_xlim([5, 750])
    # ax1.set_yticks([-5, -10, -15, -20, -25])
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.ticklabel_format(style="plain", axis="x")

    for i, name in enumerate(plotparams.keys()):
        param = plotparams[name]
        ax1.scatter(
            param["df"]["duration"].values,
            param["df"]["lumi"].values,
            # param["df"]["peakmag"].values,
            label=param["l"],
            marker=param["m"],
            s=param["s"],
            c=param["c"],
            zorder=param["zorder"],
            alpha=param["a"],
        )
        # ax1.scatter(param["df"]["duration"].values, param["df"]["peakabs"].values, label=param["l"], marker=param["m"], s=7, edgecolors=param["c"], c="None", alpha=param["a"])

    for obj in SPECIAL_OBJECTS:
        if SPECIAL_OBJECTS[obj]["distnr"] <= MAX_DISTNR:
            ax1.scatter(
                SPECIAL_OBJECTS[obj]["duration"],
                SPECIAL_OBJECTS[obj]["lumi"],
                color="black",
                marker="*",
                alpha=1,
            )
            plt.annotate(
                SPECIAL_OBJECTS[obj]["label"],
                SPECIAL_OBJECTS[obj]["loc"],
                color="black",
                fontsize=ANNOTATION_FONTSIZE,
            )
    # plt.annotate(
    #     f"max distnr: {MAX_DISTNR:.3f} arcsec",
    #     (6, -9),
    #     color="black",
    #     fontsize=ANNOTATION_FONTSIZE,
    # )
    # plt.annotate(
    #     f"max distnr: {MAX_DISTNR:.3f} arcsec",
    #     (6, 3e-24),
    #     color="black",
    #     fontsize=ANNOTATION_FONTSIZE,
    # )

    legend = plt.legend(fontsize=ANNOTATION_FONTSIZE-1, loc=4)
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    if STEPS == 1:
        outfile = os.path.join(PLOTDIR, f"population_{INTERVALS[0]}_flux_duration.png")
    else:
        outfile = os.path.join(PLOTDIR_DISTNR, f"{global_i:04d}.png")
    plt.tight_layout()
    fig.savefig(outfile)
    plt.close()
