#!/usr/bin/env python3
# Author: Sjoert van Velzen, Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, time, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from astropy.time import Time
from astropy import constants as const
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from modelSED import utilities
import matplotlib
from ztfquery import lightcurve

# CUT PARAMS
rise_error_ratio = 1
fade_error_ratio = 1
max_chisq = 999
min_peakmag = 0

max_offset_weighted_r = 0.5
max_offset_weighted_g = 0.5

min_ndetections_g = 10
min_ndetections_r = 10

max_negative_detections_ratio = 0.5

box_minrise = 15
box_maxrise = 50
box_minfade = 30
box_maxfade = 500


XRT_COLUMN = "flux0310_bb_25eV"
nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)

REDSHIFT = 0.267
FIG_WIDTH = 6
BIG_FONTSIZE = 14
SMALL_FONTSIZE = 8
GOLDEN_RATIO = 1.618
DPI = 400

PLOT_POPULATION = True
PLOT_LIGHTCURVES = False

CLASSIFICATIONS_TO_REMOVE = [
    "duplicate",
    "bogus",
    "bogus?",
    "Star",
    "star",
    "star?",
    "stars",
    "varstar",
    "varstar?",
    "VarStar",
    "CV",
    "CV?",
    "var",
    "vaster",
    "CV Candidate CRTS",
    "CV Candidate",
    "stellar?",
    "XRB",
    "CV MasterOT",
    "galaxy",
    "LBV",
]

# CLASSIFICATIONS_SN = ['SN', 'SN Ib', 'SN IIP', 'SN IIn', 'SLSN-I', 'SN Ia 02cx-like', 'SN Ia-CSM', 'SN Ia 91T-like', 'SN IIb', 'SLSN-II', 'SN Ibn', 'SN Ic', 'SN Ia-pec', 'SN II', 'SNIa', 'SN Ic-BL', 'CV Candidate', 'SN Ia pec', 'Ic-BL', 'SN Ia-91T', 'SN Ia', 'Gap I Ca-rich', "SNIc"]

# CLASSIFICATIONS_AGN = ["Off-nuclear AGN", "quasar", "AGN", "NLS1", "QSO", "CLAGN", "blazar", "Blazar"]

# CLASSIFICATIONS_UNKNOWN = ['unknown', 'none', np.nan, "AGN?", "blazar?", "NLSy1?", "CLAGN?", 'SLSN-I?', 'SLSN-II?', 'SN Ic?', 'SNII?', 'SN Ia?', 'SN II?', 'SN?', "QSO?", "None"]

TYWIN_AND_LANCEL = ["ZTF19aatubsj", "ZTF19aaejtoy"]


if __name__ == "__main__":
    CURRENT_FILE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "data"))
    PLOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "plots"))

    infile = os.path.join(DATA_DIR, "ZTF-I_Nuclear_Transients_extra.dat")

    df = pd.read_csv(infile, sep=" ")

    # print(f"Available classifications: {df['classification'].unique()}")

    # for classification in df["classification"].unique():
    #     if classification not in CLASSIFICATIONS_TO_REMOVE and classification not in CLASSIFICATIONS_SN and classification not in CLASSIFICATIONS_AGN and classification not in CLASSIFICATIONS_UNKNOWN:
    #         print(classification)

    # Now we massage the data a bit

    ireal_flare = np.repeat(True, len(df))
    ireal_flare *= (df["offset_weighted_r"] < max_offset_weighted_r) + (
        df["offset_weighted_g"] < max_offset_weighted_g
    )
    ireal_flare *= (
        df["ndetections_negative"] / df["ndetections"] < max_negative_detections_ratio
    )
    ireal_flare *= np.array((df["ndetections_g"] >= min_ndetections_g)) + np.array(
        (df["ndetections_r"] >= min_ndetections_r)
    )
    ireal_flare *= df["chi2"] != 0

    # t = 0
    # f = 0
    # for i in ireal_flare:
    #     if i == True:
    #         t += 1
    #     if i == False:
    #         f += 1
    # print(f"true == {t}")
    # print(f"false == {f}")
    # quit()

    iclass_ok = np.repeat(True, len(df))

    for classification in CLASSIFICATIONS_TO_REMOVE:
        iclass_ok *= df["classification"] != classification

    iclass_ok *= df["name"] != "ZTF18abcsvxu"  # another varstar
    iclass_ok *= df["name"] != "ZTF19aaqaluu"  # another varstar
    iclass_ok *= df["name"] != "ZTF19aafnogq"  # another varstar

    isel_full = (
        (df["peak_mag"] > min_peakmag)
        * np.array((df["sigma_rise"] / df["e_sigma_rise"] > rise_error_ratio))
        * np.array((df["sigma_fade"] / df["e_sigma_fade"] > fade_error_ratio))
        * (df["chi2"] < max_chisq)
    )

    isel_box = (
        isel_full
        * (df["sigma_rise"] * np.sqrt(2) > box_minrise)
        * (df["sigma_rise"] * np.sqrt(2) < box_maxrise)
        * (df["sigma_fade"] > box_minfade)
        * (df["sigma_fade"] < box_maxfade)
    )

    isel_full *= ireal_flare * iclass_ok

    df_box = df[isel_box]
    df_full = df[isel_full]

    tywin = df.query("name == 'ZTF19aatubsj'")
    tywin_mag = tywin["peak_mag"].values[0]

    df_agn = df_full.query("qclass == 'AGN' and name != 'ZTF19aaejtoy'")
    df_agn_flux_cut = df_agn.query("peak_mag <= @tywin_mag")

    df_unknown = df_full.query("qclass == 'Unknown'")
    df_unknown_flux_cut = df_unknown.query("peak_mag <= @tywin_mag")

    df_sn = df_full.query("qclass == 'SN'")
    df_sn_flux_cut = df_sn.query("peak_mag <= @tywin_mag")

    df_tde = df_full.query("qclass == 'TDE'")
    df_tde_flux_cut = df_tde.query("peak_mag <= @tywin_mag")

    df_prob_tde = df_full.query("qclass == 'lowM'")
    df_prob_tde_flux_cut = df_prob_tde.query("peak_mag <= @tywin_mag")

    print(f"full AGN sample: {len(df_agn)}")
    print(f"full SN sample: {len(df_sn)}")
    print(f"full TDE sample: {len(df_tde)}")
    print(f"full TDE? sample: {len(df_prob_tde)}")
    print(f"full unknown sample: {len(df_unknown)}")
    print("----------------------------------")
    print(f"objects surviving flux cut: {len(df_full.query('peak_mag <= @tywin_mag'))}")
    print("----------------------------------")
    print(f"flux cut AGN sample: {len(df_agn_flux_cut)}")
    print(f"flux cut SN sample: {len(df_sn_flux_cut)}")
    print(f"flux cut TDE sample: {len(df_tde_flux_cut)}")
    print(f"flux cut TDE? sample: {len(df_prob_tde_flux_cut)}")
    print(f"flux cut unknown sample: {len(df_unknown_flux_cut)}")

    all_objects = []

    dfs = [df_agn_flux_cut, df_unknown_flux_cut]

    for df in dfs:
        for entry in df["name"].values:
            all_objects.append(entry)

    if PLOT_POPULATION:

        plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO))

        ax1 = plt.subplot(111)

        ax1.set_xlim(2, 2e3)
        ax1.set_ylim(3, 1e4)
        ax1.set_yscale("log")
        ax1.set_xscale("log")

        data = {
            "AGN": [df_agn_flux_cut, "s", "none", "gray", 0.75, None, "AGN"],
            "Unknown": [df_unknown_flux_cut, "o", "none", "pink", 0.9, 10, "Unknown"],
            "SN": [df_sn_flux_cut, "o", "none", "darkgreen", 0.9, None, "Supernova"],
            "TDE": [df_tde_flux_cut, "P", "none", "midnightblue", 0.9, None, "TDE"],
            "TDE?": [
                df_prob_tde_flux_cut,
                "h",
                "none",
                "midnightblue",
                0.9,
                None,
                "TDE?",
            ],
            "tywin_lancel": [
                df_full.query("name in @TYWIN_AND_LANCEL"),
                "h",
                "tab:blue",
                "tab:blue",
                0.9,
                None,
                None,
            ],
            "bran": [
                df_full.query("name == 'ZTF19aapreis'"),
                "P",
                "tab:blue",
                "tab:blue",
                0.9,
                None,
                None,
            ],
        }

        for entry in data.values():
            ax1.scatter(
                entry[0]["sigma_rise"] * np.sqrt(2),
                entry[0]["sigma_fade"],
                marker=entry[1],
                facecolors=entry[2],
                edgecolors=entry[3],
                alpha=entry[4],
                s=entry[5],
                label=entry[6],
            )

        rect = patches.Rectangle(
            (box_minrise, box_minfade),
            box_maxrise - box_minrise,
            box_maxfade - box_minfade,
            linewidth=1,
            edgecolor="black",
            facecolor="none",
            ls="dashed",
        )

        ax1.add_patch(rect)
        ax1.legend(loc="lower right")

        ax1.set_ylabel("Fade e-folding time [day]", fontsize=BIG_FONTSIZE)
        ax1.set_xlabel("Rise e-folding time [day]", fontsize=BIG_FONTSIZE)

        plt.tight_layout()
        outpath = os.path.join(PLOT_DIR, "risefade.pdf")
        plt.savefig(outpath)

    if PLOT_LIGHTCURVES:

        from ztfquery import fritz

        # ztf_ids = df_tde.name.values
        ztf_ids = all_objects

        failed_objects = []

        outpath = os.path.join(PLOT_DIR, "sample_lightcurves")
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        outfile = os.path.join(outpath, "agn_and_unknown_in_box.pdf")

        with PdfPages(outfile) as pdf:

            for i, ztf_id in enumerate(ztf_ids):

                print(f"Plotting {ztf_id} (lightcurve {i+1} of {len(ztf_ids)})")

                try:
                    lc = fritz.download_lightcurve(ztf_id, get_object=True)

                    plt.figure(figsize=[5, 3], dpi=DPI)
                    ax = plt.subplot()
                    lc.show(ax=ax)

                    classification = df_full.query(f"name == '{ztf_id}'")[
                        "classification"
                    ].values[0]
                    ax.set_title(f"{ztf_id} / class: {classification}")

                    # outfile = os.path.join(outpath, f"{ztf_id}.png")

                    ax.set_ylim([22, 14.5])
                    # plt.savefig(outfile)
                    pdf.savefig()
                    plt.close()

                except OSError:
                    failed_objects.append(ztf_id)

        print(
            f"Plotting finished; {len(failed_objects)} were unsuccessful: {failed_objects}"
        )

        # lcq = lightcurve.LCQuery.download_data(circle=[298.0025,29.87147,0.0014], bandname="g")

        # from ztfquery import marshal
        # marshal.download_lightcurve("ZTF18abcdef")
        # lcdataframe = marshal.get_local_lightcurves("ZTF19aatubsj")
        # # Plot it
        # marshal.plot_lightcurve(lcdataframe)
