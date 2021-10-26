#!/usr/bin/env python3
# Author: Sjoert van Velzen, Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, time, argparse, json
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
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
from ztfquery import lightcurve, alert
from nuztf.ampel_api import ampel_api_name
from nuztf.plot import lightcurve_from_alert
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from astropy.cosmology import FlatLambdaCDM

# Run params
PLOT_POPULATION = True
PLOT_LIGHTCURVES = False
LOAD_LIGHTCURVES = True

# Cut params
first_detection_cut = "2018-01-01"

magcut_mag = 18.187388475116148
# magcut_mag = 19

rise_error_ratio = 1
fade_error_ratio = 1
max_chisq = 999
min_peakmag = 0

max_offset_weighted_r = 0.5
max_offset_weighted_g = 0.5

min_ndetections_g = 10
min_ndetections_r = 10

max_negative_detections_ratio = 0.5

# box_minrise = 10 * np.sqrt(2)
# box_maxrise = 55 * np.sqrt(2)
box_minrise = 10
box_maxrise = 55
# box_maxrise = 100
box_minfade = 30
box_maxfade = 500


peak_mag_column = "mag_peak"


XRT_COLUMN = "flux0310_bb_25eV"
nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)

# For absolute magnitude calculation
GENERIC_COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)

REDSHIFT = 0.267
FIG_WIDTH = 6
BIG_FONTSIZE = 14
SMALL_FONTSIZE = 8
GOLDEN_RATIO = 1.618
DPI = 400

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

# HIGHLIGHTED_OBJECTS = ["ZTF19aatubsj", "ZTF19aaejtoy", "ZTF19aapreis"]
# TYWIN_LANCEL = ["ZTF19aatubsj", "ZTF19aaejtoy"]
HIGHLIGHTED_OBJECTS = ["ZTF19aatubsj", "ZTF19aapreis"]
TYWIN_LANCEL = ["ZTF19aatubsj"]

BRAN = ["ZTF19aapreis"]

BAD_OBJECTS = ["ZTF18adaktri"]


if __name__ == "__main__":
    CURRENT_FILE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "data"))
    PLOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "plots"))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    infile = os.path.join(DATA_DIR, "ZTF-I_Nuclear_Transients_extra_with_dates.csv")

    df = pd.read_csv(infile)

    df["sigma_rise"] = df["sigma_rise"] * np.sqrt(2)
    df["e_sigma_rise"] = df["e_sigma_rise"] * np.sqrt(2)

    logger.info(f"Complete dataset: {len(df)} entries.")

    df = df.query("name not in @BAD_OBJECTS")

    first_detection_cut_jd = Time(first_detection_cut, format="isot").jd

    df = df.query("jd_start_hist >= @first_detection_cut_jd")

    logger.info(f"Surviving jd_start cut: {len(df)}")

    # Now we massage the data a bit
    ireal_flare = np.repeat(True, len(df))
    ireal_flare *= (df["offset_weighted_r"] <= max_offset_weighted_r) + (
        df["offset_weighted_g"] <= max_offset_weighted_g
    )
    ireal_flare *= (
        df["ndetections_negative"] / df["ndetections"] < max_negative_detections_ratio
    )
    ireal_flare *= np.array((df["ndetections_g"] >= min_ndetections_g)) + np.array(
        (df["ndetections_r"] >= min_ndetections_r)
    )
    ireal_flare *= df["chi2"] != 0

    df_temp = df
    df_temp = df_temp[ireal_flare]
    logger.info(f"Surviving quality cuts: {len(df_temp)}")

    plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO))

    ax1 = plt.subplot(111)
    ax1.hist(Time(df_temp["flare_peak_jd"].values, format="jd").datetime)
    plt.savefig("test.png")
    quit()

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
        * (df["sigma_rise"] > box_minrise)
        * (df["sigma_rise"] < box_maxrise)
        * (df["sigma_fade"] > box_minfade)
        * (df["sigma_fade"] < box_maxfade)
    )

    isel_full *= ireal_flare * iclass_ok

    df_box = df[isel_box]
    df_full = df[isel_full]

    logger.info(f"Survive quality and classification cuts: {len(df_full)}")

    tywin = df.query("name == 'ZTF19aatubsj'")
    tywin_mag = tywin[peak_mag_column].values[0]

    df_agn = df_full.query("qclass == 'AGN'")
    df_agn_flux_cut = df_agn.query(f"{peak_mag_column} <= @magcut_mag")
    df_agn_flux_cut_box = df_agn_flux_cut.query(
        "sigma_rise >= @box_minrise and sigma_rise <= @box_maxrise and sigma_fade >= @box_minfade and sigma_fade <= @box_maxfade"
    )

    df_unknown = df_full.query("qclass == 'Unknown'")
    df_unknown_flux_cut = df_unknown.query(f"{peak_mag_column} <= @magcut_mag")
    df_unknown_flux_cut_box = df_unknown_flux_cut.query(
        "sigma_rise >= @box_minrise and sigma_rise <= @box_maxrise and sigma_fade >= @box_minfade and sigma_fade <= @box_maxfade"
    )

    df_sn = df_full.query("qclass == 'SN'")
    df_sn_flux_cut = df_sn.query(f"{peak_mag_column} <= @magcut_mag")
    df_sn_flux_cut_box = df_sn_flux_cut.query(
        "sigma_rise >= @box_minrise and sigma_rise <= @box_maxrise and sigma_fade >= @box_minfade and sigma_fade <= @box_maxfade"
    )

    df_tde = df_full.query("qclass == 'TDE'")
    df_tde_flux_cut = df_tde.query(f"{peak_mag_column} <= @magcut_mag")
    df_tde_flux_cut_box = df_tde_flux_cut.query(
        "sigma_rise >= @box_minrise and sigma_rise <= @box_maxrise and sigma_fade >= @box_minfade and sigma_fade <= @box_maxfade"
    )

    df_prob_tde = df_full.query("qclass == 'lowM'")
    df_prob_tde_flux_cut = df_prob_tde.query(f"{peak_mag_column} <= @magcut_mag")
    df_prob_tde_flux_cut_box = df_prob_tde_flux_cut.query(
        "sigma_rise >= @box_minrise and sigma_rise <= @box_maxrise and sigma_fade >= @box_minfade and sigma_fade <= @box_maxfade"
    )

    all_surviving = (
        len(df_agn) + len(df_tde) + len(df_prob_tde) + len(df_sn) + len(df_unknown)
    )

    flux_cut_surviving = (
        len(df_agn_flux_cut)
        + len(df_tde_flux_cut)
        + len(df_prob_tde_flux_cut)
        + len(df_sn_flux_cut)
        + len(df_unknown_flux_cut)
    )

    box_surviving = (
        len(df_agn_flux_cut_box)
        + len(df_tde_flux_cut_box)
        + len(df_prob_tde_flux_cut_box)
        + len(df_sn_flux_cut_box)
        + len(df_unknown_flux_cut_box)
    )

    logger.info("----------------------------------")
    logger.info(f"objects surviving quality and classification cuts: {all_surviving}")
    logger.info("----------------------------------")
    logger.info(f"full AGN sample: {len(df_agn)}")
    logger.info(f"full SN sample: {len(df_sn)}")
    logger.info(f"full TDE sample: {len(df_tde)}")
    logger.info(f"full TDE? sample: {len(df_prob_tde)}")
    logger.info(f"full unknown sample: {len(df_unknown)}")
    logger.info("----------------------------------")
    logger.info(f"objects surviving flux cut: {flux_cut_surviving}")
    logger.info("----------------------------------")
    logger.info(f"flux cut AGN sample: {len(df_agn_flux_cut)}")
    logger.info(f"flux cut SN sample: {len(df_sn_flux_cut)}")
    logger.info(f"flux cut TDE sample: {len(df_tde_flux_cut)}")
    logger.info(f"flux cut TDE? sample: {len(df_prob_tde_flux_cut)}")
    logger.info(f"flux cut unknown sample: {len(df_unknown_flux_cut)}")
    logger.info(f"flux cut ALL: {flux_cut_surviving}")
    logger.info("----------------------------------")
    logger.info(f"Objects in Box: {box_surviving}")
    logger.info("----------------------------------")
    logger.info(f"flux cut AGN sample in box: {len(df_agn_flux_cut_box)}")
    logger.info(f"flux cut SN sample in box: {len(df_sn_flux_cut_box)}")
    logger.info(f"flux cut TDE sample in box: {len(df_tde_flux_cut_box)}")
    logger.info(f"flux cut TDE? sample in box: {len(df_prob_tde_flux_cut_box)}")
    logger.info(f"flux cut unknown sample in box: {len(df_unknown_flux_cut_box)}")
    logger.info("----------------------------------")

    if PLOT_POPULATION:

        plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO))

        ax1 = plt.subplot(111)

        ax1.set_xlim(4, 2e3)
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
                df_full.query("name in @TYWIN_LANCEL"),
                "h",
                "tab:blue",
                "tab:blue",
                0.9,
                None,
                None,
            ],
            "bran": [
                df_full.query("name in @BRAN"),
                "P",
                "tab:blue",
                "tab:blue",
                0.9,
                None,
                None,
            ],
            # "extra": [
            #     df_full.query("name == 'ZTF19abzrhgq'"),
            #     "P",
            #     "black",
            #     "black",
            #     1,
            #     20,
            #     None,
            # ]
        }

        # check = df_full.query("name == 'ZTF19adcddzk'")
        # print(check[["sigma_rise", "e_sigma_rise", "sigma_fade", "e_sigma_fade", peak_mag_column, "qclass"]])
        # quit()

        # remove highlighted ones so not drawn doubly
        for entry in data:
            if entry != "tywin_lancel" and entry != "bran":
                data[entry][0] = data[entry][0].query(
                    "name not in @HIGHLIGHTED_OBJECTS"
                )

        for entry in data.values():
            ax1.scatter(
                entry[0]["sigma_rise"],
                entry[0]["sigma_fade"],
                marker=entry[1],
                facecolors=entry[2],
                edgecolors=entry[3],
                alpha=entry[4],
                s=entry[5],
                label=entry[6],
            )

        # df_temp_box = df_full.query(f"{peak_mag_column} <= @magcut_mag and sigma_rise >= @box_minrise and sigma_rise <= @box_maxrise and sigma_fade >= @box_minfade and sigma_fade <= @box_maxfade")

        # ax1.scatter(
        #     df_temp_box["sigma_rise"],
        #     df_temp_box["sigma_fade"],
        # )

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
        outpath = os.path.join(
            PLOT_DIR, "sample_lightcurves", f"risefade_{magcut_mag:.2f}.pdf"
        )
        plt.savefig(outpath)

    df_full = df_full.replace("lowM", "TDE?")

    df_magcut = df_full.query(f"{peak_mag_column} <= @magcut_mag")
    df_box = df_magcut.query(
        "sigma_rise >= @box_minrise and sigma_rise <= @box_maxrise and sigma_fade >= @box_minfade and sigma_fade <= @box_maxfade"
    )

    logger.info(f"\nIn the box there are {len(df_box)} objects")

    ztf_ids = df_box["name"].values

    ztf_ids_sjoert = [
        "ZTF19aatubsj",
        "ZTF18acgqweq",
        "ZTF19aaciohh",
        "ZTF19aapreis",
        "ZTF18acrygry",
        "ZTF19aawlgne",
        "ZTF18aanlzzf",
        "ZTF18abtswjk",
        "ZTF18aapzqup",
        "ZTF19abclykm",
        "ZTF18abjhbss",
        "ZTF19aaejtoy",
        "ZTF17aaazdba",
    ]

    for ztfid in ztf_ids_sjoert:
        df_temp1 = df_box.query("name == @ztfid")
        if len(df_temp1) < 1:
            df_temp2 = df_full.query("name == @ztfid")
            if len(df_temp2) < 1:
                logger.info(f"not making the quality cuts: {ztfid}")
            else:
                logger.info(ztfid)
                logger.info(df_temp2[peak_mag_column])

    if LOAD_LIGHTCURVES:
        logger.info(f"Downloading and plotting lightcurves")
    else:
        logger.info(f"Plotting lightcurves")

    failed_objects = []

    outpath = os.path.join(PLOT_DIR, "sample_lightcurves")
    json_path = os.path.join(DATA_DIR, "sample_lightcurves")
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = os.path.join(outpath, f"box_magcut_{magcut_mag:.2f}_peculiar.pdf")

    names = []
    classifications = []
    redshifts = []
    peak_mags = []
    peak_absmags = []

    # red = ["ZTF18acrygry", "ZTF18abjhbss", "ZTF19aafltef", "ZTF18aabicqe", "ZTF18abbuwwg", "ZTF18aabeeiz", "ZTF18abfhgug", "ZTF18aarutmj", "ZTF18aahvkxq"]

    # green = ["ZTF19aarioci",  "ZTF19aaiqmgl", "ZTF19abvgxrq", "ZTF19aaejtoy", "ZTF19aaciohh", "ZTF19aamjjcx", "ZTF19aapreis", "ZTF19abzrhgq", "ZTF19abclykm", "ZTF18aanlzzf", "ZTF18aapzqup", "ZTF19aatubsj"]

    three_peculiars = ["ZTF19aatubsj", "ZTF18aanlzzf", "ZTF19adcddzk"]

    ztf_ids = three_peculiars

    with PdfPages(outfile) as pdf:

        for i, ztf_id in enumerate(tqdm(ztf_ids)):
            json_path_lc = os.path.join(json_path, f"{ztf_id}.json")

            if not os.path.isfile(json_path_lc):
                logger.info(f"Querying API for {ztf_id}")
                query_res = ampel_api_name(ztf_name=ztf_id, with_history=True)
                json.dump(query_res, open(json_path_lc, "w"))
            else:
                query_res = json.load(open(json_path_lc))

            classification = df_full.query(f"name == '{ztf_id}'")["qclass"].values[0]

            name = ztf_id

            if ztf_id == "ZTF19aafljiq":
                classification = "SLSN-II"

            if ztf_id == "ZTF19aapreis":
                name += " (Bran)"

            if ztf_id == "ZTF19abclykm":
                name += " (Steffon)"

            if ztf_id == "ZTF19aaejtoy":
                name += " (Lancel)"

            if ztf_id == "ZTF19aatubsj":
                name += " (Tywin)"

            if ztf_id == "ZTF19abovsqr":
                classification += " (BUT FROM WHERE?)"

            redshift = df_full.query(f"name == '{ztf_id}'")["redshift"].values[0]

            peak_mag = df_full.query(f"name == '{ztf_id}'")[peak_mag_column].values[0]

            if not np.isnan(redshift):
                dist_l = GENERIC_COSMOLOGY.luminosity_distance(redshift).to(u.pc).value
                peak_absmag = peak_mag - 5 * (np.log10(dist_l) - 1)
            else:
                peak_absmag = None
            peak_absmags.append(peak_absmag)

            text = f"{name} / class: {classification} / z: {redshift}"

            # if PLOT_LIGHTCURVES:

            fig, axes = lightcurve_from_alert(
                alert=query_res,
                title=text,
                include_ulims=True,
                include_cutouts=False,
                mag_range=[16, 21],
                z=redshift,
                grid_interval=100,
            )

            pdf.savefig()
            plt.close()

            names.append(name)
            redshifts.append(redshift)
            classifications.append(classification)
            peak_mags.append(peak_mag)

    # Create overview
    df_overview = pd.DataFrame()
    df_overview["names"] = names
    df_overview["classification"] = classifications
    df_overview["redshift"] = redshifts
    df_overview["peak_mag"] = peak_mags
    df_overview["peak_absmag"] = peak_absmags

    logger.info(df_overview)
    outpath = os.path.join(
        DATA_DIR, "sample_lightcurves", f"final_sample_magcut_{magcut_mag:.2f}_new.csv"
    )
    df_overview.to_csv(outpath)
