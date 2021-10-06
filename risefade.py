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
from nuztf.ampel_api import ampel_api_name, reassemble_alert
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from astropy.cosmology import FlatLambdaCDM

# Run params
PLOT_POPULATION = True
PLOT_LIGHTCURVES = True
RELOAD_LIGHTCURVES = False

# For absolute magnitude calculation
GENERIC_COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)

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

box_minrise = 10
box_maxrise = 50
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

TYWIN_AND_LANCEL = ["ZTF19aatubsj", "ZTF19aaejtoy"]

BAD_OBJECTS = ["ZTF18adaktri"]


def lightcurve_from_alert(
    alert: dict,
    figsize: list = [6.47, 4],
    title: str = None,
    include_ulims: bool = False,
    mag_range: list = None,
    z: float = None,
    logger=None,
):
    """plot AMPEL alerts as lightcurve"""

    if np.isnan(z):
        z = None

    # ZTF color and naming scheme
    BAND_NAMES = {1: "ZTF g", 2: "ZTF r", 3: "ZTF i"}
    BAND_COLORS = {1: "green", 2: "red", 3: "orange"}

    name = alert[0]["objectId"]
    candid = alert[0]["candidate"]
    prv_candid = alert[0]["prv_candidates"]

    if logger is not None:
        logger.debug(f"Plotting {name}")

    df = pd.DataFrame(candid, index=[0])
    df_ulims = pd.DataFrame()

    # Filter out images with negative difference flux
    i = 0
    for prv in prv_candid:
        # Go through the alert history
        if "magpsf" in prv.keys() and "isdiffpos" in prv.keys():
            i += 1
            ser = pd.Series(prv, name=i)
            df = df.append(ser)
        else:
            df_ulims = df_ulims.append(prv, ignore_index=True)
            i += 1

    df["mjd"] = df["jd"] - 2400000.5
    df_ulims["mjd"] = df_ulims["jd"] - 2400000.5

    # Helper functions for the axis conversion (from MJD to days from today)
    def t0_dist(obsmjd):
        t0 = Time(time.time(), format="unix", scale="utc").mjd
        return obsmjd - t0

    def t0_to_mjd(dist_to_t0):
        t0 = Time(time.time(), format="unix", scale="utc").mjd
        return t0 + dist_to_t0

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    fig.subplots_adjust(top=0.8)
    ax2 = ax.secondary_xaxis("top", functions=(t0_dist, t0_to_mjd))

    # If redshift is given, calculate absolute magnitude via luminosity distance
    # and plot as right axis
    if z is not None:

        dist_l = GENERIC_COSMOLOGY.luminosity_distance(z).to(u.pc).value

        def mag_to_absmag(mag):
            absmag = mag - 5 * (np.log10(dist_l) - 1)
            return absmag

        def absmag_to_mag(absmag):
            mag = absmag + 5 * (np.log10(dist_l) - 1)
            return mag

        ax3 = ax.secondary_yaxis("right", functions=(mag_to_absmag, absmag_to_mag))
        ax3.set_ylabel(f"Absolute Magnitude [AB]")

    # Get time now as UTC time
    ts = time.time()
    utc_now = datetime.utcfromtimestamp(ts)
    utc_string = utc_now.strftime("%Y-%m-%d")
    ax2.set_xlabel(f"Days from {utc_string}")

    # Give the figure a title
    if title is None:
        fig.suptitle(f"{name}", fontweight="bold")
    else:
        fig.suptitle(title, fontweight="bold")

    # grid line every 100 days
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.grid(b=True, axis="both", alpha=0.5)
    ax.set_xlabel("MJD")
    ax.set_ylabel("Magnitude [AB]")

    if mag_range is None:
        ax.set_ylim([23, 15])
    else:
        ax.set_ylim([np.max(mag_range), np.min(mag_range)])

    for fid in BAND_NAMES.keys():

        # Plot datapoints
        df_temp = df.query("fid == @fid")
        ax.errorbar(
            df_temp["mjd"],
            df_temp["magpsf"],
            df_temp["sigmapsf"],
            color=BAND_COLORS[fid],
            fmt=".",
            label=BAND_NAMES[fid],
            mec="black",
            mew=0.5,
        )

        # Plot upper limits
        if include_ulims:
            df_temp2 = df_ulims.query("fid == @fid")
            ax.scatter(
                df_temp2["mjd"],
                df_temp2["diffmaglim"],
                c=BAND_COLORS[fid],
                marker="v",
                s=1.3,
                alpha=0.5,
            )

    plt.tight_layout()

    if z is not None:
        axes = [ax, ax2, ax3]
    else:
        axes = [ax, ax2]

    return fig, axes


if __name__ == "__main__":
    CURRENT_FILE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "data"))
    PLOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "plots"))

    infile = os.path.join(DATA_DIR, "ZTF-I_Nuclear_Transients_extra.dat")

    df = pd.read_csv(infile, sep=" ")
    df = df.query("name not in @BAD_OBJECTS")

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
    tywin_mag = tywin[peak_mag_column].values[0]

    df_agn = df_full.query("qclass == 'AGN' and name != 'ZTF19aaejtoy'")
    df_agn_flux_cut = df_agn.query(f"{peak_mag_column} <= @tywin_mag")

    df_unknown = df_full.query("qclass == 'Unknown'")
    df_unknown_flux_cut = df_unknown.query(f"{peak_mag_column} <= @tywin_mag")

    df_sn = df_full.query("qclass == 'SN'")
    df_sn_flux_cut = df_sn.query(f"{peak_mag_column} <= @tywin_mag")

    df_tde = df_full.query("qclass == 'TDE'")
    df_tde_flux_cut = df_tde.query(f"{peak_mag_column} <= @tywin_mag")

    df_prob_tde = df_full.query("qclass == 'lowM'")
    df_prob_tde_flux_cut = df_prob_tde.query(f"{peak_mag_column} <= @tywin_mag")

    print(f"full AGN sample: {len(df_agn)}")
    print(f"full SN sample: {len(df_sn)}")
    print(f"full TDE sample: {len(df_tde)}")
    print(f"full TDE? sample: {len(df_prob_tde)}")
    print(f"full unknown sample: {len(df_unknown)}")
    print("----------------------------------")
    print(
        f"objects surviving flux cut: {len(df_full.query(f'{peak_mag_column} <= @tywin_mag'))}"
    )
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

    from ztfquery import fritz

    df_magcut = df_full.query(f"{peak_mag_column} <= @tywin_mag")

    df_box = df_magcut.query(
        "sigma_rise >= @box_minrise and sigma_rise <= @box_maxrise and sigma_fade >= @box_minfade and sigma_fade <= @box_maxfade"
    )

    df_box = df_box.replace("lowM", "TDE?")

    print(df_box[["name", peak_mag_column]])

    print(f"In the box there are {len(df_box)} objects")

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
                print(f"not making the quality cuts: {ztfid}")
            else:
                print(ztfid)
                print(df_temp2[peak_mag_column])

    if RELOAD_LIGHTCURVES:
        print(f"Downloading and plotting lightcurves")
    else:
        print(f"Plotting lightcurves")

    failed_objects = []

    outpath = os.path.join(PLOT_DIR, "sample_lightcurves")
    json_path = os.path.join(DATA_DIR, "sample_lightcurves")
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = os.path.join(outpath, "box_magcut.pdf")

    # ztf_ids = ["ZTF19aapreis"]

    names = []
    classifications = []
    redshifts = []
    peak_mags = []
    peak_absmags = []

    # with PdfPages(outfile) as pdf:

    for i, ztf_id in enumerate(tqdm(ztf_ids)):
        json_path_lc = os.path.join(json_path, f"{ztf_id}.json")

        if RELOAD_LIGHTCURVES:
            query_res = ampel_api_name(ztf_name=ztf_id, with_history=True)
            with open(json_path_lc, "w") as outfile:
                json.dump(query_res, outfile)
        else:
            with open(json_path_lc) as json_file:
                query_res = json.load(json_file)

        classification = df_box.query(f"name == '{ztf_id}'")["qclass"].values[0]

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

        redshift = df_box.query(f"name == '{ztf_id}'")["redshift"].values[0]

        peak_mag = df_box.query(f"name == '{ztf_id}'")[peak_mag_column].values[0]

        if not np.isnan(redshift):
            dist_l = GENERIC_COSMOLOGY.luminosity_distance(redshift).to(u.pc).value
            peak_absmag = peak_mag - 5 * (np.log10(dist_l) - 1)
        else:
            peak_absmag = None
        peak_absmags.append(peak_absmag)

        text = f"{name} / class: {classification} / z: {redshift}"

        # if PLOT_LIGHTCURVES:

        # fig, axes = lightcurve_from_alert(
        #     alert=query_res,
        #     title=text,
        #     include_ulims=True,
        #     z=redshift,
        # )

        # pdf.savefig()
        # plt.close()

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

    print(df_overview)
    outpath = os.path.join(DATA_DIR, "sample_lightcurves", "final_sample.csv")
    df_overview.to_csv(outpath)
