#!/usr/bin/env python3
# Author: Sjoert van Velzen, Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, time, argparse, json, math, pickle
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import constants as const
from astropy.table import Table
from astropy import units as u
import matplotlib
from nuztf.ampel_api import ampel_api_name
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

first_detection_cut = "2018-01-01"

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


if __name__ == "__main__":
    BASEDIR = os.path.join("/", "Users", "simeon", "tywin")
    DATA_DIR = os.path.join(BASEDIR, "data")
    PLOT_DIR = os.path.join(BASEDIR, "plots")

    infile = os.path.join(DATA_DIR, "ZTF-I_Nuclear_Transients_extra.dat")
    outfile = os.path.join(DATA_DIR, "ZTF-I_Nuclear_Transients_extra_with_dates.csv")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # df = pd.read_csv(infile, sep=" ")

    # ztf_ids_checked = []
    # jdstarthists = []
    # jdendhists = []

    # for ztf_id in tqdm(df["name"].values):
    #     if not isinstance(ztf_id, str) or ztf_id == "ZTF18aaiyzuz":
    #         print("not a ZTF ID")
    #         ztf_ids_checked.append(None)
    #         jdstarthists.append(None)
    #         jdendhists.append(None)
    #     else:
    #         print(f"Querying API for {ztf_id}")
    #         outfile = os.path.join(DATA_DIR, "temp", "alerts", f"{ztf_id}.p")

    #         if os.path.isfile(outfile):
    #             query_res = pickle.load(open(outfile, "rb"))
    #         else:
    #             query_res = ampel_api_name(ztf_id, with_history=False, logger=logger)

    #         jdstarthist = query_res[0]["candidate"]["jdstarthist"]
    #         jdendhist = query_res[0]["candidate"]["jdendhist"]
    #         jdstarthists.append(jdstarthist)
    #         jdendhists.append(jdendhist)
    #         ztf_ids_checked.append(ztf_id)

    #         if not os.path.isfile(outfile):
    #             pickle.dump(query_res, open(outfile, "wb"))

    # df["jd_start_hist"] = jdstarthists
    # df["jd_end_hist"] = jdendhists

    # df.to_csv(outfile)
    # del df

    df_full = pd.read_csv(outfile)

    df = df_full.dropna(subset=["name"])

    # first_detection_cut_jd = Time(first_detection_cut, format="isot").jd

    # df = df.query("jd_start_hist >= @first_detection_cut_jd")

    date_start_hist = []
    date_end_hist = []

    for i, entry in tqdm(enumerate(df["jd_start_hist"].values)):
        date_start_hist.append(Time(entry, format="jd").datetime)
        date_end_hist.append(Time(df["jd_end_hist"].values[i], format="jd").datetime)

    df["date_start_hist"] = date_start_hist
    df["date_end_hist"] = date_end_hist

    first_datapoint_in_sample = np.min(df["jd_start_hist"].values)
    first_datapoint_in_sample_date = Time(
        first_datapoint_in_sample, format="jd"
    ).datetime
    print(f"First datapoint in sample: {first_datapoint_in_sample_date}")

    plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO))
    ax1 = plt.subplot(111)
    ax1.tick_params(axis="x", which="major", labelsize=6)
    ax1.set_xlabel("Date of first datapoint in lightcurve")
    ax1.set_ylabel("Entries")
    ax1.hist(df["date_start_hist"])
    outfile = os.path.join(PLOT_DIR, "nuclear_sample_startdate.png")
    plt.savefig(outfile)
    plt.close()

    plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO))
    ax1 = plt.subplot(111)
    ax1.tick_params(axis="x", which="major", labelsize=6)
    ax1.set_xlabel("Date of last datapoint in lightcurve")
    ax1.set_ylabel("Entries")
    ax1.hist(df["date_end_hist"])
    outfile = os.path.join(PLOT_DIR, "nuclear_sample_enddate.png")
    plt.savefig(outfile)
    plt.close()
