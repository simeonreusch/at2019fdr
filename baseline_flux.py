#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

REDSHIFT = 0.267
FIG_WIDTH = 7
BIG_FONTSIZE = 12
SMALL_FONTSIZE = 8
GOLDEN_RATIO = 1.618
DPI = 400

LIGHTCURVE_DIR = os.path.join("data", "lightcurves")
PLOT_DIR = "plots"

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)


lightcurve_infile = os.path.join(LIGHTCURVE_DIR, "ZTF19aatubsj_SNT_5.0.csv")

df = pd.read_csv(lightcurve_infile)

def get_dataframe_for_band(df, band):
    df = df.query(f"filter == '{band}'").reset_index()
    df.drop(columns=["index", "Unnamed: 0"], inplace=True)
    return df

df_i = get_dataframe_for_band(df, "ZTF_i")
df_i_prepeak = df_i.query("obsmjd <= 58600 ")

text = f"Mean prepeak i-band flux: {np.mean(df_i_prepeak.ampl):.2f} \nMedian prepeak i-band flux: {np.median(df_i_prepeak.ampl):.2f}"


fig = plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH/GOLDEN_RATIO))
ax1 = plt.subplot(111)

colors = {"ZTF_g": "green", "ZTF_r": "red", "ZTF_i": "orange"}
# bands = ["ZTF_g", "ZTF_r", "ZTF_i"]
bands = ["ZTF_i"]

for band in bands:
    df_band = get_dataframe_for_band(df, band)
    ax1.errorbar(
        x=df_band.obsmjd,
        y=df_band.ampl,
        yerr=df_band["ampl.err"],
        color=colors[band],
        fmt="."
    )
ax1.axvline(58600, color="black", ls="dotted")
bbox = dict(boxstyle="round", fc="w", ec="gray")
ax1.annotate(text, (58190,300), bbox=bbox, fontsize=SMALL_FONTSIZE+1)

ax1.set_xlabel("Date [MJD]", fontsize=BIG_FONTSIZE)
ax1.set_ylabel("Flux", fontsize=BIG_FONTSIZE)
outfile = os.path.join(PLOT_DIR, "baseline.png")
plt.tight_layout()
fig.savefig(outfile)







