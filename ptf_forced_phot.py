#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy
from astropy.table import Table
from modelSED import utilities

FIG_WIDTH = 8
GOLDEN_RATIO = 1.618

for band in ["g", "r"]:

    infile = os.path.join("data", "ptf", f"{band}.txt")

    t = Table.read(infile, format="ascii")

    df = t.to_pandas()
    print(f"min MJD: {np.min(df.MJD.values)}")
    print(f"max MJD: {np.max(df.MJD.values)}")

    abmags = []

    fig = plt.figure(dpi=300, figsize=(FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.errorbar(df.MJD, y=df.flux, yerr=df.sigflux, color=band, fmt=".")
    outfile = os.path.join("plots", "ptf", f"{band}.pdf")
    plt.savefig(outfile)
    plt.close()

infile = os.path.join("data", "ptf", f"r_viraj.txt")
t = Table.read(infile, format="ascii")
df = t.to_pandas()

fig = plt.figure(dpi=300, figsize=(FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO))
ax1 = fig.add_subplot(1, 1, 1)
ax1.errorbar(df.MJD, y=df.flux, yerr=df.sigflux, color=band, fmt=".")
outfile = os.path.join("plots", "ptf", f"r_viraj.pdf")
plt.savefig(outfile)
plt.close()
