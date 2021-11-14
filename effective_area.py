#!/usr/bin/env python3

import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d


DPI = 400
FIG_WIDTH = 6
GOLDEN_RATIO = 1 / 1.618

BASEPATH = os.path.join("data", "effective_area")
PLOT_DIR = os.path.join("plots", "effective_area")

corona = pd.read_csv(os.path.join(BASEPATH, "corona.csv"), index_col=0)
jet = pd.read_csv(os.path.join(BASEPATH, "jet.csv"), index_col=0)
wind = pd.read_csv(os.path.join(BASEPATH, "wind.csv"), index_col=0)
Aeff_ehe = pd.read_csv(
    os.path.join(BASEPATH, "Aeff_ehe_gfu.csv"), names=("E_Tev", "Aeff_m2")
).apply(lambda x: np.log10(x))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(corona["log10E_GeV"], corona["E2_Fmu_GeV"], "r", label="Disk-corona")
ax1.plot(
    jet["log10E_GeV"], jet["E2_Fmu_GeV"], linestyle="dashdot", label="Relativistic Jet"
)
ax1.plot(wind["log10E_GeV"], wind["E2_Fmu_GeV"], "g--", label="Sub-relativistic wind")
ax2.plot(Aeff_ehe["E_Tev"] + 3, Aeff_ehe["Aeff_m2"], label="Aeff")

ax1.legend()
ax2.legend()
ax1.axis([2, 10, -6, -1])
ax2.axis([2, 10, -2, 3])
ax1.set_xlabel("E [GeV]")
ax2.set_xlabel("E [GeV]")
ax1.set_ylabel("$E^{2}F [GeV/cm^{2}]$")
ax2.set_ylabel("$A_{eff} [m^{2}]$")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "1.pdf"))
plt.close()

Model = corona
Model["Fmu_m-2GeV-1"] = np.log10(
    (1 / 1e-4) * (10 ** Model["E2_Fmu_GeV"] / ((10 ** Model["log10E_GeV"]) ** 2))
)

f = interp1d(Model["log10E_GeV"], Model["Fmu_m-2GeV-1"])

maxlog = np.max(Model["log10E_GeV"])
minlog = np.min(Aeff_ehe["E_Tev"] + 3)
xnew = np.linspace(minlog, maxlog, 10000)

fig, ax1 = plt.subplots(1, 1)
# plt.figure(5)
ax1.plot(Model["log10E_GeV"], Model["Fmu_m-2GeV-1"], marker=".")
ax1.plot(xnew, f(xnew))
ax1.set_xlabel(r"log10 E$_{\nu}$ [GeV]")
ax1.set_ylabel(r"F$_{\mu}$ [m$^{-2}$ GeV$^{-1}$]")
plt.savefig(os.path.join(PLOT_DIR, "2.pdf"))


tck = interpolate.splrep(Aeff_ehe["E_Tev"] + 3, Aeff_ehe["Aeff_m2"], s=1)

maxpoint = np.min([np.max(Aeff_ehe["E_Tev"] + 3), np.max(Model["log10E_GeV"])])
xnew = np.linspace(np.min(Aeff_ehe["E_Tev"] + 3), maxpoint, 100000)
Aeff = interpolate.splev(xnew, tck)

fig, ax1 = plt.subplots(1, 1)
ax1.plot(Aeff_ehe["E_Tev"] + 3, Aeff_ehe["Aeff_m2"], marker=".")
ax1.plot(xnew, Aeff)
ax1.set_xlabel(r"log10 E$_{\nu}$ [TeV]")
ax1.set_ylabel(r"log10 ")
plt.savefig(os.path.join(os.path.join(PLOT_DIR, "3.pdf")))
plt.close()

N = np.trapz((10 ** f(xnew)) * (10 ** Aeff), x=10 ** xnew)
print(N)


def calculate_N(Aeff, Model):
    Model["Fmu_m-2GeV-1"] = np.log10(
        (1 / 1e-4) * (10 ** Model["E2_Fmu_GeV"] / ((10 ** Model["log10E_GeV"]) ** 2))
    )
    f_nu_interp = interp1d(Model["log10E_GeV"], Model["Fmu_m-2GeV-1"])
    tck = interpolate.splrep(Aeff["E_Tev"] + 3, Aeff["Aeff_m2"], s=1)
    maxpoint = np.min([np.max(Aeff["E_Tev"] + 3), np.max(Model["log10E_GeV"])])
    minpoint = np.max([np.min(Aeff["E_Tev"] + 3), np.min(Model["log10E_GeV"])])
    xnew = np.linspace(minpoint, maxpoint, 100000)
    Aeff_interp = interpolate.splev(xnew, tck)
    N = np.trapz((10 ** f_nu_interp(xnew)) * (10 ** Aeff_interp), x=10 ** xnew)
    return N


print(
    calculate_N(Aeff_ehe, corona),
    calculate_N(Aeff_ehe, jet),
    calculate_N(Aeff_ehe, wind),
)

print(f"corona: {calculate_N(Aeff_ehe,corona):.4f}")
print(f"wind: {calculate_N(Aeff_ehe,wind):.4f}")
print(f"jet: {calculate_N(Aeff_ehe,jet):.4f}")
