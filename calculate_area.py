#!/usr/bin/env python
# coding: utf-8

from mmapy.neutrino_scanner import NeutrinoScanner
from ampel.log.AmpelLogger import AmpelLogger
import pandas as pd
import numpy as np
from ztf_plan_obs import gcn_parser

RECALC = True

if RECALC:

    logger = AmpelLogger()

    rectangular_areas = []
    corrected_areas = []
    uncorrected_areas = []
    good_neutrinos = []
    bad_neutrinos = []

    neutrinos = [
        "IC190503A",
        "IC190619A",
        "IC190730A",
        "IC190922B",
        "IC191001A",
        "IC200107A",
        "IC200109A",
        "IC200117A",
        "IC200512A",
        "IC200530A",
        "IC200620A",
        "IC200916A",
        "IC200926A",
        "IC200929A",
        "IC201007A",
        "IC201021A",
        "IC201130A",
        "IC201209A",
        "IC201222A",
        "IC210210A",
        "IC210510A",
        "IC210629A",
        "IC210811A",
    ]

    for i, neutrino in enumerate(neutrinos):

        print(f"Processing {neutrino} ({i+1} of {len(neutrinos)})")

        try:

            nu = NeutrinoScanner(neutrino, logger=logger)
            gcn_no = nu.find_gcn_no(neutrino)
            gcn_info = gcn_parser.parse_gcn_circular(gcn_no)
            RA = gcn_info["ra"]
            Dec = gcn_info["dec"]
            RA_max = nu.ra_max
            RA_min = nu.ra_min
            Dec_max = nu.dec_max
            Dec_min = nu.dec_min

            Dec_0 = np.mean([Dec_max, Dec_min])

            RA_width = RA_max - RA_min
            Dec_width = Dec_max - Dec_min
            correction = np.cos(np.radians(Dec_0))

            rectangular_area = RA_width * Dec_width * correction

            nu.plot_overlap_with_observations(first_det_window_days=4)

            corrected_area = nu.area
            covered_prob = nu.overlap_prob
            uncorrected_area = corrected_area / (covered_prob / 100)

            print(f"Rectangular Area = {rectangular_area}")
            print(f"Uncorrected Area = {uncorrected_area}")
            print(f"Corrected Area = {corrected_area}")
            print(f"Covered percentage = {covered_prob}")

            good_neutrinos.append(neutrino)
            rectangular_areas.append(rectangular_area)
            corrected_areas.append(corrected_area)
            uncorrected_areas.append(uncorrected_area)

        except:

            bad_neutrinos.append(neutrino)
            rectangular_areas.append(None)
            corrected_areas.append(None)
            uncorrected_areas.append(None)

    print(f"Parsing or other error, please recheck {bad_neutrinos}:")

    print(neutrinos)
    print(rectangular_areas)
    print(corrected_areas)
    print(uncorrected_areas)

    df = pd.DataFrame()
    df["neutrino"] = neutrinos
    df["rectangular_area"] = rectangular_areas
    df["uncorrected_area"] = uncorrected_areas
    df["corrected_area"] = corrected_areas

    print(df)
    df.to_csv("areas.csv")

else:

    print(df)
    df = pd.read_csv("areas.csv").drop(columns=["Unnamed: 0"])


# areas = df["corrected_area"].values

# areas = areas[~np.isnan(areas)]

# area_total = np.sum(areas)

# print(area_total)
