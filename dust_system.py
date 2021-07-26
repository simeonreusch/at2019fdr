#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

from astropy import units as u
from astropy import constants as const
import pandas as pd
import numpy as np

infile_fitdf = "fit_lumi_radii.csv"

df_fit = pd.read_csv(infile_fitdf)

opt_ir_delay_day = (59074.01 - 58709.84)*u.day

opt_ir_delay_s = opt_ir_delay_day.to(u.s)

light_travel_distance = opt_ir_delay_s*const.c

light_travel_distance = light_travel_distance.to(u.cm)


fitted_max_optical_radius = np.max(df_fit["optical_radius"].values)
fitted_max_optical_radius = fitted_max_optical_radius * u.cm

fitted_max_ir_radius = np.max(df_fit["infrared_radius"].values)
fitted_max_ir_radius = fitted_max_ir_radius * u.cm

print(f"time delay between optical and IR peak: {opt_ir_delay_day:.0f}")
print(f"dust distance inferred from light travel time: {light_travel_distance:.2e}")
print(f"dust distance inferred from BB fit: {fitted_max_ir_radius:.2e}")
