#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause


import numpy as np


# IC200530A
# RA = 255.37
# RA_err = [2.48, -2.56]
# Dec = 26.61
# Dec_err = [2.33, -3.28]

# IC190503A
# RA = 120.28
# RA_err = [0.57, -0.77]
# Dec = 6.35
# Dec_err = [0.76, -0.7]

# IC190619A
RA = 343.26
RA_err = [4.08, -2.63]
Dec = 10.73
Dec_err = [1.51, -2.61]


Dec_0 = np.mean([Dec + Dec_err[0], Dec + Dec_err[1]])

RA_width = (RA + RA_err[0]) - (RA + RA_err[1])
Dec_width = (Dec + Dec_err[0]) - (Dec + Dec_err[1])
correction = np.cos(np.radians(Dec_0))

print(RA_width)
print(Dec_width)
print(Dec_0)
print(correction)

area = RA_width * Dec_width * correction

print(f"The rectangular area reported is {area:.2f} sq. deg.")
