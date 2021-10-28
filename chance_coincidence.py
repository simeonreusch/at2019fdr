#!/usr/bin/env python3
# Author: Sjoert van Velzen

import scipy.stats
import numpy as np

N_nu = 1  # number of neutrinos
N_flare = 12  # Tywin-like events

flare_duration = 1  # yr (typical)
search_window = 2.5  # yr

ztf_sky = 28e3  # deg2
icecube_90sky = 154.33  # deg2

eff_dens = (
    N_flare / ztf_sky * flare_duration / search_window
)  # effective source density (deg^-2)
mu = eff_dens * icecube_90sky
# expectation value for number of neutrinos

p_pois = 1 - scipy.stats.poisson.cdf(N_nu - 1, mu)

print("effective source density  	: {0:0.2e} per deg2".format(p_pois))
print("neutrino expectation value	: {0:0.3f}".format(mu))
print("Poisson probability       	: {0:0.2e}".format(p_pois))

ff = np.linspace(1, 4)
nsig = np.interp(p_pois, 1 - (scipy.stats.norm.cdf(ff, 0, 1))[::-1], ff[::-1])

print("That is {0:0.2f} sigma".format(nsig))
