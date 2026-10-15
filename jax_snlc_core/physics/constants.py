"""Physical constants (SNANA conventions), shared by the simulator and
fitter's flux/photon and magnitude conversions."""

import math

LIGHT_A = 2.99792458e18   # Angstroms per second
LIGHT_KM = 2.99792458e5   # km per second (SNANA LIGHT_km)
PLANCK = 6.6260755e-27    # erg s
HC_ERG_A = LIGHT_A * PLANCK
BOLTZMANN_ERG_K = 1.380649e-16   # erg per kelvin (CODATA 2018)

LN10 = math.log(10.0)
# 10^(-0.4*x) = exp(NEG_0_4_LN10 * x)
NEG_0_4_LN10 = -0.4 * LN10
