import numpy as N
import scipy.sparse as sp
from scipy.integrate import ode
import re
import rate_coef as RC

from pychemik import solve_ODE

##############################
##### global variables #######
##############################
Te_last_coeffs = 1.73
Te_last = 1.73
conc_srov = 0

##############################
##### physical constants #####
##############################
from scipy.constants import Boltzmann, elementary_charge, epsilon_0, physical_constants
Q0 = elementary_charge
k_b = Boltzmann
AMU = physical_constants["atomic mass constant"][0]
DE_CRR = 0.13#*1.6e-19
CRR_factor = 3.8e-9

# Helium atomic mass
ram_He = 4.0026

# neutral elastic collision rate coeff
difu = 1.26076522957e-12

##############################
###### input files ###########
##############################
#soubor_rozdel = "data/collisions/eedf_He_H2_14Td_300K.txt"
#soubor_rozdel = "data/collisions/eedf_He_H2_14Td_77K.txt"
file_Edist = "data/collisions/elendif/eedf_He_H2_14Td_100_1000.txt" # electron energy distribution function
file_species = "data/species.txt"       # definition of species with initial concentrations
file_reaction_data = "data/collisions/electron.txt"   # reactions + temperature dependent data + cross sections (optional, can be None)
file_reaction_coeffs = "data/collisions/reaction.txt" # reactions with rate coefficients

Tn = 77
Te = 20000

time = 20e-3
time_step = time/10e2


r = 7.5e-3
l = (r/2.405)**2


concentrations, cas, vyvoj, speci, Te = solve_ODE(time, time_step, file_species, file_reaction_data, file_Edist, file_reaction_coeffs, Te, False)

for i in range(len(speci)):
    print(speci[i].name, ": \t %e" % speci[i].conc)

print(Te)
time_step = 1e-6
time = 1e-5
concentrations, cas, vyvoj, speci, Te = solve_ODE(time, time_step, file_species, file_reaction_data, None, file_reaction_coeffs, Te, True)

for i in range(len(speci)):
    print(speci[i].name, ": \t %e" % speci[i].conc)

import matplotlib.pyplot as plt
f, ax = plt.subplots()
for i in range(len(speci)):
    ax.plot(cas, vyvoj[:,i], label=speci[i].name)
    ax.set_yscale("log")
plt.savefig("result.pdf")
