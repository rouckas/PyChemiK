import numpy as np
np.seterr(invalid = "raise", divide = "raise", over = "raise", under = "ignore")
import matplotlib.pyplot as plt
from pychemik import solve_ODE, State, load_species
from rate_coef import load_reaction_data

##############################
###### input files ###########
##############################
file_species = "data_H5plus/species.txt"       # definition of species with initial concentrations
file_reaction_data = "data_H5plus/collisions.txt"   # reactions + temperature dependent data + cross sections (optional, can be None)

Tn = 50.

time = 200000e-3
time_step = time/1e2

state = State(Tn, None, None)
state.electron_cooling = False

reaction_list = load_reaction_data(file_reaction_data)
species_list, _ = load_species(file_species)


concentrations, cas, vyvoj, speci, Te = solve_ODE(time, time_step, species_list, reaction_list, state, method="solve_ivp")

for i in range(len(speci)):
    print(speci[i].name, ": \t %e" % speci[i].conc)

f, ax = plt.subplots()
for i in range(len(speci)):
    ax.plot(cas, vyvoj[:,i], label=speci[i].name)
    ax.set_yscale("log")
ax.legend()
plt.savefig("result.pdf")
