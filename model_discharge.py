import numpy as np
import matplotlib.pyplot as plt
from pychemik import solve_ODE, State, load_species
from rate_coef import load_reaction_data

##############################
###### input files ###########
##############################
#soubor_rozdel = "data/collisions/eedf_He_H2_14Td_300K.txt"
#soubor_rozdel = "data/collisions/eedf_He_H2_14Td_77K.txt"
file_EEDF = "data/collisions/elendif/eedf_He_H2_14Td_100_1000.txt" # electron energy distribution function
file_species = "data/species.txt"       # definition of species with initial concentrations
file_reaction_data = "data/collisions/electron.txt"   # reactions + temperature dependent data + cross sections (optional, can be None)

Tn = 77
Te = 20000

time = 20e-3
time_step = time/10e2


r = 7.5e-3
diffusion_length = (r/2.405)**2

state = State(Tn, Te, file_EEDF)
state.diffusion_length = diffusion_length
state.electron_cooling = False

reaction_list = load_reaction_data(file_reaction_data)
species_list, _ = load_species(file_species)


concentrations, cas, vyvoj, speci, Te = solve_ODE(time, time_step, species_list, reaction_list, state)

for i in range(len(speci)):
    print(speci[i].name, ": \t %e" % speci[i].conc)

print(Te)
time_step = 1e-6
time = 1e-5
state.electron_cooling = True

species_list, _ = load_species(file_species)

concentrations, cas, vyvoj, speci, Te = solve_ODE(time, time_step, species_list, reaction_list, state)

for i in range(len(speci)):
    print(speci[i].name, ": \t %e" % speci[i].conc)

f, ax = plt.subplots()
for i in range(len(speci)):
    ax.plot(cas, vyvoj[:,i], label=speci[i].name)
    ax.set_yscale("log")
plt.savefig("result.pdf")
