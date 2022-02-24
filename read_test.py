import numpy as np
import rate_coef as RC


##############################
###### input files ###########
##############################
file_Edist = "data/collisions/elendif/eedf_He_H2_14Td_100_1000.txt" # electron energy distribution function
file_reaction_data_old = "data/collisions/electron_old.txt"  # reactions + temperature dependent data + cross sections (optional, can be None)
file_reaction_coeffs_old = "data/collisions/reaction_old.txt" # reactions with rate coefficients
file_reaction_coeffs_EEDF_old = "data/collisions/reaction_EEDF_old.txt" # reactions with rate coefficients

file_reaction_data = "data/collisions/electron.txt"  # reactions + temperature dependent data + cross sections (optional, can be None)
file_reaction_coeffs = "data/collisions/reaction.txt" # reactions with rate coefficients
file_reaction_coeffs_EEDF = "data/collisions/reaction_EEDF.txt" # reactions with rate coefficients

Tn = 77
Te = 77


# test_thermal

rlist = RC.load_reaction_data(file_reaction_data)
RC.print_reaction_coeffs_file(rlist, RC.State(77., 77., None), file_reaction_coeffs)

RC.read_file(file_reaction_data_old, None, file_reaction_coeffs_old, Te, maxwell=True)

# test EEDF
EEDF = np.loadtxt(file_Edist)
RC.print_reaction_coeffs_file(rlist, RC.State(77., 77., EEDF), file_reaction_coeffs_EEDF)

RC.read_file(file_reaction_data_old, file_Edist, file_reaction_coeffs_EEDF_old, Te, maxwell=False)
