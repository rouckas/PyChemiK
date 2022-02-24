import numpy as N
import scipy.sparse as sp
from scipy.integrate import ode
import re
import rate_coef as RC


class Prvky:
    def __init__(self, name = 0, conc = 0, mass = 0, radii = 0):
        self.name = name
        self.conc = conc
        self.mass = mass
        self.radii = radii
        
class Rovnice:
    def __init__(self, electron = False, k_c = 0, Eloss = 0):
        self.electron = electron
        self.k_c = k_c
        self.Eloss = Eloss
        #self.reakce = reakce
  
def load_species(fname):
    species = open(fname,"r")
    species_list = []	
    species_numbers = {}
    index = 0			
    for line in species:
        line = line.partition("#")[0]
        try:
            name, init_conc, mass, radii = line.split()
            species_list.append(Prvky(name, float(init_conc), float(mass), float(radii)))
            species_numbers[name] = len(species_list)-1
        except: continue
    species.close()
    return species_list, species_numbers
        
def load_reactions(fname, specdict, speclist):
    """load the list of reactions from file 'fname'

    Returns:
        k_c[i]: list of reaction rate coefficients
        REACT[i, j]: matrix(#reactions, #species), sparse matrix that specifies
            the reactants of each species. If REACT[i, j] == N, then N molecules of
            species j participate as reactants in reaction j (typically, N=0 or 1).
        Z[i, j]: number of molecules of species j produced in reaction i.
            Z[i, j] < 0 if species j is destroyed in reaction i.
        Eloss[i]: electron energy loss in inelastic collisions
        Elastic[k]: list elastic electron collisions. Elastic[k] = [i, j], where
            i = index of the reaction
            j = the second interacting species
        R_special[reaction_type]: dictionary of lists of special reactions with
            non-constant reaction rate coefficients. E.g.
            R_special["ambi_dif"][0] = [i, m], where i is and index of an effective
            ambipolar diffusion reaction and m is the mass of the ionic species
    """
    reaction = open(fname,"r")
    index = 0
    k_c = []
    REACT = []
    PROD = []
    Eloss = []
    Elastic = []
    R_special = {"difuze":[], "ambi_dif":[], "Stevefelt":[]}
    for line in reaction:             
        try:
            rovnice = line.split()
            if len(rovnice) == 0: continue

            if ("ENERGY LOSS = " in line):
                E_L = re.search("ENERGY LOSS =\s+(\S+)", line).group(1)
                Eloss.append(float(E_L))
            else:
                Eloss.append(0)
                
            try:
                k = float(rovnice[0])
            except:
                R_type = rovnice[0]
                record = (index, speclist[specdict[rovnice[1]]].mass)
                R_special[R_type] = R_special.get(R_type, []) + [record]
                k = 0
            k_c.append(k)

            rozdel = rovnice.index("=>")
            pozn = rovnice.index("//")
            reactant=(rovnice[1:rozdel])
            product=(rovnice[rozdel+1:pozn])
            REACTp = N.zeros(len(specdict),int)
            PRODp = N.zeros(len(specdict),int)
        except: continue
        for i in range(len(reactant)):
            if (REACTp[specdict[reactant[i]]] == 0):      # - zbytecna podminka .. ?
                j = reactant.count(reactant[i])
                REACTp[specdict[reactant[i]]] = j
                if (i > 0):
                    if ("typ reakce = elastic" in line):
                        Elastic.append([index, specdict[reactant[i]]])
        REACT.append(REACTp)
        for i in range(len(product)):
            if (PRODp[specdict[product[i]]] == 0):      # - zbytecna podminka .. ?
                j = product.count(product[i])
                PRODp[specdict[product[i]]] = j
        PROD.append(PRODp)             
        index += 1
    reaction.close()
    k_c = N.array(k_c)
    REACT = N.array(REACT)
    PROD = N.array(PROD)
    Z = PROD - REACT
    REACT = sp.csr_matrix(REACT)
    Eloss = N.array(Eloss)
    
    Z = N.transpose(Z)      
    Z = sp.csr_matrix(Z)                    

    return k_c, REACT, Z, Eloss, Elastic, R_special
    
    
def difuze(d, con, mass): 
    nu = d * con 
    Di = k_b*Tn / (mass*AMU*nu)
    tau = l/Di
    return 1/tau  

def rate_langevin(mass):
    reduced_mass = mass * ram_He * AMU / (mass + ram_He)
    alpha = 0.228044e-40  # C2m2 J-1 polarizability
    # Langevin rate coeff.
    rate = elementary_charge / (2*epsilon_0) * (alpha / reduced_mass)**0.5
    return rate * 1e6 # cm^3 s^-1
    
def ambi_dif(rate, con, mass, Te):
    nu = rate * con #* c2
    Di = k_b*Tn / (AMU * mass*nu)
    D_a = Di * (1 + Te/Tn)
    tau = l/D_a
    return 1/tau


def Stevefelt_formula(conc, Te):
    return (3.8e-9)*(Te**(-4.5))*conc + (1.55e-10) * (Te**(-0.63)) + (6e-9) * (Te**(-2.18))*(conc**(0.37))

def Q_elastic(Te, nn, Mn, Me, rate):  
    tau = 1/(nn * rate)    
    tau_cool = tau * Mn * AMU/(2*Me)
    return 1.5  * k_b * (Tn - Te) / tau_cool 

def actual_rate(k_c, file_reaction_coeffs):
    reaction = open(file_reaction_coeffs,"r")
    index = 0
    for line in reaction:             
            rovnice = line.split()
            if (rovnice[0] == "Stevefelt"):
                index += 1
                continue
            elif (rovnice[0] == "ambi_dif"):
                index += 1
                continue                                         
            elif (rovnice[0] == "difuze"):
                index += 1
                continue               
            else:
                try:
                    k_c[index] = (float(rovnice[0]))
                    index += 1                    
                except: 
                    continue
    return k_c

# CRR ternary recombination rate
def K_CRR(Te, CRR_factor):
    return CRR_factor*Te**(-4.5) *1e-12     # [m^6/s] ... prevod na cm -> *1e12

# cooling by coulombic collisions with H3+
def Q_ei_coulombic(Te, n, nn, mass):
    TeeV = k_b*Te/Q0
    Lambda_ei = 23 - N.log((n)**0.5*TeeV**(-3/2.0))
    nu_ei_cool_NRL = 3.2e-9*Lambda_ei/(mass*TeeV**1.5)*n
    return 1.5*k_b*nu_ei_cool_NRL*(Tn - Te)/Q0
    
    
def calculate_E_loss(Te, f, concentration, k_c, pomoc, speci, Eloss, Elastic):
    Eloss_Ela = []
    for i in range(len(Elastic)):
        Eloss_Ela.append(Q_elastic(Te, concentration[Elastic[i][1]], speci[Elastic[i][1]].mass, speci[pomoc["e-"]].mass, k_c[Elastic[i][0]])  )

    Eloss_Ela = N.array(Eloss_Ela)
    Eloss_Ela = sum(Eloss_Ela)


    E_loss = N.dot(f, Eloss) / concentration[pomoc["e-"]]

    Eloss_Ela = Eloss_Ela/Q0

    coulombic =\
            Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["H3+"]], speci[pomoc["H3+"]].mass)\
            + Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["H+"]], speci[pomoc["H+"]].mass)\
            + Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["H2+"]],speci[pomoc["H2+"]].mass)\
            + Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["He+"]], speci[pomoc["He+"]].mass)\
            + Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["He2+"]], speci[pomoc["He2+"]].mass)\
            + Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["Ar+"]], speci[pomoc["Ar+"]].mass)\
            + Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["ArH+"]], speci[pomoc["ArH+"]].mass)\
            + Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["HeH+"]], speci[pomoc["HeH+"]].mass)\

    E_loss = E_loss + Eloss_Ela #+ CRR

    E_loss += coulombic
    return E_loss


def create_ODE(t, concentration, k_c, Z, REACT, pomoc, speci, Eloss, maxwell, Elastic, R_special):
    # sestaveni rovnice pro resic lsoda; nevyzaduje vypocet jakobianu 

    global Te_last_coeffs
    global Te_last #2

    if maxwell:
        # recalculate coefficients if needed
        TeK =concentration[-1] * Q0 / k_b
        TeeV = concentration[-1]

        if (TeeV != Te_last_coeffs):
                RC.read_file(file_reaction_data, file_Edist, file_reaction_coeffs, TeK, maxwell)
                k_c = actual_rate(k_c, file_reaction_coeffs)
                Te_last_coeffs = TeeV

    # calculate effective rate coefficients (dependent on concentrations)
    for r in R_special["difuze"]:
        k_c[r[0]] = difuze(difu, concentration[pomoc["He"]], r[1])
    for r in R_special["ambi_dif"]:
        # calculate the loss rate due to ambipolar diffusion
        # This is not correct if the diffusion coefficients of different ions and significantly different
        # It is rough approximation, calculation of spatial distribution would be needed for accuracy.
        k_c[r[0]] = ambi_dif(rate_langevin(r[1]) , concentration[pomoc["He"]], r[1], concentration[-1] * Q0 / k_b)
    rate_st = Stevefelt_formula(concentration[pomoc["e-"]], concentration[-1] * Q0 / k_b)
    for r in R_special["Stevefelt"]:
        k_c[r[0]] = rate_st


    # calculate the vector of reaction rates
    concentration[concentration < 1e-12] = 0
    f = N.exp(REACT * N.log(concentration[:-1])) * k_c

    if maxwell:
        E_loss = calculate_E_loss(TeK, f, concentration, k_c, pomoc, speci, Eloss, Elastic)
    else:
        E_loss = 0
    global vibr_T

    f = Z * f 
    f = N.hstack((f, E_loss))

    Te_last = concentration[-1]
    return f
   
    
def solve_ODE(t1, dt, file_species, file_reaction_data, file_Edist, file_reaction_coeffs, Te, maxwell):

    speci, pomoc = load_species(file_species)
    
    # integrate the reaction rate coeffs and save if needed
    if file_reaction_data != None:
        rlist = RC.load_reaction_data(file_reaction_data)
        if not maxwell:
            EEDF = N.loadtxt(file_Edist)
        else:
            EEDF = None
        RC.print_reaction_coeffs_file(rlist, RC.State(Tn, Te, EEDF), file_reaction_coeffs)
        #
        #RC.read_file(file_reaction_data, file_Edist, file_reaction_coeffs, Te, maxwell)

    # load the saved reaction rate coeffs
    k_c, REACT, Z, Eloss, Elastic, R_special = load_reactions(file_reaction_coeffs, pomoc, speci)

    t0 = 0
    y0 = N.array([s.conc for s in speci] + [Te * k_b / Q0])

    vyvoj = [y0]
    cas = [0]   
    global conc_srov
    global Te_last_coeffs
    global Te_last
    Te_last_coeffs = y0[-1]
    Te_last = y0[-1]
    r = ode(create_ODE).set_integrator('lsoda')
    stopni = 0

    create_ODE(1.27e-22, y0, k_c, Z, REACT, pomoc, speci, Eloss, maxwell, Elastic, R_special)
    r.set_initial_value(y0, t0).set_f_params(k_c, Z, REACT, pomoc, speci, Eloss, maxwell, Elastic, R_special)
    while r.successful() and r.t < t1:
        try:
            r.integrate(r.t+dt)
            cas.append(r.t)
            vyvoj.append(r.y)
            for i in range(len(r.y)-1):        
                if (r.y[i] <= 1e-10): r.y[i] = 0
            
            if (r.t > 1e-4):
                if N.all(N.abs((conc_srov / r.y - 1)) < 1e-5):
                    print("zastaveno v case ", r.t)
                    break
            conc_srov = r.y
        except: 
            print("stop")
            break
        
    vyvoj = N.array(vyvoj)
    cas = N.array(cas)

    for i in range(len(speci)):
        speci[i].conc = r.y[i]
    print(r.y[-1] * Q0 / k_b, r.y[-1])
    Te =r.y[-1] * Q0 / k_b

    return r, cas, vyvoj, speci, Te

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
    print speci[i].name, ": \t %e" % speci[i].conc

print(Te)
time_step = 1e-6
time = 1e-4
concentrations, cas, vyvoj, speci, Te = solve_ODE(time, time_step, file_species, file_reaction_data, None, file_reaction_coeffs, Te, True)

for i in range(len(speci)):
    print(speci[i].name, ": \t %e" % speci[i].conc)

import matplotlib.pyplot as plt
f, ax = plt.subplots()
print(vyvoj)
for i in range(len(speci)):
    ax.plot(cas, vyvoj[:,i], label=speci[i].name)
    ax.set_yscale("log")
plt.savefig("result.pdf")
