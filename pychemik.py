import numpy as N
import scipy.sparse as sp
from scipy.integrate import ode
import re
import rate_coef as RC
from rate_coef import State

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


class Prvky:
    def __init__(self, name = 0, conc = 0, mass = 0, radii = 0):
        self.name = name
        self.conc = conc
        self.mass = mass
        self.radii = radii
        
def load_species(fname):
    species = open(fname,"r")
    species_list = []	
    species_dict = {}
    index = 0			
    for line in species:
        line = line.partition("#")[0]
        try:
            name, init_conc, mass, radii = line.split()
            species_list.append(Prvky(name, float(init_conc), float(mass), float(radii)))
            species_dict[name] = len(species_list)-1
        except: continue
    species.close()
    return species_list, species_dict
        
def calculate_k(rlist, state):
    """
    Returns:
        k_c[i]: list of reaction rate coefficients
    """
    k_c = N.zeros((len(rlist),))
    for i, r in enumerate(rlist):
        k = r.k(state)
        if k is not None:
            k_c[i] = k

    return k_c

def analyze_reaction_network(rlist, specdict, speclist):
    """load the list of reactions from file 'fname'

    Returns:
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
    REACT = N.zeros((len(rlist), len(specdict)), int)
    PROD = N.zeros((len(rlist), len(specdict)), int)
    Eloss = []
    Elastic = []
    R_special = {"diffusion_ar":[], "ambi_dif":[], "Stevefelt":[]}
    for index, r in enumerate(rlist):
        Eloss.append((-r.energy_change) if r.energy_change is not None else 0.)

        if r.type in R_special.keys():
            record = (index, speclist[specdict[r.reactants[0]]].mass)
            R_special[r.type] = R_special.get(r.type, []) + [record]

        if r.type == "elastic":
            Elastic.append([index, specdict[r.reactants[1]]])

        for re in r.reactants:
            REACT[index, specdict[re]] += 1

        for pr in r.products:
            PROD[index, specdict[pr]] += 1

    Z = PROD - REACT
    REACT = sp.csr_matrix(REACT)
    Eloss = N.array(Eloss)
    
    Z = N.transpose(Z)      
    Z = sp.csr_matrix(Z)                    

    return REACT, Z, Eloss, Elastic, R_special
    
    
def difuze(d, con, mass, l, Tn): 
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
    
def ambi_dif(rate, con, mass, l, Tn, Te):
    nu = rate * con #* c2
    Di = k_b*Tn / (AMU * mass*nu)
    D_a = Di * (1 + Te/Tn)
    tau = l/D_a
    return 1/tau


def Stevefelt_formula(conc, Te):
    return (3.8e-9)*(Te**(-4.5))*conc + (1.55e-10) * (Te**(-0.63)) + (6e-9) * (Te**(-2.18))*(conc**(0.37))

def Q_elastic(Tn, Te, nn, Mn, Me, rate):  
    tau = 1/(nn * rate)    
    tau_cool = tau * Mn * AMU/(2*Me)
    return 1.5  * k_b * (Tn - Te) / tau_cool 

# CRR ternary recombination rate
def K_CRR(Te, CRR_factor):
    return CRR_factor*Te**(-4.5) *1e-12     # [m^6/s] ... prevod na cm -> *1e12

# cooling by coulombic collisions with H3+
def Q_ei_coulombic(Tn, Te, n, nn, mass):
    TeeV = k_b*Te/Q0
    Lambda_ei = 23 - N.log((n)**0.5*TeeV**(-3/2.0))
    nu_ei_cool_NRL = 3.2e-9*Lambda_ei/(mass*TeeV**1.5)*n
    return 1.5*k_b*nu_ei_cool_NRL*(Tn - Te)/Q0
    
    
def calculate_E_loss(Tn, Te, f, concentration, k_c, pomoc, speci, Eloss, Elastic):
    Eloss_Ela = []
    for i in range(len(Elastic)):
        Eloss_Ela.append(Q_elastic(Tn, Te, concentration[Elastic[i][1]], speci[Elastic[i][1]].mass, speci[pomoc["e-"]].mass, k_c[Elastic[i][0]])  )

    Eloss_Ela = N.array(Eloss_Ela)
    Eloss_Ela = sum(Eloss_Ela)


    E_loss = N.dot(f, Eloss) / concentration[pomoc["e-"]]

    Eloss_Ela = Eloss_Ela/Q0

    coulombic =\
            Q_ei_coulombic(Tn, Te, concentration[pomoc["e-"]],concentration[pomoc["H3+"]], speci[pomoc["H3+"]].mass)\
            + Q_ei_coulombic(Tn, Te, concentration[pomoc["e-"]],concentration[pomoc["H+"]], speci[pomoc["H+"]].mass)\
            + Q_ei_coulombic(Tn, Te, concentration[pomoc["e-"]],concentration[pomoc["H2+"]],speci[pomoc["H2+"]].mass)\
            + Q_ei_coulombic(Tn, Te, concentration[pomoc["e-"]],concentration[pomoc["He+"]], speci[pomoc["He+"]].mass)\
            + Q_ei_coulombic(Tn, Te, concentration[pomoc["e-"]],concentration[pomoc["He2+"]], speci[pomoc["He2+"]].mass)\
            + Q_ei_coulombic(Tn, Te, concentration[pomoc["e-"]],concentration[pomoc["Ar+"]], speci[pomoc["Ar+"]].mass)\
            + Q_ei_coulombic(Tn, Te, concentration[pomoc["e-"]],concentration[pomoc["ArH+"]], speci[pomoc["ArH+"]].mass)\
            + Q_ei_coulombic(Tn, Te, concentration[pomoc["e-"]],concentration[pomoc["HeH+"]], speci[pomoc["HeH+"]].mass)\

    E_loss = E_loss + Eloss_Ela #+ CRR

    E_loss += coulombic
    return E_loss


def create_ODE(t, concentration, rlist, k_c, Z, REACT, pomoc, speci, Eloss, Elastic, R_special, state, epsilon=1e-12):
    # sestaveni rovnice pro resic lsoda; nevyzaduje vypocet jakobianu 

    global Te_last_coeffs

    if state.electron_cooling:
        # recalculate coefficients if needed
        TeK =concentration[-1] * Q0 / k_b
        TeeV = concentration[-1]

        if (TeeV != Te_last_coeffs):
            k_c = calculate_k(rlist, RC.State(state.Tg, TeK, None))
            Te_last_coeffs = TeeV

    concentration[concentration < epsilon] = 0.

    # calculate effective rate coefficients (dependent on concentrations)
    for r in R_special["diffusion_ar"]:
        k_c[r[0]] = difuze(difu, concentration[pomoc["He"]], r[1], state.diffusion_length, state.Tg)
    for r in R_special["ambi_dif"]:
        # calculate the loss rate due to ambipolar diffusion
        # This is not correct if the diffusion coefficients of different ions and significantly different
        # It is rough approximation, calculation of spatial distribution would be needed for accuracy.
        k_c[r[0]] = ambi_dif(rate_langevin(r[1]) , concentration[pomoc["He"]], r[1], state.diffusion_length, state.Tg, concentration[-1] * Q0 / k_b)
    rate_st = Stevefelt_formula(concentration[pomoc["e-"]], concentration[-1] * Q0 / k_b)
    for r in R_special["Stevefelt"]:
        k_c[r[0]] = rate_st


    # calculate the vector of reaction rates
    from numpy import errstate
    with errstate(divide="ignore"):
        f = N.exp(REACT * N.log(concentration[:-1])) * k_c

    if state.electron_cooling:
        E_loss = calculate_E_loss(state.Tg, TeK, f, concentration, k_c, pomoc, speci, Eloss, Elastic)
    else:
        E_loss = 0

    f = Z * f 
    f = N.hstack((f, E_loss))

    return f
   

def solve_ODE(t1, dt, species_list, reaction_list, state, method="ode"):

    species_dict = {s.name:i for i, s in enumerate(species_list)}

    # load the saved reaction rate coeffs
    REACT, Z, Eloss, Elastic, R_special = analyze_reaction_network(reaction_list, species_dict, species_list)
    k_c = calculate_k(reaction_list, state)

    t0 = 0
    y0 = N.array([s.conc for s in species_list] + [state.Te * k_b / Q0])

    global Te_last_coeffs
    Te_last_coeffs = y0[-1]

    if method == "ode":
        from scipy.integrate import ode
        vyvoj = [y0]
        cas = [0]
        global conc_srov
        #r = ode(create_ODE).set_integrator('lsoda', atol=1e-3)
        r = ode(create_ODE).set_integrator('vode', method='bdf', atol=1e-2)

        # test run to detect errors, because the lsoda/vode methods ignore exceptions
        create_ODE(1.27e-22, y0, reaction_list, k_c, Z, REACT, species_dict, species_list, Eloss, Elastic, R_special, state)
        r.set_initial_value(y0, t0).set_f_params(reaction_list, k_c, Z, REACT, species_dict, species_list, Eloss, Elastic, R_special, state)
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

        if not r.successful():
            raise(RuntimeError(("Integration failed. Stopping at t=%f"%r.t)))
        
        y = N.array(vyvoj)
        t = N.array(cas)

    elif method == "solve_ivp":
        from scipy.integrate import solve_ivp
        r = solve_ivp(
                create_ODE, (t0, t1), y0, method="LSODA", t_eval=t0 + dt*N.arange(0, (t1-t0)//dt),
                args=(reaction_list, k_c, Z, REACT, species_dict, species_list, Eloss, Elastic, R_special, state),
                atol=1e-5
                )
        y = r.y.T
        t = r.t

    elif method == "solve_ivp_log":
        """attempt at solving for log(y) instead of y. Does not seem to improve stability"""
        eps=1e-10
        y0_log = N.log(y0 + eps)

        def RHS_log(t, y_log, *args):
            y = N.exp(y_log)
            dlogydt = create_ODE(t, y, *args)/y
            return dlogydt

        from scipy.integrate import solve_ivp
        r = solve_ivp(
                RHS_log, (t0, t1), y0_log, method="LSODA", t_eval=t0 + dt*N.arange(0, (t1-t0)//dt),
                args=(reaction_list, k_c, Z, REACT, species_dict, species_list, Eloss, Elastic, R_special, state),
                atol=1e-3
                )

        r.y = N.exp(r.y)
    else:
        raise ValueError("Unknown integration method " + str(method))

    for i in range(len(species_list)):
        species_list[i].conc = y[-1, i]
    Te =y[-1, -1] * Q0 / k_b

    return r, t, y, species_list, Te

##############################
##### global variables #######
##############################
Te_last_coeffs = 1.73
conc_srov = 0

