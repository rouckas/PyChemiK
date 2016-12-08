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
  
def species(soubor):
    species = file(soubor,"r")
    speci = []	
    pomoc = {}
    index = 0			
    for line in species:
        if ("#" in line): continue
        try:
            name, init_conc, mass, radii = line.split()
            speci.append(1)
            speci[index] = Prvky(name, float(init_conc), float(mass), float(radii))
            pomoc[name] = index
            index += 1
        except: continue
    species.close()
    return speci, pomoc
        
def reactions(soubor, pomoc, speci):
    reaction = file(soubor,"r")
    index = 0
    k_c = []
    REACT = []
    PROD = []
    Eloss = []
    Elastic = []
    for line in reaction:             
        try:
            rovnice = line.split()

            if ("ENERGY LOSS = " in line):
                E_L = re.search("ENERGY LOSS =\s+(\S+)", line).group(1)
                Eloss.append(float(E_L))
            else:
                Eloss.append(0)
                
            if (rovnice[0] == "Stevefelt"):
                St.append(index)
                k_c.append(0)
            elif (rovnice[0] == "ambi_dif"):
                ambi_di.append([index,speci[pomoc[rovnice[1]]].mass])
                k_c.append(0)            
                               
            elif (rovnice[0] == "difuze"):
                dif_in.append([index,speci[pomoc[rovnice[1]]].mass])
                k_c.append(0)  
               
            else:
                try:
                    k_c.append(float(rovnice[0]))
                except: 
                    continue
                                                  
            rozdel = rovnice.index("=>")
            pozn = rovnice.index("//")
            reactant=(rovnice[1:rozdel])
            product=(rovnice[rozdel+1:pozn])
            REACTp = N.zeros(len(pomoc),int)
            PRODp = N.zeros(len(pomoc),int)
        except: continue
        for i in range(len(reactant)):
            if (REACTp[pomoc[reactant[i]]] == 0):      # - zbytecna podminka .. ?
                j = reactant.count(reactant[i])
                REACTp[pomoc[reactant[i]]] = j
                if (i > 0):
                    if ("typ reakce = elastic" in line):
                        Elastic.append([index, pomoc[reactant[i]]])
        REACT.append(REACTp)
        for i in range(len(product)):
            if (PRODp[pomoc[product[i]]] == 0):      # - zbytecna podminka .. ?
                j = product.count(product[i])
                PRODp[pomoc[product[i]]] = j
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

    return k_c, REACT, Z, Eloss, Elastic
    
    
def difuze(d, con, mass): 
    nu = d * con 
    Di = k_b*Tn / (mass*AMU*nu)
    tau = l/Di
    return 1/tau  

def rate_ambi_dif(mass):     
    r_mass = mass * ram_He * AMU / (mass + ram_He)
    alpha = 0.228044e-40  # C2m2 J-1
    rate = (Q0 / (2*permitivita)) * (N.sqrt(alpha / r_mass)**0.5)
    return rate * 1e6
    
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

def actual_rate(k_c, soubor):
    reaction = file(soubor,"r")
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
    
    
    

def create_ODE(t, concentration, k_c, Z, REACT, pomoc, ambi_rate, speci, Eloss, maxwell,Elastic):
    # sestaveni rovnice pro resic lsoda; nevyzaduje vypocet jakobianu 

    global Te_old   
    global Te_old2

    if (maxwell == True): 
        Te =concentration[-1] * Q0 / k_b
        a = concentration[-1]

        if (a != Te_old2):
            #b = a/Te_old
            #if ((b < 0.9) or (b > 1.1)):                
                RC.read_file(soubor_reakce, soubor_rozdel, Te, maxwell)
                k_c = actual_rate(k_c, file_reactions)
                Te_old = a
                #Te_old2 = concentration[-1]

    global jedna
    if (jedna == 0):
        if (concentration[-1] <= 0.00664): 
            print t
            jedna = jedna + 1

    for i in range(len(dif_in)):
        k_c[dif_in[i][0]] = difuze(difu, concentration[pomoc["He"]], dif_in[i][1])
        
    for i in range(len(ambi_di)):
        k_c[ambi_di[i][0]] = ambi_dif(rate_ambi_dif(speci[i].mass) , concentration[pomoc["He"]], ambi_di[i][1], concentration[len(concentration)-1] * Q0 / k_b)

    if len(St) > 0:
        rate_st = Stevefelt_formula(concentration[pomoc["e-"]], concentration[len(concentration)-1] * Q0 / k_b)
        for i in range(len(St)):
            k_c[St[i]] = rate_st


    concentration[concentration < 1e-12] = 0
    f = concentration[:len(concentration)-1]
    f = N.log(f)
    f = REACT * f
    f = N.exp(f)
    f = N.multiply(f,k_c)
    global vypis
    global ind_vyvoj
    global opak    
    E_loss = 0
    if (maxwell == True):
        
        Eloss_Ela = []
        for i in range(len(Elastic)):
            Eloss_Ela.append(Q_elastic(Te, concentration[Elastic[i][1]], speci[Elastic[i][1]].mass, speci[pomoc["e-"]].mass, k_c[Elastic[i][0]])  )

        Eloss_Ela = N.array(Eloss_Ela)
        Eloss_Ela = sum(Eloss_Ela)


        E_loss = N.dot(f, Eloss) / concentration[pomoc["e-"]]

        Eloss_Ela = Eloss_Ela/Q0  

        coulombic = Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["H3+"]] , speci[pomoc["H3+"]].mass)         + Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["H+"]] , speci[pomoc["H+"]].mass)        +Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["H2+"]] ,speci[pomoc["H2+"]].mass)          +Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["He+"]] , speci[pomoc["He+"]].mass)         +Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["He2+"]] , speci[pomoc["He2+"]].mass)          +Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["Ar+"]] , speci[pomoc["Ar+"]].mass)         +Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["ArH+"]] , speci[pomoc["ArH+"]].mass)         +Q_ei_coulombic(Te, concentration[pomoc["e-"]],concentration[pomoc["HeH+"]] , speci[pomoc["HeH+"]].mass)        
     
        E_loss = E_loss + Eloss_Ela #+ CRR

        E_loss += coulombic

    global vibr_T

    f = Z * f 
    f = N.hstack((f, E_loss))

    if N.any(concentration < 0): print concentration     

    Te_old2 = concentration[-1]
    return f
   
    
def solve_ODE(t1, dt, pomoc, speci,  soubor_reakce, soubor_rozdel, Te, maxwell):
    
    RC.read_file(soubor_reakce, soubor_rozdel, Te, maxwell)    
    k_c, REACT, Z, Eloss, Elastic = reactions(file_reactions, pomoc, speci)

    t0 = 0
    global at_mass
    at_mass = N.array(at_mass)
    ambi_rate = rate_ambi_dif(at_mass)   
    y0 = []
    for i in range(len(speci)):
        y0.append(speci[i].conc)

    y0 = N.array(y0)
    y0 = N.append(y0, [TeV])

    vyvoj = [y0]
    cas = [0]   
    global vypis
    global conc_srov
    global ind_vyvoj
    global Te_old
    global Te_old2
    Te_old = y0[-1]
    Te_old2 = Te_old
    r = ode(create_ODE).set_integrator('lsoda')
    stopni = 0

    r.set_initial_value(y0, t0).set_f_params(k_c, Z, REACT, pomoc, ambi_rate, speci, Eloss, maxwell,Elastic)
    while r.successful() and r.t < t1:
        try:
            r.integrate(r.t+dt)
            if (r.t > 1e-4):
                for i in range(len(speci)):
                    if (N.abs((conc_srov[i] / r.y[i] - r.y[i]/r.y[i])) < 1e-5): stopni += 1
            cas.append(r.t)
            vyvoj.append(r.y)
            for i in range(len(r.y)-1):        
                if (r.y[i] <= 1e-10): r.y[i] = 0
            ind_vyvoj += 20e-2
            
            if stopni == len (speci): 
                print "zastaveno v case ", r.t
                break
            stopni = 0
            conc_srov = r.y
        except: 
            print "stop"
            break
        
    vyvoj = N.array(vyvoj)
    cas = N.array(cas)

    for i in range(len(speci)):
        speci[i].conc = r.y[i]
    print r.y[-1] * Q0 / k_b, r.y[-1]
    Te =r.y[-1] * Q0 / k_b

    return r, cas, vyvoj, speci, Te


soubor_reakce = "data/collisions/electron.txt"  # reakce + data pro CS, k_rate
#soubor_rozdel = "data/collisions/eedf_He_H2_14Td_300K.txt"
#soubor_rozdel = "data/collisions/eedf_He_H2_14Td_77K.txt"
soubor_rozdel = "data/collisions/elendif/eedf_He_H2_14Td_100_1000.txt" #rozdelovaci funkce

Te_old = 1.73
Te_old2 = 1.73
St = []   
at_mass = []
ambi_di = [] 
dif_in = []

difu = 1.26076522957e-12
k_b = 1.38e-23
Tn = 77
Te = 20000
m_H = 1.00794 * 1.667e-27
AMU = 1.667e-27
Q0 = 1.602189e-19
TeV = Te * k_b / Q0
file_species = "data/species.txt"       # vstupni koncentrace a parametry castic
file_reactions = "data/collisions/reaction.txt" # vygenerovane reakce + k_rate

time = 20e-3
time_step = time/10e2
vypis = 0#.1 * int(time / time_step)
ind_vyvoj = 0
opak = ind_vyvoj

speci, pomoc = species(file_species)
jedna = 0 
conc_srov = N.zeros(len(speci))

DE_CRR = 0.13#*1.6e-19
CRR_factor = 3.8e-9

r = 7.5e-3
l = (r/2.405)**2
ram_He = 4.0026
permitivita = 8.854187817620e-12 

Te = 22700




concentrations, cas, vyvoj, speci, Te = solve_ODE(time, time_step, pomoc, speci, soubor_reakce, soubor_rozdel, Te, False)
TeV = Te * k_b / Q0

for i in range(len(speci)):
    print speci[i].name, ": \t %e" % speci[i].conc


print Te
time_step = time_step /1e3
#time = time * 1e1
concentrations, cas, vyvoj, speci, Te = solve_ODE(time, time_step, pomoc, speci, soubor_reakce, soubor_rozdel, Te, True)

for i in range(len(speci)):
    print speci[i].name, ": \t %e" % speci[i].conc