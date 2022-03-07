import numpy as N
import numpy as np
from math import pi
import sys

M_EL =  9.109534e-31
K_B = 1.380662e-23
Q_EL = -1.602189e-19
Q0 = -Q_EL
AMU = 1.67e-27
#Te = 20000
h = 6.62606957e-34  # m^2 kg s^-1
c = 299792458.      # m s^-1
cm2J = 100*h*c
omega = 4401

def rate_coef(f, CS_E, Te, maxwell):
    CS_E = N.array(CS_E)

    ## resample to finer grid
    from scipy.interpolate import interp1d
    if (maxwell == False):
        if (CS_E[0,0] != 0): 
            print(CS_E[0])
            CS_E = N.insert(CS_E,0,0, axis =0)            
        points = f[:,0]      
        points = points.T.reshape((-1,1))
        data = interp1d(CS_E[:,0], CS_E[:,1])(points)
        CS_E = N.hstack((points, data))
    else:
        left = CS_E[:-1,0]
        right = CS_E[1:,0]
        points = left
        if (len(CS_E) < 100): 
            N_sub = 100
            for i in range(N_sub-1):
                xnew = left + (right-left)*(i+1.)/N_sub
                points = N.vstack((points, xnew))
            points = points.T.reshape((-1,1))
            data = interp1d(CS_E[:,0], CS_E[:,1])(points)
            CS_E = N.hstack((points, data))

    # obtain sigma as function of velocity
    if (maxwell == False):        
        EeV = f[:,0]		# eV = 1e-19 kg. m^2. s^-2
        fe = f[:,1]
    else:
        EeV = CS_E[:,0]		# eV = 1e-19 kg. m^2. s^-2
        fe = mxw_E(CS_E[:,0], Te*K_B/Q0)
    CS = CS_E[:,1] 	# 1e-20 m^2 = 1e-16 cm^2
    M_EL = 9.109e-31    	# kg
    # convert energy to velocity and calculate f(v)*sigma(v)*v
    v = N.sqrt(2*EeV*Q0/M_EL)	# m . s-1
    integrand = fe * CS * v

    # approximate the integral by trapezoid rule
    left = N.arange(0, len(EeV)-1)
    right = left+1
    dE = EeV[right] - EeV[left]
    integral = 0.5*sum(dE*(integrand[left]+integrand[right]))
    #rate = integral * 1e-14   # Phelps  1e-16 cm^2
    rate = integral * 1e2  #Fusion cm^2
    return rate

def CS_balance(rovnice,r_rate, T):
    if ("v=" in rovnice):
        mez = rovnice.index("=>")
        pozn_i = rovnice.index("//")
        pozn = rovnice[pozn_i:]
        if ("ENERGY LOSS = -" in pozn):
            ro = pozn.index("= ")
            pozn = pozn.replace(pozn[ro+2], " ")
            
        if ("v=" in rovnice[:mez]):
            v0 = int(rovnice[rovnice.index("v=")+2])
        else:   v0 = 0
        new_rovnice = rovnice[mez+2:pozn_i] + "=>" + rovnice[:mez] + "// dopocteno, " + pozn[2:]
        if ("v=" in rovnice[mez:]):
            v1 = int(new_rovnice[new_rovnice.index("v=")+2])
        else:   v1 = 0
        try:
            rate = N.exp(N.log(r_rate) + ((v1-v0)*omega*cm2J/Q0)/T)
            if (N.isnan(rate) == True): rate = 0
        except: rate = 0

    return new_rovnice, rate

    
        

def mxw_E(E,T,M=M_EL):   
    if N.any(T < 0): raise(ValueError("Negative value of temperature not allowed"))
    E_max = T
    with N.errstate(under="ignore"):
        return 2*pi*(N.sqrt(1/pi/E_max)**3)*N.sqrt(E)*N.exp(-E/E_max)

class State:
    def __init__(self, Tg, Te=None, EEDF=None):
        self.Tg = Tg
        self.Te = Te
        if isinstance(EEDF, str):
            self.EEDF = np.loadtxt(EEDF)
        else:
            self.EEDF = EEDF

    def __repr__(self):
        res = "State(" + str(self.Tg) + ", " + str(self.Te) + ", " + str(self.EEDF) + ")"
        return res

class Reaction:
    def __init__(self, reactants, products, 
            reaction_type=None, k=None, k_hydhel=None, k_arrh=None, CS=None,
            energy_change=None, comment=""):
        self.reactants = reactants
        self.products = products
        self.type = reaction_type
        self.comment = comment

        self.energy_change = energy_change

        self.k_type = None
        if k is not None:
            self._k = k
            self.k_type = "k"

        if k_hydhel is not None:
            if self.k_type is not None:
                raise RuntimeError("reaction" + " ".join(reactants + ["=>"] + products) + "has multiple values of k")
            self.k_hydhel = k_hydhel
            self.k_type = "k_hydhel"

        if k_arrh is not None:
            if self.k_type is not None:
                raise RuntimeError("reaction" + " ".join(reactants + ["=>"] + products) + "has multiple values of k")
            self.k_arrh = k_arrh
            self.k_type = "k_arrh"

        if CS is not None:
            if self.k_type is not None:
                raise RuntimeError("reaction" + " ".join(reactants + ["=>"] + products) + "has multiple values of k")
            self.CS = CS
            self.k_type = "CS"

    def __repr__(self):
        res = "\t".join(["reaction"] + self.reactants + ["=>"] + self.products)
        return res

    def k(self, state):

        if self.type in ["Stevefelt", "ambi_dif", "diffusion_ar"]:
            # effective r. rate coeffs, need to know num. dens of other species first
            return None

        if self.k_type == "k":
            return self._k

        if self.k_type == "k_hydhel": # function of Te
            suma = 0
            for i, ai in enumerate(self.k_hydhel):
                suma += ai * ((np.log(state.Te*K_B/Q0))**i)
            return np.exp(suma)

        if self.k_type == "k_arrh":  # function of Tg
            k0, Ta = self.k_arrh
            return k0*np.exp(-Ta/state.Tg)

        if self.k_type == "CS":       # function of EEDF or Te
            return rate_coef(state.EEDF, self.CS, state.Te, maxwell = state.EEDF is None)

class InverseReaction(Reaction):
    def __init__(self, reactants, products, **kwargs):
        if kwargs.get("energy_change", None) is not None:
            kwargs["energy_change"] *= -1

        Reaction.__init__(self, products, reactants, **kwargs)

    def __repr__(self):
        res = "\t".join(["reaction"] + self.products + ["=>"] + self.reactants)
        return res

    def k(self, state):
        g = 1 # statistical factor, not yet configurable. OK for vibrational trans. only
        kforward = Reaction.k(self, state)*g*np.exp(self.energy_change*Q0/(K_B*state.Te))
        return kforward

def load_reaction_data_simple(fname):
    rlist = []

    reaction = {}
    inverse = False

    f = open(fname)
    for line in f:
        line, _, comment = line.partition("#") # skip comments
        toks = line.split()
        if len(toks) == 0: continue

        reaction = {}

        try:
            reaction["k"] = float(toks[0])
        except:
            reaction["reaction_type"] = toks[0]


        arrow = toks.index("=>")
        reaction["reactants"] = toks[1:arrow]
        reaction["products"] = toks[arrow+1:]
        reaction["comment"] = comment.strip()

        rlist.append(Reaction(**reaction))

    return rlist

def load_reaction_data(fname, format="full"):
    if format == "simple":
        return load_reaction_data_simple(fname)

    rlist = []

    state = 1
    """
    state = 1: decoding reaction
    state = 2: loading cross section
    """
    reaction = {}
    inverse = False

    f = open(fname)
    for line in f:
        line, _, comment = line.partition("#") # skip comments
        toks = line.split()
        if len(toks) == 0: continue

        if state == 1:
            if toks[0] == "reaction":
                if reaction != {}:
                    rlist.append(Reaction(**reaction))
                    # save previous record
                    if inverse:
                        rlist.append(InverseReaction(**reaction))
                        inverse = False
                    pass

                state = 1
                reaction = {}
                arrow = toks.index("=>")
                reaction["reactants"] = toks[1:arrow]
                reaction["products"] = toks[arrow+1:]
                reaction["comment"] = comment.strip()
                continue

            elif toks[0] == "k":
                reaction["k"] = float(toks[1])
                continue

            elif toks[0] == "k_hydhel":
                reaction["k_hydhel"] = list(map(float, toks[1:]))
                continue

            elif toks[0] == "k_arrh":
                reaction["k_arrh"] = list(map(float, toks[1:3]))
                continue

            elif toks[0] == "type":
                reaction["reaction_type"] = toks[1]
                continue

            elif toks[0] == "inverse":
                inverse = True
                continue

            elif toks[0] == "energy_change":
                reaction["energy_change"] = float(toks[1])
                continue

            if toks[0] == "cross_section":
                state = 2
                CS = []
                continue

        elif state == 2:
            if toks[0] == "end":
                reaction["CS"] = CS
                state = 1
                continue

            else:
                f1, f2 = toks
                CS.append([float(f1), float(f2)])


    # save last record
    rlist.append(Reaction(**reaction))
    if inverse:
        rlist.append(InverseReaction(**reaction))
    pass


    return rlist


def print_reaction_coeffs_file(rlist, state, filename):
    with open(filename, "w") as f:
        print_reaction_coeffs(rlist, state, f)


def print_reaction_coeffs(rlist, state, file=sys.stdout):
    for r in rlist:
        k = r.k(state)
        if k is None:
            k = r.type
        comment = r.comment
        if r.type == "elastic":
            comment += " typ reakce = elastic"
        print("\t".join([str(k)] + r.reactants + ["=>"] + r.products), "//", comment, file=file)

    
def read_file(f_CS, f_distr, f_out, Te, maxwell = False):
    if not maxwell:
        file_rozdel = open(f_distr)
    file_prurez = open(f_CS)
    file_reakce = open(f_out, "w")

    if maxwell:
        nsampl = 1000
        f = N.zeros((nsampl, 2))
        f[:,0] = N.linspace(0, Te*K_B/Q0*10, nsampl)
        f[:,1] = mxw_E(f[:,0], Te*K_B/Q0)   # Maxwellowo rozdeleni    
    else:
        f = []
        for line in file_rozdel:
            m1, m2 = line.split()
            f.append([float(m1), float(m2)])
            #f.append([float(m1), mxw_E(float(m1),Te*K_B/Q0)])   # cely vypocet s Maxwellovym rozdelenim
        f = N.array(f)
        del m1, m2
    CS_E = []
    inverse = False
    for line in file_prurez:        
        if (len(line) < 2): continue            
        if ("begin" in line):
            rovnice = line[5:]   
            if ("inverse=True" in line):
                inverse = True
            continue
        if ("end" in line):
            r_rate = rate_coef(f, CS_E, Te, maxwell)
            file_reakce.write(str(r_rate) + rovnice)
            if (inverse == True):
                rovnice, rate = CS_balance(rovnice, r_rate, Te*K_B/Q0)
                file_reakce.write(str(rate) + rovnice)
            rovnice = ""
            CS_E = []
            inverse = False
            continue

        try:    
            m1, m2 = line.split()
            if (m1 == "koef"):
                file_reakce.write(m2 + rovnice)
                continue
            else:
                if ((float(m1) > 25) and CS_E[-1][0] >= 25): continue
                CS_E.append([float(m1), float(m2)])
        except:
            A = line.split()
            suma = 0
            for i in range(len(A)):
                suma += float(A[i]) * ((N.log(Te*K_B/Q0))**i)
            r_rate = N.exp(suma)
            file_reakce.write(str(r_rate) + rovnice)
            if (inverse == True):
                rovnice, rate = CS_balance(rovnice, r_rate, Te*K_B/Q0)
                file_reakce.write(str(rate) + rovnice)
           
            rovnice = ""
            del A, suma
            inverse = False
    file_reakce.close()
