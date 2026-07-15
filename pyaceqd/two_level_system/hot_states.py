import numpy as np
import pyaceqd.constants as constants
from pyaceqd.general_system.general_system import GeneralSystemACE
from pyaceqd.helpers.ace_operators import ketbra

hbar = constants.hbar  # meV*ps
kb = constants.kB # meV/K

class TwoHotStates(GeneralSystemACE): #
    def __init__(self, dt=0.1, gamma_e=1/100, lindblad=True, phonons=False, ae=5, temperature=4, 
                 delta_E1=5.1, gamma_01=0.001, d1=2, delta_E2=7.09, gamma_02=0.01, d2=4,
                 verbose=False, pt_file=None, J_to_file=None, J_file=None, threshold=8, 
                 factor_ah=None, pt_dir="", rho0=ketbra(0,0,4)):
        system_prefix = "two_hot_state" ## |0>  = G, |1> = X, |2> = x*, |3> = x**
        threshold = str(int(threshold))  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        system_op = delta_E1*ketbra(2,2,4) + delta_E2*ketbra(3,3,4)
        boson_op = ketbra(1,1,4) + ketbra(2,2,4) + ketbra(3,3,4)

        n_BE1 = 1/(np.exp(delta_E1/(kb*temperature))-1)
        n_BE2 = 1/(np.exp(delta_E2/(kb*temperature))-1)
        lindblad_ops = [[ketbra(0,1,4), gamma_e], 
                        [ketbra(1,2,4), gamma_01*(1+n_BE1)], [ketbra(2,1,4), d1*gamma_01*n_BE1], 
                        [ketbra(1,3,4), gamma_02*(1+n_BE2)], [ketbra(3,1,4), d2*gamma_02*n_BE2]]  
        
        modes = {"x": ketbra(1,0,4)}  # operator |1><0|_2 couples to x-polarized light
        rf_op = ketbra(1,1,4)  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        colors = ["#0000FF", "#FF0000"]
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                          threshold=threshold, boson_e_max=boson_e_max, system_op=system_op, modes=modes, rf_op=rf_op, rho0=rho0,
                          boson_op=boson_op, lindblad_ops=lindblad_ops, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir,
                          dim_prod=[4], colors=colors, lindblad=lindblad)