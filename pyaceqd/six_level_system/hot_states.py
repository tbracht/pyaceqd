import numpy as np
import pyaceqd.constants as constants
from pyaceqd.general_system.general_system import GeneralSystemACE
from pyaceqd.helpers.ace_operators import ketbra

hbar = constants.hbar  # meV*ps
kb = constants.kB # meV/K

def n_BE(delta_E, temperature):
    return 1/(np.exp(delta_E/(kb*temperature))-1)

class BiexcitonHotStates(GeneralSystemACE):
    """
    Docstring for BiexcitonHotStates

    states:
    |0>  = G
    |1> = X
    |2> = Y
    |3> = B
    |4> = D
    |5> = X*
    |6> = B*
    """
    def __init__(self, dt=0.1, gamma_x=1/231, gamma_xh=1/10000, gamma_b=1/129, gamma_ph0=1/1000,
                 lindblad=True, phonons=False, ae=5, temperature=4,
                 delta_fss=0, delta_b=4, delta_xd=0.110, delta_E1=3.7, delta_E2=3.7, 
                 multipl_x=4, multipl_d=2, 
                 verbose=False, pt_file=None, J_to_file=None, J_file=None, threshold=8, 
                 factor_ah=None, pt_dir="", rho0=ketbra(0,0,7), propagate_Taylor=None):
        system_prefix = "biexciton_hot_state" ## |0>  = G, |1> = X, |2> = Y, |3> = B, |4> = D, |5> = X*, |6> = B*
        threshold = str(int(threshold))  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        # delta_fss: fine structure between X and Y. rotating frame stays at energy=0. original paper uses delta_fss=0.
        # delta_b: biexciton binding energy, with reference to unshifted exciton energy
        # delta_xd: energy of dark exciton state with respect to unshifted exciton energy
        # delta_E1: energy of hot exciton state with respect to unshifted exciton energy
        # delta_E2: energy of hot biexciton state with respect to biexciton. In the original paper this is the same as delta_E1
        # note that rotating frame is centered around biexciton energy, eg E_B = 0.
        system_op = (delta_b/2 + delta_fss/2) * ketbra(1,1,7) + (delta_b/2 - delta_fss/2) * ketbra(2,2,7) + (delta_b/2 - delta_xd) * ketbra(4,4,7) \
                       + (delta_b/2 + delta_E1) * ketbra(5,5,7) + delta_E2 * ketbra(6,6,7)
        # system_op = delta_fss/2*ketbra(1,1,7) - delta_fss/2*ketbra(2,2,7) - delta_b*ketbra(3,3,7) - delta_xd*ketbra(4,4,7) \
        #             + delta_E1*ketbra(5,5,7) + (-delta_b + delta_E2)*ketbra(6,6,7)   # energies in RF with resepct to unshifted X/Y energy. 
        boson_op = ketbra(1,1,7) + ketbra(2,2,7) + 2 * ketbra(3,3,7) \
                 + ketbra(4,4,7) + ketbra(5,5,7) + 2 * ketbra(6,6,7)  # B-states couple twice as strong

        n_BE1 = 1/(np.exp(delta_E1/(kb*temperature))-1)
        n_BE2 = 1/(np.exp(delta_E2/(kb*temperature))-1)

        gamma_ph = (1 + n_BE(delta_E1, temperature)) * gamma_ph0
        gamma_phstar = n_BE(delta_E1, temperature) * gamma_ph0

        gamma_ph2 = (1 + n_BE(delta_E2, temperature)) * gamma_ph0
        gamma_ph2star = n_BE(delta_E2, temperature) * gamma_ph0

        gamma_phd = (1 + n_BE(delta_E1 + delta_xd, temperature)) * gamma_ph0
        gamma_phdstar = n_BE(delta_E1 + delta_xd, temperature) * gamma_ph0

        lindblad_ops = [[ketbra(0,1,7), gamma_x], [ketbra(0,2,7), gamma_x], [ketbra(1,3,7), gamma_b/2], [ketbra(2,3,7), gamma_b/2],  # bright states
                        [ketbra(1,6,7), gamma_xh], [ketbra(2,6,7), gamma_xh], [ketbra(5,6,7), gamma_x], [ketbra(4,6,7), gamma_xh], [ketbra(0,5,7), gamma_xh],  # temperature independent hot state rates
                        [ketbra(1,5,7), gamma_ph], [ketbra(2,5,7), gamma_ph], [ketbra(5,1,7), multipl_x*gamma_phstar], [ketbra(5,2,7), multipl_x*gamma_phstar],  # X/Y-X_hot transitions
                        [ketbra(3,6,7), gamma_ph2], [ketbra(6,3,7), multipl_x*gamma_ph2star],  # B-B_hot transitions
                        [ketbra(4,5,7), multipl_d*gamma_phd], [ketbra(5,4,7), multipl_x*gamma_phdstar]]  # D-X_hot transitions
        
        modes = {"x": ketbra(1,0,7)+ketbra(3,1,7), "y": ketbra(2,0,7)+ketbra(3,2,7)}  # eg. operator |1><0|+|3><1| couples to x-polarized light
        rf_op = ketbra(1,1,7)  # caution, RF not implemented here.
        colors = ["#0000FF", "#FF0000"]
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                          threshold=threshold, boson_e_max=boson_e_max, system_op=system_op, modes=modes, rf_op=rf_op, rho0=rho0,
                          boson_op=boson_op, lindblad_ops=lindblad_ops, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir,
                          dim_prod=[4], colors=colors, lindblad=lindblad, propagate_Taylor=None)
