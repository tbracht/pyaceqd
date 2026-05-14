from pyaceqd.tools import read_calibration_file
from pyaceqd.general_system.general_system import GeneralSystemACE
from pyaceqd.helpers.ace_operators import ketbra
import pyaceqd.constants as constants

hbar = constants.hbar  # meV*ps
d0 = 0.25  # meV
d1 = 0.12
d2 = 0.05
mu_b = 5.7882818012e-2   # meV/T
g_ex = -0.65  # in plane electron g factor
g_ez = -0.8  # out of plane electron g factor
g_hx = -0.35  # in plane hole g factor
g_hz = -2.2  # out of plane hole g factor

def energies_linear(d0=0.25, d1=0.12, d2=0.05, delta_B=4, delta_E=0.0):
    E_X = delta_E + (d0 + d1)/2.0 
    E_Y = delta_E + (d0 - d1)/2.0 
    E_S = delta_E - (d0 - d2)/2.0 
    E_F = delta_E - (d0 + d2)/2.0 
    E_B = 2.*delta_E - delta_B
    return E_X, E_Y, E_S, E_F, E_B

class SixLevelLinearSystem(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, gamma_b=None, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None, lindblad=True,
                 J_to_file=None, J_file=None, factor_ah=None, pt_dir="", rho0=ketbra(0,0,6), calibration_file=None, delta_b=4,
                d0=0.25, d1=0.2, d2=0.05, bx=0, bz=0):
        system_prefix = "sixls_linear" 
        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        if calibration_file is not None:
            E_X, E_Y, E_S, E_F, E_B, gamma_e, gamma_b, gamma_d, g_ex, g_hx, g_ez, g_hz = read_calibration_file(calibration_file)
        else:
            E_X, E_Y, E_S, E_F, E_B = energies_linear(delta_B=delta_b, d0=d0, d1=d1, d2=d2)
            g_ex = -0.65  # in plane electron g factor
            g_ez = -0.8  # out of plane electron g factor
            g_hx = -0.35  # in plane hole g factor
            g_hz = -2.2  # out of plane hole g factor
        system_op = [E_X*ketbra(1,1,6) + E_Y*ketbra(2,2,6) + E_S*ketbra(3,3,6) + E_F*ketbra(4,4,6) + E_B*ketbra(5,5,6)]
        # bright-dark coupling depending on Bx
        if bx != 0:
            system_op.append(-0.5*mu_b*bx*(g_ex+g_hx) * (ketbra(1,3,6) + ketbra(3,1,6)))
            system_op.append(-0.5*mu_b*bx*(g_ex-g_hx) * (ketbra(2,4,6) + ketbra(4,2,6)))
        # bright-bright and dark-dark coupling depending on Bz
        if bz != 0.0:
            system_op.append(-1j*(-0.5*mu_b*bz*(g_ez-3*g_hz)) * (ketbra(2,1,6) - ketbra(1,2,6)))
            system_op.append(-1j*(+0.5*mu_b*bz*(g_ez+3*g_hz)) * (ketbra(4,3,6) - ketbra(3,4,6)))
        boson_op = ketbra(1,1,6)+ketbra(2,2,6)+ketbra(3,3,6)+ketbra(4,4,6) + 2*ketbra(5,5,6)  # operator that couples to phonons
        if gamma_b is None:
            gamma_b = gamma_e
        lindblad_ops = [[ketbra(0,1,6),gamma_e],[ketbra(0,2,6),gamma_e],
                        [ketbra(1,5,6),gamma_b],[ketbra(2,5,6),gamma_b]]
                        # [ketbra(0,3,6),gamma_d],[ketbra(0,4,6),gamma_d]]
        modes = {"x": ketbra(1,0,6)+ketbra(5,1,6), "y": ketbra(2,0,6)+ketbra(5,2,6)}  # operator |1><0|_6+|5><1|_6 couples to x-polarized light
        rf_op = ketbra(1,1,6)+ketbra(2,2,6)+ketbra(3,3,6)+ketbra(4,4,6)+2*ketbra(5,5,6)  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        colors = ["#0000cf", "#45b0ee", "#ff0022", "#9966cc", "#009e00", "#ffde39"]
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                          threshold=threshold, boson_e_max=boson_e_max, system_op=system_op, modes=modes, rf_op=rf_op, rho0=rho0,
                          boson_op=boson_op, lindblad_ops=lindblad_ops, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir,
                          dim_prod=[6], colors=colors, lindblad=lindblad)
