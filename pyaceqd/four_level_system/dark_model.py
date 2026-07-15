import pyaceqd.constants as constants
from pyaceqd.general_system.general_system import GeneralSystemACE
from pyaceqd.helpers.ace_operators import ketbra

hbar = constants.hbar  # meV*ps
g = ketbra(0,0,5)
x = ketbra(1,1,5)
y = ketbra(2,2,5)
d = ketbra(3,3,5)
b = ketbra(4,4,5)

p_gx = ketbra(0,1,5)
p_gy = ketbra(0,2,5)
p_gd = ketbra(0,3,5)
p_xb = ketbra(1,4,5)
p_yb = ketbra(2,4,5)


class BiexcitonSingleDark(GeneralSystemACE):
    """
    Five-level biexciton with a single dark state (G,X,Y,D,B) based on darkmodel_new.
    Dark state is coupled to the ground state and the biexciton state, but not to the bright exciton states.
    Dark state can be addressed by using "d" polarization. 
    """
    def __init__(self, dt=0.1, gamma_e=1/100, gamma_b=None, delta_xd=0, delta_b=4, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
                 rho0=ketbra(0,0,5), lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", gamma_d=None):
        system_prefix = "b_single_dark"
        # |0> = G, |1> = X, |2> = Y, |3> = D, |4> = B
        system_op = [-delta_b*b - delta_xd*d]
        boson_op = x + y + d + 2*b
        lindblad_ops = []
        self.gamma_e = gamma_e
        if lindblad:
            if gamma_b is None:
                gamma_b = gamma_e
            lindblad_ops = [[p_gx, gamma_e], [p_gy, gamma_e], [p_xb, gamma_b], [p_yb, gamma_b]]
            if gamma_d is not None:
                lindblad_ops.append([p_gd, gamma_d])  # dark state decay

        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        # coupling modes for x and y polarization
        modes = {"x": p_gx.T + p_xb.T, "y": p_gy.T + p_yb.T,  "d": ketbra(3,0,5) + ketbra(4,3,5)} 
        # rotating-frame operator (counts excitations, B has factor 2)
        rf_op = x + y + d + 2*b
        dim_prod = [5]
        colors = ["#0000FF", "#FF0000", "#00FF00", "#FF00FF", "#AAAA00"]

        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                         threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                         lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod, colors=colors)


