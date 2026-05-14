import pyaceqd.constants as constants
from pyaceqd.general_system.general_system import GeneralSystemACE
from pyaceqd.helpers.ace_operators import ketbra

hbar = constants.hbar  # meV*ps

class BiexcitonSingleDark(GeneralSystemACE):
    """Five-level biexciton with a single dark state (G,X,Y,D,B) based on darkmodel_new.
    """
    def __init__(self, dt=0.1, gamma_e=1/100, gamma_b=None, delta_xd=0, delta_b=4, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
                 initial="|0><0|_5", lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", gamma_d=None):
        system_prefix = "b_single_dark"
        # |0> = G, |1> = X, |2> = Y, |3> = D, |4> = B
        system_op = ["{}*|4><4|_5".format(-delta_b), "{}*|3><3|_5".format(-delta_xd)]
        boson_op = "1*(|1><1|_5 + |2><2|_5 + |3><3|_5) + 2*|4><4|_5"
        lindblad_ops = []
        if lindblad:
            if gamma_b is None:
                gamma_b = gamma_e
            lindblad_ops = [["|0><1|_5", gamma_e], ["|0><2|_5", gamma_e], ["|1><4|_5", gamma_b], ["|2><4|_5", gamma_b]]
            if gamma_d is not None:
                lindblad_ops.append(["|0><3|_5", gamma_d])  # dark state decay

        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        # coupling modes for x and y polarization
        modes = {"x": ketbra(1,0,5) + ketbra(4,1,5), "y": ketbra(3,0,5) + ketbra(4,3,5)}
        # rotating-frame operator (counts excitations, B has factor 2)
        rf_op = ketbra(1,1,5) + ketbra(2,2,5) + ketbra(3,3,5) + 2 * ketbra(4,4,5)
        dim_prod = [5]
        colors = ["#0000FF", "#FF0000", "#00FF00", "#FF00FF", "#AAAA00"]

        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                         threshold=threshold, boson_e_max=boson_e_max, initial=initial, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                         lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod, colors=colors)


