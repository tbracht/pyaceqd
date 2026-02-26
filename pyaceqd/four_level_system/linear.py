from pyaceqd.general_system.general_system_new import GeneralSystemACE
import pyaceqd.constants as constants
from pyaceqd.helpers.ace_operators import ketbra, kron, id
import numpy as np

hbar = constants.hbar  # meV*ps

class Biexciton(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, gamma_b=None, delta_xy=0, delta_b=4, shift_x=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
                 initial="|0><0|_4", lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir=""):
        system_prefix = "b_linear"
        system_op = []
        if shift_x:
            # |0> = G, |1> = X, |2> = Y, |3> = B
            # shift X and Y symmetrically around 0
            system_op = ["{}*|3><3|_4".format(-delta_b),"{}*|1><1|_4".format(-delta_xy/2),"{}*|2><2|_4".format(delta_xy/2)]
        else:
            # only shift Y, X stays at E=0
            # this is just a different rotating frame
            system_op = ["{}*|3><3|_4".format(-delta_b),"{}*|2><2|_4".format(delta_xy)]
        if lindblad:
            if gamma_b is None:
                gamma_b = gamma_e
            lindblad_ops = [["|0><1|_4",gamma_e],["|0><2|_4",gamma_e],
                            ["|1><3|_4",gamma_b],["|2><3|_4",gamma_b]]

        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = "1*(|1><1|_4 + |2><2|_4) + 2*|3><3|_4"  # operator that couples to phonons
        modes = {"x": ketbra(1,0,4)+ketbra(3,1,4), "y": ketbra(2,0,4)+ketbra(3,2,4)}  # coupling to x and y polarized light
        rf_op = ketbra(1,1,4) + ketbra(2,2,4) + 2*ketbra(3,3,4)  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[4]  # subsystem dimensions
        colors = ["#0000FF", "#FF0000", "#00FF00", "#FF00FF"]  # just some example colors
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                            threshold=threshold, boson_e_max=boson_e_max, initial=initial, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                            lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod, colors=colors)

class BiexcitonSensors(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, gamma_b=None, delta_xy=0, delta_s1=0, delta_s2=0, epsilon=1e-3, linewidth1=0.01, linewidth2=0.01,
                 delta_b=4, shift_x=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
                 initial="|0><0|_4 otimes |0><0|_2 otimes |0><0|_2", lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="",
                 propagate_Taylor=None):
        system_prefix = "b_linear_sensors"
        system_op = []
        if shift_x:
            # |0> = G, |1> = X, |2> = Y, |3> = B
            # shift X and Y symmetrically around 0
            system_op = ["{}*|3><3|_4 otimes Id_2 otimes Id_2".format(-delta_b),"{}*|1><1|_4 otimes Id_2 otimes Id_2".format(-delta_xy/2),"{}*|2><2|_4 otimes Id_2 otimes Id_2".format(delta_xy/2)]
        else:
            # only shift Y, X stays at E=0
            # this is just a different rotating frame
            system_op = ["{}*|3><3|_4 otimes Id_2 otimes Id_2".format(-delta_b),"{}*|2><2|_4 otimes Id_2 otimes Id_2".format(delta_xy)]
        
        lindblad_ops = []
        if lindblad:
            if gamma_b is None:
                gamma_b = gamma_e
            lindblad_ops = [["|0><1|_4 otimes Id_2 otimes Id_2",gamma_e],["|0><2|_4 otimes Id_2 otimes Id_2",gamma_e],
                            ["|1><3|_4 otimes Id_2 otimes Id_2",gamma_b],["|2><3|_4 otimes Id_2 otimes Id_2",gamma_b]]

        system_op.append("{} * (Id_4 otimes |1><1|_2 otimes Id_2)".format(delta_s1))  # sensor 1 Hamiltonian
        system_op.append("{} * (Id_4 otimes Id_2 otimes |1><1|_2)".format(delta_s2))  # sensor 2 Hamiltonian
        # sensor coupling
        # coupling to G-Y
        system_op.append("{} * (|2><0|_4 otimes |0><1|_2 otimes Id_2 + |0><2|_4 otimes |1><0|_2 otimes Id_2)".format(epsilon))
        # coupling to Y-B
        system_op.append("{} * (|3><2|_4 otimes |0><1|_2 otimes Id_2 + |2><3|_4 otimes |1><0|_2 otimes Id_2)".format(epsilon))
        # coupling to G-X
        system_op.append("{} * (|1><0|_4 otimes Id_2 otimes |0><1|_2 + |0><1|_4 otimes Id_2 otimes |1><0|_2)".format(epsilon))    
        # coupling to X-B
        system_op.append("{} * (|3><1|_4 otimes Id_2 otimes |0><1|_2 + |1><3|_4 otimes Id_2 otimes |1><0|_2)".format(epsilon))

        # sensor loss
        if linewidth2 is None:
            linewidth2 = linewidth1
        lindblad_ops.append(["Id_4 otimes |0><1|_2 otimes Id_2", linewidth1])
        lindblad_ops.append(["Id_4 otimes Id_2 otimes |0><1|_2", linewidth2])

        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = "1*(|1><1|_4 otimes Id_2 otimes Id_2 + |2><2|_4 otimes Id_2 otimes Id_2) + 2*(|3><3|_4 otimes Id_2 otimes Id_2) "
        modes = {"x": kron(ketbra(1,0,4)+ketbra(3,1,4), id(2), id(2)), "y": kron(ketbra(2,0,4)+ketbra(3,2,4), id(2), id(2))}  # coupling to x and y polarized light
        rf_op = kron(ketbra(1,1,4) + ketbra(2,2,4) + 2*ketbra(3,3,4), np.eye(2), np.eye(2))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[4,2,2]  # subsystem dimensions
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                            threshold=threshold, boson_e_max=boson_e_max, initial=initial, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                            lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod,
                            propagate_Taylor=propagate_Taylor)
        
class BiexcitonPhotons(GeneralSystemACE):
        def __init__(self, dt=0.1, gamma_e=1/100, n_phot1=2, n_phot2=2, gamma_b=None, delta_xy=0, delta_b=4, shift_x=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
                 initial=None, lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", propagate_Taylor=None, cav_coupl=0.06, cav_loss=0.12/hbar, delta_cx=-2):
            system_prefix = "b_linear_cavity"
            n1 = n_phot1 + 1
            n2 = n_phot2 + 1
            # initial state and output operators with correct number of photons
            if initial is None:
                initial = "|0><0|_4 otimes |0><0|_{} otimes |0><0|_{}".format(n1,n2)
            # |0> = G, |1> = X, |2> = Y, |3> = B
            system_op = ["-{}*|3><3|_4 otimes Id_{} otimes Id_{}".format(delta_b,n1,n2),
                         "-{}*|1><1|_4 otimes Id_{} otimes Id_{}".format(delta_xy/2,n1,n2),
                         "{}*|2><2|_4 otimes Id_{} otimes Id_{}".format(delta_xy/2,n1,n2)]
            boson_op = "|1><1|_4 otimes Id_{} otimes Id_{} + |2><2|_4 otimes Id_{} otimes Id_{} + 2*|3><3|_4 otimes Id_{} otimes Id_{}".format(n1,n2,n1,n2,n1,n2)
            lindblad_ops = []
            # QD decay outside of the cavity
            if lindblad:
                if gamma_b is None:
                    gamma_b = gamma_e
                lindblad_ops = [["|0><1|_4 otimes Id_{} otimes Id_{}".format(n1,n2),gamma_e],
                                ["|0><2|_4 otimes Id_{} otimes Id_{}".format(n1,n2),gamma_e],
                                ["|1><3|_4 otimes Id_{} otimes Id_{}".format(n1,n2),gamma_b],
                                ["|2><3|_4 otimes Id_{} otimes Id_{}".format(n1,n2),gamma_b]]
            # cavity loss
            lindblad_ops.append(["Id_4 otimes b_{} otimes Id_{}".format(n1,n2), cav_loss])
            lindblad_ops.append(["Id_4 otimes Id_{} otimes b_{}".format(n1,n2), cav_loss])
            # cavity-qd coupling
            # cavity energy/detuning
            system_op.append("{} * (Id_4 otimes n_{} otimes Id_{}) ".format(delta_cx,n1,n2))
            system_op.append("{} * (Id_4 otimes Id_{} otimes n_{}) ".format(delta_cx,n1,n2))
            # X-cavity
            system_op.append("{} * (|1><0|_4 otimes b_{} otimes Id_{} + |0><1|_4 otimes bdagger_{} otimes Id_{})".format(cav_coupl,n1,n2,n1,n2))
            system_op.append("{} * (|3><1|_4 otimes b_{} otimes Id_{} + |1><3|_4 otimes bdagger_{} otimes Id_{})".format(cav_coupl,n1,n2,n1,n2))
            # Y-cavity
            system_op.append("{} * (|2><0|_4 otimes Id_{} otimes b_{} + |0><2|_4 otimes Id_{} otimes bdagger_{})".format(cav_coupl,n1,n2,n1,n2))
            system_op.append("{} * (|3><2|_4 otimes Id_{} otimes b_{} + |2><3|_4 otimes Id_{} otimes bdagger_{})".format(cav_coupl,n1,n2,n1,n2))
            threshold = "8"  # threshold for PT generation
            boson_e_max = 7  # maximum boson energy in meV
            boson_op = f"1*(|1><1|_4 otimes Id_{n1} otimes Id_{n2} + |2><2|_4 otimes Id_{n1} otimes Id_{n2}) + 2*(|3><3|_4 otimes Id_{n1} otimes Id_{n2}) "
            modes = {"x": kron(ketbra(1,0,4)+ketbra(3,1,4), id(n1), id(n2)), "y": kron(ketbra(2,0,4)+ketbra(3,2,4), id(n1), id(n2))}  # coupling to x and y polarized light
            rf_op = kron(ketbra(1,1,4) + ketbra(2,2,4) + 2*ketbra(3,3,4), id(n1), id(n2))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
            dim_prod=[4,n1,n2]  # subsystem dimensions
            super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                            threshold=threshold, boson_e_max=boson_e_max, initial=initial, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                            lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod,
                            propagate_Taylor=propagate_Taylor)
