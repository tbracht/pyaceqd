import numpy as np
from pyaceqd.helpers.ace_operators import ketbra, kron, id
from pyaceqd.general_system.general_system_new import GeneralSystemACE

import pyaceqd.constants as constants

hbar = constants.hbar  # meV*ps
kB = constants.kB  # meV/K

class TLS_(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, lindblad=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None, J_to_file=None, J_file=None,
                 factor_ah=None, pt_dir="", e_x=0, initial="|0><0|_2", threshold=8):
        system_prefix = "tls" 
        threshold = str(int(threshold))  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        system_op = ["{}*|1><1|_2".format(e_x)]  # system hamiltonian operator. default uses rotating frame where E_X = 0
        boson_op = "|1><1|_2"  # operator that couples to phonons
        lindblad_ops = [["|0><1|_2", gamma_e]]  # decay of excited state to ground state
        modes = {"x": ketbra(1,0,2)}  # operator |1><0|_2 couples to x-polarized light
        rf_op = ketbra(1,1,2)  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        colors = ["#0000FF", "#FF0000"]
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                          threshold=threshold, boson_e_max=boson_e_max, system_op=system_op, modes=modes, rf_op=rf_op, initial=initial,
                          boson_op=boson_op, lindblad_ops=lindblad_ops, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir,
                          dim_prod=[2], colors=colors, lindblad=lindblad)
        
class TLSOneSensor(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, phonons=False, delta_s=0, epsilon=1e-3, linewidth=0.01, ae=5, temperature=4, verbose=False, pt_file=None,
                 initial="|0><0|_2 otimes |0><0|_2", lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", propagate_Taylor=None):
        system_prefix = "tls_one_sensor"
        system_op = []
        system_op.append("{} * (Id_2 otimes |1><1|_2)".format(delta_s))  # sensor Hamiltonian
        system_op.append("{} * (|1><0|_2 otimes |0><1|_2 + |0><1|_2 otimes |1><0|_2)".format(epsilon))  # sensor coupling
        lindblad_ops = [["|0><1|_2 otimes Id_2", gamma_e]]  # decay of excited state to ground state
        lindblad_ops.append(["Id_2 otimes |0><1|_2", linewidth])  # sensor loss
        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = "|1><1|_2 otimes Id_2"  # operator that couples to phonons
        modes = {"x": kron(ketbra(1,0,2), id(2))}  # operator |1><0|_2 otimes Id_2 couples to x-polarized light
        rf_op = kron(ketbra(1,1,2), id(2))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[2,2]  # TLS + sensor dimensions
        colors = ["#0000FF", "#FF0000", "#00FF00", "#FF00FF"]  # just some example colors
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix, 
                         threshold=threshold, boson_e_max=boson_e_max, initial=initial, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops, 
                         lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod, colors=colors,
                         propagate_Taylor=propagate_Taylor)

class TLSTwoSensor(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, phonons=False, delta_s1=0, delta_s2=0, epsilon=1e-3, linewidth1=0.01, linewidth2=0.01, ae=5, temperature=4, verbose=False, pt_file=None,
                 initial="|0><0|_2 otimes |0><0|_2 otimes |0><0|_2", lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", propagate_Taylor=None):
        system_prefix = "tls_two_sensor"
        system_op = []
        system_op.append("{} * (Id_2 otimes |1><1|_2 otimes Id_2)".format(delta_s1))  # sensor 1 Hamiltonian
        system_op.append("{} * (Id_2 otimes Id_2 otimes |1><1|_2)".format(delta_s2))  # sensor 2 Hamiltonian
        system_op.append("{} * (|1><0|_2 otimes |0><1|_2 otimes Id_2 + |0><1|_2 otimes |1><0|_2 otimes Id_2)".format(epsilon))  # sensor 1 coupling
        system_op.append("{} * (|1><0|_2 otimes Id_2 otimes |0><1|_2 + |0><1|_2 otimes Id_2 otimes |1><0|_2)".format(epsilon))  # sensor 2 coupling
        lindblad_ops = [["|0><1|_2 otimes Id_2 otimes Id_2", gamma_e]]  # decay of excited state to ground state
        lindblad_ops.append(["Id_2 otimes |0><1|_2 otimes Id_2", linewidth1])  # sensor 1 loss
        lindblad_ops.append(["Id_2 otimes Id_2 otimes |0><1|_2", linewidth2])  # sensor 2 loss
        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = "|1><1|_2 otimes Id_2 otimes Id_2"  # operator that couples to phonons
        modes = {"x": kron(ketbra(1,0,2), id(2), id(2))}  # operator |1><0|_2 otimes Id_2 otimes Id_2 couples to x-polarized light
        rf_op = kron(ketbra(1,1,2), id(2), id(2))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[2,2,2]  # TLS + sensor1 + sensor2 dimensions
        colors = ["#0000FF", "#FF0000", "#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#888888", "#000000"]  # just some example colors
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                            threshold=threshold, boson_e_max=boson_e_max, initial=initial, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                            lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod, colors=colors,
                            propagate_Taylor=propagate_Taylor)

class TLSSensorBig(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, phonons=False, delta_s1=0, delta_s2=0, epsilon=1e-3, linewidth1=0.01, linewidth2=0.01, ae=5, temperature=4, verbose=False, pt_file=None,
                 initial="|0><0|_2 otimes |0><0|_3", lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", propagate_Taylor=None):
        system_prefix = "tls_big_sensor"
        system_op = []
        system_op.append("{} * (Id_2 otimes |1><1|_3)".format(delta_s1))  # sensor 1 Hamiltonian
        system_op.append("{} * (Id_2 otimes |2><2|_3)".format(delta_s1+delta_s2))  # sensor 2 Hamiltonian
        system_op.append("{} * (|1><0|_2 otimes |0><1|_3 + |0><1|_2 otimes |1><0|_3)".format(np.sqrt(2)*epsilon))  # sensor 1 coupling
        # system_op.append("{} * (|1><0|_2 otimes |0><2|_3 + |0><1|_2 otimes |2><0|_3)".format(epsilon))  # sensor 2 coupling
        system_op.append("{} * (|1><0|_2 otimes |1><2|_3 + |0><1|_2 otimes |2><1|_3)".format(np.sqrt(2)*epsilon))  # inter sensor coupling
        lindblad_ops = [["|0><1|_2 otimes Id_3", gamma_e]]  # decay of excited state to ground state
        lindblad_ops.append(["Id_2 otimes |0><1|_3", linewidth1])  # sensor 1 loss
        lindblad_ops.append(["Id_2 otimes |0><2|_3", 2*linewidth2])  # sensor 2 loss
        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = "|1><1|_2 otimes Id_3"  # operator that couples to phonons
        modes = {"x": kron(ketbra(1,0,2), id(3))}  # operator |1><0|_2 otimes Id_2 otimes Id_2 couples to x-polarized light
        rf_op = kron(ketbra(1,1,2), id(3))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[2,3]  # TLS + sensor1
        colors = ["#0000FF", "#FF0000", "#00FF00", "#FF00FF", "#00FFFF", "#FFFF00"]  # just some example colors
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                            threshold=threshold, boson_e_max=boson_e_max, initial=initial, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                            lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod, colors=colors,
                            propagate_Taylor=propagate_Taylor)
