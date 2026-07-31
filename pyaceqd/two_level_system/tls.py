import numpy as np
from pyaceqd.helpers.ace_operators import ketbra, kron, id, b_op, bdag_op, n_op
from pyaceqd.general_system.general_system import GeneralSystemACE
from pyaceqd.helpers.reduce_dimension import get_remove_indices, remove_dim_numbers, get_union_indices, filter_n_excitations
import hashlib
import pyaceqd.constants as constants

hbar = constants.hbar  # meV*ps
kB = constants.kB  # meV/K

g = ketbra(0,0,2)  # ground state operator
x = ketbra(1,1,2)  # exciton operator
p_gx = ketbra(0,1,2)  # |0><1| operator

class TLS(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, lindblad=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None, J_to_file=None, J_file=None,
                 factor_ah=None, pt_dir="", e_x=0, rho0=ketbra(0,0,2), threshold=8):
        system_prefix = "tls" 
        threshold = str(int(threshold))  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        system_op = x*e_x  # system hamiltonian operator. default uses rotating frame where E_X = 0
        boson_op = x  # operator that couples to phonons
        lindblad_ops = [[p_gx, gamma_e]]  # decay of excited state to ground state
        modes = {"x": p_gx.T, "energy": x}  # operator |1><0|_2 couples to x-polarized light, and operator |1><1|_2 for time-dependent energy fluctuations
        rf_op = x  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        colors = ["#0000FF", "#FF0000"]
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                          threshold=threshold, boson_e_max=boson_e_max, system_op=system_op, modes=modes, rf_op=rf_op, boson_op=boson_op, 
                          lindblad_ops=lindblad_ops, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir,
                          dim_prod=[2], colors=colors, lindblad=lindblad, rho0=rho0)

class TLSOneSensor(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, phonons=False, delta_s=0, epsilon=1e-3, linewidth=0.01, ae=5, temperature=4, verbose=False, pt_file=None,
                 rho0=kron(ketbra(0,0,2),ketbra(0,0,2)), lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", propagate_Taylor=None):
        system_prefix = "tls_one_sensor"
        system_op = []
        system_op.append(kron(id(2),ketbra(1,1,2))*delta_s)  # sensor Hamiltonian
        system_op.append(epsilon * (kron(p_gx.T,p_gx)+kron(p_gx,p_gx.T)))  # sensor coupling: |0><1| otimes |1><0| + |1><0| otimes |0><1|
        lindblad_ops = [[kron(p_gx,id(2)), gamma_e], # decay of excited state to ground state
                        [kron(id(2),p_gx), linewidth]]  # sensor loss
        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = kron(x,id(2))  # operator that couples to phonons
        modes = {"x": kron(p_gx.T, id(2)), "energy": kron(x, id(2))}  # operator |1><0|_2 otimes Id_2 couples to x-polarized light
        rf_op = kron(x, id(2))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[2,2]  # TLS + sensor dimensions
        colors = ["#0000FF", "#FF0000", "#00FF00", "#FF00FF"]  # just some example colors
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix, 
                         threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops, 
                         lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod, colors=colors,
                         propagate_Taylor=propagate_Taylor)

class TLSTwoSensor(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, phonons=False, delta_s1=0, delta_s2=0, epsilon=1e-3, linewidth1=0.01, linewidth2=0.01, ae=5, temperature=4, verbose=False, pt_file=None,
                 rho0=kron(ketbra(0,0,2),ketbra(0,0,2),ketbra(0,0,2)), lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", propagate_Taylor=None):
        system_prefix = "tls_two_sensor"
        system_op = []
        system_op.append(delta_s1 * kron(id(2),x,id(2)))  # sensor 1 Hamiltonian
        system_op.append(delta_s2 * kron(id(2),id(2),x))  # sensor 2 Hamiltonian
        system_op.append(epsilon*(kron(p_gx,p_gx.T,id(2)) + kron(p_gx.T,p_gx,id(2))))  # sensor 1 coupling
        system_op.append(epsilon*(kron(p_gx,id(2),p_gx.T) + kron(p_gx.T,id(2),p_gx)))  # sensor 2 coupling
        lindblad_ops = [[kron(p_gx,id(2),id(2)), gamma_e]]  # decay of excited state to ground state
        lindblad_ops.append([kron(id(2),p_gx,id(2)), linewidth1])  # sensor 1 loss
        lindblad_ops.append([kron(id(2),id(2),p_gx), linewidth2])  # sensor 2 loss
        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = kron(x,id(2),id(2))  # operator that couples to phonons
        modes = {"x": kron(p_gx.T, id(2), id(2))}  # operator |1><0|_2 otimes Id_2 otimes Id_2 couples to x-polarized light
        rf_op = kron(x, id(2), id(2))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[2,2,2]  # TLS + sensor1 + sensor2 dimensions
        colors = ["#0000FF", "#FF0000", "#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#888888", "#000000"]  # just some example colors
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                            threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                            lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod, colors=colors,
                            propagate_Taylor=propagate_Taylor)

class TLSSensorBig(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, phonons=False, delta_s1=0, delta_s2=0, epsilon=1e-3, linewidth1=0.01, linewidth2=0.01, ae=5, temperature=4, verbose=False, pt_file=None,
                 rho0=kron(ketbra(0,0,2),ketbra(0,0,3)), lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", propagate_Taylor=None):
        system_prefix = "tls_big_sensor"
        system_op = []
        system_op.append(delta_s1*kron(id(2),ketbra(1,1,3)))  # sensor 1 Hamiltonian
        system_op.append(delta_s1+delta_s2*kron(id(2),ketbra(2,2,3)))  # sensor 2 Hamiltonian
        system_op.append(np.sqrt(2)*epsilon*(kron(p_gx,ketbra(1,0,3)) + kron(p_gx.T,ketbra(0,1,3))))  # sensor 1 coupling
        system_op.append(np.sqrt(2)*epsilon*(kron(p_gx,ketbra(2,1,3)) + kron(p_gx.T,ketbra(1,2,3))))  # sensor 2 coupling
        lindblad_ops = [[kron(p_gx,id(3)), gamma_e]]  # decay of excited state to ground state
        lindblad_ops.append([kron(id(2),ketbra(0,1,3)), linewidth1])  # sensor 1 loss
        lindblad_ops.append([kron(id(2),ketbra(0,2,3)), 2*linewidth2])  # sensor 2 loss
        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = kron(x,id(3))  # operator that couples to phonons
        modes = {"x": kron(p_gx.T, id(3))}  # operator |1><0|_2 otimes Id_2 otimes Id_2 couples to x-polarized light
        rf_op = kron(x, id(3))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[2,3]  # TLS + sensor1
        colors = ["#0000FF", "#FF0000", "#00FF00", "#FF00FF", "#00FFFF", "#FFFF00"]  # just some example colors
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                            threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                            lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod, colors=colors,
                            propagate_Taylor=propagate_Taylor)

class TLSPhoton(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, phonons=False, n_phot=2, cav_coupl=0.06, cav_loss=0.12, delta_cx=0,
                 laser_cav_coupl=None, ae=5, temperature=4, verbose=False, pt_file=None,
                 rho0=None, lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="",
                 propagate_Taylor=None, threshold=8, benchmark=False, expand_pt=None, propagate_Taylor_threshold=1e-8):
        n1 = n_phot + 1  # cavity Fock space dimension
        if rho0 is None:
            rho0 = kron(ketbra(0,0,2), ketbra(0,0,n1))

        system_prefix = "tls_cavity_n{}".format(n_phot)

        I_tls = id(2)
        I_cav = id(n1)
        b = b_op(n1)
        bd = bdag_op(n1)
        num = n_op(n1)

        system_op = []
        # cavity detuning
        system_op.append(delta_cx * kron(I_tls, num))
        # Jaynes-Cummings coupling
        system_op.append(cav_coupl * (kron(p_gx.T, b) + kron(p_gx, bd)))

        lindblad_ops = [[kron(p_gx, I_cav), gamma_e]]  # TLS decay
        lindblad_ops.append([kron(I_tls, b), cav_loss])  # cavity loss

        # x-polarized light drives TLS; optionally also drives cavity directly
        modes = {"x": kron(p_gx.T, I_cav)}
        if laser_cav_coupl is not None:
            modes["x"] = modes["x"] + laser_cav_coupl * kron(I_tls, bd)

        rf_op = kron(x, I_cav) + kron(I_tls, num)  # rotating frame: TLS + cavity

        boson_e_max = 7
        boson_op = kron(x, I_cav)  # phonon coupling on TLS
        dim_prod = [2, n1]

        threshold = str(int(threshold))  # threshold for PT generation
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                         threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                         lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op,
                         dim_prod=dim_prod, propagate_Taylor=propagate_Taylor, expand_pt=expand_pt, propagate_Taylor_threshold=propagate_Taylor_threshold)

class TLSPhotonTwoSensor(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, phonons=False, n_phot=2, cav_coupl=0.06, cav_loss=0.12, delta_cx=0,
                 delta_s1=0, delta_s2=None, epsilon=1e-3, linewidth1=0.01, linewidth2=None,
                 ae=5, temperature=4, verbose=False, pt_file=None,
                 rho0=None, lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="",
                 propagate_Taylor=None, n_sensor_2=2):
        n1 = n_phot + 1  # cavity Fock space dimension
        if delta_s2 is None:
            delta_s2 = delta_s1
        if linewidth2 is None:
            linewidth2 = linewidth1
        if rho0 is None:
            rho0 = kron(ketbra(0,0,2), ketbra(0,0,n1), ketbra(0,0,2), ketbra(0,0,n_sensor_2))

        system_prefix = "tls_cavity_two_sensor"

        # shorthand identity operators
        I_tls = id(2)
        I_cav = id(n1)
        I_s = id(2)
        b = b_op(n1)
        bd = bdag_op(n1)
        num = n_op(n1)

        I_s2 = id(n_sensor_2)
        b2 =b_op(n_sensor_2)
        bd2 = bdag_op(n_sensor_2)
        num2 = n_op(n_sensor_2)

        system_op = []
        # cavity detuning
        system_op.append(delta_cx * kron(I_tls, num, I_s, I_s2))
        # cavity coupling
        system_op.append(cav_coupl * (kron(p_gx.T, b, I_s, I_s2) + kron(p_gx, bd, I_s, I_s2)))
        # sensor 1 Hamiltonian
        system_op.append(delta_s1 * kron(I_tls, I_cav, ketbra(1,1,2), I_s2))
        # sensor 2 Hamiltonian
        system_op.append(delta_s2 * kron(I_tls, I_cav, I_s, num2))
        # sensor 1 coupling to cavity
        system_op.append(epsilon * (kron(I_tls, bd, ketbra(0,1,2), I_s2) + kron(I_tls, b, ketbra(1,0,2), I_s2)))
        # sensor 2 coupling to cavity
        system_op.append(epsilon * (kron(I_tls, bd, I_s, b2) + kron(I_tls, b, I_s, bd2)))

        lindblad_ops = [[kron(p_gx, I_cav, I_s, I_s2), gamma_e]]   # TLS decay
        lindblad_ops.append([kron(I_tls, b, I_s, I_s2), cav_loss])   # cavity loss
        lindblad_ops.append([kron(I_tls, I_cav, p_gx, I_s2), linewidth1])  # sensor 1 loss
        lindblad_ops.append([kron(I_tls, I_cav, I_s, b2), linewidth2])  # sensor 2 loss

        threshold = "8"
        boson_e_max = 7
        boson_op = kron(x, I_cav, I_s, I_s2)  # phonon coupling on TLS
        modes = {"x": kron(p_gx.T, I_cav, I_s, I_s2)}  # x-polarized light drives TLS
        rf_op = kron(x, I_cav, I_s, I_s2)
        dim_prod = [2, n1, 2, n_sensor_2]

        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                         threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                         lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op,
                         dim_prod=dim_prod, propagate_Taylor=propagate_Taylor)

class TLSPhotonTwoSensorReduced(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, phonons=False, n_phot=2, cav_coupl=0.06, cav_loss=0.12, delta_cx=0,
                 delta_s1=0, delta_s2=None, epsilon=1e-3, linewidth1=0.01, linewidth2=None,
                 ae=5, temperature=4, verbose=False, pt_file=None,
                 rho0=None, lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="",
                 propagate_Taylor=None, n_sensor_2=2, max_excitations=None, remove_sensor_states=False, remove_states=None):
        n1 = n_phot + 1  # cavity Fock space dimension
        if delta_s2 is None:
            delta_s2 = delta_s1
        if linewidth2 is None:
            linewidth2 = linewidth1

        states_to_remove = []
        if max_excitations is not None:
            n_system = x
            n_phot1 = n_op(n1)
            total_excitations = kron(n_system, id(n1), id(2), id(n_sensor_2)) + kron(id(2), n_phot1, id(2), id(n_sensor_2))
            if remove_sensor_states:
                n_sensor_1 = n_op(2)
                n_sensor_2_op = n_op(n_sensor_2)
                total_excitations += kron(id(2), id(n1), n_sensor_1, id(n_sensor_2)) + kron(id(2), id(n1), id(2), n_sensor_2_op)
            _remove = filter_n_excitations(total_excitations, max_excitations)
            states_to_remove.extend(_remove)

        hash_states = ""
        if remove_states is not None:
            states_to_remove = get_union_indices(states_to_remove, get_remove_indices(remove_states))
        if remove_states is not None or max_excitations is not None:
            if verbose:
                print(f"Removing states: {states_to_remove}")
            states_to_remove = np.array(states_to_remove)
            hash_states = hashlib.sha1(states_to_remove.data.tobytes()).hexdigest()[:10]  # create a hash of the removed states for the filename
            hash_states = f"_remove_{hash_states}"
        self.states_to_remove = states_to_remove
        system_prefix = "tls_cavity_two_sensor_{}".format(hash_states)

        if rho0 is None:
            rho0 = kron(ketbra(0,0,2), ketbra(0,0,n1), ketbra(0,0,2), ketbra(0,0,n_sensor_2))

        # shorthand identity operators
        I_tls = id(2)
        I_cav = id(n1)
        I_s = id(2)
        b = b_op(n1)
        bd = bdag_op(n1)
        num = n_op(n1)

        I_s2 = id(n_sensor_2)
        b2 =b_op(n_sensor_2)
        bd2 = bdag_op(n_sensor_2)
        num2 = n_op(n_sensor_2)

        system_op = []
        # cavity detuning
        system_op.append(delta_cx * kron(I_tls, num, I_s, I_s2))
        # cavity coupling
        system_op.append(cav_coupl * (kron(p_gx.T, b, I_s, I_s2) + kron(p_gx, bd, I_s, I_s2)))
        # sensor 1 Hamiltonian
        system_op.append(delta_s1 * kron(I_tls, I_cav, ketbra(1,1,2), I_s2))
        # sensor 2 Hamiltonian
        system_op.append(delta_s2 * kron(I_tls, I_cav, I_s, num2))
        # sensor 1 coupling to cavity
        system_op.append(epsilon * (kron(I_tls, bd, ketbra(0,1,2), I_s2) + kron(I_tls, b, ketbra(1,0,2), I_s2)))
        # sensor 2 coupling to cavity
        system_op.append(epsilon * (kron(I_tls, bd, I_s, b2) + kron(I_tls, b, I_s, bd2)))

        lindblad_ops = [[kron(p_gx, I_cav, I_s, I_s2), gamma_e]]   # TLS decay
        lindblad_ops.append([kron(I_tls, b, I_s, I_s2), cav_loss])   # cavity loss
        lindblad_ops.append([kron(I_tls, I_cav, p_gx, I_s2), linewidth1])  # sensor 1 loss
        lindblad_ops.append([kron(I_tls, I_cav, I_s, b2), linewidth2])  # sensor 2 loss

        threshold = "8"
        boson_e_max = 7
        boson_op = kron(x, I_cav, I_s, I_s2)  # phonon coupling on TLS
        modes = {"x": kron(p_gx.T, I_cav, I_s, I_s2)}  # x-polarized light drives TLS
        rf_op = kron(x, I_cav, I_s, I_s2)

        # filter dimensions
        rho0 = remove_dim_numbers(rho0, states_to_remove)
        for i in range(len(system_op)):
            system_op[i] = remove_dim_numbers(system_op[i], states_to_remove)
        for i in range(len(lindblad_ops)):
            lindblad_ops[i][0] = remove_dim_numbers(lindblad_ops[i][0], states_to_remove)
        for key in modes:
            modes[key] = remove_dim_numbers(modes[key], states_to_remove)
        rf_op = remove_dim_numbers(rf_op, states_to_remove)
        boson_op = remove_dim_numbers(boson_op, states_to_remove)

        dim_prod = [rho0.shape[0]]  # only one dimension after filtering

        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                         threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                         lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op,
                         dim_prod=dim_prod, propagate_Taylor=propagate_Taylor)

