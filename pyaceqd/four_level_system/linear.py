from pyaceqd.general_system.general_system import GeneralSystemACE
import pyaceqd.constants as constants
from pyaceqd.helpers.ace_operators import ketbra, kron, id, b_op, bdag_op, n, zeros
from pyaceqd.helpers.reduce_dimension import get_remove_indices, remove_dim_numbers, get_union_indices, filter_n_excitations
import numpy as np
hbar = constants.hbar  # meV*ps
import hashlib

# system operators
g = ketbra(0,0,4)
x = ketbra(1,1,4)
y = ketbra(2,2,4)
b = ketbra(3,3,4)
p_gx = ketbra(0,1,4)
p_xb = ketbra(1,3,4)
p_gy = ketbra(0,2,4)
p_yb = ketbra(2,3,4)
i4 = id(4)

# sensor operators
s = ketbra(1,1,2)
s_0 = ketbra(0,0,2)
p_s = ketbra(0,1,2)  # sensor lowering operator
i2 = id(2)

class Biexciton(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, gamma_b=None, delta_xy=0, delta_b=4, shift_x=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
                 rho0=ketbra(0,0,4), lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir=""):
        system_prefix = "b_linear"
        system_op = []
        if shift_x:
            # |0> = G, |1> = X, |2> = Y, |3> = B
            # shift X and Y symmetrically around 0
            system_op = [-delta_b*b,
                         -delta_xy/2 * x,
                         delta_xy/2 * y]
        else:
            # only shift Y, X stays at E=0
            # this is just a different rotating frame
            system_op = [-delta_b*b,delta_xy * y]
        if lindblad:
            if gamma_b is None:
                gamma_b = gamma_e
            lindblad_ops = [[p_gx,gamma_e],[p_gy,gamma_e],
                            [p_xb,gamma_b],[p_yb,gamma_b]]

        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = x + y + 2*b  # operator that couples to phonons
        modes = {"x": p_gx.T+p_xb.T, "y": p_gy.T+p_yb.T}  # coupling to x and y polarized light
        rf_op = x + y + 2*b  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[4]  # subsystem dimensions
        colors = ["#0000FF", "#FF0000", "#00FF00", "#FF00FF"]  # just some example colors
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix, rho0=rho0,
                            threshold=threshold, boson_e_max=boson_e_max, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                            lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod, colors=colors)

class BiexcitonSensors(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, gamma_b=None, delta_xy=0, delta_s1=0, delta_s2=0, epsilon=1e-3, linewidth1=0.01, linewidth2=0.01,
                 delta_b=4, shift_x=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
                 rho0=kron(ketbra(0,0,4),ketbra(0,0,2),ketbra(0,0,2)), lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="",
                 propagate_Taylor=None):
        system_prefix = "b_linear_sensors"
        system_op = []
        if shift_x:
            # |0> = G, |1> = X, |2> = Y, |3> = B
            # shift X and Y symmetrically around 0
            system_op = [-delta_b * kron(b,i2,i2),
                         -delta_xy/2 * kron(x,i2,i2),
                         delta_xy/2 * kron(y,i2,i2)]
        else:
            # only shift Y, X stays at E=0
            # this is just a different rotating frame
            system_op = [-delta_b * kron(b,i2,i2),
                         delta_xy * kron(y,i2,i2)]
        
        lindblad_ops = []
        if lindblad:
            if gamma_b is None:
                gamma_b = gamma_e
            lindblad_ops = [[kron(p_gx,i2,i2),gamma_e],
                            [kron(p_gy,i2,i2),gamma_e],
                            [kron(p_xb,i2,i2),gamma_b],
                            [kron(p_yb,i2,i2),gamma_b]]

        system_op.append(delta_s1 * kron(i4,s,i2))  # sensor 1 Hamiltonian
        system_op.append(delta_s2 * kron(i4,i2,s))  # sensor 2 Hamiltonian
        # sensor coupling: sensor 1 to Y and B, sensor 2 to X and B
        # coupling to G-Y
        system_op.append(epsilon * (kron(p_gy.T,p_s,i2) + kron(p_gy,p_s.T,i2)))
        # coupling to Y-B
        system_op.append(epsilon * (kron(p_yb.T,p_s,i2) + kron(p_yb,p_s.T,i2)))
        # coupling to G-X
        system_op.append(epsilon * (kron(p_gx.T,i2,p_s) + kron(p_gx,i2,p_s.T)))
        # coupling to X-B
        system_op.append(epsilon * (kron(p_xb.T,i2,p_s) + kron(p_xb,i2,p_s.T)))

        # sensor loss
        if linewidth2 is None:
            linewidth2 = linewidth1
        lindblad_ops.append([kron(i4,p_s,i2), linewidth1])
        lindblad_ops.append([kron(i4,i2,p_s), linewidth2])

        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = kron(x,i2,i2) + kron(y,i2,i2) + 2*kron(b,i2,i2)
        modes = {"x": kron(p_gx.T+p_xb.T, i2, i2), "y": kron(p_gy.T+p_yb.T, i2, i2)}  # coupling to x and y polarized light
        rf_op = kron(x + y + 2*b, i2, i2)  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[4,2,2]  # subsystem dimensions
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                            threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                            lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod,
                            propagate_Taylor=propagate_Taylor)
        
class BiexcitonPhotons(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, n_phot1=2, n_phot2=2, gamma_b=None, delta_xy=0, delta_b=4, shift_x=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
             rho0=None, lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", propagate_Taylor=None, cav_coupl=0.06, cav_loss=0.12/hbar, delta_cx=-2):
        n1 = n_phot1 + 1
        n2 = n_phot2 + 1
        system_prefix = f"b_linear_cavity_{n1}_{n2}"
        if rho0 is None:
            rho0 = kron(g,ketbra(0,0,n1),ketbra(0,0,n2))
        
        # |0> = G, |1> = X, |2> = Y, |3> = B
        system_op = [-delta_b*kron(b,id(n1),id(n2)),
                     -0.5*delta_xy*kron(x,id(n1),id(n2)),
                     0.5*delta_xy*kron(y,id(n1),id(n2))]
        if not shift_x:  # if not shifting X, then shift Y by full delta_xy instead of symmetrically
            system_op = [-delta_b*kron(b,id(n1),id(n2)),
                     delta_xy*kron(y,id(n1),id(n2))]
        lindblad_ops = []
        # QD decay outside of the cavity
        if lindblad:
            if gamma_b is None:
                gamma_b = gamma_e
            lindblad_ops = [[kron(p_gx,id(n1),id(n2)), gamma_e],
                            [kron(p_gy,id(n1),id(n2)), gamma_e],
                            [kron(p_xb,id(n1),id(n2)), gamma_b],
                            [kron(p_yb,id(n1),id(n2)), gamma_b]]
        # cavity loss
        lindblad_ops.append([kron(i4,b_op(n1),id(n2)), cav_loss])
        lindblad_ops.append([kron(i4,id(n1),b_op(n2)), cav_loss])
        # cavity-qd coupling
        # cavity energy/detuning
        system_op.append(delta_cx*kron(i4,n(n1),id(n2)))
        system_op.append(delta_cx*kron(i4,id(n1),n(n2)))
        # X-cavity
        system_op.append(cav_coupl * ( kron(p_gx.T,b_op(n1),id(n2)) + kron(p_gx,bdag_op(n1),id(n2))))  # |1><0| otimes b + h.c.
        system_op.append(cav_coupl * ( kron(p_xb.T,b_op(n1),id(n2)) + kron(p_xb,bdag_op(n1),id(n2))))  # |3><1| otimes b + h.c.
        # Y-cavity
        system_op.append(cav_coupl * ( kron(p_gy.T,id(n1),b_op(n2)) + kron(p_gy,id(n1),bdag_op(n2))))  # |2><0| otimes b + h.c.
        system_op.append(cav_coupl * ( kron(p_yb.T,id(n1),b_op(n2)) + kron(p_yb,id(n1),bdag_op(n2))))  # |3><2| otimes b + h.c.
        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = kron(x,id(n1),id(n2)) + kron(y,id(n1),id(n2)) + 2*kron(b,id(n1),id(n2))
        modes = {"x": kron(p_gx.T+p_xb.T, id(n1), id(n2)), "y": kron(p_gy.T+p_yb.T, id(n1), id(n2))}  # coupling to x and y polarized light
        rf_op = kron(x + y + 2*b, id(n1), id(n2))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[4,n1,n2]  # subsystem dimensions
        # initial = f"|0><0|_{np.prod(dim_prod)}"  # just to give the PT the correct dimension.
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                        threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                        lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod,
                        propagate_Taylor=propagate_Taylor)
            
class BiexcitonPhotonsReduced(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, n_phot1=2, n_phot2=2, gamma_b=None, delta_xy=0, delta_b=4, shift_x=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
             rho0=None, lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", propagate_Taylor=None, cav_coupl=0.06, cav_loss=0.12/hbar, delta_cx=-2, remove_states=None,
             max_excitations=None):
        n1 = n_phot1 + 1
        n2 = n_phot2 + 1
        states_to_remove = []

        if max_excitations is not None:
            n_system = x + y + 2*b  # operator that counts the number of excitations in the system
            n_photons1 = n(n1)  # operator that counts the number of photons in cavity 1
            n_photons2 = n(n2)  # operator that counts the number of photons in cavity 2
            total_excitations = kron(n_system,id(n1),id(n1)) + kron(id(4),n_photons1,id(n2)) + kron(id(4),id(n1),n_photons2)  # total number of excitations in the system
            _remove = filter_n_excitations(total_excitations, max_excitations)
            states_to_remove = get_union_indices(states_to_remove, _remove)  # get indices of states to remove based on max_excitations
        
        hash_states = ""
        if remove_states is not None:
            states_to_remove = get_union_indices(states_to_remove, get_remove_indices(remove_states))
        if remove_states is not None or max_excitations is not None:
            if verbose:
                print(f"Removing states with indices {states_to_remove}")
            states_to_remove = np.array(states_to_remove)
            hash_states = hashlib.sha1(states_to_remove.data.tobytes()).hexdigest()[:10]  # create a hash of the removed states for the filename
            hash_states = f"_remove_{hash_states}"
        self.states_to_remove = states_to_remove
        
        system_prefix = f"b_linear_cavity_{n1}_{n2}{hash_states}"
        if rho0 is None:
            rho0 = kron(g,ketbra(0,0,n1),ketbra(0,0,n2))
        
        # |0> = G, |1> = X, |2> = Y, |3> = B
        system_op = [-delta_b*kron(b,id(n1),id(n2)),
                     -0.5*delta_xy*kron(x,id(n1),id(n2)),
                     0.5*delta_xy*kron(y,id(n1),id(n2))]
        if not shift_x:  # if not shifting X, then shift Y by full delta_xy instead of symmetrically
            system_op = [-delta_b*kron(b,id(n1),id(n2)),
                     delta_xy*kron(y,id(n1),id(n2))]
        lindblad_ops = []
        # QD decay outside of the cavity
        if lindblad:
            if gamma_b is None:
                gamma_b = gamma_e
            lindblad_ops = [[kron(p_gx,id(n1),id(n2)), gamma_e],
                            [kron(p_gy,id(n1),id(n2)), gamma_e],
                            [kron(p_xb,id(n1),id(n2)), gamma_b],
                            [kron(p_yb,id(n1),id(n2)), gamma_b]]
        # cavity loss
        lindblad_ops.append([kron(i4,b_op(n1),id(n2)), cav_loss])
        lindblad_ops.append([kron(i4,id(n1),b_op(n2)), cav_loss])
        # cavity-qd coupling
        # cavity energy/detuning
        system_op.append(delta_cx*kron(i4,n(n1),id(n2)))
        system_op.append(delta_cx*kron(i4,id(n1),n(n2)))
        # X-cavity
        system_op.append(cav_coupl *  (kron(p_gx.T,b_op(n1),id(n2)) + kron(p_gx,bdag_op(n1),id(n2))))  # |1><0| otimes b + h.c.
        system_op.append(cav_coupl *  (kron(p_xb.T,b_op(n1),id(n2)) + kron(p_xb,bdag_op(n1),id(n2))))  # |3><1| otimes b + h.c.
        # Y-cavity
        system_op.append(cav_coupl *  (kron(p_gy.T,id(n1),b_op(n2)) + kron(p_gy,id(n1),bdag_op(n2))))  # |2><0| otimes b + h.c.
        system_op.append(cav_coupl *  (kron(p_yb.T,id(n1),b_op(n2)) + kron(p_yb,id(n1),bdag_op(n2))))  # |3><2| otimes b + h.c.
        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = kron(x,id(n1),id(n2)) + kron(y,id(n1),id(n2)) + 2*kron(b,id(n1),id(n2))  # operator that couples to phonons
        modes = {"x": kron(p_gx.T+p_xb.T, id(n1), id(n2)), "y": kron(p_gy.T+p_yb.T, id(n1), id(n2))}  # coupling to x and y polarized light
        rf_op = kron(x + y + 2*b, id(n1), id(n2))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)

        # filter dimensions
        rho0 = remove_dim_numbers(rho0, states_to_remove)
        for i in range(len(lindblad_ops)):
            lindblad_ops[i][0] = remove_dim_numbers(lindblad_ops[i][0], states_to_remove)
        for i in range(len(system_op)):
            system_op[i] = remove_dim_numbers(system_op[i], states_to_remove)
        boson_op = remove_dim_numbers(boson_op, states_to_remove)
        for mode in modes:
            modes[mode] = remove_dim_numbers(modes[mode], states_to_remove)
        rf_op = remove_dim_numbers(rf_op, states_to_remove)
        dim_prod = [rho0.shape[0]]  # subsystem dimensions

        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                        threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                        lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod,
                        propagate_Taylor=propagate_Taylor)

class BiexcitonFourSensors(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, gamma_b=None, delta_xy=0, sensor_detunings=[0,0,0,0], epsilon=1e-3, sensor_linewidths=[0.01,0.01,0.01,0.01], 
                 delta_b=4, shift_x=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
                 rho0=None, lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="",
                 propagate_Taylor=None):
        system_prefix = "b_four_sensors"
        system_op = []

        if rho0 is None:
            rho0 = kron(g, s_0, s_0, s_0, s_0)  # start in ground state of the system and sensors

        if shift_x:
            # |0> = G, |1> = X, |2> = Y, |3> = B
            # shift X and Y symmetrically around 0
            system_op = [-delta_b*kron(b,i2,i2,i2,i2),
                         -delta_xy/2*kron(x,i2,i2,i2,i2),
                         delta_xy/2*kron(y,i2,i2,i2,i2)]
        else:
            # only shift Y, X stays at E=0
            # this is just a different rotating frame
            system_op = [-delta_b*kron(b,i2,i2,i2,i2),
                         delta_xy*kron(y,i2,i2,i2,i2)]
        
        lindblad_ops = []
        if lindblad:
            if gamma_b is None:
                gamma_b = gamma_e
            lindblad_ops = [[kron(p_gx,i2,i2,i2,i2), gamma_e],
                            [kron(p_gy,i2,i2,i2,i2), gamma_e],
                            [kron(p_xb,i2,i2,i2,i2), gamma_b],
                            [kron(p_yb,i2,i2,i2,i2), gamma_b]]

        system_op.append(sensor_detunings[0]*kron(i4,s,i2,i2,i2))  # sensor Hamiltonian
        system_op.append(sensor_detunings[1]*kron(i4,i2,s,i2,i2))
        system_op.append(sensor_detunings[2]*kron(i4,i2,i2,s,i2))
        system_op.append(sensor_detunings[3]*kron(i4,i2,i2,i2,s))
        # sensor coupling: sensor 1 to GY, sensor 2 to YB, sensor 3 to GX and sensor 4 to XB
        # coupling to G-Y
        system_op.append(epsilon * (kron(p_gy,p_s.T,i2,i2,i2) + kron(p_gy.T,p_s,i2,i2,i2)))  # sensor1
        # coupling to Y-B
        system_op.append(epsilon * (kron(p_yb,i2,p_s.T,i2,i2) + kron(p_yb.T,i2,p_s,i2,i2)))  # sensor2
        # coupling to G-X
        system_op.append(epsilon * (kron(p_gx,i2,i2,p_s.T,i2) + kron(p_gx.T,i2,i2,p_s,i2)))  # sensor3
        # coupling to X-B
        system_op.append(epsilon * (kron(p_xb,i2,i2,i2,p_s.T) + kron(p_xb.T,i2,i2,i2,p_s)))  # sensor4

        # sensor loss
        lindblad_ops.append([kron(i4,p_s,i2,i2,i2), sensor_linewidths[0]])
        lindblad_ops.append([kron(i4,i2,p_s,i2,i2), sensor_linewidths[1]])
        lindblad_ops.append([kron(i4,i2,i2,p_s,i2), sensor_linewidths[2]])
        lindblad_ops.append([kron(i4,i2,i2,i2,p_s), sensor_linewidths[3]])

        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = kron(x + y + 2*b, i2, i2, i2, i2)  # operator that couples to phonons
        modes = {"x": kron(p_gx.T+p_xb.T, i2, i2, i2, i2), "y": kron(p_gy.T+p_yb.T, i2, i2, i2, i2)}  # coupling to x and y polarized light
        rf_op = kron(x + y + 2*b, i2, i2, i2, i2)  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)
        dim_prod=[4,2,2,2,2]  # subsystem dimensions
        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                            threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                            lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod,
                            propagate_Taylor=propagate_Taylor)

class BiexcitonPhotonsSensor(GeneralSystemACE):
    def __init__(self, dt=0.1, gamma_e=1/100, n_phot1=2, n_phot2=2, gamma_b=None, delta_xy=0, delta_b=4, shift_x=True, phonons=False, ae=5, temperature=4, verbose=False, pt_file=None,
             rho0=None, lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", propagate_Taylor=None, cav_coupl=0.06, cav_loss=0.12/hbar, delta_cx=-2, remove_states=None,
             max_excitations=None, sensor_detunings=[0,0], sensor_linewidths=[0.01,0.01], epsilon=1e-3, n_sensor_2=2):
        n1 = n_phot1 + 1
        n2 = n_phot2 + 1
        states_to_remove = []
        if n_sensor_2 > 2:
            print("Sensor 2 more than 2 levels, this is not standard for sensor systems.")

        if max_excitations is not None:
            # sensors dont contribute to excitation number
            n_system = x + y + 2*b  # operator that counts the number of excitations in the system
            n_photons1 = n(n1)  # operator that counts the number of photons in cavity 1
            n_photons2 = n(n2)  # operator that counts the number of photons in cavity 2
            total_excitations = kron(n_system,id(n1),id(n1),id(2),id(n_sensor_2)) + kron(id(4),n_photons1,id(n2),id(2),id(n_sensor_2)) + kron(id(4),id(n1),n_photons2,id(2),id(n_sensor_2))  # total number of excitations in the system
            _remove = filter_n_excitations(total_excitations, max_excitations)
            states_to_remove = get_union_indices(states_to_remove, _remove)  # get indices of states to remove based on max_excitations
        
        hash_states = ""
        if remove_states is not None:
            states_to_remove = get_union_indices(states_to_remove, get_remove_indices(remove_states))
        if remove_states is not None or max_excitations is not None:
            if verbose:
                print(f"Removing states with indices {states_to_remove}")
            states_to_remove = np.array(states_to_remove)
            hash_states = hashlib.sha1(states_to_remove.data.tobytes()).hexdigest()[:10]  # create a hash of the removed states for the filename
            hash_states = f"_remove_{hash_states}"
        self.states_to_remove = states_to_remove
        
        system_prefix = f"b_linear_cavity_{n1}_{n2}{hash_states}"
        if rho0 is None:
            rho0 = kron(g,ketbra(0,0,n1),ketbra(0,0,n2),ketbra(0,0,2),ketbra(0,0,n_sensor_2))
        
        # |0> = G, |1> = X, |2> = Y, |3> = B
        system_op = [-delta_b*kron(b,id(n1),id(n2),id(2),id(n_sensor_2)),
                     -0.5*delta_xy*kron(x,id(n1),id(n2),id(2),id(n_sensor_2)),
                     0.5*delta_xy*kron(y,id(n1),id(n2),id(2),id(n_sensor_2))]
        if not shift_x:  # if not shifting X, then shift Y by full delta_xy instead of symmetrically
            system_op = [-delta_b*kron(b,id(n1),id(n2),id(2),id(n_sensor_2)),
                     delta_xy*kron(y,id(n1),id(n2),id(2),id(n_sensor_2))]
        system_op.append(sensor_detunings[0]*kron(i4,id(n1),id(n2),s,id(n_sensor_2)))  # sensor Hamiltonian
        if n_sensor_2 == 2:
            system_op.append(sensor_detunings[1]*kron(i4,id(n1),id(n2),i2,s))
        # sensor coupling: sensor 1 and 2 to cavity 1 (X-pol cavity)
        system_op.append(epsilon * (kron(i4,b_op(n1),id(n2),p_s.T,id(n_sensor_2)) + kron(i4,bdag_op(n1),id(n2),p_s,id(n_sensor_2))))  # sensor-cavity coupling
        if n_sensor_2 == 2:
            system_op.append(epsilon * (kron(i4,b_op(n1),id(n2),id(2),p_s.T) + kron(i4,bdag_op(n1),id(n2),id(2),p_s)))  # sensor-cavity coupling
        lindblad_ops = []
        # QD decay outside of the cavity
        if lindblad:
            if gamma_b is None:
                gamma_b = gamma_e
            lindblad_ops = [[kron(p_gx,id(n1),id(n2),id(2),id(n_sensor_2)), gamma_e],
                            [kron(p_gy,id(n1),id(n2),id(2),id(n_sensor_2)), gamma_e],
                            [kron(p_xb,id(n1),id(n2),id(2),id(n_sensor_2)), gamma_b],
                            [kron(p_yb,id(n1),id(n2),id(2),id(n_sensor_2)), gamma_b]]
        # cavity loss
        lindblad_ops.append([kron(i4,b_op(n1),id(n2),id(2),id(n_sensor_2)), cav_loss])
        lindblad_ops.append([kron(i4,id(n1),b_op(n2),id(2),id(n_sensor_2)), cav_loss])
        # sensor loss
        lindblad_ops.append([kron(i4,id(n1),id(n2),p_s,id(n_sensor_2)), sensor_linewidths[0]])
        if n_sensor_2 == 2:
            lindblad_ops.append([kron(i4,id(n1),id(n2),id(2),p_s), sensor_linewidths[1]])
        # cavity-qd coupling
        # cavity energy/detuning
        system_op.append(delta_cx*kron(i4,n(n1),id(n2),id(2),id(n_sensor_2)))
        system_op.append(delta_cx*kron(i4,id(n1),n(n2),id(2),id(n_sensor_2)))
        # X-cavity
        system_op.append(cav_coupl *  (kron(p_gx.T,b_op(n1),id(n2),id(2),id(n_sensor_2)) + kron(p_gx,bdag_op(n1),id(n2),id(2),id(n_sensor_2))))  # |1><0| otimes b + h.c.
        system_op.append(cav_coupl *  (kron(p_xb.T,b_op(n1),id(n2),id(2),id(n_sensor_2)) + kron(p_xb,bdag_op(n1),id(n2),id(2),id(n_sensor_2))))  # |3><1| otimes b + h.c.
        # Y-cavity
        system_op.append(cav_coupl *  (kron(p_gy.T,id(n1),b_op(n2),id(2),id(n_sensor_2)) + kron(p_gy,id(n1),bdag_op(n2),id(2),id(n_sensor_2))))  # |2><0| otimes b + h.c.
        system_op.append(cav_coupl *  (kron(p_yb.T,id(n1),b_op(n2),id(2),id(n_sensor_2)) + kron(p_yb,id(n1),bdag_op(n2),id(2),id(n_sensor_2))))  # |3><2| otimes b + h.c.
        threshold = "8"  # threshold for PT generation
        boson_e_max = 7  # maximum boson energy in meV
        boson_op = kron(x,id(n1),id(n2),id(2),id(n_sensor_2)) + kron(y,id(n1),id(n2),id(2),id(n_sensor_2)) + 2*kron(b,id(n1),id(n2),id(2),id(n_sensor_2))  # operator that couples to phonons
        modes = {"x": kron(p_gx.T+p_xb.T, id(n1), id(n2),id(2),id(n_sensor_2)), "y": kron(p_gy.T+p_yb.T, id(n1), id(n2),id(2),id(n_sensor_2))}  # coupling to x and y polarized light
        rf_op = kron(x + y + 2*b, id(n1), id(n2),id(2),id(n_sensor_2))  # rotating frame operator, if an RF is used (primarily for calculation of dressed states)

        # filter dimensions
        rho0 = remove_dim_numbers(rho0, states_to_remove)
        for i in range(len(lindblad_ops)):
            lindblad_ops[i][0] = remove_dim_numbers(lindblad_ops[i][0], states_to_remove)
        for i in range(len(system_op)):
            system_op[i] = remove_dim_numbers(system_op[i], states_to_remove)
        boson_op = remove_dim_numbers(boson_op, states_to_remove)
        for mode in modes:
            modes[mode] = remove_dim_numbers(modes[mode], states_to_remove)
        rf_op = remove_dim_numbers(rf_op, states_to_remove)
        dim_prod = [rho0.shape[0]]  # subsystem dimensions

        super().__init__(dt=dt, phonons=phonons, ae=ae, temperature=temperature, verbose=verbose, pt_file=pt_file, system_prefix=system_prefix,
                        threshold=threshold, boson_e_max=boson_e_max, rho0=rho0, system_op=system_op, boson_op=boson_op, lindblad_ops=lindblad_ops,
                        lindblad=lindblad, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, pt_dir=pt_dir, modes=modes, rf_op=rf_op, dim_prod=dim_prod,
                        propagate_Taylor=propagate_Taylor)
