import numpy as np
import os
import subprocess
from pyaceqd.tools import export_csv
import pyaceqd.constants as constants
import time
import itertools
import sys
import multiprocessing as mp
# from contextlib import redirect_stdout
# import io

hbar = constants.hbar  # meV*ps
temp_dir = constants.temp_dir
sys.path.append(constants.pybind_path)  # path to pybinds for ACE
from ACEutils import Parameters, FreePropagator, ProcessTensors, InitialState, OutputPrinter, TimeGrid, Simulation, read_outfile, DynamicalMap

def generate_rf(t, pulses, firstonly=False):
    """
    prepares file for rotating frame
    also re-generates pulse files for rotating frame
    """
    rf = pulses[0].get_frequency(t)
    rf = np.array(rf)
    # copy pulses
    new_pulses = []
    for p in pulses:
        new_pulses.append(p.copy())
    # substract e_start from all pulses
    e_start0,_ = new_pulses[0].get_energy()
    for i in range(len(new_pulses)):
        e_start,_ = new_pulses[i].get_energy()
        # substract e_start0 from all pulses, also set chirps to zero.
        new_pulses[i].set_energy(e_start-e_start0,0)
    if firstonly:
        new_pulses = [new_pulses[0]]
    return t, rf, new_pulses

def _gen_pt_worker(_generate_file):
    # this gets called in a separate process to generate the PT file
    # so we need to import ACEutils here as well
    from ACEutils import Parameters, FreePropagator, ProcessTensors, TimeGrid
    param_w = Parameters(_generate_file)
    fprop_w = FreePropagator(param_w)
    tgrid_w = TimeGrid(param_w)
    PT_w = ProcessTensors(param_w)
    _ = (param_w, fprop_w, tgrid_w, PT_w)

def _get_pt_name(system_prefix, ae, temperature, threshold, dt, J_file):
    if J_file is not None:
        pt_file = "{}_{}_{}k_th{}_dt{}.ptr".format(system_prefix,os.path.splitext(J_file)[0],temperature,threshold,dt)
    else:
        pt_file = "{}_{}nm_{}k_th{}_dt{}.pt".format(system_prefix,ae,temperature,threshold,dt)
    return pt_file

def _calc_PT_file(dt, threshold, ae, factor_ah, temperature, boson_op, filename, boson_e_max=7, verbose=False, J_file=None, J_to_file=False):
    params = []
    params += ["dt {}".format(dt)]
    params += ["te {}".format(20)]
    params += ["threshold 1e-{}".format(threshold)]
    params += ["use_Gaussian_infinite true"]
    params += ["infinite_normalize_iter 200"]
    params += ["Boson_subtract_polaron_shift true"]
    params += ["Boson_E_min {}".format(0)]
    params += ["Boson_E_max {}".format(boson_e_max)]
    if J_file is not None:
        params += ["Boson_J_from_file {}".format(J_file)]
    else:
        params += ["Boson_SysOp {{ {} }}".format(boson_op)]
        params += ["Boson_J_type QDPhonon"]
        params += ["Boson_J_a_e {}".format(ae)]
        if factor_ah is not None:
            params += ["Boson_J_a_h {}".format(ae/factor_ah)]
    if J_to_file:
        params += ["Boson_J_print {} 0 15 2000".format(J_to_file)]
    params += ["temperature {}".format(temperature)]
    params += ["dont_propagate true"]
    params += ["write_PT {}".format(filename)]
    if verbose:
        print("Calculating PT file with parameters:")
        for line in params:
            print(line)

    ctx = mp.get_context('fork')
    proc = ctx.Process(target=_gen_pt_worker, args=(params,))
    proc.start()
    while not os.path.exists(filename+"_initial"):
        time.sleep(0.2)
        if not proc.is_alive() and not os.path.exists(filename+"_initial"):
            break
    # Ensure worker exits cleanly
    proc.join(timeout=3)
    if proc.is_alive():
        proc.terminate()
        proc.join()
    if verbose:
        print("PT file generated: {}".format(filename))
    full_names = [filename+"_initial",filename+"_initial_0", filename+"_repeated", filename+"_repeated_0"]
    # make read-only
    for name in full_names:
        subprocess.run(["chmod", "444", name])
    time.sleep(1)

# phonon_default_params = {"ae": 5.0, 
#                          "temperature": 4.0, 
#                          "threshold": "10", 
#                          "boson_e_max": 7, 
#                          "factor_ah": 1.0, 
#                          "J_file": None, 
#                          "J_to_file": None
#                          }

def system_ace(t_start, t_end, *pulses, dt=0.01, phonons=False, ae=3.0, temperature=1, verbose=False, pt_file=None, \
                  multitime_op=None, system_prefix="", threshold="10", boson_e_max=7, system_op=None, boson_op=None, initial=None, lindblad_ops=None, output_ops=[], prepare_only=False, rf_op=None, rf_array=None, firstonly=False, \
                  J_to_file=None, J_file=None, factor_ah=None, print_H=False, calc_dynmap=False, rho0=None):
    """
    ACE: separate calculation for the process tensor, which can be used to simulate way longer time scales.
    """
    # check for multi-time operations
    if multitime_op is not None:
        # make sure it's a list, if only one MTO is given as dict.
        if isinstance(multitime_op, dict):
            multitime_op = [multitime_op]

    if phonons:
        # determine pt_file name
        if pt_file is None:
            pt_file = _get_pt_name(system_prefix, ae, temperature, threshold, dt, J_file)
        if verbose and os.path.exists(pt_file+"_initial") and J_to_file is None:
            print("using pt_file " + pt_file)
        # try to detect pt_file, else calculate it
        if not os.path.exists(pt_file+"_initial") or J_to_file is not None:
            print("{} not found. Calculating...".format(pt_file))
            _calc_PT_file(dt, threshold, ae, factor_ah, temperature, boson_op, pt_file, boson_e_max=boson_e_max, verbose=verbose, J_file=J_file, J_to_file=J_to_file)

    plist = []
    plist += ["dt {}".format(dt)]
    plist += ["ta {}".format(t_start)]
    plist += ["te {}".format(t_end)]
    plist += ["use_symmetric_Trotter true"]
    if phonons:
        plist += ["add_PT {}".format(pt_file)]
    # initial state
    if initial is not None:
        plist += ["initial {{ {} }}".format(initial)]
    # hamiltonian of the system
    if system_op is not None:
        for _op in system_op:
            plist += ["add_Hamiltonian {{ {} }}".format(_op)]
    # lindblad operators
    if lindblad_ops is not None:
        for _op in lindblad_ops:
            # assume lindblad_ops contains tuples of (operator, rate), ex:("|0><1|_2",1/100)
            plist += ["add_Lindblad {} {{ {} }}".format(_op[1],_op[0])]
    # multitime operators, left, right or sandwitched
    if multitime_op is not None:
        for _mto in multitime_op:
            # apply_Operator 20 {|0><1|_2} would apply the operator |0><1|_2 at t=20 from the left and the h.c. on the right on the density matrix
            # note the Operator is applied at time t, i.e., in this example at t=20, so its effect is only visible at t=20+dt
            # if applyBefore ist true, the effect is visible at t=20
            plist += ["apply_Operator{applyFrom} {time} {{ {operator} }} {applyBefore}\n".format(**_mto)]
    # output 
    for _op in output_ops:
        plist += ["add_Output {{ {} }}".format(_op)]

    if prepare_only:
        for line in plist:
            print(line)
        return [np.array([0,0]) for i in range(1+len(output_ops))]
    
    param = Parameters(plist)
    if rho0 is not None:
        initial_state = InitialState(rho0)
    else:
        initial_state = InitialState(param)

    fprop = FreePropagator(param)
    tgrid = TimeGrid(param)
    t = np.round(np.array(tgrid.get_all()), decimals=10)
    PT = ProcessTensors(param)
    
    sim = Simulation(param)
    # rotating frame: changes the energies of the hamiltonian, using the operator in rf_op
    # this is done using the add_Pulse function of ACE, as this can time-dependently change the system hamiltonian
    # note that it automatically adds the hermitian conjugate of rf_op as well, so a factor of 1/2 is needed
    # the rf_operator should usually be diagonal in the system hamiltonian basis, so it should not be complex valued.
    if rf_op is not None:
        if rf_array is None:
            # Caution: This also re-generates the pulses, removing the temporal
            # oscillation of (at least) the first pulse.
            _, rf_array, new_pulses = generate_rf(t=t, pulses=pulses, firstonly=firstonly)
            pulses = new_pulses
        fprop.add_Pulse((t,-0.5*hbar*rf_array), rf_op)
            
    # after potential RF, add pulses
    if firstonly:
        if verbose:
            print("only using first pulse for dynamics")
        pulses = [pulses[0]]
    
    for pulse in pulses:
        if verbose:
            print("Adding pulse: {}".format(pulse))
        # each pulse needs to have the correct interaction_op assigned
        fprop.add_Pulse((t, -0.5*hbar*np.pi*pulse.get_total(t)), pulse.interaction_op)                    

    if print_H:
        dim = fprop.get_dim()
        H_total = np.empty((len(t), dim, dim), dtype=complex)
        for i, ti in enumerate(t):
            H_total[i] = fprop.get_Htot(ti)
        return t, H_total

    if calc_dynmap:
        dynmap = DynamicalMap(fprop, PT, sim, tgrid)
        _dm = np.array(dynmap.E)
        return [t], _dm
    
    outp = OutputPrinter(param)
    outp.do_extract = True
    sim.run(fprop, PT, initial_state, tgrid, outp)
    # if calc_dynmap:
    #     # if get_M_t is not None:
    #     #     fprop.update(get_M_t,dt)
    #     #     return fprop.M
    #     dynmap = DynamicalMap(fprop, PT, sim, tgrid)
    #     _dm = np.array(dynmap.E)
            
    result = outp.extract()
    # t = result[0]
    # reshaped = result[1].T
    reshaped = np.vstack([result[0][np.newaxis, :], result[1].T])
    # if calc_dynmap:
    #     return reshaped, _dm
    # return t, reshaped
    return reshaped


class GeneralSystemACE:
    def __init__(self, dt=0.1, phonons=False, ae=5.0, temperature=4, verbose=False, pt_file=None, system_prefix="", threshold="10", boson_e_max=7, initial=None,
                 system_op=["0*|1><1|_2"], boson_op=None, lindblad_ops=None,J_to_file=None, J_file=None, factor_ah=None, pt_dir="", modes=None, rf_op=None):
        """
        ACE: separate calculation for the process tensor, which can be used to simulate long time scales with interaction to the environment.
        """
        self.verbose = verbose
        self.plist_base = []  # parameters that will be used in each simulation
        if system_op is None:
            raise ValueError("system_op must be provided")
        for _op in system_op:
            self.plist_base += ["add_Hamiltonian {{ {} }}".format(_op)]
        self.dt = dt
        self.plist_base += ["dt {}".format(dt)]

        self.phonons = phonons
        if self.phonons:
            # parameters for process tensor calculation
            self.ae = ae
            self.temperature = temperature
            if boson_op is None:
                raise ValueError("boson_op must be provided when phonons=True")
            self.boson_op = boson_op
            self.boson_e_max = boson_e_max
            self.threshold = threshold
            self.system_prefix = system_prefix  # for pt name
            self.J_to_file = J_to_file
            self.J_file = J_file
            self.factor_ah = factor_ah
            self.pt_file = pt_file
            if self.pt_file is None:
                self.pt_file = _get_pt_name(pt_dir+system_prefix, ae, temperature, threshold, dt, J_file)
            if verbose and os.path.exists(self.pt_file+"_initial") and J_to_file is None:
                print("using pt_file " + self.pt_file)
            # try to detect pt_file, else calculate it
            if not os.path.exists(self.pt_file+"_initial") or J_to_file is not None:
                print("{} not found. Calculating...".format(self.pt_file))
                _calc_PT_file(dt, threshold, ae, factor_ah, temperature, boson_op, self.pt_file, boson_e_max=boson_e_max, verbose=verbose, J_file=J_file, J_to_file=J_to_file)
            # add to plist
            self.plist_base += ["add_PT {}".format(self.pt_file)]
        
        if modes is None:
            print("No modes specified, assuming no interaction")
        self.modes = modes  # list of operators that can induce transitions and are mapped to light modes, eg |1><0|_2 could be mapped to x-polarized light in a TLS
        # determine dimension
        param_base = Parameters(self.plist_base)
        base_fprop = FreePropagator(param_base)
        self.dim = base_fprop.get_dim()
        if self.verbose:
            print("System dimension: {}".format(self.dim))
        # shared Process Tensor object
        self.PT = ProcessTensors(param_base)
        self.lindblad_ops = lindblad_ops
        self.rf_op = rf_op
        self.inital = initial

    def run(self, t_start, t_end, *pulses, lindblad=True, multitime_op=None, initial=None, output_ops=[], prepare_only=False, rho0=None, calc_dynmap=False,
            print_H=False, rf_op=None, rf_array=None):
        """
        runs a simulation with the given parameters and the base parameters defined in the class init.
        rho0: initial density matrix as numpy array, overrides 'initial' parameter.
        """
        run_plist = self.plist_base.copy()
        run_plist += ["ta {}".format(t_start)]
        run_plist += ["te {}".format(t_end)]
        run_plist += ["use_symmetric_Trotter true"]

        if self.lindblad_ops is not None and lindblad:
            for _op in self.lindblad_ops:
                # assume lindblad_ops contains tuples of (operator, rate), ex:("|0><1|_2",1/100)
                run_plist += ["add_Lindblad {} {{ {} }}".format(_op[1],_op[0])]

        # initial state
        if initial is None:
            initial = self.inital
        if initial is not None:
            run_plist += ["initial {{ {} }}".format(initial)]

        # multitime operators, left or right
        if multitime_op is not None:
            # make sure it's a list, if only one MTO is given as dict.
            if isinstance(multitime_op, dict):
                multitime_op = [multitime_op]
            for _mto in multitime_op:
                run_plist += ["apply_Operator_{applyFrom} {time} {{ {operator} }} {applyBefore}\n".format(**_mto)]

        # output
        for _op in output_ops:
            run_plist += ["add_Output {{ {} }}".format(_op)]

        # for testing: just print the plist
        if prepare_only:
            for line in run_plist:
                print(line)
            return [np.array([0,0]) for i in range(1+len(output_ops))]
        
        param = Parameters(run_plist)
        # initial state
        if rho0 is not None:
            initial_state = InitialState(rho0)
        else:
            initial_state = InitialState(param)
        
        fprop = FreePropagator(param)
        tgrid = TimeGrid(param)
        t = np.round(np.array(tgrid.get_all()), decimals=10)

        # rotating frame: changes the energies of the hamiltonian, using the operator in rf_op
        # this is done using the add_Pulse function of ACE, as this can time-dependently change the system hamiltonian
        # note that it automatically adds the hermitian conjugate of rf_op as well, so a factor of 1/2 is needed
        # the rf_operator should usually be diagonal in the system hamiltonian basis, so it should not be complex valued.
        if rf_op is not None:
            if rf_array is None:
                # Caution: This also re-generates the pulses, removing the temporal
                # oscillation of (at least) the first pulse.
                _, rf_array, new_pulses = generate_rf(t=t, pulses=pulses)
                pulses = new_pulses
            fprop.add_Pulse((t,-0.5*hbar*rf_array), rf_op)

        # after potential RF, add pulses
        for pulse in pulses:
            if self.verbose:
                print("Adding pulse: {}".format(pulse))
            # each pulse needs to have a polarization assigned
            if pulse.polarization not in self.modes:
                raise ValueError("Pulse polarization {} not in system modes {}".format(pulse.polarization, self.modes.keys()))
            # add pulse
            fprop.add_Pulse((t, -0.5*hbar*np.pi*pulse.get_total(t)), self.modes[pulse.polarization])     

        # option to return Hamiltonian
        if print_H:
            H_total = np.empty((len(t), self.dim, self.dim), dtype=complex)
            for i, ti in enumerate(t):
                H_total[i] = fprop.get_Htot(ti)
            return t, H_total       
        
        sim = Simulation(param)
        # calculate dynamical maps
        if calc_dynmap:
            dynmap = DynamicalMap(fprop, self.PT, sim, tgrid)
            _dm = np.array(dynmap.E)
            return [t], _dm
             
        outp = OutputPrinter(param)
        outp.do_extract = True
        # lets see if we can share the PT object
        sim.run(fprop, self.PT, initial_state, tgrid, outp)
        result = outp.extract()
        #reshaped = np.vstack([result[0][np.newaxis, :].real, result[1].T])
        return result[0].real, result[1].T
    

# a = GeneralSystemACE(dt=0.1, phonons=True, ae=5.0, temperature=4, verbose=True, system_prefix="test_system", system_op=["0*|1><1|_2"], boson_op="|1><1|_2", lindblad_ops=[( "|0><1|_2", 1/100 )])
