import numpy as np
import os
import subprocess
from pyaceqd.tools import export_csv

hbar = 0.6582173  # meV*ps

def sanity_checks(system_op,phonons,boson_op,initial,interaction_ops,verbose):
    if system_op is None and verbose:
        print("System operator not supplied, assuming TLS")
    if phonons and boson_op is None:
        print("using phonons, but boson operator not specified")
        exit(1)
    if initial is None and verbose:
        print("No initial state specified")
    # allow not using an interaction hamiltonian
    if interaction_ops is None and verbose:
        print("No interaction hamiltonian ")

def check_multitime(multitime_op,verbose):
    # multitime_op must be a dictionary like {"operator": "|1><0|_4","time": 10,"applyFrom": "", "applyBefore":"false"}
    if verbose:
        print("multitime operator: {}".format(multitime_op))
    if multitime_op is not None:
        if "operator" not in multitime_op or "time" not in multitime_op:
            print("supply 'operator' and 'time' for multitime")
            exit(0)
        if "applyFrom" not in multitime_op:
            # "" corresponds to: apply operator from left and the h.c. from the right
            multitime_op["applyFrom"] = ""
        if "applyBefore" not in multitime_op:
            # apply after or before "time". If apply after: effect visible only at time+dt
            multitime_op["applyBefore"] = "false"
        # for using the correct option in ACE
        #if multitime_op["applyFrom"] is "_left" or multitime_op["applyFrom"] is "_right":
        #    multitime_op["applyFrom"] = "_"+multitime_op["applyFrom"]
        # catch illegal options
        if multitime_op["applyFrom"] is not "_left" and multitime_op["applyFrom"] is not "_right" and multitime_op["applyFrom"] is not "":
            print(multitime_op)
            print('give "_left" or "_right" or "" for multitime')
            exit(0)
        if "applyBefore" not in multitime_op:
            multitime_op["applyBefore"] = "false"
    #return multitime_op

def generate_pulsefiles(t, pulses, temp_dir, system_prefix, suffix, abs_only=False):
    pulse_file_x = temp_dir + "{}_pulse_x_{}.dat".format(system_prefix, suffix)
    pulse_file_y = temp_dir + "{}_pulse_y_{}.dat".format(system_prefix, suffix)
    pulse_x = np.zeros_like(t, dtype=complex)
    pulse_y = np.zeros_like(t, dtype=complex)
    for _p in pulses:
        if abs_only:
            pulse_x = pulse_x + _p.polar_x * np.abs(_p.get_total(t))
            pulse_y = pulse_y + _p.polar_y * np.abs(_p.get_total(t))
        else:
            pulse_x = pulse_x + _p.polar_x * _p.get_total(t)
            pulse_y = pulse_y + _p.polar_y * _p.get_total(t)
    # this exports to a format that is readable by ACE.
    # note the precision
    export_csv(pulse_file_x, t, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
    export_csv(pulse_file_y, t, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')
    return pulse_file_x, pulse_file_y

def generate_rf_file(t, pulses, temp_dir, system_prefix, suffix, firstonly=False):
    """
    prepares file for rotating frame
    also re-generates pulse files for rotating frame
    """
    rf_file = temp_dir + "{}_rf_{}.dat".format(system_prefix, suffix)
    if len(pulses) > 1:
        print("Warning: more than one pulse supplied, only the first one is used for rf")
        print("Note that also, chirping more than the first pulse is not supported")
    rf = pulses[0].get_frequency(t)
    rf = np.array(rf)
    export_csv(rf_file, t, rf.real, rf.imag, precision=8, delimit=' ')
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
        # only first pulse is considered in the dynamics
        # This is useful, if only the first pulse shall be used
        # for the composition of the dressed states
        generate_pulsefiles(t, [new_pulses[0]], temp_dir, system_prefix, suffix, abs_only=False)
    else:
        generate_pulsefiles(t, new_pulses, temp_dir, system_prefix, suffix, abs_only=False)
    return rf_file

def read_result(data,n):
    t = data[:,0]
    result = np.empty([1+n,len(t)], dtype=complex)
    result[0] = t
    for i in range(n):
        result[i+1] = data[:,2*i+1] + 1j*data[:,2*i+2]
    return result

def read_result_1d(data):
    t = data[:,0]
    n = int((data.shape[1]-1)/2)
    result = np.empty([1+n,len(t)], dtype=complex)
    result[0] = t
    for i in range(n):
        result[i+1] = data[:,2*i+1] + 1j*data[:,2*i+2]
    return result

def system_ace(t_start, t_end, *pulses, dt=0.1, phonons=False, generate_pt=False, t_mem=10, ae=3.0, temperature=1, verbose=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
                  multitime_op=None, nintermediate=10, pulse_file_x=None, pulse_file_y=None, system_prefix="", threshold="7",
                  system_op=None, boson_op=None, initial=None, lindblad_ops=None, interaction_ops=None, output_ops=[], prepare_only=False):
    
    duration = np.abs(t_end)+np.abs(t_start)  # time interval of simulation
    tmp_file = temp_dir + "{}_{}.param".format(system_prefix, suffix)  # parameter file
    out_file = temp_dir + "{}_{}.out".format(system_prefix, suffix)  # file ACE writes to
    # sanity checks
    sanity_checks(system_op=system_op,phonons=phonons,boson_op=boson_op,initial=initial,interaction_ops=interaction_ops,verbose=verbose)
    # check for multi-time operations
    check_multitime(multitime_op=multitime_op,verbose=verbose)
    # for phonons: check if process tensor is present
    if pt_file is None:
        pt_file = "{}_{}ps_{}nm_{}k_th{}_tmem{}_dt{}.pt".format(system_prefix,duration,ae,temperature,threshold,t_mem,dt)
    if phonons:
        # try to detect pt_file, else calculate it
        if not os.path.exists(pt_file):
            print("{} not found. Calculating...".format(pt_file))
            generate_pt = True  # if pt_file is not present, set to verbose and calculate it
            verbose = True
    # pulse file generation
    t = np.arange(1.1*t_start,1.1*t_end,step=dt/(10*nintermediate))
    # if a specific pulse file is supplied, do not delete it after the calculation.
    # this allows re-using the pulse file, for example for multi-time correlation functions
    # where the pulse is not changed for many calculations
    _remove_pulse_file = False
    if pulse_file_x is None:
        _remove_pulse_file = True
        pulse_file_x, pulse_file_y = generate_pulsefiles(t=t, pulses=pulses, temp_dir=temp_dir, system_prefix=system_prefix, suffix=suffix)
    try:
        # write the simulation param file
        with open(tmp_file,'w') as f:
            # time axis of the simulation
            f.write("ta    {}\n".format(t_start))
            f.write("te    {}\n".format(t_end))
            f.write("dt    {}\n".format(dt))
            f.write("Nintermediate    {}\n".format(nintermediate))
            f.write("use_symmetric_Trotter true\n")
            # process tensor generation for phonons
            if generate_pt:
                f.write("t_mem    {}\n".format(t_mem))
                f.write("threshold 1e-{}\n".format(threshold))
                f.write("use_Gaussian true\n")
                f.write("Boson_SysOp    {{ {} }}\n".format(boson_op))
                f.write("Boson_J_type         QDPhonon\n")
                f.write("Boson_J_a_e    {}\n".format(ae))
                f.write("Boson_temperature    {}\n".format(temperature))
                f.write("Boson_subtract_polaron_shift       true\n")
            # read existing process tensor
            if phonons and not generate_pt:
                # process tensor path has to be given or in current dir!
                f.write("read_PT    {}\n".format(pt_file))
                f.write("Boson_subtract_polaron_shift       true\n")
            # initial state
            if initial is not None:
                f.write("initial    {{ {} }}\n".format(initial))
            # hamiltonian of the system
            if system_op is not None:
                for _op in system_op:
                    f.write("add_Hamiltonian {{ {} }}\n".format(_op))
            # lindblad operators
            if lindblad_ops is not None:
                for _op in lindblad_ops:
                    # assume lindblad_ops contains tuples of (operator, rate), ex:("|0><1|_2",1/100)
                    f.write("add_Lindblad {:.5f}  {{ {} }}\n".format(_op[1],_op[0]))  
            # pulse interaction
            if interaction_ops is not None:
                for _op in interaction_ops:
                    # distinguish different polarizations
                    # standard is x
                    p_file = pulse_file_x
                    # op has to be tuple of ("operator","polarization")
                    if _op[1]=="y":
                        p_file = pulse_file_y
                        if pulse_file_y is None:
                            print("Pulse file y not given")
                            exit(1)
                    f.write("add_Pulse file {}  {{ -0.5*pi*hbar*({}) }}\n".format(p_file,_op[0]))
            # multitime operators, left, right or sandwitched
            if multitime_op is not None:
                # apply_Operator 20 {|0><1|_2} would apply the operator |0><1|_2 at t=20 from the left and the h.c. on the right on the density matrix
                # note the Operator is applied at time t, i.e., in this example at t=20, so its effect is only visible at t=20+dt
                f.write("apply_Operator{applyFrom} {time} {{ {operator} }} {applyBefore}\n".format(**multitime_op))
            # output 
            for _op in output_ops:
                f.write("add_Output {{ {} }}\n".format(_op))
            if generate_pt:
                f.write("write_PT {}\n".format(pt_file))
            f.write("outfile {}\n".format(out_file))
        # param file is now written, start ACE
        if prepare_only:
            _remove_pulse_file = False
            print("prepared file {}, exiting.".format(tmp_file))
            return [0 for i in range(len(output_ops))]
        if not verbose:
            subprocess.check_output(["ACE",tmp_file])
        else:
            subprocess.check_call(["ACE",tmp_file])

        data = np.genfromtxt(out_file)
        result = read_result(data, len(output_ops))
    finally:
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        if not prepare_only:
            os.remove(tmp_file)
        if _remove_pulse_file:
            os.remove(pulse_file_x)
            if pulse_file_y is not None:
                os.remove(pulse_file_y)
    return result

def system_ace_stream(t_start, t_end, *pulses, dt=0.01, phonons=False, t_mem=20.48, ae=3.0, temperature=1, verbose=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
                  multitime_op=None, pulse_file_x=None, pulse_file_y=None, system_prefix="", threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7, \
                  system_op=None, boson_op=None, initial=None, lindblad_ops=None, interaction_ops=None, output_ops=[], prepare_only=False, LO_params=None, dressedstates=False, rf_op=None, rf_file=None, firstonly=False):
    """
    ACE_stream: separate calculation for the process tensor, which can be used to simulate way longer time scales.
    """
    t_start = t_start#-dt
    tmp_file = temp_dir + "{}_{}.param".format(system_prefix, suffix)  # parameter file
    out_file = temp_dir + "{}_{}.out".format(system_prefix, suffix)  # file ACE writes to
    # sanity checks
    sanity_checks(system_op=system_op,phonons=phonons,boson_op=boson_op,initial=initial,interaction_ops=interaction_ops, verbose=verbose)
    # check for multi-time operations
    if multitime_op is not None:
        if isinstance(multitime_op, dict):
            multitime_op = [multitime_op]
        for _mto in multitime_op:
            check_multitime(multitime_op=_mto,verbose=verbose)
    if pt_file is None:
        pt_file = "{}_{}nm_{}k_th{}_tmem{}_dt{}.ptr".format(system_prefix,ae,temperature,threshold,t_mem,dt)
    if phonons:
        if verbose and os.path.exists(pt_file+"_initial"):
            print("using pt_file " + pt_file)
        # try to detect pt_file, else calculate it
        if not os.path.exists(pt_file+"_initial"):
            print("{} not found. Calculating...".format(pt_file))
            verbose = True
            generate_file = temp_dir + "{}_generate_{}.param".format(system_prefix, suffix)  # parameter file
            # calculate the PT file 
            with open(generate_file,'w') as f:
                f.write("dt {}\n".format(dt))
                f.write("te {}\n".format(2*t_mem))
                f.write("t_mem {}\n".format(t_mem))
                f.write("buffer_blocksize {}\n".format(buffer_blocksize))
                f.write("threshold 1e-{}\n".format(threshold))
                f.write("select_threshold_ratio {}\n".format(threshold_ratio))
                f.write("dict_zero 1e-{}\n".format(dict_zero))
                f.write("Gaussian_precalc_FFT  true\n")
                f.write("use_Gaussian_repeat true\n")
                f.write("Boson_subtract_polaron_shift true\n")
                f.write("Boson_E_max {}\n".format(boson_e_max))
                f.write("Boson_SysOp    {{ {} }}\n".format(boson_op))
                f.write("Boson_J_type         QDPhonon\n")
                f.write("Boson_J_a_e    {}\n".format(ae))
                f.write("temperature    {}\n".format(temperature))
                f.write("dont_propagate        true\n")
                f.write("write_PT {}\n".format(pt_file))
            if not prepare_only:
                subprocess.check_call(["ACE_stream",generate_file])
                os.remove(generate_file)
            if prepare_only:
                print("wrote {}".format(generate_file))
            try:
                os.remove("ACE.out")
            except FileNotFoundError:
                pass
    # pulse file generation
    t = np.arange(t_start,t_end,step=dt/10)
    # if a specific pulse file is supplied, do not delete it after the calculation.
    # this allows re-using the pulse file, for example for multi-time correlation functions
    # where the pulse is not changed for many calculations
    _remove_pulse_file = False
    _remove_rf_file = False
    if pulse_file_x is None:
        _remove_pulse_file = True
        pulse_file_x, pulse_file_y = generate_pulsefiles(t=t, pulses=pulses, temp_dir=temp_dir, system_prefix=system_prefix, suffix=suffix)
    try:
        # write the simulation param file
        with open(tmp_file,'w') as f:
            # time axis of the simulation
            f.write("dt    {}\n".format(dt))
            f.write("ta    {}\n".format(t_start))
            f.write("te    {}\n".format(t_end))
            f.write("dict_zero 1e-{}\n".format(dict_zero))
            f.write("set_precision {}\n".format(precision))
            f.write("use_symmetric_Trotter true\n")
            if phonons:
                f.write("read_PT    {}\n".format(pt_file))
            # initial state
            if initial is not None:
                f.write("initial    {{ {} }}\n".format(initial))
            # hamiltonian of the system
            if system_op is not None:
                for _op in system_op:
                    f.write("add_Hamiltonian {{ {} }}\n".format(_op))
            # rotating frame: changes the energies of the hamiltonian, using the operator in rf_op
            # this is done using the add_Pulse function of ACE, as this can time-dependently change the system hamiltonian
            # note that it automatically adds the hermitian conjugate of rf_op as well, so a factor of 1/2 is needed
            if rf_op is not None:
                if rf_file is None:
                    # Caution: This also re-generates the pulse file, removing the temporal
                    # oscillation of (at least) the first pulse.
                    # If you give a custom pulse file and want to use a rotating frame,
                    # make sure to also supply the rf_file
                    _remove_rf_file = True
                    rf_file = generate_rf_file(t=t, pulses=pulses, temp_dir=temp_dir, system_prefix=system_prefix, suffix=suffix, firstonly=firstonly)
                f.write("add_Pulse file {} {{ -0.5*hbar*({}) }}\n".format(rf_file,rf_op))
            # lindblad operators
            if lindblad_ops is not None:
                for _op in lindblad_ops:
                    # assume lindblad_ops contains tuples of (operator, rate), ex:("|0><1|_2",1/100)
                    f.write("add_Lindblad {:.5f}  {{ {} }}\n".format(_op[1],_op[0]))  
            # single modes
            if LO_params is not None:
                for _LO_param in LO_params:
                    _energy = _LO_param[0]
                    _coupling = _LO_param[1]
                    f.write("add_single_mode {{ {}*(Id_2 otimes n_3) + {}*(|1><1|_2 otimes bdagger_3 + |1><1|_2 otimes b_3)}} {{|0><0|_3}}\n".format(_energy,_coupling))
            # pulse interaction
            if interaction_ops is not None:
                for _op in interaction_ops:
                    # distinguish different polarizations
                    # standard is x
                    p_file = pulse_file_x
                    # op has to be tuple of ("operator","polarization")
                    if _op[1]=="y":
                        p_file = pulse_file_y
                        if pulse_file_y is None:
                            print("Pulse file y not given")
                            exit(1)
                    f.write("add_Pulse file {}  {{ -0.5*pi*hbar*({}) }}\n".format(p_file,_op[0]))
            # multitime operators, left, right or sandwitched
            if multitime_op is not None:
                for _mto in multitime_op:
                    # apply_Operator 20 {|0><1|_2} would apply the operator |0><1|_2 at t=20 from the left and the h.c. on the right on the density matrix
                    # note the Operator is applied at time t, i.e., in this example at t=20, so its effect is only visible at t=20+dt
                    # if applyBefore ist true, the effect is visible at t=20
                    f.write("apply_Operator{applyFrom} {time} {{ {operator} }} {applyBefore}\n".format(**_mto))
            # output 
            for _op in output_ops:
                f.write("add_Output {{ {} }}\n".format(_op))    
            f.write("outfile {}\n".format(out_file))
         # param file is now written, start ACE
        if prepare_only:
            _remove_pulse_file = False
            _remove_rf_file = False
            print("prepared file {}, exiting.".format(tmp_file))
            return [np.array([0,0]) for i in range(1+len(output_ops))]
        if dressedstates:
            if not verbose:
                subprocess.check_output(["timedep_eigenstates",tmp_file])
            else:
                subprocess.check_call(["timedep_eigenstates",tmp_file])
            dressed_data = np.genfromtxt(out_file+'.ds')
            dressed_result = read_result_1d(dressed_data)
            return dressed_result
        if not verbose:
            subprocess.check_output(["ACE_stream",tmp_file])
        else:
            subprocess.check_call(["ACE_stream",tmp_file])
        data = np.genfromtxt(out_file, usecols=[i for i in range(1+2*len(output_ops))]) # skip_header=1)
        result = read_result(data, len(output_ops))
    finally:
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        if not prepare_only:
            os.remove(tmp_file)
            pass
        if _remove_pulse_file:
            os.remove(pulse_file_x)
            if pulse_file_y is not None:
                os.remove(pulse_file_y)
        if _remove_rf_file:
            os.remove(rf_file)
    return result
