import numpy as np
import os
import subprocess
from pyaceqd.tools import export_csv, basis_states
import pyaceqd.constants as constants
import time
import itertools
import sys
import multiprocessing as mp
from tabulate import tabulate
from pyaceqd.helpers.color_tools import select_equally_spaced_colors, hex_to_rgba
from pyaceqd.helpers.order_eigenstates import order_eigenstates
from pyaceqd.helpers.ace_operators import matrix_to_op
hbar = constants.hbar  # meV*ps
temp_dir = constants.temp_dir
# sys.path.append(constants.pybind_path)  # path to pybinds for ACE
from ACE._ACE import Parameters, FreePropagator, ProcessTensors, InitialState, OutputPrinter, TimeGrid, Simulation, DynamicalMap, StringToMatrix

def compose_dm(outputs, dim=2):
    """
    composes a density matrix from the output of ACE, with every output-array being the time dynamics for the corresponding output operator
    """
    # dim is the dimension of the system
    t, outputs = outputs  # unpack
    # print("Outputs shape:", outputs.shape)
    rho = np.zeros((len(outputs[0]),dim,dim),dtype=np.complex128)
    for i in range(len(outputs[0])):
        rho[i] = np.reshape(outputs[:,i], (dim,dim))
    t = np.real(outputs[0])
    return t, rho

def generate_rf(t, pulses, firstonly=False, correct_phase=True):
    """
    prepares file for rotating frame
    also re-generates pulses for rotating frame
    """
    rf = pulses[0].get_frequency(t)
    rf = np.array(rf)
    # copy pulses
    new_pulses = []
    for p in pulses:
        new_pulses.append(p.copy())
    # substract e_start from all pulses
    e_start0,_ = new_pulses[0].get_energy()
    print("e_start0:", e_start0)
    for i in range(len(new_pulses)):
        e_start,_ = new_pulses[i].get_energy()
        # substract e_start0 from all pulses, also set chirps to zero.
        new_pulses[i].set_energy(e_start-e_start0,0)
        if correct_phase:
            # if the pulses have different t0, we have to account for the phase difference
            # that arises from the energy difference during the delay between the pulses.
            # This is given by (e_start0 / hbar) * tau, where tau is the delay.
            new_pulses[i].phase = new_pulses[i].phase - (e_start0/hbar)*(new_pulses[i].get_tcenter() - new_pulses[0].get_tcenter())
        print("new e_start for pulse {}: {}".format(i, e_start-e_start0))
    if firstonly:
        new_pulses = [new_pulses[0]]
    return t, rf, new_pulses

def _gen_pt_worker(_generate_params, system_op=None):
    # this gets called in a separate process to generate the PT file
    # so we need to import from ACE here as well
    from ACE._ACE import Parameters, FreePropagator, ProcessTensors, TimeGrid
    if system_op is not None and isinstance(system_op, np.ndarray) and system_op.ndim == 2:
        system_op = [system_op]
    param_w = Parameters(_generate_params)
    fprop_w = FreePropagator(param_w)
    first = True
    for _op in system_op:
        if first:
            fprop_w.set_Hamiltonian(_op)
            first=False
        else:
            fprop_w.add_Hamiltonian(_op)
        
    tgrid_w = TimeGrid(param_w)
    PT_w = ProcessTensors(param_w)
    _ = (param_w, fprop_w, tgrid_w, PT_w)

def _get_pt_name(system_prefix, ae, temperature, threshold, dt, J_file):
    ae = ae * 1.0  # ensure float
    temperature = temperature * 1.0
    if J_file is not None:
        pt_file = "{}_{}_{}k_th{}_dt{}.ptr".format(system_prefix,os.path.splitext(J_file)[0],temperature,threshold,dt)
    else:
        pt_file = "{}_{}nm_{}k_th{}_dt{}.pt".format(system_prefix,ae,temperature,threshold,dt)
    return pt_file

def _calc_PT_file(dt, threshold, ae, factor_ah, temperature, boson_op, filename, boson_e_max=7, system_op=None, verbose=False, J_file=None, J_to_file=False):
    if system_op is None:
        raise ValueError("system_op must be provided to calculate PT file, as the system Hamiltonian is needed for the calculation.")
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
        boson_operator = StringToMatrix(boson_op) if isinstance(boson_op, str) else boson_op
        params += ["Boson_SysOp {{ {} }}".format(matrix_to_op(boson_operator))]
        params += ["Boson_J_type QDPhonon"]
        params += ["Boson_J_a_e {}".format(ae*1.0)]
        if factor_ah is not None:
            params += ["Boson_J_a_h {}".format(1.0*ae/factor_ah)]
    if J_to_file:
        params += ["Boson_J_print {} 0 15 2000".format(J_to_file)]
    params += ["temperature {}".format(temperature*1.0)]
    params += ["dont_propagate true"]
    params += ["write_PT {}".format(filename)]
    if verbose:
        print("Calculating PT file with parameters:")
        for line in params:
            print(line)
    ctx = mp.get_context('fork')
    proc = ctx.Process(target=_gen_pt_worker, args=(params,system_op))
    proc.start()
    while not os.path.exists(filename+"_initial"):
        time.sleep(0.2)
        if not os.path.exists(filename+"_initial"):
            if not proc.is_alive():
                raise RuntimeError("PT generation process terminated unexpectedly. There may be an error in the declaration of your system, or a different ACE Error.")
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


class GeneralSystemACE:
    def __init__(self, dt=0.1, phonons=False, ae=5.0, temperature=4, verbose=False, pt_file=None, system_prefix="", threshold="10", boson_e_max=7,
                 system_op=None, boson_op=None, lindblad_ops=None, lindblad=True, J_to_file=None, J_file=None, factor_ah=None, pt_dir="", modes=None, rf_op=None, dim_prod=None,
                 colors=None, propagate_Taylor=None, rho0=None, expand_pt=None):
        """
        ACE: separate calculation for the process tensor, which can be used to simulate long time scales with interaction to the environment.
        """
        self.verbose = verbose
        self.plist_base = []  # parameters that will be used in each simulation
        if system_op is None:
            raise ValueError("system_op must be provided")

        if propagate_Taylor is not None:
            if verbose:
                print("Using {}-th order Taylor expansion for propagation".format(propagate_Taylor))
            self.plist_base += ["propagate_Taylor {}".format(propagate_Taylor)]  # massive speedup for systems with hilbert space dims larger than around 8
        
        self.dt = dt
        self.lindblad = lindblad
        if isinstance(system_op,list):
            _system_op = system_op[0]
            for i in range(1, len(system_op)):
                _system_op += system_op[i]
            system_op = _system_op
        self.system_op = system_op
        self.lindblad_ops = lindblad_ops

        self.phonons = phonons
        self.expand_pt = expand_pt
        if self.phonons:
            # parameters for process tensor calculation
            if boson_op is None:
                raise ValueError("boson_op must be provided when phonons=True")
            self.pt_file = pt_file
            if self.pt_file is not None:
                self.pt_file = os.path.join(pt_dir, self.pt_file)
            if self.pt_file is None:
                self.pt_file = _get_pt_name(pt_dir+system_prefix, ae, temperature, threshold, dt, J_file)
            if verbose and os.path.exists(self.pt_file+"_initial") and J_to_file is None:
                print("using pt_file " + self.pt_file)
            # try to detect pt_file, else calculate it
            if not os.path.exists(self.pt_file+"_initial") or J_to_file is not None:
                print("system:", self.system_op)
                print("{} not found. Calculating...".format(self.pt_file))
                _calc_PT_file(dt, threshold, ae, factor_ah, temperature, boson_op,
                            self.pt_file, boson_e_max=boson_e_max, verbose=verbose,
                            system_op=self.system_op, J_file=J_file, J_to_file=J_to_file)

        if modes is None:
            raise UserWarning("No modes specified, assuming no interaction")
        self.modes = modes  # list of operators that can induce transitions and are mapped to light modes, eg |1><0|_2 could be mapped to x-polarized light in a TLS

        self.dim_prod = dim_prod  # provides dimension of system, eg. for TLS coupled to two 3LS it would be [2,3,3], so total dim is 2*3*3 = 18
        if self.dim_prod is None:
            raise UserWarning("No dim_prod provided, which is beneficial if you want dressed state analysis. It is recommended to provide dim_prod as a list of dimensions of subsystems, ex: [2,3,3] for TLS coupled to two 3LS.")
        self.dim = np.shape(system_op[0])[0]  # dimension of the system, inferred from the first system operator. assumes all system operators have the same dimension.
        if self.verbose:
            print("System dimension: {}".format(self.dim))

        self.rf_op = rf_op
        self.colors = colors  # optional colors for plotting
        self.rho0 = rho0  # initial density matrix as numpy array

        # dict to store all args
        self.args = {
            "dt": dt,
            "phonons": phonons,
            "ae": ae,
            "temperature": temperature,
            "verbose": verbose,
            "pt_file": pt_file,
            "system_prefix": system_prefix,
            "threshold": threshold,
            "boson_e_max": boson_e_max,
            "system_op": system_op,
            "boson_op": boson_op,
            "lindblad_ops": lindblad_ops,
            "lindblad": lindblad,
            "J_to_file": J_to_file,
            "J_file": J_file,
            "factor_ah": factor_ah,
            "pt_dir": pt_dir,
            "modes": modes,
            "rf_op": rf_op,
            "dim_prod": dim_prod,
            "colors": colors,
            "propagate_Taylor": propagate_Taylor,
            "rho0": rho0
        }

    def copy(self):
        """
        creates a copy of the class instance, which can be used to run simulations with the same parameters but different pulses or time ranges.
        """
        return GeneralSystemACE(**self.args)

    def run(self, t_start, t_end, *pulses, multitime_op=None, output_ops=[], prepare_only=False, rho0=None, calc_dynmap=False,
            return_H=False, rf=False, rf_array=None, get_M_t=None, benchmark=False):
        """
        runs a simulation with the given parameters and the base parameters defined in the class init.
        rho0: initial density matrix as numpy array, overrides 'initial' parameter.
        """
        run_plist = self.plist_base.copy()
        run_plist += ["use_symmetric_Trotter true"]
        run_plist += ["outfile /dev/null"]  # supress creation of "ACE.out" file

        # multitime operators, left or right
        if multitime_op is not None:
            # make sure it's a list, if only one MTO is given as dict.
            if isinstance(multitime_op, dict):
                multitime_op = [multitime_op]
            for _mto in multitime_op:
                # check if applyFrom is either "left", "right"
                if _mto["applyFrom"] not in ["left", "right"]:
                    raise UserWarning("applyFrom must be either 'left' or 'right', got: {}".format(_mto["applyFrom"]))
                if isinstance(_mto['operator'], str):
                    _mto['operator'] = StringToMatrix(_mto['operator'])
                    if self.verbose:
                        print("Converted multitime operator from string to matrix: {}".format(_mto['operator']))
                    # run_plist += ["apply_Operator_{applyFrom} {time} {{ {operator} }} {applyBefore}\n".format(**_mto)]

        # output
        # check if output_ops are strings, empty or arrays
        outputs_are_strings = False
        if len(output_ops) == 0:
            # if empty, returns full density matrix if OutputPrinter is called with param only
            outputs_are_strings = True
        else:
            if isinstance(output_ops[0], str):
                if self.verbose:
                    print("Output operators are strings")
                outputs_are_strings = True
                for _op in output_ops:
                    run_plist += ["add_Output {{ {} }}".format(_op)]

        # for testing: just print the plist
        if prepare_only:
            for line in run_plist:
                print(line)
            return [np.array([0,0]) for i in range(1+len(output_ops))]
        
        param = Parameters(run_plist)
        # initial state
        if rho0 is None and self.rho0 is not None:
            rho0 = self.rho0
        if rho0 is not None:
            initial_state = InitialState(rho0)
        else:
            raise ValueError("No initial state provided, and no default initial state set in class. Provide an initial state as a numpy array using the rho0 parameter.")
        
        fprop = FreePropagator(param)
        fprop.set_dim(self.dim)
        fprop.set_Hamiltonian(self.system_op)
        if self.lindblad_ops is not None and self.lindblad:
            for _op in self.lindblad_ops:
                # assume lindblad_ops contains tuples of (operator, rate), ex:(ketbra(0,1,2),1/100)
                fprop.add_Lindblad(_op[1], _op[0])
                # fprop.add_Lindblad(1/100, ketbra(0,1,2))
        tgrid = TimeGrid(t_start, t_end, self.dt)
        t = np.round(np.array(tgrid.get_all()), decimals=10)

        # multitime operators that were not given as strings
        if multitime_op is not None:
            for _mto in multitime_op:
                if isinstance(_mto['operator'], np.ndarray):
                    if _mto['operator'].shape != (self.dim, self.dim):
                        raise ValueError("multitime operator has wrong shape: {}, expected: ({},{})".format(_mto['operator'].shape, self.dim, self.dim))
                    if _mto["applyFrom"] == "left":
                        fprop.apply_Operator_left(_mto['time'], _mto['operator'], _mto['applyBefore'])
                    elif _mto["applyFrom"] == "right":
                        fprop.apply_Operator_right(_mto['time'], _mto['operator'], _mto['applyBefore'])

        # rotating frame: changes the energies of the hamiltonian, using the operator in rf_op
        # this is done using the add_Pulse function of ACE, as this can time-dependently change the system hamiltonian
        # note that it automatically adds the hermitian conjugate of rf_op as well, so a factor of 1/2 is needed
        # the rf_operator should usually be diagonal in the system hamiltonian basis, so it should not be complex valued.
        if rf:
            if len(pulses) == 0:
                pass
            else:
                if rf_array is None:
                    # Caution: This also re-generates the pulses, removing the temporal
                    # oscillation of (at least) the first pulse.
                    _, rf_array, new_pulses = generate_rf(t=t, pulses=pulses)
                    pulses = new_pulses
                fprop.add_Pulse((t,-0.5*hbar*rf_array), self.rf_op)

        # after potential RF, add pulses
        for pulse in pulses:
            t_pulse = t
            # check if pulse is a dict
            # assumes the dict contains 'polarization', "time" and 'total' keys, for ex.:
            # pulse = {"polarization": "x", "total": total_pulse_array, "time": t_array}
            # instead of "polarization" you can also use "channel" as a key
            # total_pulse_array must contain the complex pulse amplitude at each time point in t_array
            if isinstance(pulse, dict):
                if self.verbose:
                    print("Adding pulse dict")
                
                polarization = pulse.get("polarization", None)
                if polarization is None:
                    polarization = pulse.get("channel", None)
                t_pulse = pulse.get("time", None)
                total_pulse = pulse.get("total", None) * np.pi  # note that we multiply by pi here, so setting the pulse area to 1 corresponds to a pi-pulse.

                if t_pulse is None:
                    raise ValueError("Pulse dict must contain 'time' key")
                if polarization is None:
                    raise ValueError("Pulse dict must contain 'polarization' or 'channel' key")
                if total_pulse is None:
                    raise ValueError("Pulse dict must contain 'total' key")
                
                if polarization not in self.modes:
                    raise ValueError("Pulse polarization/channel: {} not in system modes: {}".format(polarization, self.modes.keys()))
                interaction_op = self.modes[polarization]
                
                if len(total_pulse) != len(t_pulse):
                    raise ValueError("Pulse 'total' length does not match 'time' length")
            # else, assume it's a Pulse object
            else:
                if self.verbose:
                    print("Adding pulse: {}".format(pulse))
                # each pulse needs to have a polarization assigned
                if pulse.polarization not in self.modes:
                    raise ValueError("Pulse polarization {} not in system modes {}".format(pulse.polarization, self.modes.keys()))
                total_pulse = pulse.get_total(t) * np.pi  # note that we multiply by pi here, so setting the pulse area to 1 corresponds to a pi-pulse.
                interaction_op = self.modes[pulse.polarization]
            # add pulse
            fprop.add_Pulse((t_pulse, -0.5*hbar*total_pulse), interaction_op)  # time, complex pulse amplitude, operator    

        # option to return Hamiltonian
        if return_H:
            H_total = np.empty((len(t), self.dim, self.dim), dtype=complex)
            for i, ti in enumerate(t):
                H_total[i] = fprop.get_Htot(ti)
            return t, H_total       
        
        sim = Simulation()
        PT = ProcessTensors()
        if self.phonons:
            if self.expand_pt is not None:
                PT.add_PT(self.pt_file, 0, self.expand_pt)  # expand_pt = N dims are added after the system dim. 
            else:
                PT.add_PT(self.pt_file)
        # calculate dynamical maps
        if calc_dynmap:
            if get_M_t is not None:  # option to return Propagator at specific time.
                fprop.update(get_M_t,self.dt)
                return fprop.M
            dynmap = DynamicalMap(fprop, PT, sim, tgrid)
            _dm = np.array(dynmap.E)
            # _dm = dynmap.get_E()
            return t, _dm
        
        if not outputs_are_strings:
            # check output operator dimensions
            for _op in output_ops:
                if _op.shape != (self.dim, self.dim):
                    raise ValueError("Output operator has wrong shape: {}, expected: ({},{})".format(_op.shape, self.dim, self.dim))
            outp = OutputPrinter(output_ops)
        else:
            # output operators are strings eg "|0><1|_2"
            outp = OutputPrinter(param)  # empyt output_ops will return full density matrix
        outp.do_extract = True
        if benchmark:
            start = time.time()
        sim.run(fprop, PT, initial_state, tgrid, outp)
        if benchmark:
            end = time.time()
        result = outp.extract()
        #reshaped = np.vstack([result[0][np.newaxis, :].real, result[1].T])
        if benchmark:
            return result[0].real, result[1].T, end - start
        return result[0].real, result[1].T
    
    def dressed_states(self, t_start, t_end, *pulses, rho0=None, rf=True, rf_array=None, firstonly=False, no_pulse=False, colors=None,
                       visible_states=None, print_states_t=None, plot=True, t_lim=None, filename="dressed", e_lim=None, return_eigenvectors=False, fix_order=True):
        """
        calculates dressed states using the density matrix from run()
        rho0: initial density matrix as numpy array, overrides 'initial' parameter.
        rf: use rotating frame of laser pulses
        firstonly: only use first pulse to calculate the dressed states, but use all pulses for the density matrix calculation.
        no_pulse: do not use any pulse to calculate the dressed states, only the system hamiltonian (useful if system is not in its eigenstates, e.g. due to magnetic field)
                  all pulses are still used for the density matrix calculation.
        rf_array: use this rf_array instead of generating it from the pulses.
        """
        t, rho = compose_dm(self.run(t_start, t_end, *pulses, rho0=rho0, rf=rf, rf_array=rf_array), dim=self.dim)
        if firstonly:
            pulses = [pulses[0]]
        if no_pulse:
            pulses = []
        t, H_total = self.run(t_start, t_end, *pulses, rho0=rho0, return_H=True, rf=rf, rf_array=rf_array)
        if self.colors is None:
            self.colors = select_equally_spaced_colors(n=self.dim)

        dim_prod = self.dim_prod
        # if dim_prod is None:
        #     dim_prod = [self.dim]

        if plot:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.ylim(-0.1,1.1)
            labels = basis_states(dim_prod)
            for i in range(self.dim):
                plt.plot(t, rho[:,i,i].real, label=labels[i], color=self.colors[i])
            if t_lim is not None:
                plt.xlim(t_lim[0],t_lim[1])
            plt.xlabel("t (ps)")
            plt.ylabel("occupation")
            plt.legend()
            plt.savefig(filename + "_rho.png")
            plt.clf()
        
        e_vectors = np.zeros((len(t),self.dim,self.dim),dtype=np.complex128)
        e_values = np.zeros((len(t),self.dim))

        for i in range(len(t)):
            # diagonalize current Hamiltonian
            e_values[i], e_vectors[i] = np.linalg.eigh(H_total[i])

        # order eigenstates to prevent jumps between branches
        if fix_order:            
            e_values, e_vectors = order_eigenstates(e_values, e_vectors)

        for i in range(len(t)):
            e_vectors[i] = e_vectors[i].T  # so that e_vectors[i,j] is the j-th component of the i-th eigenvector

        # first fix the phase of the eigenvectors
        for i in range(len(t)):
            # if first component of first EV is not real and smaller than 0:
            # multiply all EVs with exp(-1j*angle)
            angle=0
            if (np.imag(e_vectors[i,0,0]) !=0 or e_vectors[i,0,0] < 0):
                angle = np.angle(e_vectors[i,0,0])
            e_vectors[i,:,:] = e_vectors[i,:,:]*np.exp(-1j*angle)        

        if print_states_t is not None:
            _t = print_states_t
            i = np.argmin(np.abs(t-_t))
            header = basis_states(dim_prod)
            # add column in front for the dressed state index
            header.insert(0,"t:{:.2f}".format(t[i]))
            header.append("Energy")
            table = []
            for j in range(self.dim):
                row = ["|Ψ"+str(j+1)+"⟩"]
                row.extend(np.abs(e_vectors[i,j])**2)
                row.extend([e_values[i,j]])
                table.append(row)
            print(tabulate(table,headers=header,floatfmt=".2f"))
            # print(tabulate(np.abs(e_vectors[i])**2,headers=header,floatfmt=".2f"))

        n_colors = np.empty([self.dim,e_values.shape[0]])  # for gnuplot
        if len(self.colors) != self.dim:
            print("Error: Number of colors does not match number of dressed states.")
            return

        s_colors = []  # stores color values
        r_array = np.zeros(self.dim)
        g_array = np.zeros(self.dim)
        b_array = np.zeros(self.dim)
        a_array = np.zeros(self.dim)
        a_array_gp = np.zeros(self.dim)  # for gnuplot
        for i in range(self.dim):
            r_array[i] = hex_to_rgba(self.colors[i])[0]/255
            g_array[i] = hex_to_rgba(self.colors[i])[1]/255
            b_array[i] = hex_to_rgba(self.colors[i])[2]/255
            if visible_states is None:
                a_array[i] = hex_to_rgba(self.colors[i])[3]/255
                a_array_gp[i] = 1-hex_to_rgba(self.colors[i])[3]/255

        if visible_states is not None:
            # check that no value will be OOB
            if np.max(visible_states) > self.dim-1:
                print("Error: Visible states out of bounds.")
                return
            a_array[visible_states] = 1
            a_array_gp[visible_states] = 0

        for i in range(self.dim):
            colors = []
            for j in range(e_values.shape[0]):
                e = np.abs(e_vectors[j,i])**2
                r = int(np.clip(np.dot(r_array,e),0,1)*255)
                g = int(np.clip(np.dot(g_array,e),0,1)*255)
                b = int(np.clip(np.dot(b_array,e),0,1)*255)
                a = int(np.clip(np.dot(a_array,e),0,1)*255)
                agp = int(np.clip(np.dot(a_array_gp,e),0,1)*255)
                n_colors[i,j] = 65536*r + 256*g + b + agp*16777216  # can be used in gnuplot with 'lc rgb variable'
                colors.append("#{:02x}{:02x}{:02x}{:02x}".format(r,g,b,a))
            s_colors.append(colors)
            if plot:
                plt.scatter(t,e_values[:,i],c=colors)
        if plot:
            if t_lim is not None:
                plt.xlim(t_lim[0],t_lim[1])
            if e_lim is not None:
                plt.ylim(e_lim[0],e_lim[1])
            for i in range(self.dim):
                plt.plot(t,e_values[:,i],label="ds{}".format(i+1))
            plt.legend()
            plt.xlabel("t (ps)")
            plt.ylabel("E (meV)")
            plt.savefig(filename + "_ds.png")
            plt.clf()

        # dressed state occupations
        # we use the following formula for the occupation of a dressed state |psi>:
        # <|psi><psi|> = sum_ij a_i * a_j^* * <|phi_i><phi_j|>
        # where |phi_i> are the states of the system and a_i are the components of |psi> in the basis of |phi_i>
        # <|phi_i><phi_j|> is the density matrix rho
        ds_occ = np.zeros([len(t),self.dim])
        for i in range(len(t)):
            for j in range(self.dim):
                ds_ij = e_vectors[i,j][:,None]*e_vectors[i,j].conj()  # ai * aj^*
                ds_occ[i,j] = np.sum(ds_ij*rho[i]).real  # sum_ij ai * aj^* * <|phi_i><phi_j|>
        if plot:
            plt.clf()
            plt.ylim(-0.1,1.1)
            if t_lim is not None:
                plt.xlim(t_lim[0],t_lim[1])
            for i in range(self.dim):
                plt.scatter(t,ds_occ[:,i],c=s_colors[i])
            for i in range(self.dim):
                plt.plot(t,ds_occ[:,i],label="ds{}".format(i+1))
            plt.xlabel("t (ps)")
            plt.ylabel("occupation (dressed state)")
            plt.legend()
            plt.savefig(filename + "_ds_occ.png")
            plt.clf()
        populations = np.diagonal(rho, axis1=1, axis2=2)
        if return_eigenvectors:
            return t, populations, e_values, ds_occ, s_colors, n_colors, e_vectors, rho
        return t, populations, e_values, ds_occ, s_colors, n_colors
