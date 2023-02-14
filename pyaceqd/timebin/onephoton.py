import re
import numpy as np
import matplotlib.pyplot as plt
from pyaceqd.tools import export_csv
from pyaceqd.tools import construct_t, simple_t_gaussian
import tqdm
from concurrent.futures import ThreadPoolExecutor, wait
import os

# exemplary options-dict:
options_example = {"verbose": False, "delta_xd": 4, "gamma_e": 1/65, "lindblad": True,
 "temp_dir": '/mnt/temp_data/', "phonons": False, "pt_file": "tls_dark_3.0nm_4k_th10_tmem20.48_dt0.02.ptr"}

class OnePhotonTimebin():
    def __init__(self, system, sigma_x, *pulses, dt=0.02, tb=800, simple_exp=True, gaussian_t=None, verbose=False, workers=15, options={}) -> None:
        self.system = system  # system that is used for the simulation
        self.dt = dt  # timestep during simulation
        self.options = options
        self.options["dt"] = dt  # also save it in the options dict
        self.tb = tb  # timebin width
        self.simple_exp = simple_exp  # use exponential timestepping
        self.gaussian_t = gaussian_t  # use gaussian timestepping during pulse
        self.pulses = pulses
        self.workers = workers  # number of threads spawned by ThreadPoolExecutor
        try:
            self.temp_dir = options["temp_dir"]
        except KeyError:
            print("temp_dir not included in options, setting to /mnt/temp_data/")
            self.options["temp_dir"] = "/mnt/temp_data/"
            self.temp_dir = self.options["temp_dir"]
        self.prepare_pulsefile(verbose=verbose)
        self.options["pulse_file_x"] = self.pulse_file_x  # put pulse files in options dict
        self.options["pulse_file_y"] = self.pulse_file_y
        # prepare the operators used in output/multitime
        self.prepare_operators(sigma_x=sigma_x, verbose=verbose)

    def calc_densitymatrix(self, first_abs=False):
        """
        if first_abs=True, takes the absolute value of the G1 function before integration. 
        this kills all phase-related effects.
        """
        rho_ee = self.rho_ee()
        rho_ll = self.rho_ll()
        norm = rho_ee+rho_ll  # for normalization (trace of the density matrix)
        t1, rho_el_g1 = self.rho_el()
        rho_el = np.abs(np.trapz(rho_el_g1,t1))
        if first_abs:
            rho_el = np.trapz(np.abs(rho_el_g1),t1)
        
        print("ee:{}, ll:{}, el:{}".format(rho_ee,rho_ll,rho_el))
        print("ee:{}, ll:{}, el:{}".format(rho_ee/norm,rho_ll/norm,rho_el/norm))
        return rho_ee, rho_ll, rho_el

    def prepare_pulsefile(self, verbose=False):
        # 2*tb is the maximum simulation length, 0 is the start of the simulation
        _t_pulse = np.arange(0,2.1*self.tb,step=self.dt/5)  # notice that for usual propagation, dt/10 is used
        # different polarizations
        self.pulse_file_x = self.temp_dir + "G1_pulse_x.dat"
        self.pulse_file_y = self.temp_dir + "G1_pulse_y.dat"
        pulse_x = np.zeros_like(_t_pulse, dtype=complex)
        pulse_y = np.zeros_like(_t_pulse, dtype=complex)
        for _p in self.pulses:
            pulse_x = pulse_x + _p.polar_x*_p.get_total(_t_pulse)
            pulse_y = pulse_y + _p.polar_y*_p.get_total(_t_pulse)
        export_csv(self.pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ', verbose=verbose)
        export_csv(self.pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ', verbose=verbose)

    def prepare_operators(self, sigma_x, verbose=False):
        # for ex.: sigma = |g><x|, i.e., |0><1|_2
        pattern = "^\|([0-9]*)><([0-9]*)\|_([1-9]*)"  # catches the thee relevant numbers in 3 capture groups 
        re_result = re.search(pattern=pattern, string=sigma_x)
        lower_state = re_result.group(1)
        upper_state = re_result.group(2)
        dimension = re_result.group(3)
        # define sigma_x and its conjugate
        self.sigma_x = "|{}><{}|_{}".format(lower_state,upper_state,dimension)
        self.sigma_xdag = "|{}><{}|_{}".format(upper_state,lower_state,dimension)
        self.x_op = "|{}><{}|_{}".format(upper_state,upper_state,dimension)
        if verbose:
            print("sigma_x: {}, sigma_xdag: {}, x_op: {}".format(self.sigma_x, self.sigma_xdag, self.x_op))
        
    def rho_ee(self):
        output_ops = [self.x_op]
        t,x = self.system(0,self.tb,output_ops=output_ops,suffix="ee",**self.options)
        x = np.real(x)
        t = np.real(t)
        rho_ee = np.trapz(x,t)
        return rho_ee
    
    def rho_ll(self):
        output_ops = [self.x_op]
        t,x = self.system(0,2*self.tb,output_ops=output_ops,suffix="ll",**self.options)
        x = np.real(x)
        t = np.real(t)
        # timesteps only during the second timebin are of relevance
        n_t = int(self.tb/self.dt)
        relevant_x = x[-n_t:]
        relevant_t = t[-n_t:]
        rho_ee = np.trapz(relevant_x,relevant_t)
        return rho_ee

    def rho_el(self):
        output_ops = [self.sigma_x]
        multitime_op = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}

        if self.gaussian_t is not None:
            t1 = simple_t_gaussian(0,self.gaussian_t,self.tb,5*self.dt,50*self.dt,*self.pulses)
        else:
            t1 = construct_t(0, self.tb, 5*self.dt, 50*self.dt, *self.pulses, simple_exp=self.simple_exp)
        
        _G1 = np.zeros([len(t1)],dtype=complex)
        with tqdm.tqdm(total=len(t1)) as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    multitime_op_new = dict(multitime_op)  # must make a copy of the dict
                    multitime_op_new["time"] = t1[i]
                    tend = t1[i] + self.tb
                    _e = executor.submit(self.system,0,tend,multitime_op=multitime_op_new, suffix=i, output_ops=output_ops, **self.options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                futures[i] = futures[i].result()
            # futures now contains t,pgx for every i
            for i in range(len(t1)):
                # pgx
                _G1[i] = futures[i][1][-1]
        return t1, _G1

    def __del__(self):
        os.remove(self.pulse_file_x)
        os.remove(self.pulse_file_y)
