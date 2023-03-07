import re
import numpy as np
from pyaceqd.tools import construct_t, simple_t_gaussian
from pyaceqd.timebin.timebin import TimeBin
import tqdm
from concurrent.futures import ThreadPoolExecutor, wait

# exemplary options-dict:
options_example = {"verbose": False, "delta_xd": 4, "gamma_e": 1/65, "lindblad": True,
 "temp_dir": '/mnt/temp_data/', "phonons": False, "pt_file": "tls_dark_3.0nm_4k_th10_tmem20.48_dt0.02.ptr"}

class OnePhotonTimebin(TimeBin):
    def __init__(self, system, sigma_x, *pulses, dt=0.02, tb=800, simple_exp=True, gaussian_t=None, verbose=False, workers=15, options={}) -> None:
        super().__init__(system, *pulses, dt=dt, tb=tb, simple_exp=simple_exp, gaussian_t=gaussian_t, verbose=verbose, workers=workers, options=options)
        # prepare the operators used in output/multitime
        self.prepare_operators(sigma_x=sigma_x, verbose=verbose)
        try:
            self.gamma_e = self.options["gamma_e"]
        except KeyError:
            print("gamma_e not supplied in options.")
            exit(1)

    def calc_densitymatrix(self, first_abs=False, verbose=False):
        """
        if first_abs=True, takes the absolute value of the G1 function before integration. 
        this kills all phase-related effects.
        """
        rho_ee = self.rho_ee() * self.gamma_e
        rho_ll = self.rho_ll() * self.gamma_e
        norm = rho_ee+rho_ll  # for normalization (trace of the density matrix)
        t1, rho_el_g1 = self.rho_el()
        rho_el = np.abs(np.trapz(rho_el_g1,t1))
        if first_abs:
            rho_el = np.trapz(np.abs(rho_el_g1),t1)
        rho_el = rho_el * self.gamma_e
        if verbose:
            print("ee:{}, ll:{}, el:{}".format(rho_ee,rho_ll,rho_el))
            print("ee:{}, ll:{}, el:{}".format(rho_ee/norm,rho_ll/norm,rho_el/norm))
        return rho_ee, rho_ll, rho_el, norm

    def prepare_operators(self, sigma_x, verbose=False):
        # for ex.: sigma = |g><x|, i.e., |0><1|_2
        pattern = "^\|([0-9]*)><([0-9]*)\|_([1-9]*)"  # catches the three relevant numbers in 3 capture groups 
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

    def rho_el(self, dt_small=0.1):
        output_ops = [self.sigma_x]
        multitime_op = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}

        if self.gaussian_t is not None:
            t1 = simple_t_gaussian(0, self.gaussian_t, self.tb, dt_small, 10*dt_small, *self.pulses)
        else:
            t1 = construct_t(0, self.tb, dt_small, 10*dt_small, *self.pulses, simple_exp=self.simple_exp)
        
        _G1 = np.zeros([len(t1)],dtype=complex)
        with tqdm.tqdm(total=len(t1),leave=None) as tq:
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
