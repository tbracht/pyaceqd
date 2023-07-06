import re
import numpy as np
import matplotlib.pyplot as plt
from pyaceqd.tools import construct_t, simple_t_gaussian
from pyaceqd.timebin.timebin import TimeBin
import tqdm
from concurrent.futures import ThreadPoolExecutor, wait
import os

# exemplary options-dict:
options_example = {"verbose": False, "delta_xd": 4, "gamma_e": 1/65, "lindblad": True,
 "temp_dir": '/mnt/temp_data/', "phonons": False, "pt_file": "tls_dark_3.0nm_4k_th10_tmem20.48_dt0.02.ptr"}

class TwoPhotonTimebin(TimeBin):
    def __init__(self, system, sigma_gx, sigma_xb, *pulses, dt=0.02, tb=800, dt_small=0.1, simple_exp=True, gaussian_t=None, verbose=False, workers=15, options={}) -> None:
        super().__init__(system, *pulses, dt=dt, tb=tb, simple_exp=simple_exp, gaussian_t=gaussian_t, verbose=verbose, workers=workers, options=options)
        # prepare the operators used in output/multitime
        self.gamma_e = options["gamma_e"]
        self.prepare_operators(sigma_gx=sigma_gx, sigma_xb=sigma_xb, verbose=verbose)
        if self.gaussian_t is not None:
            self.t1 = simple_t_gaussian(0,self.gaussian_t,self.tb,dt_small,10*dt_small,*self.pulses,decimals=1)
        else:
            self.t1 = construct_t(0, self.tb, dt_small, 10*dt_small, *self.pulses, simple_exp=self.simple_exp)


    def calc_densitymatrix(self):
        density_matrix = np.zeros([4,4], dtype=complex)
        # trace
        _,_,density_matrix[0,0] = self.rho_ee_ee()
        _,_,density_matrix[1,1] = self.rho_el_el()
        _,_,density_matrix[2,2] = self.rho_le_le()
        _,_,density_matrix[3,3] = self.rho_ll_ll()
        # ee_xx
        _,_,density_matrix[0,1] = self.rho_ee_el()
        density_matrix[1,0] = np.conj(density_matrix[0,1])
        density_matrix[0,2] = 0 # self.rho_ee_le()
        density_matrix[2,0] = np.conj(density_matrix[0,2])
        _,_,density_matrix[0,3] = self.rho_ee_ll()
        density_matrix[3,0] = np.conj(density_matrix[0,3])
        # el_xx
        density_matrix[1,2] = 0  # self.rho_el_le()
        density_matrix[2,1] = np.conj(density_matrix[1,2])
        _,_,density_matrix[1,3] = self.rho_el_ll()
        density_matrix[3,1] = np.conj(density_matrix[1,3])
        # le_ll
        density_matrix[2,3] = 0  # self.rho_le_ll()
        density_matrix[3,2] = np.conj(density_matrix[2,3])
        # normalize 
        norm = np.trace(density_matrix)
        # still output both, because the diagonal contains the number of coincidence measurments
        return density_matrix/norm, density_matrix

    def prepare_operators(self, sigma_gx, sigma_xb, verbose=False):
        """
        this function does not take into account if the transition operators contain multiple transitions
        i.e., it does not work if for example sigma_gx = |0><1|_3 + |1><2|_3
        """
        # for ex.: sigma_gx = |g><x|, i.e., |0><1|_2
        pattern = "^\|([0-9]*)><([0-9]*)\|_([1-9]*)"  # catches the three relevant numbers in 3 capture groups 
        # first, sigma_x
        re_result = re.search(pattern=pattern, string=sigma_gx)
        lower_state1 = re_result.group(1)
        upper_state1 = re_result.group(2)
        dimension = re_result.group(3)
        # define sigma_x and its conjugate
        self.sigma_x = "|{}><{}|_{}".format(lower_state1,upper_state1,dimension)
        self.sigma_xdag = "|{}><{}|_{}".format(upper_state1,lower_state1,dimension)
        self.x_op = "|{}><{}|_{}".format(upper_state1,upper_state1,dimension)
        # next, sigma_b
        re_result = re.search(pattern=pattern, string=sigma_xb)
        lower_state2 = re_result.group(1)
        upper_state2 = re_result.group(2)
        dimension = re_result.group(3)
        # define sigma_b and its conjugate
        self.sigma_b = "|{}><{}|_{}".format(lower_state2,upper_state2,dimension)
        self.sigma_bdag = "|{}><{}|_{}".format(upper_state2,lower_state2,dimension)
        self.b_op = "|{}><{}|_{}".format(upper_state2,upper_state2,dimension)
        # sigma_gb
        self.gb_op = "|{}><{}|_{}".format(lower_state1,upper_state2,dimension)
        self.gbdag_op = "|{}><{}|_{}".format(upper_state1,lower_state2,dimension)
        if verbose:
            print("sigma_x: {}, sigma_xdag: {}, x_op: {}".format(self.sigma_x, self.sigma_xdag, self.x_op))
            print("sigma_b: {}, sigma_bdag: {}, b_op: {}".format(self.sigma_b, self.sigma_bdag, self.b_op))
            print("gb: {}, gbdag: {}".format(self.gb_op, self.gbdag_op))

    # first the four functions for the trace of the density matrix
    def rho_ee_ee(self, dt_small=0.1):
        """
        calculates G2 assuming an XX-emission triggers the coincidence measurment at time t1, following an X at time t2, i.e.:
        sigma^dagger_XX(t1) sigma^dagger_X(t2) sigma_x(t2) sigma_xx(t1)>
        here, t1 and t2 are both in the same time-bin, i.e. t1<=t2<=tb
        """
        output_ops = [self.x_op, self.b_op]
        # at t1, apply sigma_b from left and sigma_bbdag from right
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}

        t1 = self.t1
        n_tau = int((self.tb)/self.dt)
        # simulation time-axis
        t2 = np.linspace(0, self.tb, n_tau + 1)
        _G2 = np.zeros([len(t1)])
        tend = self.tb  # always the same
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    sigma_b_new = dict(sigma_b)  # must make a copy of the dict
                    sigma_bdag_new = dict(sigma_bdag)
                    sigma_b_new["time"] = t1[i]
                    sigma_bdag_new["time"] = t1[i]
                    # apply sigma_b from left and sigma_bbdag from right
                    multitme_ops = [sigma_b_new,sigma_bdag_new]
                    _e = executor.submit(self.system,0,tend,multitime_op=multitme_ops, suffix=i, output_ops=output_ops, **self.options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                futures[i] = futures[i].result()
            # futures now contains [t,x,b] for every i
            for i in range(len(t1)):
                # t2 = t1,...,tb
                n_t2 = n_tau - int((t1[i])/self.dt)
                temp_t2 = np.zeros(n_t2+1)
                # special case tau=0:
                # as Tr(sigma^dagger*sigma^dagger*sigma*sigma * rho) = x, G2(t,0) = x(t), which is the value with index [-(n_tau+1)]
                temp_t2[0] = np.abs(futures[i][2][-(n_t2+1)])
                # futures[i][2] are the b values , [1] are the x values
                # here, we want the x-values for every t2=t1,..,tb
                if n_t2 > 0: 
                    temp_t2[1:n_t2+1] = np.abs(futures[i][1][-n_t2:])
                t_new = t2[:len(temp_t2)]
                # plt.clf()
                # plt.plot(t_new,np.real(temp_t2),'r-')
                # plt.plot(t_new,np.imag(temp_t2),'b-')
                # plt.savefig("aa_tests/plot_{}.png".format(i))
                # integrate over t_new
                _G2[i] = np.trapz(temp_t2,t_new)
        return t1, _G2, np.trapz(_G2,t1)*self.gamma_e**2

    def rho_ll_ll(self,dt_small=0.1):
        """
        calculates G2 assuming an XX-emission triggers the coincidence measurment at time t1, following an X at time t2, i.e.:
        sigma^dagger_XX(t1) sigma^dagger_X(t2) sigma_x(t2) sigma_xx(t1)>
        here, t1 and t2 are both in the second time-bin, i.e. tb<t1<=t2<=2*tb
        """
        output_ops = [self.x_op, self.b_op]
        # at t1, apply sigma_b from left and sigma_bbdag from right
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}

        t1 = self.t1 

        n_tau = int((self.tb)/self.dt)
        # simulation time-axis
        t2 = np.linspace(0, self.tb, n_tau + 1)
        _G2 = np.zeros([len(t1)])
        tend = 2*self.tb  # always the same
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    sigma_b_new = dict(sigma_b)  # must make a copy of the dict
                    sigma_bdag_new = dict(sigma_bdag)
                    sigma_b_new["time"] = self.tb+t1[i]
                    sigma_bdag_new["time"] = self.tb+t1[i]
                    # apply sigma_b from left and sigma_bbdag from right
                    multitme_ops = [sigma_b_new,sigma_bdag_new]
                    _e = executor.submit(self.system,0,tend,multitime_op=multitme_ops, suffix=i, output_ops=output_ops, **self.options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                futures[i] = futures[i].result()
            # futures now contains [t,x,b] for every i
            for i in range(len(t1)):
                # t2 = t1,...,2tb
                n_t2 = n_tau - int((t1[i])/self.dt)
                temp_t2 = np.zeros(n_t2+1)
                # special case tau=0:
                # as Tr(sigma^dagger*sigma^dagger*sigma*sigma * rho) = x, G2(t,0) = x(t), which is the value with index [-(n_tau+1)]
                temp_t2[0] = np.abs(futures[i][2][-(n_t2+1)])
                # futures[i][2] are the b values , [1] are the x values
                # here, we want the x-values for every t2=t1,..,tend
                if n_t2 > 0: 
                    # take the last n_t2 values of the array
                    temp_t2[1:n_t2+1] = np.abs(futures[i][1][-n_t2:])
                t_new = t2[:len(temp_t2)]
                # integrate over t_new
                _G2[i] = np.trapz(temp_t2,t_new)
        return t1, _G2, np.trapz(_G2,t1)*self.gamma_e**2

    def rho_el_el(self, dt_small=0.1):
        """
        calculates G2 assuming an XX-emission triggers the coincidence measurment at time t1, following an X at time t2, i.e.:
        sigma^dagger_XX(t1) sigma^dagger_X(t2) sigma_x(t2) sigma_xx(t1)>
        here, t1 is in the first timebin, and t2 is in the second time-bin, i.e. t1<=tb<t2<=2*tb
        """
        output_ops = [self.x_op, self.b_op]
        # at t1, apply sigma_b from left and sigma_bbdag from right
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}

        t1 = self.t1

        n_tau = int((self.tb)/self.dt)
        # simulation time-axis
        t2 = np.linspace(0, self.tb, n_tau + 1)
        _G2 = np.zeros([len(t1)])
        tend = 2*self.tb  # always the same
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    sigma_b_new = dict(sigma_b)  # must make a copy of the dict
                    sigma_bdag_new = dict(sigma_bdag)
                    sigma_b_new["time"] = t1[i]
                    sigma_bdag_new["time"] = t1[i]
                    # apply sigma_b from left and sigma_bbdag from right
                    multitme_ops = [sigma_b_new,sigma_bdag_new]
                    _e = executor.submit(self.system,0,tend,multitime_op=multitme_ops, suffix=i, output_ops=output_ops, **self.options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                futures[i] = futures[i].result()
            # futures now contains [t,x,b] for every i
            for i in range(len(t1)):
                # t2 = tb,...,2*tb
                n_t2 = n_tau  #  - int((t1[i])/self.dt)
                temp_t2 = np.zeros(n_t2+1)
                # special case tau=0:
                # as Tr(sigma^dagger*sigma^dagger*sigma*sigma * rho) = x, G2(t,0) = x(t), which is the value with index [-(n_tau+1)]
                temp_t2[0] = np.abs(futures[i][2][-n_t2-1])
                # futures[i][2] are the b values , [1] are the x values
                # here, we want the x-values for every t2=tb,..,2tb
                # take the last n_t2 values of the array
                temp_t2[1:n_t2+1] = np.abs(futures[i][1][-n_t2:])
                t_new = t2[:len(temp_t2)]
                # integrate over t_new
                _G2[i] = np.trapz(temp_t2,t_new)
        return t1, _G2, np.trapz(_G2,t1)*self.gamma_e**2


    def rho_le_le(self, dt_small=0.1):
        """
        calculates G2 assuming an X-emission triggers the coincidence measurment at time t1, following an XX at time t2, i.e.:
        sigma^dagger_X(t1) sigma^dagger_XX(t2) sigma_XX(t2) sigma_X(t1)>
        here, t1 is in the first timebin, and t2 is in the second time-bin, i.e. t1<=tb<t2<=2*tb
        """
        output_ops = [self.b_op]
        # at t1, apply sigma_X from left and sigma_Xdag from right
        sigma_x = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore": "false"}
        sigma_xdag = {"operator": self.sigma_xdag ,"applyFrom": "_right", "applyBefore": "false"}
    
        t1 = self.t1 

        n_tau = int((self.tb)/self.dt)
        # simulation time-axis
        t2 = np.linspace(0, self.tb, n_tau + 1)
        _G2 = np.zeros([len(t1)])
        tend = 2*self.tb  # always the same
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    # t1 = 0,...,tb
                    _t1 = t1[i]
                    sigma_xdag_new = dict(sigma_xdag)
                    sigma_x_new = dict(sigma_x)
                    # add correct times
                    sigma_xdag_new["time"] = _t1
                    sigma_x_new["time"] = _t1
                    multitime_op_new = [sigma_xdag_new,sigma_x_new]
                    _e = executor.submit(self.system,0,tend,multitime_op=multitime_op_new, suffix=i, output_ops=output_ops, **self.options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                wait(futures)
            for k in range(len(futures)):
                # futures are still 'future' objects
                futures[k] = futures[k].result()
            # futures now contains t,xx for every i
            for i in range(len(t1)):
                # t2 = tb,...,2*tb
                n_t2 = n_tau  #  - int((t1[i])/self.dt)
                temp_t2 = np.zeros(n_t2+1)
                # special case, this should(?) also work as for t1=t2=tb, G2=0. untested
                # temp_t2 = np.abs(futures[i][1][-(n_t2+1)])
                # special case tau=0:
                # as Tr(sigmaX^dagger*sigmaXX^dagger*sigmaXX*sigmaX * rho) = 0, G2(t,0) = 0, which is the value with index [-(n_tau+1)]
                temp_t2[0] = 0  # np.abs(futures[i][1][-n_t2-1])
                # futures[i][1] are the b values
                # here, we want the b-values for every t2=tb,..,2tb
                # take the last n_t2 values of the array
                temp_t2[1:n_t2+1] = np.abs(futures[i][1][-n_t2:])
                t_new = t2[:len(temp_t2)]
                # integrate over t_new
                _G2[i] = np.trapz(temp_t2,t_new)
        return t1,_G2, np.trapz(_G2, t1)*self.gamma_e**2

    # the remaining three ee_xx
    def rho_ee_ll(self, plot_g2=False):
        """
        correlations between EE and LL states. this includes four different times, but we only have two 'time-axes',
        and one of those always starts at the end of the other. This allows us to calculate the correlation functions
        with a reasonable numerical effort. However, we make one assumption when using the gaussian-density spaced time-axis:
        the pulses in the first and second time-bin arrive (1) at the same time and (2) have the same amplitude.
        More precisely, the pulse in the first time-bin defines the time-axis for the first and second time-bin.
        This is necessary because using "+tb" for the late-timebin operators effectively just shifts the time-axis by one time-bin.
        """
        output_ops = [self.sigma_x, self.gb_op]

        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_xdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
        
        t1 = self.t1 
        _G2 = np.zeros([len(t1)], dtype=complex)
        if plot_g2:
            _g2plot = np.zeros([len(t1),len(t1)], dtype=complex)
        # loop over t1
        for i in tqdm.trange(len(t1),leave=None):
            # tau1: use the interval 0,...,tb-t1
            # i.e., if t1 = 0,...,tx then tau1 expands to absolute times of tx,...,tb
            _t1 = t1[i]
            futures = []
            with tqdm.tqdm(total=len(t1)-i, leave=None) as tq:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    # loop over t2: starts at t1
                    for j in range(len(t1)-i):
                        # j=0 is a special case that has to be addressed: here, |XX><G| has to be applied at t=t1
                        # this is addressed by taking care to use the correct order of operators in the parameter file
                        # if the time is the same, the order in the param file is the order that is used to apply the operators
                        _t2 = t1[j+i]
                        _t3 = _t1 + self.tb
                        _t4_end = _t2 + self.tb

                        sigma_bdag_new = dict(sigma_bdag)
                        sigma_xdag_new = dict(sigma_xdag)
                        sigma_b_new = dict(sigma_b)
                        # add correct times
                        sigma_bdag_new["time"] = _t1
                        sigma_xdag_new["time"] = _t2
                        sigma_b_new["time"] = _t3
                        # the order of the operators is important to catch the special case where t1=t2
                        # because then ACE applies the operator first, that is first in the parameter file
                        multitime_op_new = [sigma_bdag_new,sigma_xdag_new,sigma_b_new]
                        # support for additional offset
                        #if _t3 >= self.tb and _t4_end <= 2*self.tb:
                        _e = executor.submit(self.system,0,_t4_end,multitime_op=multitime_op_new, suffix=j, output_ops=output_ops, **self.options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
                        #else:
                        #    # print("_t3:{:.4f}, _t4_end:{:.4f}".format(_t3, _t4_end))
                        #    _e = executor.submit(lambda: np.zeros([3,1]))
                        #    _e.add_done_callback(lambda f: tq.update())
                        #    futures.append(_e)
            for k in range(len(futures)):
                # futures are still 'future' objects
                futures[k] = futures[k].result()
            # futures now contains t,pgx,pgb for every j
            t2_array = t1[i:]  # array for the second time-axis
            temp_t2 = np.zeros_like(t2_array)
            # p_gb, taking the abs. value eradicates any phase, i.e., we only get the abs. value of the dens. matrix
            temp_t2[0] = np.abs(futures[0][2][-1])
            if plot_g2: 
                _g2plot[i,0] = futures[0][2][-1]
            for k in range(1,len(t2_array)):
                # pgx
                temp_t2[k] = np.abs(futures[k][1][-1])
                if plot_g2:
                    _g2plot[i,k] = futures[k][1][-1]
            _G2[i] = np.trapz(temp_t2, t2_array)
        if plot_g2:
            return t1,_g2plot
        return t1, _G2, np.abs(np.trapz(_G2, t1))*self.gamma_e**2

    def rho_ee_ll_debug(self):
        """
        just the j=0 case, in which phase-errors occured due to numerical artifacts.
        """
        output_ops = [self.sigma_x, self.gb_op]

        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_xdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
        
        t1 = self.t1 
        _G2 = np.zeros([len(t1)], dtype=complex)
        # loop over t1
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            futures = []
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for i in range(len(t1)):
                    # j=0 is a special case that has to be addressed: here, |XX><G| has to be applied at t=t1
                    # this is addressed by taking care to use the correct order of operators in the parameter file
                    # if the time is the same, the order in the param file is the order that is used to apply the operators
                    _t1 = t1[i]
                    _t2 = t1[i]
                    _t3 = _t1 + self.tb
                    _t4_end = _t2 + self.tb
                    sigma_bdag_new = dict(sigma_bdag)
                    sigma_xdag_new = dict(sigma_xdag)
                    sigma_b_new = dict(sigma_b)
                    # add correct times
                    sigma_bdag_new["time"] = _t1
                    sigma_xdag_new["time"] = _t2
                    sigma_b_new["time"] = _t3
                    # the order of the operators is important to catch the special case where t1=t2
                    # because then ACE applies the operator first, that is first in the parameter file
                    multitime_op_new = [sigma_bdag_new,sigma_xdag_new,sigma_b_new]
                    _e = executor.submit(self.system,0,_t4_end,multitime_op=multitime_op_new, suffix=i, output_ops=output_ops, **self.options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
            for k in range(len(futures)):
            # futures are still 'future' objects
                futures[k] = futures[k].result()
            # p_gb
            for i in range(len(futures)):
                _G2[i] = futures[i][2][-1]
        return t1, _G2

    def rho_ee_el(self):
        output_ops = [self.sigma_x]

        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_xdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}

        t1 = self.t1 
        _G2 = np.zeros([len(t1)], dtype=complex)
        # loop over t1
        for i in tqdm.trange(len(t1),leave=None):
            # tau1: use the interval 0,...,tb-t1
            # i.e., if t1 = 0,...,tx then tau1 expands to absolute times of tx,...,tb
            _t1 = t1[i]
            futures = []
            with tqdm.tqdm(total=len(t1)-i, leave=None) as tq:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    # loop over t2: starts at t1
                    for j in range(len(t1)-i):
                        # j=0 is a special case that has to be addressed: here, |XX><G| has to be applied at t=t1
                        # this is addressed by taking care to use the correct order of operators in the parameter file
                        # if the time is the same, the order in the param file is the order that is used to apply the operators
                        _t2 = t1[j+i]
                        _t3_end = _t2 + self.tb

                        sigma_bdag_new = dict(sigma_bdag)
                        sigma_xdag_new = dict(sigma_xdag)
                        sigma_b_new = dict(sigma_b)
                        # add correct times
                        sigma_b_new["time"] = _t1
                        sigma_bdag_new["time"] = _t1
                        sigma_xdag_new["time"] = _t2
                        
                        # the order of the operators is important to catch the special case where t1=t2
                        # because then ACE applies the operator first, that is first in the parameter file
                        multitime_op_new = [sigma_b_new,sigma_bdag_new,sigma_xdag_new]
                        _e = executor.submit(self.system,0,_t3_end,multitime_op=multitime_op_new, suffix=j, output_ops=output_ops, **self.options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)

            for k in range(len(futures)):
                # futures are still 'future' objects
                futures[k] = futures[k].result()
            # futures now contains t,pgx for every j
            t2_array = t1[i:]  # array for the second time-axis
            temp_t2 = np.zeros_like(t2_array)
            for k in range(0,len(t2_array)):
                # pgx
                temp_t2[k] = np.abs(futures[k][1][-1])
            _G2[i] = np.trapz(temp_t2, t2_array)
        return t1, _G2, np.trapz(_G2, t1)*self.gamma_e**2

    def rho_ee_le(self):
        output_ops = [self.sigma_b]

        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_xdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_x = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}

        t1 = self.t1 
        _G2 = np.zeros([len(t1)], dtype=complex)
        # loop over t1
        for i in tqdm.trange(len(t1),leave=None):
            # tau1: use the interval 0,...,tb-t1
            # i.e., if t1 = 0,...,tx then tau1 expands to absolute times of tx,...,tb
            _t1 = t1[i]
            futures = []
            with tqdm.tqdm(total=len(t1)-i, leave=None) as tq:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    # loop over t2: starts at t1
                    for j in range(len(t1)-i):
                        # j=0 is a special case that has to be addressed: here, |XX><G| has to be applied at t=t1
                        # this is addressed by taking care to use the correct order of operators in the parameter file
                        # if the time is the same, the order in the param file is the order that is used to apply the operators
                        _t2 = t1[j+i]
                        _t3_end = _t2 + self.tb

                        sigma_bdag_new = dict(sigma_bdag)
                        sigma_xdag_new = dict(sigma_xdag)
                        sigma_x_new = dict(sigma_x)
                        # add correct times
                        sigma_x_new["time"] = _t1
                        sigma_bdag_new["time"] = _t1
                        sigma_xdag_new["time"] = _t2
                        
                        # the order of the operators is important to catch the special case where t1=t2
                        # because then ACE applies the operator first, that is first in the parameter file
                        multitime_op_new = [sigma_x_new,sigma_bdag_new,sigma_xdag_new]
                        _e = executor.submit(self.system,0,_t3_end,multitime_op=multitime_op_new, suffix=j, output_ops=output_ops, **self.options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)

            for k in range(len(futures)):
                # futures are still 'future' objects
                futures[k] = futures[k].result()
            # futures now contains t,pgx for every j
            t2_array = t1[i:]  # array for the second time-axis
            temp_t2 = np.zeros_like(t2_array)
            for k in range(0,len(t2_array)):
                # pxb
                temp_t2[k] = np.abs(futures[k][1][-1])
            _G2[i] = np.trapz(temp_t2, t2_array)
        return t1, _G2, np.trapz(_G2, t1)*self.gamma_e**2

    # remaining two el_xx
    def rho_el_le(self):
        # this is zero
        return 0,0,0

    def rho_el_ll(self):
        output_ops = [self.x_op, self.sigma_b]
        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}

        t1 = self.t1 
        _G2 = np.zeros([len(t1)], dtype=complex)
        # loop over t1
        for i in tqdm.trange(len(t1),leave=None):
            # tau1: use the interval 0,...,tb-t1
            # i.e., if t1 = 0,...,tx then tau1 expands to absolute times of tx,...,tb
            _t1 = t1[i]
            futures = []
            with tqdm.tqdm(total=len(t1)-i, leave=None) as tq:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    # loop over t2: starts at t1
                    for j in range(len(t1)-i):
                        # j=0 is a special case that has to be addressed: here, <|X><B|> has to be taken at t=t1
                        _t2 = _t1 + self.tb
                        _t3_end = t1[j+i] + self.tb

                        sigma_bdag_new = dict(sigma_bdag)
                        sigma_b_new = dict(sigma_b)
                        # add correct times
                        sigma_bdag_new["time"] = _t1
                        sigma_b_new["time"] = _t2
                        # the order of the operators is important to catch the special case where t1=t2
                        # because then ACE applies the operator first, that is first in the parameter file
                        multitime_op_new = [sigma_bdag_new,sigma_b_new]
                        _e = executor.submit(self.system,0,_t3_end,multitime_op=multitime_op_new, suffix=j, output_ops=output_ops, **self.options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
            for k in range(len(futures)):
                # futures are still 'future' objects
                futures[k] = futures[k].result()
            # futures now contains t,x,pxb for every j
            t2_array = t1[i:]  # array for the second time-axis
            temp_t2 = np.zeros_like(t2_array)
            # p_xb, j=0 special case
            temp_t2[0] = np.abs(futures[0][2][-1])
            for k in range(1,len(t2_array)):
                # x
                temp_t2[k] = np.abs(futures[k][1][-1])
            _G2[i] = np.trapz(temp_t2, t2_array)
        return t1, _G2, np.trapz(_G2, t1)*self.gamma_e**2

    def rho_el_ll_debug(self):
        # j=0 special case 
        output_ops = [self.x_op, self.sigma_b]
        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}

        t1 = self.t1 
        _G2 = np.zeros([len(t1)], dtype=complex)
        _g20 = np.zeros_like(_G2)
        # loop over t1
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            futures = []
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for i in range(len(t1)):
                # tau1: use the interval 0,...,tb-t1
                # i.e., if t1 = 0,...,tx then tau1 expands to absolute times of tx,...,tb
                    _t1 = t1[i]
                    # j=0 is a special case that has to be addressed: here, <|X><B|> has to be taken at t=t1
                    _t2 = _t1 + self.tb
                    _t3_end = t1[i] + self.tb

                    sigma_bdag_new = dict(sigma_bdag)
                    sigma_b_new = dict(sigma_b)
                        # add correct times
                    sigma_bdag_new["time"] = _t1
                    sigma_b_new["time"] = _t2
                        # the order of the operators is important to catch the special case where t1=t2
                        # because then ACE applies the operator first, that is first in the parameter file
                    multitime_op_new = [sigma_bdag_new,sigma_b_new]
                    _e = executor.submit(self.system,0,_t3_end,multitime_op=multitime_op_new, suffix=i, output_ops=output_ops, **self.options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
            for k in range(len(futures)):
                # futures are still 'future' objects
                futures[k] = futures[k].result()
            # futures now contains t,x,pxb for every j
            # p_xb, j=0 special case
            _g20[i] = futures[0][2][-1]
        return t1, _g20

    # the remaining le_ll

    def rho_le_ll(self):
        # this is zero
        return 0,0,0
    