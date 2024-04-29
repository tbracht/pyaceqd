import re
import numpy as np
import matplotlib.pyplot as plt
from pyaceqd.tools import construct_t, simple_t_gaussian, concurrence
from pyaceqd.timebin.timebin import TimeBin
import tqdm
from concurrent.futures import ThreadPoolExecutor, wait
import os

# exemplary options-dict:
options_example = {"verbose": False, "delta_xd": 4, "gamma_e": 1/65, "lindblad": True,
 "temp_dir": '/mnt/temp_data/', "phonons": False, "pt_file": "tls_dark_3.0nm_4k_th10_tmem20.48_dt0.02.ptr"}

class TwoPhotonTimebinNew(TimeBin):
    def __init__(self, system, sigma_x, sigma_xdag, sigma_b, sigma_bdag, *pulses, dt=0.02, tb=800, dt_small=0.1, simple_exp=True, gaussian_t=None, verbose=False, workers=15, options={}) -> None:
        super().__init__(system, *pulses, dt=dt, tb=tb, simple_exp=simple_exp, gaussian_t=gaussian_t, verbose=verbose, workers=workers, options=options)
        # prepare the operators used in output/multitime
        self.gamma_e = options["gamma_e"]
        self.prepare_operators(sigma_x=sigma_x, sigma_xdag=sigma_xdag, sigma_b=sigma_b, sigma_bdag=sigma_bdag, verbose=verbose)
        if self.gaussian_t is not None:
            self.t1 = simple_t_gaussian(0,self.gaussian_t,self.tb,dt_small,10*dt_small,*self.pulses,decimals=1)
        else:
            self.t1 = construct_t(0, self.tb, dt_small, 10*dt_small, *self.pulses, simple_exp=self.simple_exp)

    def calc_timedynamics(self):
        return self.system(0, 2*self.tb, *self.pulses, **self.options)


    def calc_densitymatrix(self, save_dm=False, save_all=False, filename="densitymatrix", verbose=False, reduced=False, use_second_zero=False):
        """
        calculates the density matrix of the system, using the G2 functions.
        The density matrix is calculated in the basis |ee>, |el>, |le>, |ll>
        """
        density_matrix = np.zeros([4,4], dtype=complex)
        # trace
        t,G2_EEEE,density_matrix[0,0],G2_EEEE_1,G2_EEEE_2,_ = self.rho_ee_ee(use_second_zero=use_second_zero)
        _,G2_ELEL,density_matrix[1,1] = self.rho_el_el()
        _,G2_LELE,density_matrix[2,2] = self.rho_le_le()
        _,G2_LLLL,density_matrix[3,3],G2_LLLL_1,G2_LLLL_2,_ = self.rho_ll_ll(use_second_zero=use_second_zero)
        # ee_xx
        _,G2_EELL,density_matrix[0,3],G2_EELL_1,G2_EELL_2,_ = self.rho_ee_ll(use_second_zero=use_second_zero)
        density_matrix[3,0] = np.conj(density_matrix[0,3])
        if not reduced:
            _,G2_EEEL,density_matrix[0,1],G2_EEEL_1,G2_EEEL_2 = self.rho_ee_el()
            density_matrix[1,0] = np.conj(density_matrix[0,1])
            _,G2_EELE,density_matrix[0,2],G2_EELE_1,G2_EELE_2 = self.rho_ee_le()
            density_matrix[2,0] = np.conj(density_matrix[0,2])

            # el_xx
            _,G2_ELLE,density_matrix[1,2],G2_ELLE_1,G2_ELLE_2 = self.rho_el_le()
            density_matrix[2,1] = np.conj(density_matrix[1,2])
            _,G2_ELLL,density_matrix[1,3],G2_ELLL_1,G2_ELLL_2 = self.rho_el_ll()
            density_matrix[3,1] = np.conj(density_matrix[1,3])
            # le_ll
            _,G2_LELL,density_matrix[2,3],G2_LELL_1,G2_LELL_2 = self.rho_le_ll()
            density_matrix[3,2] = np.conj(density_matrix[2,3])
        # normalize 
        norm = np.trace(density_matrix)

        if save_dm:
            np.save(filename+"_dm.npy", density_matrix)
        if save_all:
            np.save(filename+"_dm.npy", density_matrix)
            np.save(filename+"_t.npy", t)
            components = [G2_EEEE, G2_ELEL, G2_LELE, G2_LLLL, G2_EEEL, G2_EELE, G2_EELL, G2_ELLE, G2_ELLL, G2_LELL]
            components_array = np.stack(components, axis=0)
            np.save(filename+"_components.npy", components_array)

            components_1 = [G2_EEEE_1, G2_LLLL_1, G2_EEEL_1, G2_EELE_1, G2_EELL_1, G2_ELLE_1, G2_ELLL_1, G2_LELL_1]
            components_1_array = np.stack(components_1, axis=0)
            np.save(filename+"_components_1.npy", components_1_array)
    
            components_2 = [G2_EEEE_2, G2_LLLL_2, G2_EEEL_2, G2_EELE_2, G2_EELL_2, G2_ELLE_2, G2_ELLL_2, G2_LELL_2]
            components_2_array = np.stack(components_2, axis=0)
            np.save(filename+"_components_2.npy", components_2_array)
        if verbose:
            # print density matrix nicely formatted
            print("density matrix:")
            print(np.array2string(density_matrix, formatter={'complex_kind': lambda x: "%.3f+%.3fj" % (x.real, x.imag)}))
            print("normalized density matrix:")
            print(np.array2string(density_matrix/norm, formatter={'complex_kind': lambda x: "%.3f+%.3fj" % (x.real, x.imag)}))
        # still output both, because the diagonal contains the number of coincidence measurments
        return concurrence(density_matrix/norm), density_matrix

    def prepare_operators(self, sigma_x, sigma_xdag, sigma_b, sigma_bdag, verbose=False):
        """
        all operators needed to calculate the correlation functions
        """
        # define sigma_x and its conjugate
        self.sigma_x = sigma_x
        self.sigma_xdag = sigma_xdag
        self.x_op = "(" + sigma_xdag +  " * " +  sigma_x + ")"
        # define sigma_b and its conjugate
        self.sigma_b = sigma_b
        self.sigma_bdag = sigma_bdag
        self.b_op = "(" + sigma_bdag +  " * " +  sigma_b + ")"
        if verbose:
            print("sigma_x: {}, sigma_xdag: {}, x_op: {}".format(self.sigma_x, self.sigma_xdag, self.x_op))
            print("sigma_b: {}, sigma_bdag: {}, b_op: {}".format(self.sigma_b, self.sigma_bdag, self.b_op))

    # first the four functions for the trace of the density matrix
    def rho_ee_ee(self, add_time=0, use_second_zero=False):
        """
        calculates G2 assuming an XX-emission triggers the coincidence measurment at time t1, following an X at time t2, i.e.:
        <sigma^dagger_XX(t1) sigma^dagger_X(t2) sigma_x(t2) sigma_xx(t1)>
        here, t1 and t2 are both in the same time-bin, i.e. t1<=t2<=tb
        Secondly, an X-emission triggers the coincidence measurment at time t1, following an XX at time t2, i.e.:
        <sigma^dagger_X(t1) sigma^dagger_XX(t2) sigma_XX(t2) sigma_X(t1)>
        The latter can for example happen in the case of re-excitation.
        """
        t1 = self.t1
        n_tau = int((self.tb)/self.dt)
        # simulation time-axis
        t2 = np.linspace(0, self.tb, n_tau + 1)
        tend = self.tb + add_time  # always the same
        def _rho_ee_ee(output_ops, sigma_X, sigma_Xdag):
            _G2 = np.zeros([len(t1)])
            _G2_t1t2 = np.zeros([len(t1),len(t2)])
            with tqdm.tqdm(total=len(t1), leave=None) as tq:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    futures = []
                    for i in range(len(t1)):
                        sigma_X_new = dict(sigma_X)  # must make a copy of the dict
                        sigma_Xdag_new = dict(sigma_Xdag)
                        sigma_X_new["time"] = t1[i] + add_time
                        sigma_Xdag_new["time"] = t1[i] + add_time
                        # apply sigma_b from left and sigma_bbdag from right
                        multitme_ops = [sigma_X_new,sigma_Xdag_new]
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
                    # which is the value with index [-(n_tau+1)] with the second output operator 
                    temp_t2[0] = np.abs(futures[i][2][-(n_t2+1)])
                    # futures[i][2] are the values of the second output operators for tau0, [1] are the values of the first output operator
                    # here, we want the values of the first output operator for every t2=t1,..,tb
                    if n_t2 > 0: 
                        temp_t2[1:n_t2+1] = np.abs(futures[i][1][-n_t2:])
                    t_new = t2[:len(temp_t2)]
                    # plt.clf()
                    # plt.plot(t_new,np.real(temp_t2),'r-')
                    # plt.plot(t_new,np.imag(temp_t2),'b-')
                    # plt.savefig("aa_tests/plot_{}.png".format(i))
                    # integrate over t_new
                    _G2[i] = np.trapz(temp_t2,t_new)
                    _G2_t1t2[i, -len(temp_t2):] = temp_t2 
            return _G2, _G2_t1t2
        # first, case t1 <= t2
        out_op1 = self.sigma_xdag + "*" + self.sigma_x
        out_op_tau0 = self.sigma_bdag + "*" + self.sigma_xdag + "*" + self.sigma_x + "*" + self.sigma_b
        output_ops = [out_op1, out_op_tau0]
        # at t1, apply sigma_b from left and sigma_bdag from right
        sigma_left = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
        sigma_right = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        _G2_1, _G21_t1t2 = _rho_ee_ee(output_ops, sigma_left, sigma_right)
        if use_second_zero:
            return t1, t2,_G2_1, np.trapz(_G2_1,t1)*self.gamma_e**2, _G2_1, _G2_1*0,  _G21_t1t2
        # second, case t2 <= t1
        out_op1 = self.sigma_bdag + "*" + self.sigma_b
        # should be zero, as t1=t2 is already covered by the first case
        out_op_tau0 = "0*" + self.sigma_xdag # + "*" + self.sigma_bdag + "*" + self.sigma_b + "*" + self.sigma_x  # this will always evaluate to zero for a diamond-shape system
        output_ops = [out_op1, out_op_tau0]
        # at t1, apply sigma_x from left and sigma_xdag from right
        sigma_left = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}
        sigma_right = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        _G2_2, _G22_t1t2 = _rho_ee_ee(output_ops, sigma_left, sigma_right)
        # combine both
        _G2 = _G2_1 + _G2_2
        return t1, _G2, np.trapz(_G2,t1)*self.gamma_e**2, _G2_1, _G2_2, _G21_t1t2+_G22_t1t2
    
    def rho_ll_ll(self, use_second_zero=False):
        """
        same as for EE,EE, just in the second timebin
        """
        return self.rho_ee_ee(add_time=self.tb, use_second_zero=use_second_zero)

    def rho_el_el(self, output_ops=None, sigma_X=None, sigma_Xdag=None):
        """
        calculates G2 assuming an XX-emission triggers the coincidence measurment at time t1, following an X at time t2 in the second time-bin, i.e.:
        <sigma^dagger_XX(t1) sigma^dagger_X(t2+tb) sigma_x(t2+tb) sigma_xx(t1)>
        here, t1 is in the first timebin, and t2 is in the second time-bin, i.e. t1<=tb<t2<=2*tb
        The arguments of the G2 function only overlap at t1=tb & t2=0, so we only have to consider this one special case.
        """
        out_op1 = self.sigma_xdag + "*" + self.sigma_x
        out_op_tau0 = self.sigma_bdag + "*" + self.sigma_xdag + "*" + self.sigma_x + "*" + self.sigma_b
        # the default operators calculate EL,EL
        if output_ops is None:
            output_ops = [out_op1, out_op_tau0]
        # output_ops = [self.x_op, self.b_op]
        # at t1, apply sigma_b from left and sigma_bdag from right
        if sigma_X is None:
            sigma_X = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
        if sigma_Xdag is None:
            sigma_Xdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}

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
                    sigma_b_new = dict(sigma_X)  # must make a copy of the dict
                    sigma_bdag_new = dict(sigma_Xdag)
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
            # futures now contains [t, out_op1, out_op_tau0] for every i
            for i in range(len(t1)):
                # t2 = tb,...,2*tb
                n_t2 = n_tau
                temp_t2 = np.zeros(n_t2+1)
                # futures[i][2] are the out_op_tau0 values , [1] are the out_op1 = x values
                # here, we want the x-values for every t2=tb,..,2tb
                # so take the last n_t2+1 values of the array
                temp_t2[:n_t2+1] = np.abs(futures[i][1][-n_t2-1:])
                # special case tau=0:
                # the time-bins only overlap at t1=tb & t2=0, so we only have to consider 
                # this is the case for i = len(t1)-1 and index -n_t2-1
                if i == len(t1)-1:
                    temp_t2[0] = np.abs(futures[i][2][-n_t2-1])
                t_new = t2[:len(temp_t2)]
                # integrate over t_new
                _G2[i] = np.trapz(temp_t2,t_new)
        return t1, _G2, np.trapz(_G2,t1)*self.gamma_e**2


    def rho_le_le(self):
        """
        calculates G2 assuming an X-emission triggers the coincidence measurment at time t1, following an XX at time t2, i.e.:
        <sigma^dagger_X(t1) sigma^dagger_XX(t2+tb) sigma_XX(t2+tb) sigma_X(t1)>
        here, t1 is in the first timebin, and t2+tb is in the second time-bin, i.e. t1<=tb<t2<=2*tb
        The form of the equations is the same as for EL,EL, so we can use the same function, but we have to change the operators: x->b and b->x
        """
        # the operators to calculate LE,LE
        out_op1 = self.sigma_bdag + "*" + self.sigma_b
        out_op_tau0 = self.sigma_xdag + "*" + self.sigma_bdag + "*" + self.sigma_b + "*" + self.sigma_x
        output_ops = [out_op1, out_op_tau0]
        # at t1, apply sigma_x from left and sigma_xdag from right
        sigma_X = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore": "false"}
        sigma_Xdag = {"operator": self.sigma_xdag ,"applyFrom": "_right", "applyBefore": "false"}

        return self.rho_el_el(output_ops=output_ops, sigma_X=sigma_X, sigma_Xdag=sigma_Xdag)

    # the remaining three ee_xx
    def rho_ee_ll(self, use_second_zero=False):
        """
        correlations between EE and LL states. this includes four different times, but we only have two 'time-axes',
        and one of those always starts at the end of the other. This allows us to calculate the correlation functions
        with a reasonable numerical effort. However, we make one assumption when using the gaussian-density spaced time-axis:
        the pulses in the first and second time-bin arrive (1) at the same time and (2) have the same amplitude.
        More precisely, the pulse in the first time-bin defines the time-axis for the first and second time-bin.
        This is necessary because using "+tb" for the late-timebin operators effectively just shifts the time-axis by one time-bin.
        """
        # case 1: t1 <= t2
        output_ops = [self.sigma_x, self.sigma_x + "*" + self.sigma_b]
        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_xdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
        t1, _G2_1, eell_1, G21_t1t2 = self.four_time(output_ops, sigma_bdag, sigma_xdag, sigma_b)
        if use_second_zero:
            return t1, _G2_1, eell_1, _G2_1, _G2_1*0, G21_t1t2
        # case 2: t2 <= t1
        output_ops = [self.sigma_bdag, self.sigma_b + "*" + self.sigma_x]
        sigma_xdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_x = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}
        _G2_2 = _G2_1 * 0
        eell_2 = eell_1 * 0
        t1, _G2_2, eell_2, G22_t1t2 = self.four_time(output_ops, sigma_xdag, sigma_bdag, sigma_x)
        return t1, _G2_1 + _G2_2, eell_1 + eell_2, _G2_1, _G2_2, G21_t1t2 + G22_t1t2

    def rho_ee_el(self, operators=None):
        output_ops = [self.sigma_x]
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_xdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        if operators is not None:
            if len(operators) != 4:
                raise ValueError("operators must be a list of length 4")
            output_ops = [operators[0]]
            sigma_b = {"operator": operators[1], "applyFrom": "_left", "applyBefore":"false"}
            sigma_bdag = {"operator": operators[2], "applyFrom": "_right", "applyBefore":"false"}
            sigma_xdag = {"operator": operators[3], "applyFrom": "_right", "applyBefore":"false"}
            
        # t1 < t2
        def _part_t1_le_t2():
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
                temp_t2 = np.zeros_like(t2_array, dtype=complex)
                for k in range(0,len(t2_array)):
                    # pgx
                    temp_t2[k] = futures[k][1][-1]
                _G2[i] = np.trapz(temp_t2, t2_array)
            return t1, _G2, np.trapz(_G2, t1)*self.gamma_e**2

        # t2 < t1
        def _part_t2_le_t1():
            # part 2 is different in the ordering of the operators, which leads to the simulation ending on a time depending on the t1-axis instead of t2.
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
                            _t3_end = _t1 + self.tb   # note the difference here

                            sigma_bdag_new = dict(sigma_bdag)
                            sigma_xdag_new = dict(sigma_xdag)
                            sigma_b_new = dict(sigma_b)
                            # add correct times
                            sigma_b_new["time"] = _t2   # and here
                            sigma_bdag_new["time"] = _t2
                            sigma_xdag_new["time"] = _t1

                            # the order of the operators is important to catch the special case where t1=t2
                            # because then ACE applies the operator first, that is first in the parameter file
                            multitime_op_new = [sigma_xdag_new,sigma_b_new,sigma_bdag_new]  # and here
                            _e = executor.submit(self.system,0,_t3_end,multitime_op=multitime_op_new, suffix=j, output_ops=output_ops, **self.options)
                            _e.add_done_callback(lambda f: tq.update())
                            futures.append(_e)

                for k in range(len(futures)):
                    # futures are still 'future' objects
                    futures[k] = futures[k].result()
                # futures now contains t,pgx for every j
                t2_array = t1[i:]  # array for the second time-axis
                temp_t2 = np.zeros_like(t2_array, dtype=complex)
                for k in range(0,len(t2_array)):
                    # pgx
                    temp_t2[k] = futures[k][1][-1]
                _G2[i] = np.trapz(temp_t2, t2_array)
            return t1, _G2, np.trapz(_G2, t1)*self.gamma_e**2

        t1, _G21, eeel_1 = _part_t1_le_t2()
        t1, _G22, eeel_2 = _part_t2_le_t1()
        return t1, _G21 + _G22, eeel_1 + eeel_2, _G21, _G22


    def rho_ee_le(self):
        # exactly like ee_el, but with the operators exchanged X<>B (and part 1<>2)
        output_op = self.sigma_b
        operators = [output_op, self.sigma_x, self.sigma_xdag, self.sigma_bdag]
        return self.rho_ee_el(operators=operators)

    # remaining two el_xx
    def four_time(self, output_ops, sigma_1, sigma_2, sigma_3):
        t1 = self.t1
        _G2 = np.zeros([len(t1)], dtype=complex)
        _G2_t1t2 = np.zeros([len(t1),len(t1)], dtype=complex)
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
                        # j=0 is a special case that has to be addressed
                        _t2 = t1[j+i]
                        _t3_end = _t2 + self.tb
                        sigma_1_new = dict(sigma_1)
                        sigma_2_new = dict(sigma_2)
                        sigma_3_new = dict(sigma_3)
                        # add correct times
                        sigma_1_new["time"] = _t1
                        sigma_2_new["time"] = _t2
                        sigma_3_new["time"] = _t1 + self.tb
                        # the order of the operators is important to catch the special case where t1=t2
                        # because then ACE applies the operator first, that is first in the parameter file
                        multitime_op_new = [sigma_1_new,sigma_2_new,sigma_3_new]
                        _e = executor.submit(self.system,0,_t3_end,multitime_op=multitime_op_new, suffix=j, output_ops=output_ops, **self.options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
            for k in range(len(futures)):
                # futures are still 'future' objects
                futures[k] = futures[k].result()
            # futures now contains t,out_op[1],out_op[2] for every j
            t2_array = t1[i:]  # array for the second time-axis
            temp_t2 = np.zeros_like(t2_array, dtype=complex)
            # j=0 special case
            temp_t2[0] = futures[0][2][-1]
            for k in range(1,len(t2_array)):
                temp_t2[k] = futures[k][1][-1]
            _G2_t1t2[i, -len(temp_t2):] = temp_t2
            _G2[i] = np.trapz(temp_t2, t2_array)
        return t1, _G2, np.trapz(_G2, t1)*self.gamma_e**2, _G2_t1t2

    def rho_el_le(self):
        # case t1 <= t2
        output_ops = [self.sigma_xdag, self.sigma_xdag + "*" + self.sigma_b]
        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_x = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}
        sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
        t1, _G21, elle_1, _ = self.four_time(output_ops, sigma_bdag, sigma_x, sigma_b)

        # case t2 <= t1
        output_ops = [self.sigma_b, self.sigma_xdag + "*" + self.sigma_b]
        sigma_x = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}
        sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
        sigma_xdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        t1, _G22, elle_2, _ = self.four_time(output_ops, sigma_x, sigma_bdag, sigma_xdag)
        return t1, _G21 + _G22, elle_1 + elle_2, _G21, _G22

    def rho_el_ll(self, calc_lell=False):
        # case t1 <= t2
        def _part_t1_le_t2():
            output_ops = [self.sigma_xdag + "*" + self.sigma_x, self.sigma_xdag + "*" + self.sigma_x + "*" + self.sigma_b]
            sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
            sigma_b = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
            if calc_lell:
                output_ops = [self.sigma_bdag + "*" + self.sigma_b, self.sigma_bdag + "*" + self.sigma_b + "*" + self.sigma_x]
                sigma_bdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
                sigma_b = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}
            t1 = self.t1 
            _G2 = np.zeros([len(t1)], dtype=complex)
            n_tau = int((self.tb)/self.dt)
            # simulation time-axis
            t2 = np.linspace(0, self.tb, n_tau + 1)
            # loop over t1
            futures = []
            _t3_end = 2*self.tb  # end of the simulation
            with tqdm.tqdm(total=len(t1), leave=None) as tq:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    for i in tqdm.trange(len(t1),leave=None):
                        # tau1: use the interval 0,...,tb-t1
                        # i.e., if t1 = 0,...,tx then tau1 expands to absolute times of tx,...,tb
                        _t1 = t1[i]
                        _t2 = _t1 + self.tb
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
            for i in range(len(futures)):
                # futures are still 'future' objects
                futures[i] = futures[i].result()
                # futures now contains t,x,pxb for every j
            for i in range(len(t1)):
                # t2 = t1,...,tb
                n_t2 = n_tau - int((t1[i])/self.dt)
                temp_t2 = np.zeros(n_t2+1, dtype=complex)
                # special case tau=0:
                # which is the value with index [-(n_tau+1)] with the second output operator 
                temp_t2[0] = futures[i][2][-(n_t2+1)]
                # futures[i][2] are the values of the second output operators for tau0, [1] are the values of the first output operator
                # here, we want the values of the first output operator for every t2=t1,..,tb
                if n_t2 > 0: 
                    temp_t2[1:n_t2+1] = futures[i][1][-n_t2:]
                t_new = t2[:len(temp_t2)]
                # integrate over t_new
                _G2[i] = np.trapz(temp_t2,t_new)
            return t1, _G2, np.trapz(_G2, t1)*self.gamma_e**2

        # case t2 <= t1
        def _part_t2_le_t1():
            output_ops = [self.sigma_b, self.sigma_xdag + "*" + self.sigma_b + "*" + self.sigma_x]
            sigma_bdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}
            sigma_x = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}
            sigma_xdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
            if calc_lell:
                output_ops = [self.sigma_x, self.sigma_bdag + "*" + self.sigma_x + "*" + self.sigma_b]
                sigma_bdag = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
                sigma_x = {"operator": self.sigma_b, "applyFrom": "_left", "applyBefore":"false"}
                sigma_xdag = {"operator": self.sigma_bdag, "applyFrom": "_right", "applyBefore":"false"}

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
                            # j=0 is a special case that has to be addressed
                            _t2 = t1[j+i]
                            _t3_end = _t2 + self.tb

                            sigma_bdag_new = dict(sigma_bdag)
                            sigma_x_new = dict(sigma_x)
                            sigma_xdag_new = dict(sigma_xdag)
                            # add correct times
                            sigma_bdag_new["time"] = _t2
                            sigma_x_new["time"] = _t1 + self.tb
                            sigma_xdag_new["time"] = _t1 + self.tb
                            # the order of the operators is important to catch the special case where t1=t2
                            # because then ACE applies the operator first, that is first in the parameter file
                            multitime_op_new = [sigma_bdag_new,sigma_x_new,sigma_xdag_new]
                            _e = executor.submit(self.system,0,_t3_end,multitime_op=multitime_op_new, suffix=j, output_ops=output_ops, **self.options)
                            _e.add_done_callback(lambda f: tq.update())
                            futures.append(_e)
                for k in range(len(futures)):
                    # futures are still 'future' objects
                    futures[k] = futures[k].result()
                # futures now contains t,out_op[1],out_op[2] for every j
                t2_array = t1[i:]  # array for the second time-axis
                temp_t2 = np.zeros_like(t2_array, dtype=complex)
                # j=0 special case
                temp_t2[0] = futures[0][2][-1]
                for k in range(1,len(t2_array)):
                    temp_t2[k] = futures[k][1][-1]
                _G2[i] = np.trapz(temp_t2, t2_array)
            return t1, _G2, np.trapz(_G2, t1)*self.gamma_e**2
        
        t1, _G21, elle_1 = _part_t1_le_t2()
        t1, _G22, elle_2 = _part_t2_le_t1()
        return t1, _G21 + _G22, elle_1 + elle_2, _G21, _G22

    # the remaining le_ll
    def rho_le_ll(self):
        # this is very similar to EL,LL but with operators exchanged
        return self.rho_el_ll(calc_lell=True)
    