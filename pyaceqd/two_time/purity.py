# calculate purity of single-photon source
# this bascically compares the peaks of the two-time correlation function
# G2(tau=0) with the peak of the two-time correlation function G2(tau=T_pulse)
# where T_pulse is the separation of pulses in the pulse train
# this means that the simulation needs to span at least 2*T_pulse, i.e., 3 pulses in the pulse train
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, ProcessPoolExecutor
from pyaceqd.tools import construct_t, simple_t_gaussian, export_csv, calc_tl_dynmap_pseudo, extract_dms, op_to_matrix
from pyaceqd.timebin.timebin import TimeBin
from pyaceqd.pulses import PulseTrain, ChirpedPulse
from pyaceqd.two_level_system.tls import tls
from pyaceqd.two_time.correlations import tl_two_op_two_time, tl_three_op_two_time
import propagate_tau_module
import time
# import warnings
# warnings.filterwarnings('error', category=np.ComplexWarning)

class Purity(TimeBin):
    def __init__(self, system, sigma_x, sigma_xdag, *pulses, dt=0.1, tb=800, dt_small=0.1, simple_exp=True, gaussian_t=None, verbose=False, workers=15, t_simul=None, options={}, factor_t=1, factor_tau=2) -> None:
        pulse = PulseTrain(tb, 5, *pulses)
        self.factor_t = factor_t
        self.factor_tau = factor_tau
        super().__init__(system, pulse, dt=dt, tb=tb, simple_exp=simple_exp, gaussian_t=gaussian_t, verbose=verbose, workers=workers, t_simul=t_simul, options=options)
        self.sigma_x = "(" + sigma_x + ")"
        self.sigma_xdag = "(" + sigma_xdag + ")"
        
        try:
            self.gamma_e = options["gamma_e"]
        except KeyError:
            print("gamma_e not included in options, setting to 100")
            self.options["gamma_e"] = 100
            self.gamma_e = self.options["gamma_e"]
        if self.gaussian_t is not None:
            self.t1 = simple_t_gaussian(0,self.gaussian_t,self.tb,dt_small,10*dt_small,*pulses,decimals=1,exp_part=self.simple_exp)
            # _t = np.concatenate((_t,self.tb-_t))
            # sort and remove duplicates
            # _t = np.sort(np.unique(_t))
            # self.t1 = _t
            # plt.clf()
            # plt.scatter(self.t1, np.zeros_like(self.t1))
            # plt.savefig("t1.png")
            # plt.clf()
        else:
            self.t1 = construct_t(0, self.tb, dt_small, 10*dt_small, *pulses, simple_exp=self.simple_exp)
        # complete t-axis, when t1 is repeated for factor_t > 1
        t_axis_complete = np.array([])
        for i in range(factor_t):
            # TODO: 
            # maybe we need t_axis_complete[:-1] so the last value is not repeated, if we use factor_t > 1
            # usually we only use factor_t=1, so this is not a problem yet
            t_axis_complete = np.concatenate((t_axis_complete, self.t1 + i*self.tb))
        self.t_axis_complete = t_axis_complete
        # compatibility with tls, which needs no polarization
        self.options["pulse_file_x"] = self.pulse_file_x
        self.options["pulse_file_y"] = self.pulse_file_y
        # print(self.options)

    def prepare_pulsefile(self, verbose=False, t_simul=None, plot=False):
        # override prepare_pulsefile from TimeBin
        # because we need a different t_end, and also use a PulseTrain
        t_end = (self.factor_t + self.factor_tau +1)*self.tb
        if t_simul is not None:
            t_end = t_simul
        _n_t = int(t_end/self.dt) + 1  # number of time steps
        _t_pulse = np.linspace(0,t_end,_n_t)
        # different polarizations
        self.pulse_file_x = self.temp_dir + "twotime_pulse_x_{}.dat".format(id(self))
        self.pulse_file_y = self.temp_dir + "twotime_pulse_y_{}.dat".format(id(self))
        pulse_x = np.zeros_like(_t_pulse, dtype=complex)
        pulse_y = np.zeros_like(_t_pulse, dtype=complex)
        pulse_x, pulse_y = self.pulses[0].get_total_xy(_t_pulse)
        export_csv(self.pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ', verbose=verbose)
        export_csv(self.pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ', verbose=verbose)
        if plot:
            plt.clf()
            plt.plot(_t_pulse, pulse_x.real)
            plt.xlabel("t")
            plt.ylabel("E_x")
            plt.savefig("pulsetrain.png")
            plt.clf()

    def calc_timedynamics(self, output_ops=None):
        new_options = dict(self.options)
        if output_ops is not None:
            new_options["output_ops"] = output_ops
        t_end = (self.factor_t + self.factor_tau + 1)*self.tb
        return self.system(0, t_end, *self.pulses, **new_options)

    def G2(self, return_whole=False):
        sigma_left = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}
        sigma_right = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        
        out_op1 = self.sigma_xdag + "*" + self.sigma_x
        out_op_tau0 = self.sigma_xdag + "*" + self.sigma_xdag + "*" + self.sigma_x + "*" + self.sigma_x
        output_ops = [out_op1, out_op_tau0]
        t1 = self.t1
        factor_t = self.factor_t
        t_axis_complete = self.t_axis_complete
        factor_tau = self.factor_tau
        n_tau = factor_tau*int(self.tb/self.dt)
        t2 = np.linspace(0, factor_tau*self.tb, n_tau + 1)
        _G2 = np.zeros([factor_t*len(t1), len(t2)])
        with tqdm.tqdm(total=factor_t*len(t1), leave=None) as tq:
            for i in range(factor_t):
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    futures = []
                    for j in range(len(t1)):
                        tend = i*self.tb + t1[j] + factor_tau*self.tb
                        sigma_X_new = dict(sigma_left)
                        sigma_Xdag_new = dict(sigma_right)
                        sigma_X_new["time"] = i*self.tb + t1[j]
                        sigma_Xdag_new["time"] = i*self.tb + t1[ j]
                        multitime_ops = [sigma_X_new, sigma_Xdag_new]
                        _e = executor.submit(self.system, 0, tend, multitime_op=multitime_ops, suffix=j, output_ops=output_ops, **self.options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
                    wait(futures)
                for j in range(len(t1)):
                    futures[j] = futures[j].result()
                for j in range(len(t1)):
                    _G2[j+i*len(t1),1:] = np.abs(futures[j][1][-(n_tau):])
                    # special case tau=0:
                    _G2[j+i*len(t1),0] = np.abs(futures[j][2][-(n_tau+1)])
        # integrate over t1
        G2 = np.trapz(_G2, t_axis_complete, axis=0)
        if return_whole:
            return t1, t2, _G2
        return t2, G2
    
    def calc_purity(self):
        t,g2 = self.G2()
        dt = self.dt
        tb = self.tb
        n_1 = int(0.5*tb/dt)
        G21 = 2*np.trapz(g2[:n_1], t[:n_1])
        G22 = np.trapz(g2[n_1:3*n_1], t[n_1:3*n_1])
        return 1-G21/G22

class Indistinguishability(Purity):
    def __init__(self, system, sigma_x, sigma_xdag, *pulses, dt=0.1, tb=800, dt_small=0.1, simple_exp=True, gaussian_t=None, verbose=False, workers=15, t_simul=None, options={}, dm=False, sigma_x_mat=None, sigma_xdag_mat=None, t_mem=10) -> None:
        self.pulses = pulses
        self.dm = dm
        self.tl_map = None
        self.tl_dms = None
        self.t_mem = t_mem  # memory time for phonon dynamics with time-local maps
        self.sigma_x_mat = sigma_x_mat
        self.sigma_xdag_mat = sigma_xdag_mat
        if sigma_x_mat is None or sigma_xdag_mat is None:
            print("WARNING: sigma_x_mat or sigma_xdag_mat not provided, trying to convert sigma_x and sigma_xdag to matrices")
            self.sigma_x_mat = op_to_matrix(sigma_x)
            self.sigma_xdag_mat = op_to_matrix(sigma_xdag)
        self.dim = self.sigma_x_mat.shape[0]  # dimension of the Hilbert space
        super().__init__(system, sigma_x, sigma_xdag, *pulses, dt=dt, tb=tb, dt_small=dt_small, simple_exp=simple_exp, gaussian_t=gaussian_t, verbose=verbose, workers=workers, t_simul=t_simul, options=options)

    def G1(self):
        sigma_x = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}
 
        out_op1 = self.sigma_xdag
        out_op_tau0 = self.sigma_xdag + "*" + self.sigma_x
        output_ops = [out_op1, out_op_tau0]
        t1 = self.t1
        factor_t = self.factor_t
        t_axis_complete = self.t_axis_complete
        factor_tau = self.factor_tau
        n_tau = factor_tau*int(self.tb/self.dt)
        t2 = np.linspace(0, factor_tau*self.tb, n_tau + 1)
        _G1 = np.zeros([factor_t*len(t1), len(t2)], dtype=complex)
        with tqdm.tqdm(total=factor_t*len(t1), leave=None) as tq:
            for i in range(factor_t):
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    futures = []
                    for j in range(len(t1)):
                        tend = i*self.tb + t1[j] + factor_tau*self.tb
                        sigma_X_new = dict(sigma_x)
                        sigma_X_new["time"] = i*self.tb + t1[j]
                        multitime_ops = [sigma_X_new]
                        _e = executor.submit(self.system, 0, tend, multitime_op=multitime_ops, suffix=j, output_ops=output_ops, **self.options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
                    wait(futures)
                for j in range(len(t1)):
                    futures[j] = futures[j].result()
                for j in range(len(t1)):
                    _G1[j+i*len(t1),1:] = futures[j][1][-(n_tau):]
                    # special case tau=0:
                    _G1[j+i*len(t1),0] = futures[j][2][-(n_tau+1)]
        # plot _G1
        # plt.clf()
        # plt.pcolormesh(t2, t_axis_complete, np.abs(_G1)**2)
        # plt.xlabel("tau")
        # plt.ylabel("t1")
        # plt.savefig("G1.png")
        # plt.clf()
        # integrate over t1
        # return t2, self.t_axis_complete, _G1
        G1 = np.trapz(np.abs(_G1)**2, t_axis_complete, axis=0)
        return t2, G1
    
    def simple_propagation(self, return_whole=False):
        # most importantly, in all calculations, the same factor_t, factor_tau and tb must be used
        output_ops = [self.sigma_xdag + "*" + self.sigma_x]
        factor_tau = self.factor_tau
        # print(self.t_axis_complete[-1])
        tend = (self.factor_t + factor_tau)*self.tb
        n_tau = factor_tau*int(self.tb/self.dt)
        t2 = np.linspace(0, factor_tau*self.tb, n_tau + 1)
        t, val = self.system(0, tend, suffix=-1, output_ops=output_ops, **self.options)
        val = np.abs(val)
        # <x(t)>*<x(t+tau)>
        t1 = np.linspace(0, self.factor_t*self.tb, int((self.factor_t*self.tb)/self.dt) + 1)
        
        G0_tau = np.zeros(len(t2))  # Only allocate final result array
        for j in range(len(t2)):
            # Create temporary view of shifted values
            val_shifted = val[j:j+len(t1)]
            # Calculate product for this slice directly
            product = val[:len(val_shifted)] * val_shifted
            # Integrate this slice
            G0_tau[j] = np.trapz(product, t1[:len(val_shifted)])
        
        # G0 = np.zeros([len(t1), len(t2)])
        # # the following loop can efficiently implemented using numpy
        # # for i in tqdm.trange(len(t1)):
        # #     for j in range(len(t2)):
        # #         G0[i,j] = val[i]*val[i+j]
        # # efficient implementation
        # i_indices, j_indices = np.ogrid[:len(t1), :len(t2)]
        # G0 = val[i_indices] * val[i_indices + j_indices]
        # # integrate over t1
        # G0_tau = np.trapz(G0, t1, axis=0)
        # if return_whole:
        #     return t1, t2, G0
        return t2, G0_tau
    
    def simple_propagation_tl(self, return_whole=False):
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()
        factors = self.factor_t + self.factor_tau
        len_tb = int(self.tb/self.dt)
        # complete t axis for the simulation
        t_total = np.linspace(0, factors*self.tb, factors*len_tb + 1)

        n_tau = self.factor_tau*int(self.tb/self.dt)
        # tau axis for result
        t2 = np.linspace(0, self.factor_tau*self.tb, n_tau + 1) 
        # t axis for result
        t1 = np.linspace(0, self.factor_t*self.tb, int((self.factor_t*self.tb)/self.dt) + 1)

        rho0 = np.zeros((self.dim,self.dim), dtype=complex)
        rho0[0,0] = 1  # initial state, rho0 = |0><0|
        rho_t = np.ones((len(t_total), self.dim**2), dtype=complex)
        rho_t[0] = rho0.reshape(4)  # initial state, rho0 = |0><0|
        rho_t[-1] = rho0.reshape(4)  # final state, rho0 = |0><0| just for convenience, will be overwritten
        for j in range(factors):
            # from 0 to len_tb-1, we have the pulses
            # do this in each time bin
            for i in range(1,len(self.tl_dms)):
                rho_t[i+j*len_tb] = np.dot(self.tl_dms[i-1], rho_t[i-1+j*len_tb])
            # now apply the time-local dynamical map
            for i in range(len(self.tl_dms),len_tb+1):
                rho_t[i+j*len_tb] = np.dot(self.tl_map, rho_t[i-1+j*len_tb])
        
        val = np.zeros_like(t_total)
        op = self.sigma_xdag_mat @ self.sigma_x_mat
        # val = np.einsum('ij,tji->t', op, rho_t.reshape(len(t_total), self.dim, self.dim))  # calculate <x(t)> for each time step
        for i in range(len(t_total)):
            val[i] = np.real(np.trace(op@rho_t[i].reshape((self.dim, self.dim))))

        # G0 = np.zeros([len(t1), len(t2)])
        # i_indices, j_indices = np.ogrid[:len(t1), :len(t2)]
        # G0 = val[i_indices] * val[i_indices + j_indices]
        # # integrate over t1
        # G0_tau = np.trapz(G0, t1, axis=0)

        # More memory efficient version using views
        # this is in principle similar as calculating
        # an autocorrelation with a sliding window of length len(t1)
        G0_tau = np.zeros(len(t2))  # Only allocate final result array
        for j in range(len(t2)):
            # Create temporary view of shifted values
            val_shifted = val[j:j+len(t1)]
            # Calculate product for this slice directly
            product = val[:len(val_shifted)] * val_shifted
            # Integrate this slice
            G0_tau[j] = np.trapz(product, t1[:len(val_shifted)])
        return t2, G0_tau
    
    def simple_propagation_tl_phonons(self, return_whole=False):
        tl_map, dms = self.get_tl_phonons(mtos=[], t_mtos=[])
        dms_sep1 = dms[0]

        factors = self.factor_t + self.factor_tau
        len_tb = int(self.tb/self.dt)
        # complete t axis for the simulation
        t_total = np.linspace(0, factors*self.tb, factors*len_tb + 1)

        n_tau = self.factor_tau*int(self.tb/self.dt)
        # tau axis for result
        t2 = np.linspace(0, self.factor_tau*self.tb, n_tau + 1) 
        # t axis for result
        t1 = np.linspace(0, self.factor_t*self.tb, int((self.factor_t*self.tb)/self.dt) + 1)

        rho0 = np.zeros((self.dim,self.dim), dtype=complex)
        rho0[0,0] = 1  # initial state, rho0 = |0><0|
        rho_t = np.ones((len(t_total), self.dim**2), dtype=complex)
        rho_t[0] = rho0.reshape(4)  # initial state, rho0 = |0><0|
        rho_t[-1] = rho0.reshape(4)  # final state, rho0 = |0><0| just for convenience, will be overwritten
        for j in range(factors):
            # from 0 to len_tb-1, we have the pulses
            # do this in each time bin
            for i in range(1,len(dms_sep1)):
                rho_t[i+j*len_tb] = np.dot(dms_sep1[i-1], rho_t[i-1+j*len_tb])
            # now apply the time-local dynamical map
            for i in range(len(dms_sep1),len_tb+1):
                rho_t[i+j*len_tb] = np.dot(tl_map, rho_t[i-1+j*len_tb])
        
        val = np.zeros_like(t_total)
        op = self.sigma_xdag_mat @ self.sigma_x_mat
        # calculate <x(t)> for each time step
        for i in range(len(t_total)):
            val[i] = np.real(np.trace(op@rho_t[i].reshape((self.dim, self.dim))))

        G0_tau = np.zeros(len(t2))  # Only allocate final result array
        for j in range(len(t2)):
            # Create temporary view of shifted values
            val_shifted = val[j:j+len(t1)]
            # Calculate product for this slice directly
            product = val[:len(val_shifted)] * val_shifted
            # Integrate this slice
            G0_tau[j] = np.trapz(product, t1[:len(val_shifted)])
        return t2, G0_tau

    def get_tl(self, t_mem=None):
        if t_mem is None:
            t_mem = self.gaussian_t
        if t_mem is None:
            t_mem = self.tb/2
        tend = 2*t_mem
        # _options = self.options.copy()
        # _options["pulse_file_x"] = self.pulse_file_x
        # _options["pulse_file_y"] = self.pulse_file_y
       
        result, dm = self.system(0, tend, multitime_op=[], calc_dynmap=True, **self.options)
        _t = result[0]  # time axis for getting the dynamic maps
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        memory_time = self.gaussian_t if self.gaussian_t is not None else self.tb
        tl_map, dms = extract_dms(dm_tl, _t, memory_time, t_MTOs=[])

        self.tl_map = tl_map
        self.tl_dms = dms[0]

    def get_tl_phonons(self, mtos=[], t_mtos=[]):
        tmem = self.gaussian_t + self.t_mem
        tend = 2.1*tmem
        result, dm = self.system(0, tend, multitime_op=mtos, calc_dynmap=True, **self.options)
        _t = result[0]  # time axis for getting the dynamic maps
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        tl_map, dms = extract_dms(dm_tl, _t, tmem, t_MTOs=t_mtos)
        dms = np.array(dms, dtype=complex)
        return tl_map, dms

    def calc_timedynamics_tl_phonons(self, output_ops=None):
        tl_map, dms = self.get_tl_phonons(mtos=[], t_mtos=[])
        dm_sep1 = dms[0]

        factors = self.factor_t + self.factor_tau
        len_tb = int(self.tb/self.dt)
        t_total = np.linspace(0, factors*self.tb, factors*len_tb + 1)
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_t = np.ones((len(t_total), 4), dtype=complex)
        rho_t[0] = rho0.reshape(4)  # initial state, rho0 = |0><0|
        rho_t[-1] = rho0.reshape(4)  # final state, rho0 = |0><0|
        for j in range(factors):
            # from 0 to len_tb-1, we have the pulses
            # do this in each time bin
            for i in range(1,len(dm_sep1)):
                rho_t[i+j*len_tb] = np.dot(dm_sep1[i-1], rho_t[i-1+j*len_tb])
            # now apply the time-local dynamical map
            for i in range(len(dm_sep1),len_tb+1):
                rho_t[i+j*len_tb] = np.dot(tl_map, rho_t[i-1+j*len_tb])
        return t_total, rho_t.reshape((len(t_total), 2, 2))

    def calc_timedynamics_tl(self):
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()
        factors = self.factor_t + self.factor_tau
        len_tb = int(self.tb/self.dt)
        t_total = np.linspace(0, factors*self.tb, factors*len_tb + 1)
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_t = np.ones((len(t_total), 4), dtype=complex)
        rho_t[0] = rho0.reshape(4)  # initial state, rho0 = |0><0|
        rho_t[-1] = rho0.reshape(4)  # final state, rho0 = |0><0|
        self.tl_complete = np.zeros((len(t_total)-1, 4, 4), dtype=complex)
        for j in range(factors):
            # from 0 to len_tb-1, we have the pulses
            # do this in each time bin
            # print(t_total[j*len_tb])
            for i in range(1,len(self.tl_dms)):
                rho_t[i+j*len_tb] = np.dot(self.tl_dms[i-1], rho_t[i-1+j*len_tb])
                self.tl_complete[i+j*len_tb-1] = self.tl_dms[i-1]
            # now apply the time-local dynamical map
            for i in range(len(self.tl_dms),len_tb+1):
                rho_t[i+j*len_tb] = np.dot(self.tl_map, rho_t[i-1+j*len_tb])
                self.tl_complete[i+j*len_tb-1] = self.tl_map
        return t_total, rho_t.reshape((len(t_total), 2, 2))  
    
    def get_dm2_phonons(self, mtos, t_mto, suffix=1):
        mtos_new = []
        for mto in mtos:
            mto_new = mto.copy()
            mto_new["time"] = t_mto
            mtos_new.append(mto_new)
        result, dm = self.system(0, t_mto + self.gaussian_t + self.t_mem + 2*self.dt, multitime_op=mtos_new, calc_dynmap=True,suffix=suffix, **self.options)
        _t = result[0]  # time axis for getting the dynamic maps
        _t = np.round(_t, 6)  # round to 6 digits
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        _, dms = extract_dms(dm_tl, _t, self.gaussian_t + self.t_mem, t_MTOs=[t_mto])
        return dms[1]  # return the second dynamic map, which is the one we need for the phonon dynamics

    def get_dm2_phonons_advanced(self, mtos, t_mto, suffix=1):
        # in principle, we don't have to calculate the tl-maps up until t_mto + t_gaussian + self.t_mem + 2*self.dt,
        # but the correct final time depends on t_mto, as gaussian_t is absolute and t_mem is always needed. 
        # this means the maximum final time will be t_gaussian + 2 * t_mem
        # while t_mto is t_gaussian + t_mem, meaning before we used a maximum of 2*t_gaussian + 2*t_mem.
        mtos_new = []
        for mto in mtos:
            mto_new = mto.copy()
            mto_new["time"] = t_mto
            mtos_new.append(mto_new)
        t_end = self.gaussian_t + 2 * self.t_mem + 2*self.dt
        result, dm = self.system(0, t_end, multitime_op=mtos_new, calc_dynmap=True, suffix=suffix, **self.options)
        _t = result[0]  # time axis for getting the dynamic maps
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        # extracting the dynmaps is now a bit different, as we have to take into account the reduced
        # number of time steps, the non-stationary 'steps' in each local map will be different:
        # for t_mto = 0, the memory time will be t_gaussian + t_mem,
        # for t_mto = 1, memory time = t_gaussian - 1 + t_mem, 
        # for t_mto = t_gaussian, memory time = t_mem
        # from then on, it is always t_mem, as this is the minimum memory time we need
        memory_time = np.max([self.gaussian_t + self.t_mem - t_mto, self.t_mem])
        _, dms = extract_dms(dm_tl, _t, memory_time, t_MTOs=[t_mto])
        return dms[1]  # return the second dynamic map, which is the one we need for the phonon dynamics

    def G1_tl_phonons(self):
        t_apply = self.gaussian_t + self.t_mem + 5*self.dt
        _mto = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore": "false", "time": t_apply}
        tl_map, dms_sep = self.get_tl_phonons(mtos=[_mto], t_mtos=[t_apply])

        dim = np.shape(self.sigma_x_mat)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0

        tau_max=self.tb*self.factor_tau
        n_tau = int(tau_max/self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)

        t_mem_indices = np.where(self.t1 <= (self.gaussian_t + self.t_mem))[0]
        # print(f"t_mem_indices: {t_mem_indices}, t_mem: {self.t_mem}, gaussian_t: {self.gaussian_t}")
        # print(self.t1[t_mem_indices])
        # calc tl maps:
        # always let the maps have the same shape as the dms_sep[0], which is computed using a memory time of
        # self.gaussian_t + self.t_mem.
        # In principle, the shape of the time-dependent dynamical maps is a little smaller,
        # but we need to pass a 'nice' array to fortran. We just fill the rest with
        # the time-local map.  
        dms_tauc2 = np.zeros((len(t_mem_indices), *np.shape(dms_sep[0])), dtype=complex)
        dms_tauc2[:,:] = tl_map

        with tqdm.tqdm(total=len(t_mem_indices), leave=None) as tq:
            futures = []
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for i in range(len(t_mem_indices)):
                    _t_mto = self.t1[i]
                    _e = executor.submit(self.get_dm2_phonons_advanced,[_mto], _t_mto, i)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
            wait(futures)
            for i in range(len(t_mem_indices)):
                dm_part = futures[i].result()
                _len_part = np.shape(dm_part)[0]
                dms_tauc2[i,:_len_part] = dm_part

        
        # for i in tqdm.trange(len(t_mem_indices)):
        #     _t_mto = self.t1[i]
        #     dms_tauc2[i] = self.get_dm2_phonons([_mto], _t_mto)
            # _t_mto = self.t1[i]
            # # calculate the dynamic map for this time
            # mto = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore": "false", "time": _t_mto}
            # result, dm = self.system(0, _t_mto + self.gaussian_t+self.t_mem+2*self.dt, multitime_op=[mto], calc_dynmap=True, **self.options)
            # if i == 40:
            #     temp_res = result
            # _t = result[0]  # time axis for getting the dynamic maps
            # _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
            # dm_tl = calc_tl_dynmap_pseudo(dm, _t)
            # _, dms = extract_dms(dm_tl, _t, self.gaussian_t+self.t_mem, t_MTOs=[_t_mto])
            # dms_tauc2[i] = dms[1]

        # with tqdm.tqdm(total=len(t_mem_indices), leave=None) as tq:
        #     futures = []
        #     with ThreadPoolExecutor(max_workers=self.workers) as executor:
        #         for i in range(len(t_mem_indices)):
        #             _t_mto = self.t1[i]
        #             _e = executor.submit(self.get_dm2_phonons,[_mto], _t_mto, i)
        #             _e.add_done_callback(lambda f: tq.update())
        #             futures.append(_e)
        #     wait(futures)
        #     for i in range(len(t_mem_indices)):
        #         dms_tauc2[i] = futures[i].result()

        # print("dms_tauc shape:", dms_tauc2.shape)
        dm_taucs2 = np.asfortranarray(dms_tauc2.transpose(2, 3, 0, 1))
        dm_separated1 = np.asfortranarray(dms_sep[0].transpose(1, 2, 0))
        dm_separated2 = np.asfortranarray(dms_sep[1].transpose(1, 2, 0))
        dm_s = tl_map

        _tend = self.t_axis_complete[-1] + tau_max
        # the 'simulation' time axis. The time axis for the two-time correlation function
        # is self.t_axis_complete, which contains less time points so we need less propagations.
        t_axis = np.linspace(0, _tend, int(_tend/self.dt) + 1)

        opA_mat = np.identity(dim)
        opB_mat = self.sigma_xdag_mat
        opC_mat = self.sigma_x_mat

        # print("operators:")
        # print(opA_mat)
        # print(opB_mat)
        # print(opC_mat)

        # print(opB_mat@opC_mat)

        G1 = propagate_tau_module.calc_twotime_phonon_block(dm_taucs2=dm_taucs2, dm_sep1=dm_separated1, dm_sep2=dm_separated2, dm_s=dm_s,
                                                            rho_init=rho0.reshape(dim**2),n_tb=int(self.tb/self.dt),nx_tau=self.factor_tau,
                                                            dim=dim,opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t_axis_complete)
        # i = 40
        # results = np.zeros((len(tau)), dtype=complex)
        # j = 0
        # n_tb=int(self.tb/self.dt)
        # rho_t = rho0.copy().reshape(dim**2)
        # n_map = np.shape(dms_sep[0])[0]
        # while t_axis[j] <= self.t_axis_complete[i]:
        #     if j < n_map:
        #         rho_t = dms_sep[0, j] @ rho_t
        #     else:
        #         rho_t = dm_s @ rho_t
        #     j += 1
        # results[0] = np.trace(opA_mat @ opB_mat @ opC_mat @ rho_t.reshape(dim, dim))

        # j = 0
        # use_dm2 = True
        # print(dms_tauc2[0,0])
        # for k in range(1, len(tau)):
        #     if j < n_map:
        #         if use_dm2:
        #             rho_t = dms_tauc2[i,j] @ rho_t
        #         else:
        #             rho_t = dms_sep[0, j] @ rho_t
        #     else:
        #         rho_t = dm_s @ rho_t
        #     results[k] = np.trace(opB_mat @ rho_t.reshape(dim, dim))
        #     j += 1
        #     if (j == n_tb):
        #         j = 0
        #         use_dm2 = False

        # # result_f, result_rho = propagate_tau_module.calc_single_i(t_axis=t_axis,t_axis_complete=self.t_axis_complete,rho0=rho0.reshape(dim**2),dm_sep1=dms_sep[0],dm_s=dm_s,dm_tauc2=dms_tauc2[i],dim=dim,n_tb=n_tb,n_tau=len(tau),opa=opA_mat.T,opb=opB_mat.T,opc=opC_mat.T)
        # # G1=0
        # return tau, results, temp_res, G1# , result_f,result_rho


        
        #return tau, self.t_axis_complete, G1
        g1 = np.trapz(np.abs(G1)**2, self.t_axis_complete, axis=0)
        return tau, g1
    
    def G2_tl_phonons(self):
        t_apply = self.gaussian_t + self.t_mem + 5*self.dt
        _mto = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore": "false", "time": t_apply}
        _mto2 = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore": "false", "time": t_apply}
        tl_map, dms_sep = self.get_tl_phonons(mtos=[_mto,_mto2], t_mtos=[t_apply])

        dim = np.shape(self.sigma_x_mat)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0

        tau_max=self.tb*self.factor_tau
        n_tau = int(tau_max/self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)

        t_mem_indices = np.where(self.t1 <= (self.gaussian_t + self.t_mem))[0]
        # calc tl maps:
        dms_tauc2 = np.zeros((len(t_mem_indices), *np.shape(dms_sep[0])), dtype=complex)
        dms_tauc2[:,:] = tl_map

        with tqdm.tqdm(total=len(t_mem_indices), leave=None) as tq:
            futures = []
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for i in range(len(t_mem_indices)):
                    _t_mto = self.t1[i]
                    _e = executor.submit(self.get_dm2_phonons_advanced,[_mto, _mto2], _t_mto, i)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
            wait(futures)
            for i in range(len(t_mem_indices)):
                dm_part = futures[i].result()
                _len_part = np.shape(dm_part)[0]
                dms_tauc2[i,:_len_part] = dm_part
        
        # with tqdm.tqdm(total=len(t_mem_indices), leave=None) as tq:
        #     futures = []
        #     with ThreadPoolExecutor(max_workers=self.workers) as executor:
        #         for i in range(len(t_mem_indices)):
        #             _t_mto = self.t1[i]
        #             _e = executor.submit(self.get_dm2_phonons,[_mto, _mto2], _t_mto, i)
        #             _e.add_done_callback(lambda f: tq.update())
        #             futures.append(_e)
        #     wait(futures)
        #     for i in range(len(t_mem_indices)):
        #         dms_tauc2[i] = futures[i].result()

        dm_taucs2 = np.asfortranarray(dms_tauc2.transpose(2, 3, 0, 1))
        dm_separated1 = np.asfortranarray(dms_sep[0].transpose(1, 2, 0))
        dm_separated2 = np.asfortranarray(dms_sep[1].transpose(1, 2, 0))
        dm_s = tl_map

        _tend = self.t_axis_complete[-1] + tau_max
        # the 'simulation' time axis. The time axis for the two-time correlation function
        # is self.t_axis_complete, which contains less time points so we need less propagations.
        t_axis = np.linspace(0, _tend, int(_tend/self.dt) + 1)

        opA_mat = self.sigma_xdag_mat
        opB_mat = self.sigma_xdag_mat @ self.sigma_x_mat
        opC_mat = self.sigma_x_mat
        
        G2 = propagate_tau_module.calc_twotime_phonon_block(dm_taucs2=dm_taucs2, dm_sep1=dm_separated1, dm_sep2=dm_separated2, dm_s=dm_s,
                                                            rho_init=rho0.reshape(dim**2),n_tb=int(self.tb/self.dt),nx_tau=self.factor_tau,
                                                            dim=dim,opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t_axis_complete)
        g2 = np.trapz(np.abs(G2), self.t_axis_complete, axis=0)
        return tau, g2

    def G2_tl(self):
        dim = np.shape(self.sigma_x_mat)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0  # initial state, rho0 = |0><0|

        tau_max=self.tb*self.factor_tau
        n_tau = int(tau_max/self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)

        # calc and prepare tl maps:
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()
        dm_tl = np.asfortranarray(self.tl_dms.transpose(1, 2, 0))
        dm_s = self.tl_map

        _tend = self.t_axis_complete[-1] + tau_max
        # the 'simulation' time axis. The time axis for the two-time correlation function
        # is self.t_axis_complete, which contains less time points so we need less propagations.
        t_axis = np.linspace(0, _tend, int(_tend/self.dt) + 1)
        # multi-time operators as matrices
        opA_mat = self.sigma_xdag_mat
        opC_mat = self.sigma_x_mat
        opB_mat = opA_mat @ opC_mat
        #start_time = time.time()
        # print(propagate_tau_module.__doc__)
        G2 = propagate_tau_module.calc_onetime_parallel_block(dm_block=dm_tl,dm_s=dm_s,rho_init=rho0.reshape(dim**2),n_tb=int(self.tb/self.dt),nx_tau=self.factor_tau,dim=dim,opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t_axis_complete)
        #end_time = time.time()
        #print(f"Time taken for tl_two_op_two_time with dm: {end_time - start_time:.2f} seconds")
        g2 = np.trapz(np.abs(G2), self.t_axis_complete, axis=0)
        return tau, g2
    
    def G1_tl(self):
        dim = np.shape(self.sigma_x_mat)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0  # initial state, rho0 = |0><0|

        tau_max=self.tb*self.factor_tau
        n_tau = int(tau_max/self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)

        # calc tl maps:
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()
        dm_tl = np.asfortranarray(self.tl_dms.transpose(1, 2, 0))
        dm_s = self.tl_map

        _tend = self.t_axis_complete[-1] + tau_max
        t_axis = np.linspace(0, _tend, int(_tend/self.dt) + 1)
        opA_mat = np.identity(dim)
        opB_mat = self.sigma_xdag_mat
        opC_mat = self.sigma_x_mat
        #start_time = time.time()
        # print(propagate_tau_module.__doc__)
        G1 = propagate_tau_module.calc_onetime_parallel_block(dm_block=dm_tl,dm_s=dm_s,rho_init=rho0.reshape(dim**2),n_tb=int(self.tb/self.dt),nx_tau=self.factor_tau,dim=dim,opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t_axis_complete)
        #end_time = time.time()
        #print(f"Time taken for tl_two_op_two_time with dm: {end_time - start_time:.2f} seconds")
        g1 = np.trapz(np.abs(G1)**2, self.t_axis_complete, axis=0)
        return tau, g1

    def calc_indistinguishability(self):
        """
        returns indistinguishability,single-photon purity
        """
        # calculate G0, G1 and G2
        # and integrate over tau=0,...,tb/2 and tb/2,...,3tb/2
        if self.dm:
            if self.options["phonons"]:
                print("Calculating with phonons")
                t,g1 = self.G1_tl_phonons()
            else:
                t,g1 = self.G1_tl()
        else:
            t,g1 = self.G1()
        dt = self.dt
        tb = self.tb
        n_1 = int(0.5*tb/dt)
        G11 = 2*np.trapz(g1[:n_1], t[:n_1])
        G12 = np.trapz(g1[n_1:3*n_1], t[n_1:3*n_1])
        # print("G11", G11, "G12", G12)

        if self.dm:
            if self.options["phonons"]:
                t2,g2 = self.G2_tl_phonons()
            else:
                t2,g2 = self.G2_tl()
        else:
            t2,g2 = self.G2()
        G21 = 2*np.trapz(g2[:n_1], t2[:n_1])
        G22 = np.trapz(g2[n_1:3*n_1], t2[n_1:3*n_1])
        # print("G21", G21, "G22", G22)

        if self.dm:
            if self.options["phonons"]:
                t0,g0 = self.simple_propagation_tl_phonons()
            else:
                t0,g0 = self.simple_propagation_tl()
        else:
            t0,g0 = self.simple_propagation()
        # special, integrate 0,...,tb and tb,...,2tb
        # n_2 = int(tb/dt)
        G01 = 2*np.trapz(g0[:n_1], t0[:n_1])
        G02 = np.trapz(g0[n_1:3*n_1], t0[n_1:3*n_1])
        # print("G01", G01, "G02", G02)

        result = (G01-G11+G21)/(G02-G12+G22)
        return 1 - result, 1-G21/G22

# tau = 10
# t0_n = 4.5
# p1 = ChirpedPulse(tau_0=tau, e_start=0, alpha=0, t0=t0_n*tau, e0=1, polar_x=1)
# options = {"verbose": False, "gamma_e": 1/100, "lindblad": True,
#  "temp_dir": '/mnt/temp_data/', "phonons": False}

# def resample(x, y, z, s_x, s_y):
#     x_new = np.zeros(int((len(x))/s_x))
#     y_new = np.zeros(int((len(y))/s_y))
#     z_new = np.zeros((len(y_new),len(x_new)))
#     for i in range(len(x_new)):
#         for j in range(len(y_new)):
#             x_new[i] = x[int(i*s_x)]
#             y_new[j] = y[int(j*s_y)]
#             z_new[j,i] = z[int(j*s_y),int(i*s_x)]
#     return x_new, y_new, z_new

# # a = Purity(tls, "|0><1|_2", "|1><0|_2", p1, dt=0.1, tb=100, simple_exp=False, gaussian_t=None, verbose=False, workers=15, t_simul=None, options=options, factor_t=1, factor_tau=1)
# # t1,t2,g2 = a.G2(return_whole=True)
# # print(t1)
# # print(t2)
# # plt.clf()
# # plt.pcolormesh(t2, t1, np.abs(g2)**2)
# # plt.xlabel("tau")
# # plt.ylabel("t1")
# # plt.colorbar()
# # plt.savefig("g2.png")
# # plt.clf()

# sigma_x = op_to_matrix("|0><1|_2")
# sigma_xdag = op_to_matrix("|1><0|_2")
# a = Indistinguishability(tls, "|0><1|_2", "|1><0|_2", p1, dt=0.1, tb=2000, simple_exp=False, gaussian_t=2*t0_n*tau, verbose=False, workers=15, t_simul=None, options=options, dm=True,
#                          sigma_x_mat=sigma_x, sigma_xdag_mat=sigma_xdag)
# a.factor_tau = 2
# # print(a.calc_indistinguishability())
# # t11,g11 = a.G1()
# t12,g12 = a.G1_tl()
# # t13,g13,g10_1 = a.G1_tl_new()
# # t14,g14,g10_2 = a.G1_tl_newest()
# # # print("G1", t11.shape, g11.shape)
# # # print("G1_tl", t12.shape, g12.shape)
# # # print("G1_tl_new", t13.shape, g13.shape)
# plt.clf()
# # # plt.plot(t11,np.abs(g11), label="G1")
# plt.plot(t12,np.abs(g12), label="G1_tl")
# # plt.plot(t13,np.abs(g13), label="G1_tl_new")
# # plt.plot(t14,np.abs(g14), label="G1_tl_newest")
# # # plt.plot(a.t_axis_complete, np.abs(g10_1), label="G10_1")
# # # plt.plot(a.t_axis_complete, np.abs(g10_2), dashes=[2,2], label="G10_2")
# plt.xlabel("tau")
# plt.ylabel("G1")
# plt.legend()
# # plt.xlim(0, 300)
# plt.savefig("g1.png")

# # t21,g21 = a.G2()
# # t22,g22 = a.G2_tl()
# t23,g23 = a.G2_tl()
# # t24,g24,G20_2 = a.G2_tl_newest()
# plt.clf()
# # # plt.plot(t21,np.abs(g21), label="G2")
# # # plt.plot(t22,np.abs(g22), label="G2_tl")
# plt.plot(t23,np.abs(g23), label="G2_tl_new")
# # plt.plot(t24,np.abs(g24),dashes=[2,2], label="G2_tl_newest")
# # plt.plot(a.t_axis_complete, np.abs(G20_1), label="G20_1")
# # plt.plot(a.t_axis_complete, np.abs(G20_2), dashes=[2,2], label="G20_2")
# plt.xlabel("tau")
# plt.ylabel("G2")
# plt.legend()
# # plt.xlim(2995, 3005)
# plt.savefig("g2.png")

# # start = time.time()
# # t,g0 = a.simple_propagation()
# # end = time.time()
# # print(f"Time taken for simple_propagation: {end - start:.2f} seconds")
# start = time.time()
# t2,g02 = a.simple_propagation_tl()
# end = time.time()
# print(f"Time taken for simple_propagation_tl: {end - start:.2f} seconds")
# plt.clf()   
# # plt.plot(t,np.abs(g0), label="G0")
# plt.plot(t2,np.abs(g02), linestyle='dashed', label="G0_tl")
# plt.xlabel("t")
# plt.ylabel("G0")
# # plt.xlim(1500-10, 1500+10)
# plt.legend()
# plt.savefig("g0.png")

# # t,x = a.simple_propagation()
# t,x = a.calc_timedynamics(output_ops=["|1><1|_2"])
# t2, rho = a.calc_timedynamics_tl()
# x2 = rho[:,1,1]  # extract the population of state |1>
# plt.clf()
# plt.plot(t.real,x.real)
# # print(t[int(1500/0.1)+2])
# plt.plot(t2.real,x2.real, linestyle='dashed')
# # plt.plot(t,a.pulses[0].get_total(t-a.tb)/np.max(a.pulses[0].get_total(t)), label="pulse_shifted")
# # plt.plot(t,a.pulses[0].get_total(t)/np.max(a.pulses[0].get_total(t)), linestyle='dashed', label="pulse")
# # print(t2[int(1500/0.1)+2])
# # plt.xlim(10+0+1500*2, 10+20+1500*2)
# # # plt.plot(t[int(2000/0.1):]-2000,x[int(2000/0.1):])
# plt.savefig("x_train.png")



# plt.clf()
# t,g2 = a.G2()
# t2,xtau = a.simple_propagation(return_whole=False)
# t1,t2,g0 = a.simple_propagation(return_whole=True)
# print(t1.shape, t2.shape, g0.shape)
# plt.clf()
# plt.pcolormesh(*resample(t2, t1, g0, 10, 20))
# plt.xlim(0,150)
# plt.ylim(0,150)
# plt.savefig("xx_g0_1pi.png")
# tau,g0 = a.simple_propagation()
# np.save("g0_train_1pi.npy", g0)
# np.save("t_g0_train_1pi.npy", tau)
# t1,t2,g2 = a.G2(return_whole=True)
# tau,g1 = a.G1()
# np.save("g1_train_1pi.npy", g1)
# np.save("t1_g1_train_1pi.npy", tau)
# g1 = np.load("g1_train_1pi.npy")
# tau = np.load("t1_g1_train_1pi.npy")
# plt.clf()
# plt.plot(tau,g1)
# plt.xlabel("tau")
# plt.savefig("g1_train_1pi.png")

# plt.clf()
# plt.plot(t,g2)
# plt.savefig("g2_train_1pi.png")
# np.save("g2_train_1pi.npy", g2)
# np.save("t_train_1pi.npy", t)
# plt.clf()
# plt.plot(t,g2)
# plt.xlim(1900,2100)
# plt.savefig("g2_train_zoom_1pi.png")
# plt.clf()
# plt.pcolormesh(tau, t, np.abs(g2)**2)
# plt.xlabel("tau")
# plt.ylabel("t1")
# plt.colorbar()
# plt.savefig("g2_2d.png")
# np.save("g2_tau_1pi.npy", g2)
# np.save("t_tau_1pi.npy", t)
# # dt = 0.1
# # tb = 2000
# # n_1 = int(0.5*tb/dt)

# t = np.load("t_tau_1pi.npy")
# g2 = np.load("g2_tau_1pi.npy")

# # print(t[:n_1])
# # print(t[n_1:3*n_1])
# G21 = 2*np.trapz(g2[:n_1], t[:n_1])
# G22 = np.trapz(g2[n_1:3*n_1], t[n_1:3*n_1])
# print(G21)
# print(G22)
# print(1-G21/G22)
# plt.clf()
# plt.plot(t,g2,"b-")
# plt.plot(-t,g2,"b-")
# # plt.xlim(-200,200)
# plt.xlabel("tau")
# plt.ylabel("G2")
# plt.savefig("g2_tau_1pi.png")

# t,g1 = a.G1()
# np.save("g1_train2.npy", g1)
# np.save("tg1_train2.npy", t)

# g1 = np.load("g1_train2.npy")
# t = np.load("tg1_train2.npy")
# plt.clf()
# plt.plot(t,np.abs(g1))
# # plt.plot(t,np.real(g1),linestyle='dashed', label="real")
# # plt.plot(t,np.imag(g1),linestyle='dashed', label="imag")
# plt.legend()
# plt.xlabel("tau")
# plt.ylabel("G1")
# plt.savefig("g1_train2.png")


# options = {"verbose": False, "gamma_e": 1/100, "lindblad": True,
#  "temp_dir": '/mnt/temp_data/', "phonons": False}
# dtaus = 0.5
# tau_max = 20
# tau_min = 2
# sigma_x = op_to_matrix("|0><1|_2")
# sigma_xdag = op_to_matrix("|1><0|_2")
# n_tau = int((tau_max-tau_min)/dtaus)
# taus = np.linspace(tau_min, tau_max, n_tau + 1)
# indists = np.zeros(len(taus))
# purities = np.zeros(len(taus))
# t0_n = 4.5
# for i in tqdm.trange(len(taus)):
#     tau = taus[i]
#     p1 = ChirpedPulse(tau_0=tau, e_start=0, alpha=0, t0=t0_n*tau, e0=1, polar_x=1)
#     a = Indistinguishability(tls, "|0><1|_2", "|1><0|_2", p1, dt=0.1, tb=2000, simple_exp=False, gaussian_t=2*t0_n*tau, verbose=False, workers=15, t_simul=None, options=options, dm=True,
#                          sigma_x_mat=sigma_x, sigma_xdag_mat=sigma_xdag)
#     indists[i], purities[i] = a.calc_indistinguishability()

# plt.clf()
# plt.plot(taus, indists, label="indistinguishability")
# plt.xlabel("tau")
# plt.ylabel("indistinguishability")
# plt.savefig("indistinguishability.png")
# plt.clf()

# plt.plot(taus, purities, label="purity")
# plt.xlabel("tau")
# plt.ylabel("purity")
# plt.savefig("purity.png")
# plt.clf()

# tau = 20
# p1 = ChirpedPulse(tau_0=tau, e_start=0, alpha=0, t0=t0_n*tau, e0=1, polar_x=1)
# a = Indistinguishability(tls, "|0><1|_2", "|1><0|_2", p1, dt=0.1, tb=2000, simple_exp=True, gaussian_t=2*t0_n*tau, verbose=False, workers=15, t_simul=None, options=options, dm=False,
#                          sigma_x_mat=sigma_x, sigma_xdag_mat=sigma_xdag)

# print(a.calc_indistinguishability())