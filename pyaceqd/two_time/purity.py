# calculate purity of single-photon source
# this bascically compares the peaks of the two-time correlation function
# G2(tau=0) with the peak of the two-time correlation function G2(tau=T_pulse)
# where T_pulse is the separation of pulses in the pulse train
# this means that the simulation needs to span at least 2*T_pulse, i.e., 3 pulses in the pulse train
import numpy as np
import tqdm
from concurrent.futures import ThreadPoolExecutor, wait
from pyaceqd.helpers.dynamical_map import calc_tl_dynmap_pseudo, extract_dms
from pyaceqd.helpers.time_axes import UnregularTimeAxis, round_to_dt
import warnings
try:
    from pyaceqd.two_time import propagate_tau_module
except ImportError as exc:
    _PROPAGATE_TAU_TL_IMPORT_ERROR = exc
    warnings.warn("propagate_tau_module not found, time-local acceleration unavailable.",
                  ImportWarning,
                  stacklevel=2,)
    propagate_tau_module = None
import time
import pyaceqd.constants as constants

def _require_propagate_tau_tl() -> None:
    if propagate_tau_module is None:
        raise RuntimeError(
            "The optional Fortran module 'propagate_tau_module' is not available. "
            "Reinstall with Fortran build enabled to use time-local accelerated routines."
        ) from _PROPAGATE_TAU_TL_IMPORT_ERROR

temp_dir = constants.temp_dir

def get_max_pulse_t(pulses):
    times = []
    for p in pulses:
        times.append(p.t0 + 4*p.tau)
    return max(times)


class Indistinguishability:
    def __init__(self, system, sigma_x, sigma_xdag, *pulses, tb=800, t_mem=10, dt_small=0.1,
                 regular_stepping=False, variable_stepping=False, 
                 exponential_stepping=False, max_pulse_t=None, verbose=False, workers=15, t_simul=None, 
                 factor_t=1, factor_tau=2, dt_big=None, add_tend=True, use_dm=True):
        self.pulses = pulses
        self.system = system
        self.factor_t = factor_t
        self.factor_tau = factor_tau
        self.sigma_x = sigma_x
        self.sigma_xdag = sigma_xdag
        self.use_dm = use_dm
        self.max_pulse_t = max_pulse_t
        self.tb = tb
        self.verbose = verbose
        self.workers = workers
        if self.max_pulse_t is None:
            self.max_pulse_t = get_max_pulse_t(pulses)
        # print("Max pulse time for purity calculation: ", self.max_pulse_t)

        self.tl_map = None
        self.tl_dms = None
        self.t_mem = t_mem  # memory time for phonon dynamics with time-local maps
        
        self.dim = system.dim
        self.dt = system.dt
        if self.dt > dt_small:
            raise ValueError("dt_small for purity calculation cannot be smaller than system dt: {} > {}".format(dt_small, self.dt))
        if np.mod(dt_small, self.dt) != 0:
            raise ValueError("dt_small for purity calculation must be a multiple of system dt: {} % {} != 0".format(dt_small, self.dt))
        self.phonons = system.phonons
        if not system.lindblad:
            print("WARNING: system is not using lindblad operators, eg. no decay.")
        if dt_big is None:
                dt_big = 10*dt_small

        # mto checks
        if sigma_x.shape != (self.dim,self.dim) or sigma_xdag.shape != (self.dim,self.dim):
            raise ValueError("Sigma_x or sigma_xdag operator are dims: {}, {}, but system dim is {}x{}".format(sigma_x.shape, sigma_xdag.shape, self.dim, self.dim))

        time_generator = UnregularTimeAxis(0, tb, self.max_pulse_t + self.t_mem, dt_small=dt_small, dt_big=dt_big, pulses=pulses, factor_tau=factor_tau, include_tend=add_tend, round_dt=True)

        self.t1 = time_generator.time_axis_two_step(exponential_part=exponential_stepping)
        if variable_stepping:
            variable_dt_max = dt_big
            self.t1 = time_generator.time_axis_variable(exponential_part=exponential_stepping, dt_big_variable=variable_dt_max)
        if regular_stepping:
            self.t1 = time_generator.time_axis_regular()
        
        # pre-calculate pulses on the simulation time axis
        # simulation_time_generator = UnregularTimeAxis(0, (factor_t+factor_tau)*tb, dt_small=self.dt)
        # self.t_simul = simulation_time_generator.time_axis_regular()
        # self.evaluated_pulses = []
        # for p in pulses:
        #     self.evaluated_pulses.append(p.get_total_dict(self.t_simul))
        
        # complete t-axis, when t1 is repeated for factor_t > 1
        t_axis_complete = np.array([])
        for i in range(factor_t):
            # TODO: 
            # maybe we need t_axis_complete[:-1] so the last value is not repeated, if we use factor_t > 1
            # usually we only use factor_t=1, so this is not a problem yet
            t_axis_complete = np.concatenate((t_axis_complete, self.t1 + i*self.tb))
        self.t_axis_complete = t_axis_complete

        if self.verbose:
            print("t1 axis for purity calculation: ", len(self.t1), " points")
            print("complete t axis for purity calculation: ", len(self.t_axis_complete), " points")
            n_tau = self.factor_tau*int(self.tb/self.dt)
            print("tau axis for purity calculation: ", n_tau + 1, " points")
            print("Memory usage estimate: {:.2f} MB (complex128 assumed).".format((n_tau + 1)*16/(1024*1024)))

    def calc_timedynamics(self, output_ops=None, t_end=None):
        if t_end is None:
            t_end = (self.factor_t + self.factor_tau + 1)*self.tb
        return self.system.run(0, t_end, *self.pulses, output_ops=output_ops)

    def G2(self, return_whole=False, tqdm_options={}):
        sigma_left = {"operator": self.sigma_x, "applyFrom": "left", "applyBefore":False}
        sigma_right = {"operator": self.sigma_xdag, "applyFrom": "right", "applyBefore":False}
        
        out_op1 = self.sigma_xdag @ self.sigma_x
        out_op_tau0 = self.sigma_xdag @ self.sigma_xdag @ self.sigma_x @ self.sigma_x
        output_ops = [out_op1, out_op_tau0]
        t1 = self.t1
        factor_t = self.factor_t
        t_axis_complete = self.t_axis_complete
        factor_tau = self.factor_tau
        n_tau = factor_tau*int(self.tb/self.dt)
        t2 = np.linspace(0, factor_tau*self.tb, n_tau + 1)
        _G2 = np.zeros([factor_t*len(t1), len(t2)])
        with tqdm.tqdm(total=factor_t*len(t1), leave=None, **tqdm_options) as tq:
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
                        _e = executor.submit(self.system.run, 0, tend, *self.pulses, multitime_op=multitime_ops, output_ops=output_ops)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
                    wait(futures)
                for j in range(len(t1)):
                    _,futures[j] = futures[j].result()
                for j in range(len(t1)):
                    _G2[j+i*len(t1),1:] = np.abs(futures[j][0][-(n_tau):])
                    # special case tau=0:
                    _G2[j+i*len(t1),0] = np.abs(futures[j][1][-(n_tau+1)])
        if return_whole:
            return t1, t2, _G2
        # integrate over t1
        G2 = np.trapezoid(_G2, t_axis_complete, axis=0)
        return t2, G2
    
    def calc_purity(self):
        if self.use_dm:
            if self.system.phonons:
                print("Calculating with phonons")
                t,g2 = self.G2_tl_phonons()
            else:
                t,g2 = self.G2_tl()
        else:
            t,g2 = self.G2()
        dt = self.dt
        tb = self.tb
        n_1 = int(0.5*tb/dt)
        G21 = 2*np.trapezoid(g2[:n_1], t[:n_1])
        G22 = np.trapezoid(g2[n_1:3*n_1], t[n_1:3*n_1])
        return 1-G21/G22

    def G1(self):
        sigma_x = {"operator": self.sigma_x, "applyFrom": "left", "applyBefore":False}
 
        out_op1 = self.sigma_xdag
        out_op_tau0 = self.sigma_xdag @ self.sigma_x
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
                        _e = executor.submit(self.system.run, 0, tend, *self.pulses, multitime_op=multitime_ops, output_ops=output_ops)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
                    wait(futures)
                for j in range(len(t1)):
                    _,futures[j] = futures[j].result()
                for j in range(len(t1)):
                    _G1[j+i*len(t1),1:] = futures[j][0][-(n_tau):]
                    # special case tau=0:
                    _G1[j+i*len(t1),0] = futures[j][1][-(n_tau+1)]
        # plot _G1
        # plt.clf()
        # plt.pcolormesh(t2, t_axis_complete, np.abs(_G1)**2)
        # plt.xlabel("tau")
        # plt.ylabel("t1")
        # plt.savefig("G1.png")
        # plt.clf()
        # integrate over t1
        # return t2, self.t_axis_complete, _G1
        G1 = np.trapezoid(np.abs(_G1)**2, t_axis_complete, axis=0)
        return t2, G1
    
    def simple_propagation(self, return_whole=False):
        # most importantly, in all calculations, the same factor_t, factor_tau and tb must be used
        output_ops = [self.sigma_xdag @ self.sigma_x]
        factor_tau = self.factor_tau
        # print(self.t_axis_complete[-1])
        tend = (self.factor_t + factor_tau)*self.tb
        n_tau = factor_tau*int(self.tb/self.dt)
        t2 = np.linspace(0, factor_tau*self.tb, n_tau + 1)
        t, val = self.system.run(0, tend, *self.pulses, output_ops=output_ops)
        val = np.squeeze(val)
        val = np.abs(val)
        # <x(t)>*<x(t+tau)>
        t1 = np.linspace(0, self.factor_t*self.tb, int((self.factor_t*self.tb)/self.dt) + 1)
        
        # Use optimized Fortran implementation if available
        # if propagate_tau_module is not None:
        #     G0_tau = propagate_tau_module.sliding_window_correlation(
        #         val=np.ascontiguousarray(val, dtype=np.float64),
        #         time_t1=np.ascontiguousarray(t1, dtype=np.float64),
        #         n_t1=len(t1),
        #         n_tau=len(t2)-1
        #     )
        # else:
            # Fallback to Python implementation
        G0_tau = np.zeros(len(t2))  # Only allocate final result array
        for j in range(len(t2)):
            # Create temporary view of shifted values
            val_shifted = val[j:j+len(t1)]
            # if len(val_shifted) != len(t1):
            #     print(len(val_shifted), len(t1))
            # Calculate product for this slice directly
            product = val[:len(val_shifted)] * val_shifted
            # Integrate this slice
            G0_tau[j] = np.trapezoid(product, t1[:len(val_shifted)])
        return t2, G0_tau
    
    def simple_propagation_tl(self):
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
        
        # Use optimized Fortran implementation if available
        if propagate_tau_module is not None:
            dm_block = np.asfortranarray(self.tl_dms.transpose(1, 2, 0))
            op = self.sigma_xdag @ self.sigma_x
            val = propagate_tau_module.apply_dynamical_maps_and_trace(
                dm_block=dm_block,
                dm_s=self.tl_map,
                rho_init=rho0.reshape(self.dim**2),
                n_map=len(self.tl_dms),
                n_tb=len_tb,
                n_factors=factors,
                dim=self.dim,
                op=op
            )
        else:
            # Fallback to Python implementation
            rho_t = np.ones((len(t_total), self.dim**2), dtype=complex)
            rho_t[0] = rho0.reshape(self.dim**2)
            rho_t[-1] = rho0.reshape(self.dim**2)
            for j in range(factors):
                # from 0 to len_tb-1, we have the pulses
                # do this in each time bin
                for i in range(1,len(self.tl_dms)):
                    rho_t[i+j*len_tb] = np.dot(self.tl_dms[i-1], rho_t[i-1+j*len_tb])
                # now apply the time-local dynamical map
                for i in range(len(self.tl_dms),len_tb+1):
                    rho_t[i+j*len_tb] = np.dot(self.tl_map, rho_t[i-1+j*len_tb])
            
            val = np.zeros_like(t_total)
            op = self.sigma_xdag @ self.sigma_x
            # val = np.einsum('ij,tji->t', op, rho_t.reshape(len(t_total), self.dim, self.dim))  # calculate <x(t)> for each time step
            for i in range(len(t_total)):
                val[i] = np.real(np.trace(op@rho_t[i].reshape((self.dim, self.dim))))
        # calculate <x(t)>*<x(t+tau)>, integrated over t
        # Use optimized Fortran implementation if available
        if propagate_tau_module is not None:
            G0_tau = propagate_tau_module.sliding_window_correlation(
                val=np.ascontiguousarray(val, dtype=np.float64),
                time_t1=np.ascontiguousarray(t1, dtype=np.float64),
                n_t1=len(t1),
                n_tau=len(t2)-1
            )
        else:
            # Fallback to Python implementation
            G0_tau = np.zeros(len(t2))
            for j in range(len(t2)):  # loop over tau
                val_shifted = val[j:j+len(t1)]  # x(t+tau)
                product = val[:len(val_shifted)] * val_shifted
                G0_tau[j] = np.trapezoid(product, t1[:len(val_shifted)])
        return t2, G0_tau
    
    def simple_propagation_tl_phonons(self):
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
        
        # Use optimized Fortran implementation if available
        if propagate_tau_module is not None:
            dm_block = np.asfortranarray(dms_sep1.transpose(1, 2, 0))
            op = self.sigma_xdag @ self.sigma_x
            val = propagate_tau_module.apply_dynamical_maps_and_trace(
                dm_block=dm_block,
                dm_s=tl_map,
                rho_init=rho0.reshape(self.dim**2),
                n_map=len(dms_sep1),
                n_tb=len_tb,
                n_factors=factors,
                dim=self.dim,
                op=op
            )
        else:
            # Fallback to Python implementation
            rho_t = np.ones((len(t_total), self.dim**2), dtype=complex)
            rho_t[0] = rho0.reshape(self.dim**2)
            rho_t[-1] = rho0.reshape(self.dim**2)
            for j in range(factors):
                # from 0 to len_tb-1, we have the pulses
                # do this in each time bin
                for i in range(1,len(dms_sep1)):
                    rho_t[i+j*len_tb] = np.dot(dms_sep1[i-1], rho_t[i-1+j*len_tb])
                # now apply the time-local dynamical map
                for i in range(len(dms_sep1),len_tb+1):
                    rho_t[i+j*len_tb] = np.dot(tl_map, rho_t[i-1+j*len_tb])
            
            val = np.zeros_like(t_total)
            op = self.sigma_xdag @ self.sigma_x
            # calculate <x(t)> for each time step
            for i in range(len(t_total)):
                val[i] = np.real(np.trace(op@rho_t[i].reshape((self.dim, self.dim))))

        # Use optimized Fortran implementation if available
        if propagate_tau_module is not None:
            G0_tau = propagate_tau_module.sliding_window_correlation(
                val=np.ascontiguousarray(val, dtype=np.float64),
                time_t1=np.ascontiguousarray(t1, dtype=np.float64),
                n_t1=len(t1),
                n_tau=len(t2)-1
            )
        else:
            # Fallback to Python implementation
            G0_tau = np.zeros(len(t2))
            for j in range(len(t2)):
                val_shifted = val[j:j+len(t1)]
                product = val[:len(val_shifted)] * val_shifted
                G0_tau[j] = np.trapezoid(product, t1[:len(val_shifted)])
        return t2, G0_tau

    def get_tl(self, t_mem=None):
        if t_mem is None:
            t_mem = self.max_pulse_t
        tend = 2*t_mem
       
        _t, dm = self.system.run(0, tend, *self.pulses, multitime_op=[], calc_dynmap=True)
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        memory_time = self.max_pulse_t if self.max_pulse_t is not None else self.tb
        tl_map, dms = extract_dms(dm_tl, _t, memory_time, t_MTOs=[])
        self.tl_map = tl_map
        self.tl_dms = dms[0]

    def get_tl_phonons(self, mtos=[], t_mtos=[]):
        tmem = self.max_pulse_t + self.t_mem
        tend = 2.1*tmem
        _t, dm = self.system.run(0, tend, *self.pulses, multitime_op=mtos, calc_dynmap=True)
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        tl_map, dms = extract_dms(dm_tl, _t, tmem, t_MTOs=t_mtos)
        dms = np.array(dms, dtype=complex)
        return tl_map, dms

    def calc_timedynamics_tl_phonons(self):
        tl_map, dms = self.get_tl_phonons(mtos=[], t_mtos=[])
        dm_sep1 = dms[0]

        factors = self.factor_t + self.factor_tau
        len_tb = int(self.tb/self.dt)
        t_total = np.linspace(0, factors*self.tb, factors*len_tb + 1)
        rho0 = np.zeros((self.dim,self.dim), dtype=complex)
        rho0[0,0] = 1  # initial state, rho0 = |0><0|
        # rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_t = np.ones((len(t_total), self.dim**2), dtype=complex)
        rho_t[0] = rho0.reshape(self.dim**2)  # initial state, rho0 = |0><0|
        rho_t[-1] = rho0.reshape(self.dim**2)  # final state, rho0 = |0><0|
        for j in range(factors):
            # from 0 to len_tb-1, we have the pulses
            # do this in each time bin
            for i in range(1,len(dm_sep1)):
                rho_t[i+j*len_tb] = np.dot(dm_sep1[i-1], rho_t[i-1+j*len_tb])
            # now apply the time-local dynamical map
            for i in range(len(dm_sep1),len_tb+1):
                rho_t[i+j*len_tb] = np.dot(tl_map, rho_t[i-1+j*len_tb])
        return t_total, rho_t.reshape((len(t_total), self.dim, self.dim))

    def calc_timedynamics_tl(self):
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()
        factors = self.factor_t + self.factor_tau
        len_tb = int(self.tb/self.dt)
        t_total = np.linspace(0, factors*self.tb, factors*len_tb + 1)
        rho0 = np.zeros((self.dim,self.dim), dtype=complex)
        rho0[0,0] = 1  # initial state, rho0 = |0><0|
        rho_t = np.ones((len(t_total), self.dim**2), dtype=complex)
        rho_t[0] = rho0.reshape(self.dim**2)  # initial state, rho0 = |0><0|
        rho_t[-1] = rho0.reshape(self.dim**2)  # final state, rho0 = |0><0|
        self.tl_complete = np.zeros((len(t_total)-1, self.dim**2, self.dim**2), dtype=complex)
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
        return t_total, rho_t.reshape((len(t_total), self.dim, self.dim))  

    def get_dm2_phonons_advanced(self, mtos, t_mto):
        # in principle, we don't have to calculate the tl-maps up until t_mto + pulse_max_t + self.t_mem + 2*self.dt,
        # but the correct final time depends on t_mto, as max_pulse_t is absolute and t_mem is always needed. 
        # this means the maximum final time will be pulse_max_t + 2 * t_mem
        # while t_mto is pulse_max_t + t_mem, meaning before we used a maximum of 2*pulse_max_t + 2*t_mem.
        mtos_new = []
        for mto in mtos:
            mto_new = mto.copy()
            mto_new["time"] = t_mto
            mtos_new.append(mto_new)
        t_end = self.max_pulse_t + 2 * self.t_mem + 2*self.dt
        _t, dm = self.system.run(0, t_end, *self.pulses, multitime_op=mtos_new, calc_dynmap=True)
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        # extracting the dynmaps is now a bit different, as we have to take into account the reduced
        # number of time steps, the non-stationary 'steps' in each local map will be different:
        # for t_mto = 0, the memory time will be pulse_max_t + t_mem,
        # for t_mto = 1, memory time = pulse_max_t - 1 + t_mem, 
        # for t_mto = pulse_max_t, memory time = t_mem
        # from then on, it is always t_mem, as this is the minimum memory time we need
        memory_time = np.max([self.max_pulse_t + self.t_mem - t_mto, self.t_mem])
        _, dms = extract_dms(dm_tl, _t, memory_time, t_MTOs=[t_mto])
        return dms[1]  # return the second dynamic map, which is the one we need for the phonon dynamics

    def G1_tl_phonons(self):
        _require_propagate_tau_tl()
        t_apply = np.round(round_to_dt(self.max_pulse_t + self.t_mem + 5*self.dt, self.dt),6)
        _mto = {"operator": self.sigma_x, "applyFrom": "left", "applyBefore": False, "time": t_apply}
        tl_map, dms_sep = self.get_tl_phonons(mtos=[_mto], t_mtos=[t_apply])

        dim = np.shape(self.sigma_x)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0

        tau_max=self.tb*self.factor_tau
        n_tau = int(tau_max/self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)

        t_mem_indices = np.where(self.t1 <= (self.max_pulse_t + self.t_mem))[0]
        # calc tl maps:
        # always let the maps have the same shape as the dms_sep[0], which is computed using a memory time of
        # self.max_pulse_t + self.t_mem.
        # In principle, the shape of the time-dependent dynamical maps is a little smaller,
        # but we need to pass a 'nice' array to fortran. We just fill the rest with
        # the time-local map.  
        dms_tauc2 = np.zeros((len(t_mem_indices), *np.shape(dms_sep[0])), dtype=complex)
        dms_tauc2[:,:] = tl_map

        with tqdm.tqdm(total=len(t_mem_indices), leave=None) as tq:
            futures = []
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for i in range(len(t_mem_indices)):
                    _t_mto = np.round(self.t1[i],6)
                    _e = executor.submit(self.get_dm2_phonons_advanced,[_mto], _t_mto)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
            wait(futures)
            for i in range(len(t_mem_indices)):
                dm_part = futures[i].result()
                _len_part = np.shape(dm_part)[0]
                dms_tauc2[i,:_len_part] = dm_part

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
        opB_mat = self.sigma_xdag
        opC_mat = self.sigma_x

        g1 = propagate_tau_module.calc_twotime_phonon_block(dm_taucs2=dm_taucs2, dm_sep1=dm_separated1, dm_sep2=dm_separated2, dm_s=dm_s,
                                                            rho_init=rho0.reshape(dim**2),n_tb=int(self.tb/self.dt),nx_tau=self.factor_tau,
                                                            dim=dim,opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t_axis_complete,
                                                            exponent=2.0)
        return tau, g1
    
    def G2_tl_phonons(self):
        _require_propagate_tau_tl()
        t_apply = np.round(round_to_dt(self.max_pulse_t + self.t_mem + 5*self.dt, self.dt),6)
        _mto = {"operator": self.sigma_x, "applyFrom": "left", "applyBefore": False, "time": t_apply}
        _mto2 = {"operator": self.sigma_xdag, "applyFrom": "right", "applyBefore": False, "time": t_apply}
        tl_map, dms_sep = self.get_tl_phonons(mtos=[_mto,_mto2], t_mtos=[t_apply])

        dim = np.shape(self.sigma_x)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0

        tau_max=self.tb*self.factor_tau
        n_tau = int(tau_max/self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)

        t_mem_indices = np.where(self.t1 <= (self.max_pulse_t + self.t_mem))[0]
        # calc tl maps:
        dms_tauc2 = np.zeros((len(t_mem_indices), *np.shape(dms_sep[0])), dtype=complex)
        dms_tauc2[:,:] = tl_map

        with tqdm.tqdm(total=len(t_mem_indices), leave=None) as tq:
            futures = []
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for i in range(len(t_mem_indices)):
                    _t_mto = np.round(self.t1[i],6)
                    _e = executor.submit(self.get_dm2_phonons_advanced,[_mto, _mto2], _t_mto)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
            wait(futures)
            for i in range(len(t_mem_indices)):
                dm_part = futures[i].result()
                _len_part = np.shape(dm_part)[0]
                dms_tauc2[i,:_len_part] = dm_part

        dm_taucs2 = np.asfortranarray(dms_tauc2.transpose(2, 3, 0, 1))
        dm_separated1 = np.asfortranarray(dms_sep[0].transpose(1, 2, 0))  # dm from t=0 to t=t_mem
        dm_separated2 = np.asfortranarray(dms_sep[1].transpose(1, 2, 0))  # dm from t_apply to t_apply + t_mem
        dm_s = tl_map

        _tend = self.t_axis_complete[-1] + tau_max
        # the 'simulation' time axis. The time axis for the two-time correlation function
        # is self.t_axis_complete, which contains less time points so we need less propagations.
        t_axis = np.linspace(0, _tend, int(_tend/self.dt) + 1)

        opA_mat = self.sigma_xdag
        opB_mat = self.sigma_xdag @ self.sigma_x
        opC_mat = self.sigma_x
        
        g2 = propagate_tau_module.calc_twotime_phonon_block(dm_taucs2=dm_taucs2, dm_sep1=dm_separated1, dm_sep2=dm_separated2, dm_s=dm_s,
                                                            rho_init=rho0.reshape(dim**2),n_tb=int(self.tb/self.dt),nx_tau=self.factor_tau,
                                                            dim=dim,opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t_axis_complete,
                                                            exponent=1.0)
        return tau, g2

    def G2_tl(self):
        _require_propagate_tau_tl()
        dim = np.shape(self.sigma_x)[0]
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
        opA_mat = self.sigma_xdag
        opC_mat = self.sigma_x
        opB_mat = self.sigma_xdag @ self.sigma_x
        g2 = propagate_tau_module.calc_twotime_parallel_block(dm_block=dm_tl,dm_s=dm_s,rho_init=rho0.reshape(dim**2),
                                                              n_tb=int(self.tb/self.dt),nx_tau=self.factor_tau,dim=dim,
                                                              opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t_axis_complete,
                                                              exponent=1.0)
        return tau, g2
    
    def G1_tl(self):
        _require_propagate_tau_tl()
        dim = np.shape(self.sigma_x)[0]
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
        opB_mat = self.sigma_xdag
        opC_mat = self.sigma_x
        g1 = propagate_tau_module.calc_twotime_parallel_block(dm_block=dm_tl,dm_s=dm_s,rho_init=rho0.reshape(dim**2),
                                                              n_tb=int(self.tb/self.dt),nx_tau=self.factor_tau,dim=dim,
                                                              opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t_axis_complete,
                                                              exponent=2.0)
        return tau, g1

    def calc_indistinguishability(self):
        """
        returns indistinguishability,single-photon purity
        """
        # calculate G0, G1 and G2
        # and integrate over tau=0,...,tb/2 and tb/2,...,3tb/2
        if self.use_dm:
            if self.system.phonons:
                print("Calculating with phonons")
                t,g1 = self.G1_tl_phonons()
            else:
                t,g1 = self.G1_tl()
        else:
            t,g1 = self.G1()
        dt = self.dt
        tb = self.tb
        n_1 = int(0.5*tb/dt)
        G11 = 2*np.trapezoid(g1[:n_1], t[:n_1])
        G12 = np.trapezoid(g1[n_1:3*n_1], t[n_1:3*n_1])

        if self.use_dm:
            if self.system.phonons:
                t2,g2 = self.G2_tl_phonons()
            else:
                t2,g2 = self.G2_tl()
        else:
            t2,g2 = self.G2()
        G21 = 2*np.trapezoid(g2[:n_1], t2[:n_1])
        G22 = np.trapezoid(g2[n_1:3*n_1], t2[n_1:3*n_1])

        if self.use_dm:
            if self.system.phonons:
                t0,g0 = self.simple_propagation_tl_phonons()
            else:
                t0,g0 = self.simple_propagation_tl()
        else:
            t0,g0 = self.simple_propagation()
        # integrate 0,...,tb and tb,...,2tb
        G01 = 2*np.trapezoid(g0[:n_1], t0[:n_1])
        G02 = np.trapezoid(g0[n_1:3*n_1], t0[n_1:3*n_1])
        result = (G01-G11+G21)/(G02-G12+G22)
        return 1 - result, 1-G21/G22
