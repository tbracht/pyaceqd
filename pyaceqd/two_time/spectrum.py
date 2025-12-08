import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, wait, ProcessPoolExecutor
from functools import partial
from pyaceqd.tools import round_to_dt ,construct_t, simple_t_gaussian, export_csv, calc_tl_dynmap_pseudo, extract_dms, op_to_matrix
from pyaceqd.timebin.timebin import TimeBin
from pyaceqd.two_level_system.tls import tls
from pyaceqd.four_level_system.linear import biexciton
from pyaceqd.pulses import ChirpedPulse
from pyaceqd.two_time.correlations import tl_two_op_two_time, tl_three_op_two_time
try:
    from pyaceqd.two_time import propagate_tau_module
except ImportError:
    print("WARNING: propagate_tau_module not found, using pure python implementation for G2 and G1 calculations.")
    propagate_tau_module = None
import time
# import warnings
# warnings.filterwarnings('error', category=np.ComplexWarning)
import pyaceqd.constants as constants
temp_dir = constants.temp_dir
hbar = constants.hbar

def compute_s_omega_t(i, omegas, t_ax, tau_ax, g1):
    """
    helper function to compute the time-dependent spectrum for a single frequency omegas[i]
    """
    # print("t_ax.shape: ", t_ax.shape, "tau_ax.shape: ", tau_ax.shape)
    s_omega_t_i = np.zeros(len(t_ax))
    for j in range(len(t_ax)):
        # print("t index: ", j)
        _tbar_ax = t_ax[:j + 1]
        g_omega_tbar = np.zeros(len(_tbar_ax), dtype=complex)
        # get index where tau_ax[n] = t_ax[j]
        # j_index = np.where(tau_ax == t_ax[j])[0][-1]
        # if index_tau_t != j:
        #     print("Warning: index_tau_t != j, index_tau_t: {}, j: {}".format(index_tau_t, j))
        for k in range(len(_tbar_ax)):
            # k_index = np.where(tau_ax == _tbar_ax[k])[0][0]
            j_index = j
            k_index = k
            _tau_ax = tau_ax[:j_index - k_index + 1]
            _g1 = g1[k, :j_index - k_index + 1]
            g_omega_tbar[k] = np.trapezoid(_g1 * np.exp(-1j * omegas[i] * _tau_ax / hbar), _tau_ax)
        s_omega_t_i[j] = np.real(np.trapezoid(g_omega_tbar, _tbar_ax))
    return s_omega_t_i

def compute_s_t(omegas, t_ax, tau_ax, g1):
    s_omegas_t = np.zeros((len(omegas), len(t_ax)))
    for j in tqdm.trange(len(t_ax)):
        # print("t index: ", j)
        _tbar_ax = t_ax[:j + 1]
        g_omega_tbar = np.zeros((len(omegas), len(_tbar_ax)), dtype=complex)
        # get index where tau_ax[n] = t_ax[j]
        # j_index = np.where(tau_ax == t_ax[j])[0][-1]
        # if index_tau_t != j:
        #     print("Warning: index_tau_t != j, index_tau_t: {}, j: {}".format(index_tau_t, j))
        for k in range(len(_tbar_ax)):
            # k_index = np.where(tau_ax == _tbar_ax[k])[0][0]
            j_index = j
            k_index = k
            _tau_ax = tau_ax[:j_index - k_index + 1]
            _g1 = g1[k, :j_index - k_index + 1]
            g_omega_tbar[:, k] = np.trapezoid(_g1 * np.exp(-1j * omegas[:,None] * _tau_ax / hbar), _tau_ax, axis=1)
        s_omegas_t[:, j] = np.real(np.trapezoid(g_omega_tbar, _tbar_ax))
    return s_omegas_t

# def compute_s_t_incremental(omegas, t_ax, tau_ax, g1):
#     s_omegas_t = np.zeros((len(omegas), len(t_ax)))
#     dt = np.abs(t_ax[1] - t_ax[0])
#     dtau = np.abs(tau_ax[1] - tau_ax[0])
#     for j in tqdm.trange(1,len(t_ax)):
#         increment_t_g1 = np.zeros((len(omegas), j), dtype=complex)
#         for k in range(j):
#             increment_t_g1[:,k] = g1[k,j-k] * np.exp(-1j*omegas*(j-k)*dtau/hbar) #+ g1[k, j-k-1] * np.exp(-1j*omegas*(j-k-1)*dtau/hbar)
#         increment_t = dtau * np.trapezoid(increment_t_g1, axis=1, dx=dt)
#         increment_tau_g1 =  dt * dtau * ((g1[j-1,1] * np.exp(-1j*omegas*dtau/hbar) )) #+  g1[j-1,0])/2 + g1[j,0]) 
#         s_omegas_t[:,j] = np.real(s_omegas_t[:,j-1] + increment_t + increment_tau_g1)
#     return s_omegas_t

def compute_s_t_incremental(omegas, t_ax, tau_ax, g1):
    # assumes regular and identically spaced t,tau grids
    s_omegas_t = np.zeros((len(omegas), len(t_ax)))
    dt = np.abs(t_ax[1] - t_ax[0])
    dtau = np.abs(tau_ax[1] - tau_ax[0])
    s_omegas_t[:,0] = np.real(dt**2 * g1[0,0])
    for j in tqdm.trange(1,len(t_ax)):
        # sum_t_g1 = 0
        # for k in range(j):
            # sum_t_g1 += g1[k,j-k] * np.exp(-1j*omegas*(j-k)*dtau/hbar)
        k_indices = np.arange(j)
        tau_indices = j - k_indices
        sum_t_g1 = np.trapezoid(g1[k_indices, tau_indices] * np.exp(-1j*omegas[:,None]*tau_indices*dtau/hbar), axis=1)
        # if j > 1:
            # sum_t_g1 -= 0.5* g1[k_indices[0], tau_indices[0]] * np.exp(-1j*omegas*tau_indices[0]*dtau/hbar)
            # sum_t_g1 -= 0.5* g1[k_indices[-1], tau_indices[-1]] * np.exp(-1j*omegas*tau_indices[-1]*dtau/hbar)
        # sum_t_g1 -= 0.5 * (g1[0,j] * np.exp(-1j*omegas*j*dtau/hbar) + g1[j-1,1] * np.exp(-1j*omegas*dtau/hbar))  # trapezoidal correction
        s_omegas_t[:,j] = np.real(s_omegas_t[:,j-1] + dt**2 * sum_t_g1 + dt**2 * g1[j,0] )
    return s_omegas_t.real

class Spectrum(TimeBin):
    def __init__(self, system, sigma_x, sigma_xdag, *pulses, dt=0.1, tend=800, dt_small=0.1, simple_exp=True, gaussian_t=None, gaussian_dt=False, verbose=False, workers=15, options={}, dm=False, sigma_x_mat=None, sigma_xdag_mat=None, t_mem=10, dt_big=None, add_tend=True) -> None:
        self.dm = dm
        self.tend = tend
        self.tl_map = None
        self.tl_dms = None
        self.t_mem = t_mem  # memory time for phonon dynamics with time-local maps
        self.sigma_x_mat = sigma_x_mat
        self.sigma_xdag_mat = sigma_xdag_mat
        self.verbose = verbose
        if sigma_x_mat is None or sigma_xdag_mat is None:
            if self.verbose:
                print("WARNING: sigma_x_mat or sigma_xdag_mat not provided, trying to convert sigma_x and sigma_xdag to matrices.")
                print("supply as A+B+.... without brackets.")
            self.sigma_x_mat = op_to_matrix(sigma_x)
            self.sigma_xdag_mat = op_to_matrix(sigma_xdag)


        self.dim = self.sigma_x_mat.shape[0]  # dimension of the Hilbert space
        super().__init__(system, *pulses, dt=dt, tb=tend, simple_exp=simple_exp, gaussian_t=gaussian_t, verbose=verbose, workers=workers, t_simul=tend, options=options)
        self.sigma_x = "(" + sigma_x + ")"
        self.sigma_xdag = "(" + sigma_xdag + ")"
        
        try:
            self.gamma_e = options["gamma_e"]
        except KeyError:
            print("gamma_e not included in options, setting to 100")
            self.options["gamma_e"] = 100
            self.gamma_e = self.options["gamma_e"]
        if dt_big is None:
                dt_big = 10*dt_small
        if gaussian_dt and self.gaussian_t is not None:
            self.t1 = simple_t_gaussian(0,self.gaussian_t,self.tend,dt_small,dt_big,*pulses,decimals=1,exp_part=self.simple_exp,add_tend=add_tend)
        else:
            n_t = self.tend/dt_big
            self.t1 = np.linspace(0, self.tend, int(n_t)+1)
        self.options["pulse_file_x"] = self.pulse_file_x
        self.options["pulse_file_y"] = self.pulse_file_y

    def calc_timedynamics(self, output_ops=None, t_end=None):
        new_options = dict(self.options)
        if output_ops is not None:
            new_options["output_ops"] = output_ops
        if t_end is None:
            t_end = self.tend
        return self.system(0, t_end, *self.pulses, **new_options)
    
    def get_tl(self, t_mem=None):
        if t_mem is None:
            t_mem = self.gaussian_t
        if t_mem is None:
            t_mem = self.tend/2
        tend = 2*t_mem
        # _options = self.options.copy()
        # _options["pulse_file_x"] = self.pulse_file_x
        # _options["pulse_file_y"] = self.pulse_file_y
       
        result, dm = self.system(0, tend, multitime_op=[], calc_dynmap=True, **self.options)
        _t = result[0]  # time axis for getting the dynamic maps
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        memory_time = self.gaussian_t if self.gaussian_t is not None else self.tend
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

    def calc_timedynamics_tl_phonons(self):
        tl_map, dms = self.get_tl_phonons(mtos=[], t_mtos=[])
        dm_sep1 = dms[0]

        len_tb = int(self.tend/self.dt)
        t_total = np.linspace(0, self.tend, len_tb + 1)
        rho0 = np.zeros((self.dim,self.dim), dtype=complex)
        rho0[0,0] = 1  # initial state, rho0 = |0><0|
        # rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_t = np.ones((len(t_total), self.dim**2), dtype=complex)
        rho_t[0] = rho0.reshape(self.dim**2)  # initial state, rho0 = |0><0|
        rho_t[-1] = rho0.reshape(self.dim**2)  # final state, rho0 = |0><0|
        # from 0 to len_tb-1, we have the pulses
        # do this in each time bin
        for i in range(1,len(dm_sep1)):
            rho_t[i] = np.dot(dm_sep1[i-1], rho_t[i-1])
            # now apply the time-local dynamical map
        for i in range(len(dm_sep1),len_tb+1):
            rho_t[i] = np.dot(tl_map, rho_t[i-1])
        return t_total, rho_t.reshape((len(t_total), self.dim, self.dim))

    def calc_timedynamics_tl(self):
        if self.options["phonons"]:
            return self.calc_timedynamics_tl_phonons()
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()
        
        len_tb = int(self.tend/self.dt)
        t_total = np.linspace(0, self.tend, len_tb + 1)
        rho0 = np.zeros((self.dim,self.dim), dtype=complex)
        rho0[0,0] = 1  # initial state, rho0 = |0><0|
        rho_t = np.ones((len(t_total), self.dim**2), dtype=complex)
        rho_t[0] = rho0.reshape(self.dim**2)  # initial state, rho0 = |0><0|
        rho_t[-1] = rho0.reshape(self.dim**2)  # final state, rho0 = |0><0|
        self.tl_complete = np.zeros((len(t_total)-1, self.dim**2, self.dim**2), dtype=complex)
            # from 0 to len_tb-1, we have the pulses
            # do this in each time bin
            # print(t_total[j*len_tb])
        for i in range(1,len(self.tl_dms)):
            rho_t[i] = np.dot(self.tl_dms[i-1], rho_t[i-1])
            # now apply the time-local dynamical map
        for i in range(len(self.tl_dms),len_tb+1):
            rho_t[i] = np.dot(self.tl_map, rho_t[i-1])
        return t_total, rho_t.reshape((len(t_total), self.dim, self.dim))
    
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

        tau_max=self.tend
        n_tau = int(tau_max/self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)

        t_mem_indices = np.where(self.t1 <= (self.gaussian_t + self.t_mem))[0]
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
                    _t_mto = np.round(self.t1[i],6)
                    _e = executor.submit(self.get_dm2_phonons_advanced,[_mto], _t_mto, i)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
            wait(futures)
            for i in range(len(t_mem_indices)):
                dm_part = futures[i].result()
                _len_part = np.shape(dm_part)[0]
                dms_tauc2[i,:_len_part] = dm_part

        dm_taucs2 = np.asfortranarray(dms_tauc2.transpose(2, 3, 0, 1))
        dm_separated1 = np.asfortranarray(dms_sep[0].transpose(1, 2, 0))
        dm_separated2 = np.asfortranarray(dms_sep[1].transpose(1, 2, 0))
        dm_s = tl_map

        _tend = self.t1[-1] + tau_max
        # the 'simulation' time axis. The time axis for the two-time correlation function
        # is self.t_axis_complete, which contains less time points so we need less propagations.
        t_axis = np.linspace(0, _tend, int(_tend/self.dt) + 1)

        opA_mat = np.identity(dim)
        opB_mat = self.sigma_xdag_mat
        opC_mat = self.sigma_x_mat

        G1 = propagate_tau_module.calc_onetime_simple_phonon(dm_taucs2=dm_taucs2, dm_sep1=dm_separated1, dm_sep2=dm_separated2, dm_s=dm_s,
                                                            rho_init=rho0.reshape(dim**2),n_tb=int(self.tend/self.dt),
                                                            dim=dim,opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t1)
        # g1 = np.trapezoid(np.abs(G1)**2, self.t1, axis=0)
        return self.t1, tau, G1
    
    def G1_tl(self):
        if self.options["phonons"]:
            return self.G1_tl_phonons()
        dim = np.shape(self.sigma_x_mat)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0  # initial state, rho0 = |0><0|

        tau_max=self.tend
        n_tau = int(tau_max/self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)

        # calc tl maps:
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()
        dm_tl = np.asfortranarray(self.tl_dms.transpose(1, 2, 0))
        dm_s = self.tl_map

        _tend = self.t1[-1] + tau_max
        t_axis = np.linspace(0, _tend, int(_tend/self.dt) + 1)
        opA_mat = np.identity(dim)
        opB_mat = self.sigma_xdag_mat
        opC_mat = self.sigma_x_mat
        G1 = propagate_tau_module.calc_onetime_simple(dm_block=dm_tl,dm_s=dm_s,rho_init=rho0.reshape(dim**2),n_tb=int(self.tend/self.dt),dim=dim,opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t1)
        # g1 = np.trapezoid(np.abs(G1)**2, self.t1, axis=0)
        return self.t1, tau, np.conj(G1)
    
    def G1(self):
        """
        calculates G1 for two operators:
        <op2(t1+tau) op1(t1)>
        """
        op1_t = self.sigma_x
        op2_ttau = self.sigma_xdag
        # check if first char of op1_t and op2_ttau is a bracket
        if op1_t[0] != "(":
            op1_t = "(" + op1_t + ")"
            print("WARNING: added brackets to op1_t")
        if op2_ttau[0] != "(":
            op2_ttau = "(" + op2_ttau + ")"
            print("WARNING: added brackets to op2_ttau")
        tau0_op = op2_ttau + " * " + op1_t
        output_ops = [op2_ttau, tau0_op]
        # at t1, apply op1 from left
        op_1 = {"operator": op1_t, "applyFrom": "_left", "applyBefore":"false"}

        t1 = self.t1
        n_tau = int((self.tend)/self.dt)
        # simulation time-axis
        t2 = np.linspace(0, self.tend, n_tau + 1) # tau axis
        _G1 = np.zeros([len(t1),len(t2)], dtype=complex)
        tend = self.tend  # not always the same end time, see below
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    op_1_new = dict(op_1)
                    op_1_new["time"] = t1[i]
                    # t_end is always t[i]+tend, as we want to calculate G1 for tau=t[i],...,tend,
                    # i.e., always the same length for the tau axis, as these will be fourier transformed
                    # to get the spectrum
                    _e = executor.submit(self.system,0,t1[i]+tend,multitime_op=[op_1_new], suffix=i, output_ops=output_ops, **self.options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                futures[i] = futures[i].result()
            # futures now contains [t,<op2>,<op2*op1>] for every i
            for i in range(len(t1)):
                _G1[i,0] = futures[i][2][-(n_tau+1)]  # tau=0
                _G1[i,1:] = futures[i][1][-n_tau:]  # tau>0
        return t1, t2, _G1
    
    def get_spectrum(self, save_g1_dir=None, load=None, dm=True, timeit=False):
        """
        calculates the spectrum of G1 for two operators:
        <op2(t1+tau) op1(t1)>
        """
        # uses G1 to calculate the spectrum
        if load is not None and os.path.exists(load + "g1.npy"):
            t_axis = np.load(load + "t_axis.npy")
            tau_axis = np.load(load + "tau_axis.npy")
            g1 = np.load(load + "g1.npy")
        else:
            if dm:
                t_axis, tau_axis, g1 = self.G1_tl()
            else:
                t_axis, tau_axis, g1 = self.G1()
        if save_g1_dir is not None and load is None:
            np.save(save_g1_dir + "g1.npy", g1)
            np.save(save_g1_dir + "t_axis.npy", t_axis)
            np.save(save_g1_dir + "tau_axis.npy", tau_axis)
        if timeit:
            start_time = time.time()
        dtau = np.abs(tau_axis[1] - tau_axis[0])
        fft_freqs = -2*np.pi * hbar * np.fft.fftfreq(2*len(tau_axis)-1,d=dtau)
        # symmetrize g1
        # g1[:,-10:] = 0 + 0j
        # g1 = np.conj(g1)
        g1_symm = np.empty([len(t_axis),2*len(tau_axis)-1],dtype=complex)
        g1_symm[:,:len(tau_axis)] = g1[:,::-1]
        g1_symm[:,-(len(tau_axis)-1):] = np.conj(g1[:,1:])
        spectra = np.empty([len(g1_symm),len(g1_symm[0])],dtype=complex)
        for j in range(len(g1_symm)):
            # do fft for every t, along the tau axis
            spectra[j] = np.fft.fftshift(np.fft.fft(g1_symm[j]))
        spectrum = np.real(np.trapezoid(spectra.transpose(),t_axis))
        if timeit:
            end_time = time.time()
            print(f"Spectrum calculation took {end_time - start_time} seconds.")
        return np.fft.fftshift(fft_freqs), spectrum, spectra
    
    def get_time_dependent_spectrum_tl(self, tend=100, omega_min=-5, omega_max=5, domega=0.1):
        _dt = self.dt  # choose a dt that is multiple of self.dt
        tend = tend
        self.tend = tend # round_to_dt(np.array([tend]),_dt)[0]
        n_t = int(tend/_dt)
        t_axis = np.linspace(0, tend, n_t + 1)
        self.t1 = t_axis
        t_axis, tau_axis, g1 = self.G1_tl()  # g1[t_index, tau_index]
        # print("G1 calculated")
        # print(t_axis.shape, tau_axis.shape, g1.shape)
        dtau = np.abs(tau_axis[1] - tau_axis[0])
        dt = np.abs(t_axis[1] - t_axis[0])
        # tau_axis = tau_axis[::int(_dt/self.dt)]
        # print("Downsampled tau axis")
        # print(tau_axis[0], tau_axis[-1], tau_axis.shape)
        # g1 = g1[:,::int(_dt/self.dt)]
        _omega_max = (np.abs(omega_max) + np.abs(omega_min))
        n_omega = int(_omega_max/domega)
        omega_axis = np.linspace(omega_min, omega_max, n_omega + 1)
        # S_omega_t = np.zeros((len(omega_axis), len(t_axis)), dtype=complex)

        S_omega_t = compute_s_t_incremental(omega_axis, t_axis, tau_axis, g1)
        plt.clf()
        plt.pcolormesh(omega_axis,t_axis,np.log(np.abs(S_omega_t.T)+0.001),shading='gouraud',vmin=-1,vmax=3)
        plt.xlabel("Frequency (meV)")
        plt.ylabel("Time (ps)")
        plt.colorbar(label="log(S(omega,t))")
        plt.savefig("time_dep_spectrum_tl.png")

        plt.clf()
        plt.plot(omega_axis, np.log(np.abs(S_omega_t[:,-1])+0.001))
        plt.xlabel("Frequency (meV)")
        plt.ylabel("log(S(omega,tend))")
        plt.savefig("spectrum_at_tend.png")
        return S_omega_t

    def get_time_dependent_spectrum(self, tend, omega_min=-5, omega_max=5, domega=0.1):
        """
        To calculate the time-dependend spectrum S(\omega, t) = Re(\int_{0}^(t) dt' \int_{0}^{t-t'}d\tau G1(t',\tau) exp(-i \omega \tau))
        we need to calculate G1(t',\tau) on a regular timegrid which is the same for t and tau
        per defautl, the tau axis is spaced with the simulation dt, while the t axis can be anything.
        This means we choose a regular t axis that matches n*dt, and downsample the tau axis.
        We further set self.tend to tend, so the tau axis is only up to tend.
        """
        _dt = self.dt  # choose a dt that is multiple of self.dt
        self.tend = round_to_dt(np.array([tend]),_dt)[0]  # make sure that tend is multiple of _dt
        n_t = int(tend/_dt)
        t_axis = np.linspace(0, tend, n_t + 1)
        self.t1 = t_axis
        t_axis, tau_axis, g1 = self.G1_tl()
        # print("G1 calculated")
        # print(t_axis.shape, tau_axis.shape, g1.shape)
        dtau = np.abs(tau_axis[1] - tau_axis[0])
        dt = np.abs(t_axis[1] - t_axis[0])
        # print(f"dtau: {dtau}, dt: {dt}")
        tau_axis = tau_axis[::int(_dt/self.dt)]
        # print("Downsampled tau axis")
        # print(tau_axis[0], tau_axis[-1], tau_axis.shape)
        g1 = g1[:,::int(_dt/self.dt)]
        _omega_max = (np.abs(omega_max) + np.abs(omega_min))
        n_omega = int(_omega_max/domega)
        omega_axis = np.linspace(omega_min, omega_max, n_omega + 1)
        # S_omega_t = np.zeros((len(omega_axis), len(t_axis)), dtype=complex)
        
        # def run_parallel(omegas, t_ax, tau_ax, g1):
        #     s_omega_t = np.zeros((len(omegas), len(t_ax)))
        #     with ProcessPoolExecutor(max_workers=self.workers) as executor:
        #         # partial function fixes the arguments that are not changing (all but i)
        #         results = list(tqdm.tqdm(executor.map(partial(compute_s_omega_t, omegas=omegas, t_ax=t_ax, tau_ax=tau_ax, g1=g1), range(len(omegas))), total=len(omegas)))
        #     for i, result in enumerate(results):
        #         s_omega_t[i, :] = result
        #     return s_omega_t

        S_omega_t = compute_s_t_incremental(omega_axis, t_axis, tau_axis, g1)  #run_parallel(omega_axis, t_axis, tau_axis, g1)
        plt.clf()
        plt.pcolormesh(omega_axis,t_axis,np.log(np.abs(S_omega_t.T)+0.001),shading='gouraud',vmin=-1,vmax=3)
        plt.xlabel("Frequency (meV)")
        plt.ylabel("Time (ps)")
        plt.colorbar(label="log(S(omega,t))")
        plt.savefig("time_dep_spectrum.png")
        return S_omega_t



    
options = {"verbose": False, "gamma_e": 1/100, "lindblad": True, "phonons": True, "use_infinite": True, "ae": 5, "temperature": 4, "temp_dir": temp_dir}
a = Spectrum(tls, "|0><1|_2", "|1><0|_2", ChirpedPulse(tau_0=3, e_start=0, alpha=0, t0=5*3, e0=1, polar_x=1), dt=0.1, dt_small=0.1, tend=1500, gaussian_dt=True, simple_exp=True, gaussian_t=30, verbose=True, workers=23, options=options, dm=True)
t,rho = a.calc_timedynamics_tl()
plt.clf()
plt.plot(t.real, rho[:,1,1].real, label="X")
plt.xlabel("Time (a.u.)")
plt.ylabel("Population of |1>")
plt.legend()
plt.savefig("x.png")

# t, tau, g1 = a.G1_tl_phonons()
# print("G1 calculated")
# print(t.shape, tau.shape, g1.shape)
# exit()

# freqs, spectrum, spectra = a.get_spectrum()
# plt.clf()
# plt.xlim(-4, 4)
# spectral_cut = np.where((freqs >= -4) & (freqs <= 4))
# # plt.ylim(np.min(np.log(np.abs(spectrum[spectral_cut])))-1, np.max(np.log(np.abs(spectrum[spectral_cut])))+1)
# plt.plot(freqs[spectral_cut], np.log(np.abs(spectrum[spectral_cut])), label="Spectrum")
# plt.xlabel("Frequency (a.u.)")
# plt.ylabel("Spectrum")
# plt.legend()
# plt.savefig("spectrum.png")

# exit()

options = {"verbose": False, "gamma_e": 1/100, "lindblad": True, "delta_b": 2, "phonons": False, "use_infinite": True, "ae": 5, "temperature": 4, "temp_dir": temp_dir}
sigma_x_mat = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0]])
sigma_xdag_mat = sigma_x_mat.T
tau_0=8
a = Spectrum(biexciton, "|0><2|_4+|2><3|_4", "|2><0|_4+|3><2|_4", ChirpedPulse(tau_0=tau_0, e_start=-1, alpha=0, t0=4*tau_0, e0=23, polar_x=1), dt=0.1, dt_small=0.1, tend=1000, gaussian_dt=False, simple_exp=True, gaussian_t=8*tau_0, verbose=True, workers=23, options=options, dm=True,
)#sigma_x_mat=sigma_x_mat, sigma_xdag_mat=sigma_xdag_mat)
t,rho = a.calc_timedynamics_tl()
plt.clf()
plt.plot(t.real, rho[:,3,3].real, label="B")
plt.xlabel("Time (a.u.)")
plt.ylabel("Population of |3>")
plt.legend()
plt.savefig("b.png")

# a.get_time_dependent_spectrum(tend=64, omega_min=-4, omega_max=2, domega=0.05)
a.get_time_dependent_spectrum(tend=150, omega_min=-4, omega_max=2, domega=0.05)
a.get_tl()
dm_s = a.tl_map
eigenvalues, eigenvectors = np.linalg.eig(dm_s)
print("Eigenvalues of time-local map:")
for i in range(len(eigenvalues)):
    print(f"{i}: {eigenvalues[i]}")
print("Eigenvectors of time-local map:")
for i in range(len(eigenvalues)):
    print(f"{i}: {eigenvectors[:,i]}")
print("Time-local map calculated")
plt.clf()
plt.imshow((dm_s.real), cmap='viridis')
plt.colorbar(label="|DM elements|")
plt.xlabel("Column")
plt.ylabel("Row")
plt.title("Time-local dynamical map")
plt.savefig("tl_map.png")

# freqs, spectrum, spectra = a.get_spectrum(dm=True, timeit=False)
# plt.clf()
# plt.xlim(-8, 4)
# plt.plot(freqs, np.log(np.abs(spectrum)), label="Spectrum")
# plt.xlabel("Frequency (a.u.)")
# plt.ylabel("Spectrum")
# plt.legend()
# plt.savefig("spectrum.png")
