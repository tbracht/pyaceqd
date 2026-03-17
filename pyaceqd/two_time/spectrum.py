import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
from pyaceqd.helpers.dynamical_map import calc_tl_dynmap_pseudo, extract_dms
from pyaceqd.helpers.time_axes import UnregularTimeAxis, round_to_dt
try:
    from pyaceqd.two_time import propagate_tau_module
except ImportError as exc:
    _PROPAGATE_TAU_TL_IMPORT_ERROR = exc
    warnings.warn("propagate_tau_module not found, time-local acceleration unavailable.",
                  ImportWarning,
                  stacklevel=2,)
    propagate_tau_module = None
import pyaceqd.constants as constants

hbar = constants.hbar


def _require_propagate_tau_tl() -> None:
    if propagate_tau_module is None:
        raise RuntimeError(
            "The optional Fortran module 'propagate_tau_module' is not available. "
            "Reinstall with Fortran build enabled to use time-local accelerated routines."
        ) from _PROPAGATE_TAU_TL_IMPORT_ERROR


def get_max_pulse_t(pulses):
    times = []
    for p in pulses:
        times.append(p.t0 + 4*p.tau)
    return max(times)


def compute_s_omega_t(i, omegas, t_ax, tau_ax, g1):
    """
    helper function to compute the time-dependent spectrum for a single frequency omegas[i]
    """
    s_omega_t_i = np.zeros(len(t_ax))
    for j in range(len(t_ax)):
        _tbar_ax = t_ax[:j + 1]
        g_omega_tbar = np.zeros(len(_tbar_ax), dtype=complex)
        for k in range(len(_tbar_ax)):
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
        _tbar_ax = t_ax[:j + 1]
        g_omega_tbar = np.zeros((len(omegas), len(_tbar_ax)), dtype=complex)
        for k in range(len(_tbar_ax)):
            j_index = j
            k_index = k
            _tau_ax = tau_ax[:j_index - k_index + 1]
            _g1 = g1[k, :j_index - k_index + 1]
            g_omega_tbar[:, k] = np.trapezoid(
                _g1 * np.exp(-1j * omegas[:, None] * _tau_ax / hbar), _tau_ax, axis=1)
        s_omegas_t[:, j] = np.real(np.trapezoid(g_omega_tbar, _tbar_ax))
    return s_omegas_t


def compute_s_t_incremental(omegas, t_ax, tau_ax, g1):
    # assumes regular and identically spaced t,tau grids
    s_omegas_t = np.zeros((len(omegas), len(t_ax)))
    dt = np.abs(t_ax[1] - t_ax[0])
    dtau = np.abs(tau_ax[1] - tau_ax[0])
    s_omegas_t[:, 0] = np.real(dt**2 * g1[0, 0])
    for j in tqdm.trange(1, len(t_ax)):
        k_indices = np.arange(j)
        tau_indices = j - k_indices
        sum_t_g1 = np.trapezoid(
            g1[k_indices, tau_indices] * np.exp(-1j * omegas[:, None] * tau_indices * dtau / hbar),
            axis=1)
        s_omegas_t[:, j] = np.real(s_omegas_t[:, j-1] + dt**2 * sum_t_g1 + dt**2 * g1[j, 0])
    return s_omegas_t.real


class Spectrum:
    def __init__(self, system, sigma_x, sigma_xdag, *pulses, tend=800, dt_small=0.1,
                 max_pulse_t=None, t_mem=10, regular_stepping=False, variable_stepping=False,
                 exponential_stepping=False, verbose=False, workers=15, dt_big=None,
                 add_tend=True) -> None:
        self.system = system
        self.sigma_x = sigma_x
        self.sigma_xdag = sigma_xdag
        self.pulses = pulses
        self.tend = tend
        self.verbose = verbose
        self.workers = workers
        self.tl_map = None
        self.tl_dms = None
        self.t_mem = t_mem

        self.dim = system.dim
        self.dt = system.dt
        self.phonons = system.phonons
        if not system.lindblad:
            print("WARNING: system is not using lindblad operators, eg. no decay.")
        if sigma_x.shape != (self.dim, self.dim) or sigma_xdag.shape != (self.dim, self.dim):
            raise ValueError(
                "sigma_x or sigma_xdag shape mismatch: {}, {} vs system dim {}x{}".format(
                    sigma_x.shape, sigma_xdag.shape, self.dim, self.dim))

        if max_pulse_t is None:
            max_pulse_t = get_max_pulse_t(pulses)
        self.max_pulse_t = max_pulse_t

        if dt_big is None:
            dt_big = 10 * dt_small

        time_generator = UnregularTimeAxis(0, tend, self.max_pulse_t + self.t_mem,
                                           dt_small=dt_small, dt_big=dt_big, pulses=pulses,
                                           factor_tau=1, include_tend=add_tend, round_dt=True)
        self.t1 = time_generator.time_axis_two_step(exponential_part=exponential_stepping)
        if variable_stepping:
            self.t1 = time_generator.time_axis_variable(exponential_part=exponential_stepping,
                                                        dt_big_variable=dt_big)
        if regular_stepping:
            self.t1 = time_generator.time_axis_regular()

        if self.verbose:
            print("t1 axis for spectrum calculation: ", len(self.t1), " points")
            n_tau = int(self.tend / self.dt)
            print("tau axis for spectrum calculation: ", n_tau + 1, " points")

    def calc_timedynamics(self, output_ops=None, t_end=None):
        if t_end is None:
            t_end = self.tend
        return self.system.run(0, t_end, *self.pulses, output_ops=output_ops)

    def get_tl(self, t_mem=None):
        if t_mem is None:
            t_mem = self.max_pulse_t
        tend = 2 * t_mem
        _t, dm = self.system.run(0, tend, *self.pulses, multitime_op=[], calc_dynmap=True)
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        memory_time = self.max_pulse_t
        tl_map, dms = extract_dms(dm_tl, _t, memory_time, t_MTOs=[])
        self.tl_map = tl_map
        self.tl_dms = dms[0]

    def get_tl_phonons(self, mtos=[], t_mtos=[]):
        tmem = self.max_pulse_t + self.t_mem
        tend = 2.1 * tmem
        _t, dm = self.system.run(0, tend, *self.pulses, multitime_op=mtos, calc_dynmap=True)
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        tl_map, dms = extract_dms(dm_tl, _t, tmem, t_MTOs=t_mtos)
        dms = np.array(dms, dtype=complex)
        return tl_map, dms

    def calc_timedynamics_tl_phonons(self):
        tl_map, dms = self.get_tl_phonons(mtos=[], t_mtos=[])
        dm_sep1 = dms[0]

        len_tb = int(self.tend / self.dt)
        t_total = np.linspace(0, self.tend, len_tb + 1)
        rho0 = np.zeros((self.dim, self.dim), dtype=complex)
        rho0[0, 0] = 1  # initial state, rho0 = |0><0|
        rho_t = np.ones((len(t_total), self.dim**2), dtype=complex)
        rho_t[0] = rho0.reshape(self.dim**2)
        rho_t[-1] = rho0.reshape(self.dim**2)
        for i in range(1, len(dm_sep1)):
            rho_t[i] = np.dot(dm_sep1[i - 1], rho_t[i - 1])
        for i in range(len(dm_sep1), len_tb + 1):
            rho_t[i] = np.dot(tl_map, rho_t[i - 1])
        return t_total, rho_t.reshape((len(t_total), self.dim, self.dim))

    def calc_timedynamics_tl(self):
        if self.phonons:
            return self.calc_timedynamics_tl_phonons()
        if self.tl_map is None:
            self.get_tl()

        len_tb = int(self.tend / self.dt)
        t_total = np.linspace(0, self.tend, len_tb + 1)
        rho0 = np.zeros((self.dim, self.dim), dtype=complex)
        rho0[0, 0] = 1  # initial state, rho0 = |0><0|
        rho_t = np.ones((len(t_total), self.dim**2), dtype=complex)
        rho_t[0] = rho0.reshape(self.dim**2)
        rho_t[-1] = rho0.reshape(self.dim**2)
        self.tl_complete = np.zeros((len(t_total) - 1, self.dim**2, self.dim**2), dtype=complex)
        for i in range(1, len(self.tl_dms)):
            rho_t[i] = np.dot(self.tl_dms[i - 1], rho_t[i - 1])
        for i in range(len(self.tl_dms), len_tb + 1):
            rho_t[i] = np.dot(self.tl_map, rho_t[i - 1])
        return t_total, rho_t.reshape((len(t_total), self.dim, self.dim))

    def get_dm2_phonons_advanced(self, mtos, t_mto):
        # the correct final time depends on t_mto, as gaussian_t is absolute and t_mem is always needed.
        # this means the maximum final time will be gaussian_t + 2 * t_mem
        # while t_mto is gaussian_t + t_mem, meaning before we used a maximum of 2*gaussian_t + 2*t_mem.
        mtos_new = []
        for mto in mtos:
            mto_new = mto.copy()
            mto_new["time"] = t_mto
            mtos_new.append(mto_new)
        t_end = self.max_pulse_t + 2 * self.t_mem + 2 * self.dt
        _t, dm = self.system.run(0, t_end, *self.pulses, multitime_op=mtos_new, calc_dynmap=True)
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        # for t_mto = 0, memory time = gaussian_t + t_mem
        # for t_mto = gaussian_t, memory time = t_mem (minimum)
        memory_time = np.max([self.max_pulse_t + self.t_mem - t_mto, self.t_mem])
        _, dms = extract_dms(dm_tl, _t, memory_time, t_MTOs=[t_mto])
        return dms[1]

    def G1_tl_phonons(self):
        _require_propagate_tau_tl()
        t_apply = np.round(round_to_dt(self.max_pulse_t + self.t_mem + 5 * self.dt, self.dt), 6)
        _mto = {"operator": self.sigma_x, "applyFrom": "left", "applyBefore": False, "time": t_apply}
        tl_map, dms_sep = self.get_tl_phonons(mtos=[_mto], t_mtos=[t_apply])

        dim = self.dim
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0

        tau_max = self.tend
        n_tau = int(tau_max / self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)

        t_mem_indices = np.where(self.t1 <= (self.max_pulse_t + self.t_mem))[0]
        # always let the maps have the same shape as dms_sep[0], which is computed using a memory time
        # of gaussian_t + t_mem. Fill the remainder with the stationary time-local map.
        dms_tauc2 = np.zeros((len(t_mem_indices), *np.shape(dms_sep[0])), dtype=complex)
        dms_tauc2[:, :] = tl_map

        with tqdm.tqdm(total=len(t_mem_indices), leave=None) as tq:
            futures = []
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for i in range(len(t_mem_indices)):
                    _t_mto = np.round(self.t1[i], 6)
                    _e = executor.submit(self.get_dm2_phonons_advanced, [_mto], _t_mto)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
            wait(futures)
            for i in range(len(t_mem_indices)):
                dm_part = futures[i].result()
                _len_part = np.shape(dm_part)[0]
                dms_tauc2[i, :_len_part] = dm_part

        dm_taucs2 = np.asfortranarray(dms_tauc2.transpose(2, 3, 0, 1))
        dm_separated1 = np.asfortranarray(dms_sep[0].transpose(1, 2, 0))
        dm_separated2 = np.asfortranarray(dms_sep[1].transpose(1, 2, 0))
        dm_s = tl_map

        _tend = self.t1[-1] + tau_max
        t_axis = np.linspace(0, _tend, int(_tend / self.dt) + 1)

        opA_mat = np.identity(dim)
        opB_mat = self.sigma_xdag
        opC_mat = self.sigma_x

        G1 = propagate_tau_module.calc_onetime_simple_phonon(
            dm_taucs2=dm_taucs2, dm_sep1=dm_separated1, dm_sep2=dm_separated2, dm_s=dm_s,
            rho_init=rho0.reshape(dim**2), n_tb=int(self.tend / self.dt),
            dim=dim, opa=opA_mat, opb=opB_mat, opc=opC_mat, time=t_axis, time_sparse=self.t1)
        return self.t1, tau, G1

    def G1_tl(self):
        if self.phonons:
            return self.G1_tl_phonons()
        _require_propagate_tau_tl()
        dim = self.dim
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0

        tau_max = self.tend
        n_tau = int(tau_max / self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)

        if self.tl_map is None:
            self.get_tl()
        dm_tl = np.asfortranarray(self.tl_dms.transpose(1, 2, 0))
        dm_s = self.tl_map

        _tend = self.t1[-1] + tau_max
        t_axis = np.linspace(0, _tend, int(_tend / self.dt) + 1)

        opA_mat = np.identity(dim)
        opB_mat = self.sigma_xdag
        opC_mat = self.sigma_x

        G1 = propagate_tau_module.calc_onetime_simple(
            dm_block=dm_tl, dm_s=dm_s, rho_init=rho0.reshape(dim**2),
            n_tb=int(self.tend / self.dt), dim=dim,
            opa=opA_mat, opb=opB_mat, opc=opC_mat, time=t_axis, time_sparse=self.t1)
        return self.t1, tau, np.conj(G1)

    def G1(self):
        """
        Calculates G1: <sigma_xdag(t1+tau) sigma_x(t1)>
        """
        op_1 = {"operator": self.sigma_x, "applyFrom": "left", "applyBefore": False}
        output_ops = [self.sigma_xdag, self.sigma_xdag @ self.sigma_x]
        t1 = self.t1
        n_tau = int(self.tend / self.dt)
        t2 = np.linspace(0, self.tend, n_tau + 1)
        _G1 = np.zeros([len(t1), len(t2)], dtype=complex)
        tend = self.tend
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    op_1_new = dict(op_1)
                    op_1_new["time"] = t1[i]
                    _e = executor.submit(self.system.run, 0, t1[i] + tend, *self.pulses,
                                         multitime_op=[op_1_new], output_ops=output_ops)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                wait(futures)
            for i in range(len(futures)):
                _, futures[i] = futures[i].result()
            for i in range(len(t1)):
                _G1[i, 0] = futures[i][1][-(n_tau + 1)]  # tau=0: sigma_xdag @ sigma_x
                _G1[i, 1:] = futures[i][0][-n_tau:]       # tau>0: sigma_xdag
        return t1, t2, _G1

    def get_spectrum(self, save_g1_dir=None, load=None, dm=True, timeit=False, final_spectrum_only=False):
        """
        Calculates the spectrum via G1: <sigma_xdag(t1+tau) sigma_x(t1)>
        """
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
        fft_freqs = -2 * np.pi * hbar * np.fft.fftfreq(2 * len(tau_axis) - 1, d=dtau)
        # symmetrize g1
        g1_symm = np.empty([len(t_axis), 2 * len(tau_axis) - 1], dtype=complex)
        g1_symm[:, :len(tau_axis)] = g1[:, ::-1]
        g1_symm[:, -(len(tau_axis) - 1):] = np.conj(g1[:, 1:])
        if final_spectrum_only:
            dt0 = np.abs(t_axis[1] - t_axis[0])
            spectrum = np.fft.fftshift(np.fft.fft(g1_symm[0])) * 0.5 * dt0
            for j in range(1, len(g1_symm)):
                dt = np.abs(t_axis[j] - t_axis[j - 1])
                if j == len(g1_symm) - 1:
                    dt = dt * 0.5  # trapezoidal correction for the last point
                spectrum += np.fft.fftshift(np.fft.fft(g1_symm[j])) * dt
            return np.fft.fftshift(fft_freqs), np.real(spectrum)
        spectra = np.empty([len(g1_symm), len(g1_symm[0])], dtype=complex)
        for j in range(len(g1_symm)):
            spectra[j] = np.fft.fftshift(np.fft.fft(g1_symm[j]))
        spectrum = np.real(np.trapezoid(spectra.transpose(), t_axis))
        if timeit:
            end_time = time.time()
            print(f"Spectrum calculation took {end_time - start_time} seconds.")
        return np.fft.fftshift(fft_freqs), spectrum, spectra

    def get_time_dependent_spectrum_tl(self, tend=100, omega_min=-5, omega_max=5, domega=0.1, plot=False):
        _dt = self.dt
        self.tend = tend
        n_t = int(tend / _dt)
        t_axis = np.linspace(0, tend, n_t + 1)
        self.t1 = t_axis
        t_axis, tau_axis, g1 = self.G1_tl()
        _omega_max = np.abs(omega_max) + np.abs(omega_min)
        n_omega = int(_omega_max / domega)
        omega_axis = np.linspace(omega_min, omega_max, n_omega + 1)
        S_omega_t = compute_s_t_incremental(omega_axis, t_axis, tau_axis, g1)
        if plot:
            plt.clf()
            plt.pcolormesh(omega_axis, t_axis, np.log(np.abs(S_omega_t.T) + 0.001),
                            shading='gouraud', vmin=-1, vmax=3)
            plt.xlabel("Frequency (meV)")
            plt.ylabel("Time (ps)")
            plt.colorbar(label="log(S(omega,t))")
            plt.savefig("time_dep_spectrum_tl.png")
            plt.clf()
            plt.plot(omega_axis, np.log(np.abs(S_omega_t[:, -1]) + 0.001))
            plt.xlabel("Frequency (meV)")
            plt.ylabel("log(S(omega,tend))")
            plt.savefig("spectrum_at_tend.png")
        return omega_axis, t_axis, S_omega_t

    def get_time_dependent_spectrum(self, tend, omega_min=-5, omega_max=5, domega=0.1):
        """
        Compute S(omega, t) = Re(int_0^t dt' int_0^{t-t'} dtau G1(t',tau) exp(-i omega tau))
        on a regular time grid matching n*dt.
        """
        _dt = self.dt
        self.tend = round_to_dt(tend, _dt)
        n_t = int(tend / _dt)
        t_axis = np.linspace(0, tend, n_t + 1)
        self.t1 = t_axis
        t_axis, tau_axis, g1 = self.G1_tl()
        tau_axis = tau_axis[::int(_dt / self.dt)]
        g1 = g1[:, ::int(_dt / self.dt)]
        _omega_max = np.abs(omega_max) + np.abs(omega_min)
        n_omega = int(_omega_max / domega)
        omega_axis = np.linspace(omega_min, omega_max, n_omega + 1)
        S_omega_t = compute_s_t_incremental(omega_axis, t_axis, tau_axis, g1)
        plt.clf()
        plt.pcolormesh(omega_axis, t_axis, np.log(np.abs(S_omega_t.T) + 0.001),
                       shading='gouraud', vmin=-1, vmax=3)
        plt.xlabel("Frequency (meV)")
        plt.ylabel("Time (ps)")
        plt.colorbar(label="log(S(omega,t))")
        plt.savefig("time_dep_spectrum.png")
        return S_omega_t
