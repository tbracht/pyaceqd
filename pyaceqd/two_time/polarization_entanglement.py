import numpy as np
import os
import warnings
from pyaceqd.tools import concurrence
from pyaceqd.helpers.time_axes import round_to_dt, UnregularTimeAxis
from pyaceqd.helpers.dynamical_map import calc_tl_dynmap_pseudo, extract_dms
import tqdm
from concurrent.futures import ThreadPoolExecutor, wait
import matplotlib.pyplot as plt
from pyaceqd.constants import hbar
import pyaceqd.constants as constants
try:
    from pyaceqd.two_time import propagate_tau_module
except ImportError as exc:
    _PROPAGATE_TAU_TL_IMPORT_ERROR = exc
    warnings.warn("propagate_tau_module not found, time-local acceleration unavailable.",
                  ImportWarning,
                  stacklevel=2,)
    propagate_tau_module = None
temp_dir = constants.temp_dir


def _require_propagate_tau_tl() -> None:
    if propagate_tau_module is None:
        raise RuntimeError(
            "The optional Fortran module 'propagate_tau_module' is not available. "
            "Reinstall with Fortran build enabled to use time-local accelerated routines."
        ) from _PROPAGATE_TAU_TL_IMPORT_ERROR

def get_max_pulse_t(pulses):
    times = []
    for p in pulses:
        times.append(p.t_center + 4*p.tau)
    if len(times) == 0:
        return 0
    return max(times)


class PolarizationEntanglement():
    def __init__(self, system, sigma_x, sigma_y, *pulses, tend=800, 
                 dt_small=0.1, max_pulse_t=None, workers=2, t_mem=10,
                 dt_big=None, regular_stepping=False,
                 variable_stepping=False, exponential_stepping=False, verbose=False) -> None:
        """
        Parameters
        ----------
        system : System
            System that is used for the simulation
        sigma_x : str
            transition operator for x-polarization
        sigma_y : str
            transition operator for y-polarization
        pulses : list of Pulses
            Pulses that are used for the simulation
        tend : float
            final time for t and tau. This means that maximal simulation time is 2*tend for t=tend and tau=tend 
        dt_small : float, optional
            Small timestep for the correlations
        max_pulse_t : float, optional
            Gaussian timestep for the simulation
        workers : int, optional
            Number of threads spawned by ThreadPoolExecutor
        t_mem : float, optional
            Memory time for phonon dynamics, used to calculate time-local maps
        dt_big : float, optional
            Big timestep for the simulation, used for times where no pulse is present. If None, it is set to 10*dt_small
        regular_stepping : bool, optional
            If True, use a regular t-grid with spacing dt_small,
            disregarding settings for variable stepping and exponential stepping
        variable_stepping : bool, optional
            If True, use variable stepping with dt_big for large times, and dt_small for small times
        exponential_stepping : bool, optional
            If True, use exponential stepping for the t-grid, with dt_small for small times and exponentially increasing step size for larger times
        verbose : bool, optional
            Verbose output
        """
        self.system = system  # system that is used for the simulation
        self.dt = system.dt  # timestep during simulation
        self.tend = tend  # final time t for the simulation
        self.pulses = pulses
        self.t_mem = t_mem  # memory time for phonon dynamics
        self.workers = workers  # number of threads spawned by ThreadPoolExecutor
        self.verbose = verbose
        self.dim = system.dim
        self.phonons = system.phonons
        if not system.lindblad:
            print("WARNING: system is not using lindblad operators, eg. no decay.")
        self.ax = sigma_x
        self.ay = sigma_y
        self.axdag = np.conjugate(sigma_x.T)
        self.aydag = np.conjugate(sigma_y.T)
        if sigma_x.shape != (self.dim, self.dim) or sigma_y.shape != (self.dim, self.dim):
            print("Operator dimension mismatch: expected shape ({dim}, {dim}), got sigmax: {sigma_x.shape}, sigmay: {sigma_y.shape}")
            raise ValueError("Operators must have shape (dim, dim), where dim is the dimension of the system.")
        self.tl_map = None  # time-local dynamical map

        # construct time grid
        if max_pulse_t is None:
            print("Warning: max_pulse_t is not set. Default is t_center+4*tau of the last pulse.")
            max_pulse_t = get_max_pulse_t(pulses)
        self.max_pulse_t = max_pulse_t
        
        self.dt_big = dt_big
        if self.dt_big is None:
            self.dt_big = 10*dt_small
        self.dt_small = dt_small

        time_generator = UnregularTimeAxis(0, tend, self.max_pulse_t + self.t_mem,
                                           dt_small=dt_small, dt_big=dt_big, pulses=pulses,
                                           include_tend=True, round_dt=True)
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

    def get_tl(self):
        self._t_mem = round_to_dt(self.max_pulse_t, self.dt)  # use max_pulse_t as memory time
        if self.max_pulse_t is None:
            raise ValueError("max_pulse_t must be set to calculate time-local maps, it is used as 'memory time' for phonon-free dynamics.")
        tend = 2*self._t_mem
        # _options = self.options.copy()
        # _options["pulse_file_x"] = self.pulse_file_x
        # _options["pulse_file_y"] = self.pulse_file_y
       
        _t, dm = self.system.run(0, tend, *self.pulses, multitime_op=[], calc_dynmap=True)
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        # check_tlmap_frobenius(dm_tl, _t, xlim=tend-self.dt)
        # memory_time = self._t_mem # if self.max_pulse_t is not None else self.tend
        tl_map, dms = extract_dms(dm_tl, _t, self._t_mem, t_MTOs=[])
        
        self.tl_map = tl_map
        self.tl_dms = dms[0]
        self.dm = dm
        self.dm_tl = dm_tl

    def get_tl_phonons(self, mtos=[], t_mtos=[]):
        tmem = self.max_pulse_t + self.t_mem
        tend = 2.1*tmem
        _t, dm = self.system.run(0, tend, *self.pulses, multitime_op=mtos, calc_dynmap=True)
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        tl_map, dms = extract_dms(dm_tl, _t, tmem, t_MTOs=t_mtos)
        dms = np.array(dms, dtype=complex)
        return tl_map, dms
    
    def G2_tl(self, op1_t, op2_ttau, op3_ttau, op4_t, tau_max=None, time_sparse=None, return_final_tau=False):
        _require_propagate_tau_tl()
        if self.phonons:
            return self.G2_tl_phonons(op1_t, op2_ttau, op3_ttau, op4_t)
        dim = np.shape(op1_t)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0  # initial state, rho0 = |0><0|
        if tau_max is None:
            tau_max=self.tend
        n_tau = int(tau_max/self.dt)
        tau = np.linspace(0, tau_max, n_tau + 1)
        if time_sparse is None:
            time_sparse = self.t1
        # calc tl maps:
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()
        dm_tl = np.asfortranarray(self.tl_dms.transpose(1, 2, 0))
        dm_s = self.tl_map

        _tend = time_sparse[-1] + tau_max
        t_axis = np.linspace(0, _tend, int(_tend/self.dt) + 1)
        opA_mat = op1_t
        opB_mat = op2_ttau @ op3_ttau
        opC_mat = op4_t
        G2 = propagate_tau_module.calc_onetime_simple(dm_block=dm_tl,dm_s=dm_s,rho_init=rho0.reshape(dim**2),n_tb=int(tau_max/self.dt),dim=dim,
                                                      opa=opA_mat,opb=opB_mat,opc=opC_mat,time=np.round(t_axis,6),time_sparse=np.round(time_sparse,6))
        # print(np.shape(G2_ttau))
        # print("G2 shape:", np.shape(G2))
        # print("tau shape:", np.shape(tau))
        if return_final_tau:
            # return t, G2(t), G2_integrated, G2(t,tau_max)
            return time_sparse, np.trapezoid(G2, tau, axis=1), np.trapezoid(np.trapezoid(G2, tau, axis=1), time_sparse, axis=0), G2, tau
        G2 = np.trapezoid(G2, tau, axis=1)
        return time_sparse, G2, np.trapezoid(G2, time_sparse, axis=0)  # , tau, G2_ttau[50]

    def G2_analytic_integrated(self, op1_t, op2_ttau, op3_ttau, op4_t):
        dim = np.shape(op1_t)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0  # initial state, rho0 = |0><0|

        tau_max=self.tend
        n_tau = int(tau_max/self.dt)
        print("n_tau:", n_tau)
        tau = np.linspace(0, tau_max, n_tau + 1)
        # calc tl maps:
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()

        dm = self.dm  # dynamical map, NOT time local. 
        dm_s = self.tl_map  # stationary time-local map
        # dm_tl = self.dm_tl
        dm_tl = self.tl_dms  # time-local dynamical maps during pulse
        print("dm shape:", np.shape(dm))
        print("dm_tl shape:", np.shape(dm_tl))
        # dm_s = dm_tl[int(self._t_mem//self.dt)+2]  # dynamical map at t = t_mem

        # dm_s = dm[-1] @ np.linalg.inv(dm[int(self._t_mem//self.dt)])
        # _dt = self.dt * len(dm) - self._t_mem
        # print("dt for eigvals:", _dt)
        # print(np.linalg.norm(dm_s - self.tl_map))

        eigvals, T = np.linalg.eig(dm_s)  # eigvals_n = (exp(z_n*dt))
        angle = 0  # fix phase of T such that T[0,0] is real and positive
        # this makes sure that repeated runs give the same result
        if (np.imag(T[0,0]) !=0 or T[0,0] < 0):
            angle = np.angle(T[0,0])
        T = T*np.exp(-1j*angle)
        T_inv = np.linalg.inv(T)
        # part: t >= t_mem, tau >= 0
        G2 = np.zeros(len(self.t1), dtype=complex)
        # op1 and op4 are brought to matrix form, op23 are brought to vector form
        op_1 = np.kron(op1_t.T, np.eye(dim))  # op_1 is applied from right at time t
        op_23 = (op2_ttau @ op3_ttau).reshape(dim**2)
        op_4 = np.kron(np.eye(dim), op4_t)  # op_4 is applied from left at time t+tau
        # t > tmem part: start with rho_tmem

        rho_tmem = dm[int(self._t_mem//self.dt)] @ rho0.reshape(dim**2)  # using only the dm_map 0->tmem, this directly propagates rho from 0 to tmem
        
        x_n2 = T_inv @ rho_tmem
        x_n1n2 = T_inv @ op_4 @ op_1 @ T
        x_n1 = op_23.T @ T
        x_n1n2 = x_n1[:,None] * x_n1n2 * x_n2[None,:]

        z_n1 = np.log(eigvals)/self.dt  #self.dt  # z_n1 = ln(eigval_n1)/dt
        z_n1[0] = -1  # avoid division by zero, which is due to the eigenvalue exp(0) = 1 corresponding to the steady state

        x_total = x_n1n2 / (z_n1[:,None]*z_n1[None,:])
        G2 = (np.sum(x_total, axis=(0,1)))  # first part: t >= t_mem, tau >= 0

        # second part: t <= tmem, tau <= tmem
        time_small = self.t1[self.t1 <= self._t_mem]
        # use existing function to calculate this part
        _, G2_smalltau, G2_smalltau_int, G2_tau, _tau = self.G2_tl(op1_t, op2_ttau, op3_ttau, op4_t, tau_max=self._t_mem, time_sparse=time_small, return_final_tau=True)
        G2 += G2_smalltau_int

        # third part: t < tmem, tau > tmem.
        rho_t = np.zeros((len(time_small), dim*dim), dtype=complex)
        rho_t[0] = rho0.reshape(dim**2)
        _propagators = np.zeros((len(time_small), dim*dim, dim*dim), dtype=complex)
        _propagators[0] = np.eye(dim*dim)
        # propagate rho_t to time_small, and store propagators from time_small to tau_mem
        for i in range(len(time_small)-1):   
            # print(i, "/", len(time_small), time_small[i])         
            rho_t[i+1] = dm_tl[i] @ rho_t[i]
            # print(np.trace(rho_t[:,i].reshape(dim,dim)))
            # rest to propagate to tau_mem
            _propagators[i] = dm_tl[i]
            for j in range(1,len(time_small)-1):  # additionally -i if we want to only go to t_mem
                if i+j < len(dm_tl):
                    _propagators[i] = dm_tl[j+i] @ _propagators[i]
                else:
                    _propagators[i] = dm_s @ _propagators[i]
                # _propagators[i] = dm_tl[j] @ _propagators[i]
        _propagators[-1] = np.linalg.matrix_power(dm_s, int((self._t_mem)/self.dt))
        # last propagator is identity, as no propagation is needed from t=t_mem to tau=0
        # _propagators[-1] = np.eye(dim*dim)

        xx_n1n2 = np.zeros((len(time_small), dim*dim, dim*dim), dtype=complex)
        for i in range(len(time_small)):
            x_n2 = op_4 @ op_1 @ rho_t[i]
            # x_n2 = op4_matrix @ rho_t[i].reshape(dim,dim) @ op1_matrix
            # x_n2 = x_n2.reshape(dim*dim)
            # x_n2 = np.einsum("ij,j->i", op_4 @ op_1 , rho_t[i])
            x_n1n2 = T_inv @ _propagators[i]
            # x_n1n2 = np.einsum("ij,jk->ik", T_inv , _propagators[i])
            x_n1 = op_23.T @ T
            # x_n1 = np.einsum("i,ij->j", op_23 , T)
            xx_n1n2[i] = x_n1[:,None] * x_n1n2 * x_n2[None,:]


        # z_n1 = np.log(eigvals)/self.dt  #self.dt  # z_n1 = ln(eigval_n1)/dt
        # z_n1[0] = -1  # avoid division by zero, which is due to the eigenvalue exp(0) = 1 corresponding to the steady state

        G2_bigtau = np.sum(- xx_n1n2 / z_n1[None,:,None], axis=(1,2))
        G2 += np.trapezoid(G2_bigtau, time_small, axis=0)
        return G2

    def G2_tl_analytic(self, op1_t, op2_ttau, op3_ttau, op4_t):
        dim = np.shape(op1_t)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0  # initial state, rho0 = |0><0|

        tau_max=self.tend
        n_tau = int(tau_max/self.dt)
        print("n_tau:", n_tau)
        tau = np.linspace(0, tau_max, n_tau + 1)
        # calc tl maps:
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()

        dm = self.dm  # dynamical map, NOT time local. 
        dm_s = self.tl_map  # stationary time-local map
        # dm_tl = self.dm_tl
        dm_tl = self.tl_dms  # time-local dynamical maps during pulse
        print("dm shape:", np.shape(dm))
        print("dm_tl shape:", np.shape(dm_tl))
        # dm_s = dm_tl[int(self._t_mem//self.dt)+2]  # dynamical map at t = t_mem

        # dm_s = dm[-1] @ np.linalg.inv(dm[int(self._t_mem//self.dt)])
        # _dt = self.dt * len(dm) - self._t_mem
        # print("dt for eigvals:", _dt)
        # print(np.linalg.norm(dm_s - self.tl_map))

        eigvals, T = np.linalg.eig(dm_s)  # eigvals_n = (exp(z_n*dt))
        angle = 0  # fix phase of T such that T[0,0] is real and positive
        # this makes sure that repeated runs give the same result
        if (np.imag(T[0,0]) !=0 or T[0,0] < 0):
            angle = np.angle(T[0,0])
        T = T*np.exp(-1j*angle)
        T_inv = np.linalg.inv(T)
        # part: t >= t_mem, tau >= 0
        G2 = np.zeros(len(self.t1), dtype=complex)
        # op1 and op4 are brought to matrix form, op23 are brought to vector form
        op_1 = np.kron(op1_t.T, np.eye(dim))  # op_1 is applied from right at time t
        op_23 = (op2_ttau @ op3_ttau).reshape(dim**2)
        op_4 = np.kron(np.eye(dim), op4_t)  # op_4 is applied from left at time t+tau
        # t > tmem part: start with rho_tmem

        rho_tmem = dm[int(self._t_mem//self.dt)] @ rho0.reshape(dim**2)  # using only the dm_map 0->tmem
        # rho_tmem2 = rho0.reshape(dim**2)
        # for i in range(len(dm_tl)):
        #     rho_tmem2 = dm_tl[i] @ rho_tmem2
        # print(np.linalg.norm(rho_tmem - rho_tmem2))

        x_n2 = T_inv @ rho_tmem
        x_n1n2 = T_inv @ op_4 @ op_1 @ T
        x_n1 = op_23.T @ T
        x_n1n2 = x_n1[:,None] * x_n1n2 * x_n2[None,:]

        z_n1 = np.log(eigvals)/self.dt  #self.dt  # z_n1 = ln(eigval_n1)/dt
        z_n1[0] = -1  # avoid division by zero, which is due to the eigenvalue exp(0) = 1 corresponding to the steady state
      
        x_total = (- x_n1n2 / z_n1[:,None])[:,:,None] * (eigvals[None,:,None]**((self.t1-self._t_mem)/self.dt)[None,None,:])
        # x_total = (- x_n1n2 / z_n1[:,None])[:,:,None] * (np.exp(z_n1[None,:,None]*(self.t1-self._t_mem)[None,None,:]))

        G2 = (np.sum(x_total, axis=(0,1)))
        
        # _tau = np.linspace(0, 200, int(200/self.dt) + 1)
        # _t1 = self.t1[self.t1 >= self._t_mem]
        # _t1 = _t1[_t1 <= self._t_mem + 200]
        # one = np.ones_like(x_n1n2)
        # G2_analytic = np.sum(one[:,:,None,None] * (eigvals[None,:,None,None]**((_t1[None,None,:,None] +_tau[None,None,None,:] -self._t_mem)/self.dt)), axis=(0,1))
        # plt.clf()
        # plt.pcolormesh(_t1, _tau, (G2_analytic.real),
        #                   shading='auto')
        # plt.xlabel("tau")
        # plt.ylabel("t")
        # plt.colorbar(label="|G2_analytic(t,tau)|")
        # plt.savefig("test_polent_g2_t_tau_analytic.png")
        # plt.clf()


        # return self.t1, G2, np.trapezoid(G2, self.t1, axis=0)
        # part: t < t_mem, tau >= 0
        # subpart: tau < t_mem
        time_small = self.t1[self.t1 <= self._t_mem]
        # use existing function to calculate this part
        _, G2_smalltau, G2_smalltau_int, G2_tau, _tau = self.G2_tl(op1_t, op2_ttau, op3_ttau, op4_t, tau_max=self._t_mem, time_sparse=time_small, return_final_tau=True)
        # G2_taumax = G2_tau[:,-1]  # G2 at tau = t_mem
        # plt.clf()
        # plt.pcolormesh(_tau, time_small, np.abs(G2_tau), shading='auto')
        # plt.xlabel("tau")
        # plt.ylabel("t")
        # plt.colorbar(label="|G2(t,tau)|")
        # plt.savefig("test_polent_g2_t_tau.png")
        # plt.clf()

        # n_tausmall = int(self._t_mem/self.dt)
        # # print("n_tausmall:", n_tausmall)
        # tau = np.linspace(0, self._t_mem, n_tausmall + 1)
        # G2_small_t_tau = np.zeros((len(time_small), len(tau)), dtype=complex)
        # rho_tnow = np.zeros(dim*dim, dtype=complex)
        # rho_tnow = rho0.reshape(dim**2)
        # G2_small_t_tau[0,:] = 0
        # op_23_matrix = op2_ttau @ op3_ttau
        # op1_matrix = op1_t
        # op4_matrix = op4_t
        # for i in tqdm.trange(len(time_small)-1):
        #     # propagate to t
        #     rho_tnow = dm_tl[i] @ rho_tnow
        #     # propagate to tau
        #     # rho_ttau = op_4 @ (op_1 @ rho_tnow)
        #     rho_ttau = (op4_matrix @ rho_tnow.reshape(dim,dim) @ op1_matrix).reshape(dim**2)
        #     G2_small_t_tau[i+1,0] = np.trace(op_23_matrix @ np.reshape(rho_ttau, (dim,dim)))  # tau = 0
        #     # TODO: needs to be adjusted when time axis is not regular
        #     for j in range(n_tausmall-i):  # max tau such that t+tau <= t_mem
        #         if i+j < len(dm_tl):
        #             rho_ttau = dm_tl[i+j] @ rho_ttau
        #         else:
        #             rho_ttau = dm_s @ rho_ttau
        #         G2_small_t_tau[i+1,j] = np.trace(op_23_matrix @ np.reshape(rho_ttau, (dim,dim)))
        # G2_smalltau = np.trapezoid(G2_small_t_tau, tau, axis=1)

        # part t < t_mem, tau >= t_mem
        # G2_bigtau = self.dt*G2_taumax * np.sum(- eigvals[1:]**((self._t_mem)/self.dt)/ z_n1[1:] )
        # G2_bigtau = G2_taumax * 1/self.options["gamma_e"]
        # print(np.sum(z_n1[:1]))

        rho_t = np.zeros((len(time_small), dim*dim), dtype=complex)
        rho_t[0] = rho0.reshape(dim**2)
        _propagators = np.zeros((len(time_small), dim*dim, dim*dim), dtype=complex)
        _propagators[0] = np.eye(dim*dim)
        # propagate rho_t to time_small, and store propagators from time_small to tau_mem
        for i in range(len(time_small)-1):   
            # print(i, "/", len(time_small), time_small[i])         
            rho_t[i+1] = dm_tl[i] @ rho_t[i]
            # print(np.trace(rho_t[:,i].reshape(dim,dim)))
            # rest to propagate to tau_mem
            _propagators[i] = dm_tl[i]
            for j in range(1,len(time_small)-1):  # additionally -i if we want to only go to t_mem
                if i+j < len(dm_tl):
                    _propagators[i] = dm_tl[j+i] @ _propagators[i]
                else:
                    _propagators[i] = dm_s @ _propagators[i]
                # _propagators[i] = dm_tl[j] @ _propagators[i]
        _propagators[-1] = np.linalg.matrix_power(dm_s, int((self._t_mem)/self.dt))
        # last propagator is identity, as no propagation is needed from t=t_mem to tau=0
        # _propagators[-1] = np.eye(dim*dim)

        xx_n1n2 = np.zeros((len(time_small), dim*dim, dim*dim), dtype=complex)
        for i in range(len(time_small)):
            x_n2 = op_4 @ op_1 @ rho_t[i]
            # x_n2 = op4_matrix @ rho_t[i].reshape(dim,dim) @ op1_matrix
            # x_n2 = x_n2.reshape(dim*dim)
            # x_n2 = np.einsum("ij,j->i", op_4 @ op_1 , rho_t[i])
            x_n1n2 = T_inv @ _propagators[i]
            # x_n1n2 = np.einsum("ij,jk->ik", T_inv , _propagators[i])
            x_n1 = op_23.T @ T
            # x_n1 = np.einsum("i,ij->j", op_23 , T)
            xx_n1n2[i] = x_n1[:,None] * x_n1n2 * x_n2[None,:]


        # z_n1 = np.log(eigvals)/self.dt  #self.dt  # z_n1 = ln(eigval_n1)/dt
        # z_n1[0] = -1  # avoid division by zero, which is due to the eigenvalue exp(0) = 1 corresponding to the steady state

        G2_bigtau = np.sum(- xx_n1n2 / z_n1[None,:,None], axis=(1,2))
        # G2 = np.trapezoid(G2, tau, axis=1)
        G2[self.t1 <= self._t_mem] = G2_smalltau + G2_bigtau
        return self.t1, G2, np.trapezoid(G2, self.t1, axis=0)

    def get_dm2_phonons_advanced(self, mtos, t_mto):
        # in principle, we don't have to calculate the tl-maps up until t_mto + t_gaussian + self.t_mem + 2*self.dt,
        # but the correct final time depends on t_mto, as max_pulse_t is absolute and t_mem is always needed. 
        # this means the maximum final time will be t_gaussian + 2 * t_mem
        # while t_mto is t_gaussian + t_mem, meaning before we used a maximum of 2*t_gaussian + 2*t_mem.
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
        # for t_mto = 0, the memory time will be t_gaussian + t_mem,
        # for t_mto = 1, memory time = t_gaussian - 1 + t_mem, 
        # for t_mto = t_gaussian, memory time = t_mem
        # from then on, it is always t_mem, as this is the minimum memory time we need
        memory_time = np.max([self.max_pulse_t + self.t_mem - t_mto, self.t_mem])
        _, dms = extract_dms(dm_tl, _t, memory_time, t_MTOs=[t_mto])
        return dms[1]  # return the second dynamic map, which is the one we need for the phonon dynamics
    
    def G2_tl_phonons(self, op1_t, op2_ttau, op3_ttau, op4_t):
        _require_propagate_tau_tl()
        t_apply = self.max_pulse_t + self.t_mem + 5*self.dt
        _mto = {"operator": op4_t, "applyFrom": "left", "applyBefore": False, "time": t_apply}
        _mto2 = {"operator": op1_t, "applyFrom": "right", "applyBefore": False, "time": t_apply}
        tl_map, dms_sep = self.get_tl_phonons(mtos=[_mto,_mto2], t_mtos=[t_apply])

        dim = np.shape(op1_t)[0]
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0

        tau_max=self.tend
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
                    _e = executor.submit(self.get_dm2_phonons_advanced,[_mto,_mto2], _t_mto, i)
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

        opA_mat = op1_t
        opB_mat = op2_ttau @ op3_ttau
        opC_mat = op4_t

        G2 = propagate_tau_module.calc_onetime_simple_phonon(dm_taucs2=dm_taucs2, dm_sep1=dm_separated1, dm_sep2=dm_separated2, dm_s=dm_s,
                                                            rho_init=rho0.reshape(dim**2),n_tb=int(self.tend/self.dt),
                                                            dim=dim,opa=opA_mat,opb=opB_mat,opc=opC_mat,time=t_axis,time_sparse=self.t1)
        G2 = np.trapezoid(G2, tau, axis=1)
        return self.t1, G2, np.trapezoid(G2, self.t1, axis=0)

    def calc_densitymatrix(self, tl=False, return_rho=False):
        density_matrix = np.zeros([4,4], dtype=complex)
        with tqdm.tqdm(total=10, leave=None) as tq:
            _,_,density_matrix[0,0] = self.G2(self.axdag, self.axdag, self.ax, self.ax, tl=tl)  # xx,xx
            tq.update()
            _,_,density_matrix[3,3] = self.G2(self.aydag, self.aydag, self.ay, self.ay, tl=tl)  # yy,yy
            tq.update()
            _,_,density_matrix[1,1] = self.G2(self.axdag, self.aydag, self.ay, self.ax, tl=tl)  # xy,xy
            tq.update()
            _,_,density_matrix[2,2] = self.G2(self.aydag, self.axdag, self.ax, self.ay, tl=tl)  # yx,yx
            tq.update()

            _,_,density_matrix[0,1] = self.G2(self.axdag, self.axdag, self.ay, self.ax, tl=tl)  # xx,xy
            tq.update()
            density_matrix[1,0] = np.conj(density_matrix[0,1])
            _,_,density_matrix[0,2] = self.G2(self.axdag, self.axdag, self.ax, self.ay, tl=tl)  # xx,yx
            tq.update()
            density_matrix[2,0] = np.conj(density_matrix[0,2])
            _,_,density_matrix[0,3] = self.G2(self.axdag, self.axdag, self.ay, self.ay, tl=tl)  # xx,yy
            tq.update()
            density_matrix[3,0] = np.conj(density_matrix[0,3])

            _,_,density_matrix[1,2] = self.G2(self.axdag, self.aydag, self.ax, self.ay, tl=tl)  # xy,yx
            tq.update()
            density_matrix[2,1] = np.conj(density_matrix[1,2])
            _,_,density_matrix[1,3] = self.G2(self.axdag, self.aydag, self.ay, self.ay, tl=tl)  # xy,yy
            tq.update()
            density_matrix[3,1] = np.conj(density_matrix[1,3])

            _,_,density_matrix[2,3] = self.G2(self.aydag, self.axdag, self.ay, self.ay, tl=tl)  # yx,yy
            tq.update()
            density_matrix[3,2] = np.conj(density_matrix[2,3])

        norm = np.trace(density_matrix)
        density_matrix = density_matrix / norm
        if return_rho:
            return concurrence(density_matrix), density_matrix
        return concurrence(density_matrix)
    
    def G1(self, op1_t, op2_ttau):
        """
        calculates G1 for two operators:
        <op2(t1+tau) op1(t1)>
        """
        # check if first char of op1_t and op2_ttau is a bracket
        tau0_op = op2_ttau @ op1_t
        output_ops = [op2_ttau, tau0_op]
        # at t1, apply op1 from left
        op_1 = {"operator": op1_t, "applyFrom": "left", "applyBefore":False}

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
                    _e = executor.submit(self.system.run,0,t1[i]+tend,*self.pulses,multitime_op=[op_1_new], output_ops=output_ops)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                _, futures[i] = futures[i].result()
            # futures now contains [t,<op2>,<op2*op1>] for every i
            for i in range(len(t1)):
                _G1[i,0] = futures[i][1][-(n_tau+1)]  # tau=0
                _G1[i,1:] = futures[i][0][-n_tau:]  # tau>0
        return t1, t2, _G1
    
    def calc_timedynamics_tl_phonons(self):
        tl_map, dms = self.get_tl_phonons(mtos=[], t_mtos=[])
        dm_sep1 = dms[0]

        opt_mat = self.ax
        self.dim = np.shape(opt_mat)[0]

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
        if self.phonons:
            return self.calc_timedynamics_tl_phonons()
        if self.tl_map is None:
            # calculate the dynamical maps
            self.get_tl()
        
        opt_mat = self.ax
        self.dim = np.shape(opt_mat)[0]

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
    
    def calc_timedynamics(self, output_ops=None):
        new_options = dict(self.options)
        if output_ops is not None:
            new_options["output_ops"] = output_ops
        return self.system.run(0, self.tend, *self.pulses, output_ops=new_options.get("output_ops", []))
    
    def get_spectrum(self, op1_t, op2_ttau, save_g1_dir=None, load=None):
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
            t_axis, tau_axis, g1 = self.G1(op1_t, op2_ttau)
        if save_g1_dir is not None and load is None:
            np.save(save_g1_dir + "g1.npy", g1)
            np.save(save_g1_dir + "t_axis.npy", t_axis)
            np.save(save_g1_dir + "tau_axis.npy", tau_axis)
        dtau = np.abs(tau_axis[1] - tau_axis[0])
        fft_freqs = -2*np.pi * hbar * np.fft.fftfreq(2*len(tau_axis)-1,d=dtau)
        # symmetrize g1
        # g1[:,-10:] = 0 + 0j
        g1_symm = np.empty([len(t_axis),2*len(tau_axis)-1],dtype=complex)
        g1_symm[:,:len(tau_axis)] = g1[:,::-1]
        g1_symm[:,-(len(tau_axis)-1):] = np.conj(g1[:,1:])
        spectra = np.empty([len(g1_symm),len(g1_symm[0])],dtype=complex)
        for j in range(len(g1_symm)):
            # do fft for every t, along the tau axis
            spectra[j] = np.fft.fftshift(np.fft.fft(g1_symm[j]))
        spectrum = np.real(np.trapezoid(spectra.transpose(),t_axis))
        return np.fft.fftshift(fft_freqs), spectrum, spectra

    def G2(self, op1_t, op2_ttau, op3_ttau, op4_t, tl=False):
        """
        calculates G2 for four operators:
        <op1(t1) op2(t1+tau) op3(t1+tau) op4(t1)>
        returns the integral of G2 over t1 and tau
        """
        if tl:
            return self.G2_tl(op1_t, op2_ttau, op3_ttau, op4_t)
        op23_ttau = op2_ttau @ op3_ttau
        tau0_op = op1_t @ op23_ttau @ op4_t
        output_ops = [op23_ttau, tau0_op]
        # at t1, apply op4 from left and op1 from right
        op_1 = {"operator": op1_t, "applyFrom": "right", "applyBefore":False}
        op_4 = {"operator": op4_t, "applyFrom": "left", "applyBefore":False}

        t1 = self.t1
        n_tau = int((self.tend)/self.dt)
        # simulation time-axis
        t2 = np.linspace(0, self.tend, n_tau + 1)
        _G2 = np.zeros([len(t1)], dtype=complex)
        tend = self.tend  # always the same
        # _G2ttau=None
        # taunew=None
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    op_1_new = dict(op_1)  # must make a copy of the dict
                    op_4_new = dict(op_4)
                    op_1_new["time"] = t1[i]
                    op_4_new["time"] = t1[i]
                    # apply op4 from left and sigma_bbdag from right
                    multitme_ops = [op_1_new, op_4_new]
                    _e = executor.submit(self.system.run,0,tend,*self.pulses,multitime_op=multitme_ops, output_ops=output_ops)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                _, futures[i] = futures[i].result()
            # futures now contains [t,<op2*op3>,<op1*op2*op3*op4>] for every i
            for i in range(len(t1)):
                # t2 = t1,...,tend
                n_t2 = n_tau - int((t1[i])/self.dt)
                temp_t2 = np.zeros(n_t2+1, dtype=complex)
                # special case tau=0:
                # as then, Tr(op1*op2*op3*op4 * rho) = G2(t,0), which is the value with index [2][-(n_t2+1)]
                temp_t2[0] = futures[i][1][-(n_t2+1)]
                # futures[i][2] are the corresponding values, [1] are the values for tau>0, when the operators are applied separately
                # here, we want the <op2*op3>-values for every t2=t1,..,tend
                if n_t2 > 0: 
                    temp_t2[1:n_t2+1] = futures[i][0][-n_t2:]
                t_new = t2[:len(temp_t2)]
                # plt.clf()
                # plt.plot(t_new,np.real(temp_t2),'r-')
                # plt.plot(t_new,np.imag(temp_t2),'b-')
                # plt.savefig("aa_tests/plot_{}.png".format(i))
                # integrate over t_new
                _G2[i] = np.trapezoid(temp_t2,t_new)
                # if i == 50: # debug
                #     _G2ttau = temp_t2
                #     taunew = t_new
        return t1, _G2, np.trapezoid(_G2,t1)  # , taunew, _G2ttau
    
    def calc_densitymatrix_reuse(self, plot_G2=None, return_counts=False, return_rho=False):
        density_matrix = np.zeros([4,4], dtype=complex)
        with tqdm.tqdm(total=3, leave=None) as tq:
            # XX,XX; XX,XY; XY,XY
            op23s = [self.axdag @ self.ax, self.axdag @ self.ay, self.aydag @ self.ay]
            t1, G2_1_t, G2_1 = self.G2_reuse(self.axdag, op23s, self.ax)
            tq.update()
            # XX,YX; XX,YY; XY,YX; XY,YY
            op23s = [self.axdag @ self.ax, self.axdag @ self.ay, self.aydag @ self.ax, self.aydag @ self.ay]
            t2, G2_2_t, G2_2 = self.G2_reuse(self.axdag, op23s, self.ay)
            tq.update()
            # YX,YX; YX,YY; YY,YY
            op23s = [self.axdag @ self.ax, self.axdag @ self.ay, self.aydag @ self.ay]
            t3, G2_3_t, G2_3 = self.G2_reuse(self.aydag, op23s, self.ay)
            tq.update()

            density_matrix[0,0] = np.abs(G2_1[0])  # xx,xx
            density_matrix[3,3] = np.abs(G2_3[2])  # yy,yy
            density_matrix[1,1] = np.abs(G2_1[2])  # xy,xy
            density_matrix[2,2] = np.abs(G2_3[0])  # yx,yx

            density_matrix[0,1] = G2_1[1]  # xx,xy
            density_matrix[1,0] = np.conj(density_matrix[0,1])
            density_matrix[0,2] = G2_2[0]  # xx,yx
            density_matrix[2,0] = np.conj(density_matrix[0,2])
            density_matrix[0,3] = G2_2[1]  # xx,yy
            density_matrix[3,0] = np.conj(density_matrix[0,3])

            density_matrix[1,2] = G2_2[2]  # xy,yx
            density_matrix[2,1] = np.conj(density_matrix[1,2])
            density_matrix[1,3] = G2_2[3]  # xy,yy
            density_matrix[3,1] = np.conj(density_matrix[1,3])

            density_matrix[2,3] = G2_3[1]  # yx,yy
            density_matrix[3,2] = np.conj(density_matrix[2,3])

        norm = np.trace(density_matrix)

        if plot_G2 is not None:
            plt.clf()
            plt.plot(t1, np.abs(G2_1_t[0]), label="xx,xx")
            plt.plot(t1, np.abs(G2_1_t[2]), label="xy,xy")
            plt.plot(t2, np.abs(G2_2_t[1]), label="xx,yy")
            plt.plot(t3, np.abs(G2_3_t[0]), dashes=[4,4],label="yx,yx")
            plt.plot(t3, np.abs(G2_3_t[2]), dashes=[4,4],label="yy,yy")
            plt.xlabel("t (ps)")
            plt.ylabel("G2(t)")
            plt.legend()
            plt.savefig("{}.png".format(plot_G2))
            np.save("{}.npy".format(plot_G2), np.array([t1, G2_1_t[0], G2_1_t[1], G2_1_t[2], G2_2_t[0], G2_2_t[1], G2_2_t[2], G2_2_t[3], G2_3_t[0], G2_3_t[1], G2_3_t[2]]))
        if return_rho:
            return concurrence(density_matrix/norm), density_matrix
        if return_counts:
            return concurrence(density_matrix/norm), density_matrix[0,0], density_matrix[1,1], density_matrix[2,2], density_matrix[3,3], density_matrix[0,3]
        
        return concurrence(density_matrix/norm)
    

    def calc_timedep_data(self):
        with tqdm.tqdm(total=3, leave=None) as tq:
            # 0 XX,XX; 1 XX,XY; 2 XY,XY
            op23s = [self.axdag @ self.ax, self.axdag @ self.ay, self.aydag @ self.ay]
            t1, t2, _, _, G2_1_full = self.G2_reuse(self.axdag, op23s, self.ax, return_full_G2=True)
            tq.update()
            # 3 XX,YX; 4 XX,YY; 5 XY,YX; 6 XY,YY
            op23s = [self.axdag @ self.ax, self.axdag @ self.ay, self.aydag @ self.ax, self.aydag @ self.ay]
            t1, t2, _, _, G2_2_full = self.G2_reuse(self.axdag, op23s, self.ay, return_full_G2=True)
            tq.update()
            # 7 YX,YX; 8 YX,YY; 9 YY,YY
            op23s = [self.axdag @ self.ax, self.axdag @ self.ay, self.aydag @ self.ay]
            t1, t2, _, _, G2_3_full = self.G2_reuse(self.aydag, op23s, self.ay, return_full_G2=True)
            tq.update()
        return t1,t2,np.append(G2_1_full,np.append(G2_2_full,G2_3_full,axis=0),axis=0)

    def calc_timedependent_rho(self, plot_G2=None, t1=None, t2=None, G2_full=None, t=None, G2_t=None, add_norm=0, mode="t", skip=0, return_G2=False):
        if t is None or G2_t is None:
            if t1 is None or t2 is None or G2_full is None:
                t1,t2,G2_full = self.calc_timedep_data()
            if mode == "t":
                t, G2_t = self.integrate_timedep_G2(t1,t2,G2_full)
            if mode == "tau":
                t, G2_t = self.integrate_g2_tau(t1,t2,G2_full)

        t = t[skip:]
        G2_t = G2_t[:,skip:]

        density_matrix = np.zeros([len(t),4,4], dtype=complex)
        
        density_matrix[:,0,0] = np.abs(G2_t[0])  # xx,xx
        density_matrix[:,3,3] = np.abs(G2_t[9])  # yy,yy
        density_matrix[:,1,1] = np.abs(G2_t[2])  # xy,xy
        density_matrix[:,2,2] = np.abs(G2_t[7])  # yx,yx

        density_matrix[:,0,1] = G2_t[1]  # xx,xy
        density_matrix[:,1,0] = np.conj(density_matrix[:,0,1])
        density_matrix[:,0,2] = G2_t[3]  # xx,yx
        density_matrix[:,2,0] = np.conj(density_matrix[:,0,2])
        density_matrix[:,0,3] = G2_t[4]  # xx,yy
        density_matrix[:,3,0] = np.conj(density_matrix[:,0,3])

        density_matrix[:,1,2] = G2_t[5]  # xy,yx
        density_matrix[:,2,1] = np.conj(density_matrix[:,1,2])
        density_matrix[:,1,3] = G2_t[6]  # xy,yy
        density_matrix[:,3,1] = np.conj(density_matrix[:,1,3])

        density_matrix[:,2,3] = G2_t[8]  # yx,yy
        density_matrix[:,3,2] = np.conj(density_matrix[:,2,3])

        _integrated_dm = np.trapezoid(density_matrix, t, axis=0)
        _integrated_norm = np.trace(_integrated_dm).real
        integrated_concurrence = concurrence(_integrated_dm/(_integrated_norm))

        # add uncorrelated background
        density_matrix[:,0,0] += add_norm
        density_matrix[:,3,3] += add_norm
        density_matrix[:,1,1] += add_norm
        density_matrix[:,2,2] += add_norm

        norm = np.trace(density_matrix, axis1=1, axis2=2).real
        c_t = np.zeros_like(t)
        for i in range(len(t)):
            c_t[i] = concurrence(density_matrix[i]/(norm[i]))
        if plot_G2 is not None:
            np.savez("{}.npz".format(plot_G2), t1=t1, t2=t2, G2_full=G2_full)
            plt.clf()
            plt.plot(t, np.abs(G2_t[0]), label="xx,xx")
            plt.plot(t, np.abs(G2_t[2]), label="xy,xy")
            plt.plot(t, np.abs(G2_t[4]), label="xx,yy")
            plt.plot(t, np.abs(G2_t[7]), dashes=[4,4],label="yx,yx")
            plt.plot(t, np.abs(G2_t[9]), dashes=[4,4],label="yy,yy")
            plt.xlabel("t (ps)")
            plt.ylabel("G2(t)")
            plt.legend()
            plt.savefig("{}.png".format(plot_G2))
        if return_G2:
            return t, c_t, density_matrix, norm, _integrated_dm, integrated_concurrence, G2_t
        return t, c_t, density_matrix, norm, _integrated_dm, integrated_concurrence
    
    def G2_reuse(self, op1_t, op23s_ttau, op4_t, return_full_G2=False):
        """
        re-uses the same simulation for different output operators,
        which are given in op23s_ttau

        Parameters
        ----------
        op1_t : str
            First operator at time t
        op23s_ttau : list of str
            list of operators for times t+tau
        op4_t : str
            fourth operator at time t
        return_full_G2 : bool, optional
            if true, return full G2(t,tau)

        Returns
        -------
        t1 : ndarray
            time axis t
        _G2 : ndarray
            τ integrated G2 values
        G2_integrated : ndarray
            t and τ integrated G2 values 
        G2_full : ndarray, optional
            If return_full_G2=True returns the full G2(t,tau) 
        """
        tau0_ops = []  # operators for tau=0
        for op23_ttau in op23s_ttau:
            tau0_ops.append(op1_t @ op23_ttau @ op4_t)
        output_ops = op23s_ttau + tau0_ops  # concatenate lists
        # at t1, apply op4 from left and op1 from right
        op_1 = {"operator": op1_t, "applyFrom": "right", "applyBefore":False}
        op_4 = {"operator": op4_t, "applyFrom": "left", "applyBefore":False}

        t1 = self.t1
        n_tau = int((self.tend)/self.dt)
        # simulation time-axis
        t2 = np.linspace(0, self.tend, n_tau + 1)
        _G2 = np.zeros([len(op23s_ttau),len(t1)], dtype=complex)
        
        if return_full_G2:
            G2_full = np.zeros([len(op23s_ttau), len(t1), n_tau + 1], dtype=complex)
        
        tend = self.tend  # always the same
        
        with tqdm.tqdm(total=len(t1), leave=None, desc="calculating") as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    op_1_new = dict(op_1)  # must make a copy of the dict
                    op_4_new = dict(op_4)
                    op_1_new["time"] = t1[i]
                    op_4_new["time"] = t1[i]
                    # apply op4 from left and sigma_bbdag from right
                    multitme_ops = [op_1_new, op_4_new]
                    _e = executor.submit(self.system.run,0,tend,*self.pulses,multitime_op=multitme_ops, output_ops=output_ops)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                _, futures[i] = futures[i].result()
            # futures now contains [t,<op2*op3> for all op23s,<op1*op2*op3*op4> for all op23s] for every i
            # so if op_23s has length 2, futures[i] has length 5
            # and futures[i][1,...,len(op23s)] are the <op2*op3>-values
            # and futures[i][len(op23s)+1,...,2*len(op23s)] are the <op1*op2*op3*op4>-values
            for i in range(len(t1)):
                # t2 = t1,...,tend
                n_t2 = n_tau - int((t1[i])/self.dt)
                temp_t2 = np.zeros([len(op23s_ttau),n_t2+1], dtype=complex)
                for j in range(len(op23s_ttau)):
                    # special case tau=0:
                    # as then, Tr(op1*op2*op3*op4 * rho) = G2(t,0), which is the value with index [1+len(op23s)+j][-(n_t2+1)]
                    temp_t2[j,0] = futures[i][len(op23s_ttau) + j][-(n_t2+1)]
                    # futures[i][1+len(op23s)+j] are the corresponding values, [1+j] are the values for tau>0, when the operators are applied separately
                    # here, we want the <op2*op3>-values for every t2=t1,..,tend
                    if n_t2 > 0: 
                        temp_t2[j,1:n_t2+1] = futures[i][j][-n_t2:]
                    
                    if return_full_G2:
                        G2_full[j, i, :n_t2+1] = temp_t2[j]
                
                t_new = t2[:n_t2+1]
                # integrate over t_new, i.e., over the tau axis
                for j in range(len(op23s_ttau)):
                    _G2[j,i] = np.trapezoid(temp_t2[j],t_new)
                    
        if return_full_G2:
            # t1 is t axis, t2 is tau axis
            # G2_full is the full G2(t,tau) for every operator
            return t1, t2, _G2, np.trapezoid(_G2,t1,axis=1), G2_full
        else:
            return t1, _G2, np.trapezoid(_G2,t1,axis=1)

    def integrate_g2_tau(self, t1, t2, G2_full):
        """
        Calculate tau-dependent G2 function, i.e., only integrated over t.
        this is different to the usually returned G2 function by the G2 function, which we
        usually first integrate over t and then over tau.
        Thats why we need the whole G2_full still depending on t1 (t), and t2 (tau).
        G2(τ) = ∫_0^∞ dt G(2)(t,τ)
        """

        G2_tau = np.zeros((G2_full.shape[0], len(t2)), dtype=complex)

        # iterate over all tau
        for i in tqdm.trange(len(t2),desc="t integration"):
            G2_tau[:,i] = np.trapezoid(G2_full[:, :, i], t1)
        return t2, G2_tau

    #def timedep_G2(self, op1_t, op23s_ttau, op4_t):
    def integrate_timedep_G2(self, t1, t2, G2_full):
        """
        Calculates time-dependent G2 function by integration:
        G2(t) = ∫_0^t dt' ∫_0^(t-t') dτ G(2)(t',τ)

        Parameters
        ----------
        t1: t axis
        t2: tau axis
        G2_full: G2(t,tau)

        Returns
        -------
        t : ndarray
            time axis
        G2_t : ndarray
            time-dependent G2 values 
        """


        # initialize G2(t)
        G2_t = np.zeros((G2_full.shape[0], len(t1)), dtype=complex)

        # iterate over all t
        for i in tqdm.trange(len(t1), leave=None, desc="integrating"):

            # integrate over t' from 0 to t
            # t_idx = (t1 <= t)  # indices for t' <= t
            t_prime = t1[:i+1] # should be same as t1[:i+1]
            
            # G2(t') values for the current t
            G2_tprime = np.zeros([G2_full.shape[0],len(t_prime)], dtype=complex)

            if len(t_prime) == 0:
                print("No valid t' values for t =", t1[i])
                continue
                
            # For each t' integrate over τ from 0 to (t-t')
            for j, tp in enumerate(t_prime):
                tau_max = t1[i] - tp
                tau_idx = (t2 <= tau_max)  # t2 is the tau-axis
                tau = t2[tau_idx]
                
                if len(tau) == 0:
                    print("No valid τ values for t' =", tp)
                    continue
                    
                # iterate over all components in the G2 function
                #for k in range(G2_full.shape[0]):
                    # integrate the G2 function over τ for the current t' with index j
                G2_tprime[:,j] = np.trapezoid(G2_full[:,j,tau_idx], tau)
            
            G2_t[:,i] = np.trapezoid(G2_tprime, t_prime)
        
        return t1, G2_t
