import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pyaceqd.pulses import ChirpedPulse, CWLaser
from pyaceqd.tools import construct_t, simple_t_gaussian, calc_tl_dynmap_pseudo, use_tl_map, extract_dms, use_dm_block, tl_pad_stationary, op_to_matrix, tl_pad_stationary_nsteps
from pyaceqd.two_level_system.tls import tls
from concurrent.futures import ThreadPoolExecutor, wait
from pyaceqd.constants import hbar
from pyaceqd.general_system.general_system import generate_pulsefiles
from functools import reduce
import cProfile
import pstats
import propagate_tau_module
import time

"""
This module contains functions to calculate the two-time correlation functions for any system.
"""

def _ops_one_time(system, *pulses, t0=-500, t_MTO=0, tend=500, dt=0.1, options={"lindblad": True, "phonons": False}, debug=False):
    """
    internal function to calculate the two-time correlation function, for ex. G1(tau), for a system.
    multi time operators already have to be in the options dictionary.
    """
    t,out_b,out_0 = system(t0, tend, *pulses, dt=dt, **options)
    t = np.round(t, 6)  # round to 6 digits to avoid floating point errors

    if debug:
        plt.clf()
        plt.plot(t.real, np.real(out_b))
        plt.plot(t.real, np.real(out_0))
        plt.xlim(t0, tend)
        plt.xlabel("Time (ps)")
        plt.ylabel("G(t)")
        plt.title("correlation G(t) with MTO")
        plt.savefig("g_mto.png")
    # construct the tau axis 
    n_tau = int((tend - t_MTO) / dt) + 1
    tau = np.linspace(t_MTO, tend, n_tau)
    # construct correlation function from the output
    _G = np.empty(n_tau, dtype=complex)
    i_MTO = np.where(t == t_MTO)[0][0]  # find the index where the MTO is applied
    if debug:
        print("i_MTO:{:.3f}, t[i]:{:.3f}, out0[i]:{:.3f}, out0[i+1]:{:.3f}, out0[i-1]:{:.3f}, outb[i]:{:.3f}, outb[i+1]:{:.3f}, outb[i-1]:{:.3f}".format(i_MTO, t[i_MTO], out_0[i_MTO], out_0[i_MTO + 1], out_0[i_MTO - 1], out_b[i_MTO], out_b[i_MTO + 1], out_b[i_MTO - 1]))
        
    _G[0] = out_0[i_MTO]
    _G[1:] = out_b[i_MTO + 1:]
    return tau, _G

def two_op_one_time(system, *pulses, opA="|1><0|_2", opB="|0><1|_2", t0=-500, t_MTO=0, tend=500, dt=0.1, options={"lindblad": True, "phonons": False}, debug=False):
    """
    Description
    -----------
    Calculate the two-time correlation function, for ex. G1(tau), for a system.
    <A(t_MTO+tau)B(t_MTO)>
    The propagation starts at time t0, and up until t_MTO the system is propagated regularly.
    At t_MTO operator B is applied to the system from the left, and the system is propagated until tend.

    Parameters
    ----------
    system : object
        The system to calculate the correlation function for.
    opA : object
        The first operator to use in the calculation.
    opB : object
        The second operator to use in the calculation.
    t0 : float
        The start time for the calculation.
    t_MTO : float
        The time at which the MTO (Multi-Time Operator) is applied.
    tend : float
        The end time for the calculation.
    options : dict
        A dictionary of options for the calculation.

    Returns
    -------
    tuple
        A tuple (tau, G1) containing the tau array, and correlation function 
    """

    op2 = {"operator": opB,"applyFrom": "_left", "applyBefore": "false", "time": t_MTO}
    output_ops = [opA, "(" + opA + "*" + opB + ")"]  # output for the correlation function. tau=0 has to be extracted separately
    options["output_ops"] = output_ops
    options["multitime_op"] = [op2]

    return _ops_one_time(system, *pulses, t0=t0, t_MTO=t_MTO, tend=tend, dt=dt, options=options, debug=debug)

def three_op_one_time(system, *pulses, opA="|1><0|_2", opB="|1><1|_2", opC="|0><1|_2", t0=-500, t_MTO=0, tend=500, dt=0.1, options={"lindblad": True, "phonons": False}, debug=False):
    """
    Description
    -----------
    Calculate the two-time correlation function <A(t_MTO)B(t_MTO+tau)C(t_MTO)>, for ex. G2(tau), for a given system.
    In case of G2 and a two-level system, this is <sigma^dagger(t_MTO) (sigma^dagger(t_MTO+tau)sigma(t_MTO+tau)) sigma(t_MTO)>,
    i.e., opA = sigma^dagger, opB = sigma^dagger*sigma, opC = sigma.
    The propagation starts at time t0, and up until t_MTO the system is propagated regularly.
    At t_MTO operator B is applied to the system from the left, and the system is propagated until tend.

    Parameters
    ----------
    system : object
        The system to calculate the correlation function for.
    opA : object
        The first operator to use in the calculation.
    opB : object
        The second operator to use in the calculation.
    opC : object
        The third operator to use in the calculation.
    t0 : float
        The start time for the calculation.
    t_MTO : float
        The time at which the MTO (Multi-Time Operator) is applied.
    tend : float
        The end time for the calculation.
    options : dict
        A dictionary of options for the calculation.

    Returns
    -------
    tuple
        A tuple (tau, G2) containing the tau array, and correlation function 
    """

    op1 = {"operator": opA,"applyFrom": "_right", "applyBefore": "false", "time": t_MTO}
    op2 = {"operator": opC,"applyFrom": "_left", "applyBefore": "false", "time": t_MTO}
    output_ops = [opB, "(" + opA + "*" + opB + "*" + opC + ")"]  # output for the correlation function. tau=0 has to be extracted separately
    options["output_ops"] = output_ops
    options["multitime_op"] = [op1,op2]
    return _ops_one_time(system, *pulses, t0=t0, t_MTO=t_MTO, tend=tend, dt=dt, options=options, debug=debug)

def _ops_two_time(system, t_axis, *pulses, mtos=[], tau_max=500, dt=0.1, options={"lindblad": True, "phonons": False}, debug=False, workers=15, n_mto=None, t_start=0):
    if len(mtos) < n_mto:
        raise ValueError("multi-time operators are required for the two-time correlation function.")
    extra_mtos = []
    if len(mtos) > n_mto:
        if debug:
            print("using extra multi-time operators")
        for i in range(n_mto, len(mtos)):
            extra_mtos.append(mtos[i])
    if t_start > 0:
        raise ValueError("t_start > 0 is not supported yet. Use t_start<=0 to e.g. reach a stationary state before applying the MTO.")
    t1 = t_axis
    n_tau = int(tau_max / dt)
    tau = np.linspace(0, tau_max, n_tau + 1)
    t2 = tau  # just rename
    print(options["phonons"])
    _G = np.empty((len(t1), len(t2)), dtype=complex)
    with tqdm.tqdm(total=len(t1), desc="Calculating G(t1,t2)", unit="t1") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i, t1_i in enumerate(t1):
                tend = t1_i + tau_max
                _mtos = []
                # mtos with variable time
                for j in range(n_mto):
                    _op = dict(mtos[j])
                    _op["time"] = t1_i
                    _mtos.append(dict(_op))
                # mtos with fixed time
                for j in range(len(extra_mtos)):
                    _op = dict(extra_mtos[j])
                    _mtos.append(dict(_op))
                _e = executor.submit(system, t_start, tend, *pulses, dt=dt, suffix=i, multitime_op=_mtos, **options)
                _e.add_done_callback(lambda p: pbar.update(1))
                futures.append(_e)
            wait(futures)
        for j in range(len(futures)):
            futures[j] = futures[j].result()
        for j in range(len(futures)):
            if debug and j == int(len(futures) / 2):
                plt.clf()
                plt.plot(np.abs(futures[j][0]), np.abs(futures[j][1]))
                plt.plot(np.abs(futures[j][0]), np.abs(futures[j][2]))
                plt.xlim(0, tau_max)
                plt.xlabel("Time (ps)")
                plt.ylabel("G(t)")
                plt.savefig("g_mto.png")
            _G[j,1:] = futures[j][1][-n_tau:]
            _G[j,0] = futures[j][2][-(n_tau+1)]
        return t1, t2, _G

def two_op_two_time(system, t_axis, *pulses, opA="|1><0|_2", opB="|0><1|_2", tau_max=500, dt=0.1, options={"lindblad": True, "phonons": False}, debug=False, workers=15):
    """
    Description
    -----------
    Calculates the two-time correlation function <A(t+tau)B(t)>, for ex. G1(t,tau), for a system.
    The propagation starts at time 0, and up until t the system is propagated regularly.
    At t, operator B is applied to the system from the left, and the system is propagated until t+tau_end.
    The result is <A(_t)> after _t with tau = _t-t.
    0---------t--------->|<--------tau-------->|   _t axis = time axis of simulation: 0,...,t+tau_end
    
    Parameters
    ----------
    system : object
        The system to calculate the correlation function for.
    opA : str
        The first operator to use in the calculation.
    opB : str
        The second operator to use in the calculation.
    tau_max : float
        The maximum time for the calculation.
    dt : float
        The time step for the calculation.
    options : dict
        A dictionary of options for the calculation.
    debug : bool
        plot results and debug info.
    workers : int
        The number of workers to use for the calculation.
    Returns
    -------
    tuple
        A tuple (t1, t2, G1) containing the t1 and t2 arrays, and the correlation function G1.
    """

    op2 = {"operator": opB, "applyFrom": "_left", "applyBefore": "false"}
    output_ops = [opA, "(" + opA + "*" + opB + ")"]  # output for the correlation function. tau=0 has to be extracted separately
    options["output_ops"] = output_ops
    mtos = [op2]
    
    return _ops_two_time(system, t_axis, *pulses, mtos=mtos, tau_max=tau_max, dt=dt, options=options, debug=debug, workers=workers, n_mto=1)

def three_op_two_time(system, t_axis, *pulses, opA="|1><0|_2", opB="|1><1|_2", opC="|0><1|_2", tau_max=500, dt=0.1, t_start=0, options={"lindblad": True, "phonons": False}, debug=False, workers=15):
    """
    Description
    -----------
    Calculates the two-time correlation function <A(t)B(t+tau)C(t)>, for ex. G2(t,t+tau), for a system.
    The propagation starts at time 0, and up until t the system is propagated regularly.
    At t, operator B is applied to the system from the left, and the system is propagated until t+tau_end.
    The result is <A(_t)B(_t+tau)> after _t with tau = _t-t.
    0---------t--------->|<--------tau-------->|   _t axis = time axis of simulation: 0,...,t+tau_end
    
    Parameters
    ----------
    system : object
        The system to calculate the correlation function for.
    opA : str
        The first operator to use in the calculation.
    opB : str   
        The second operator to use in the calculation.
    opC : str
        The third operator to use in the calculation.
    tau_max : float
        The maximum time for the calculation.
    dt : float
        The time step for the calculation.
    t_start : float
        The start time for the calculation. If t_start<0, the system is propagated until from t_start to t=0 before applying the MTO.
    options : dict
        A dictionary of options for the calculation.
    debug : bool
        plot results and debug info.
    workers : int
        The number of workers to use for the calculation.
    Returns
    -------
    tuple
        A tuple (t1, t2, G2) containing the t1 and t2 arrays, and the correlation function G2.

    """
    op1 = {"operator": opA, "applyFrom": "_right", "applyBefore": "false"}
    op2 = {"operator": opC, "applyFrom": "_left", "applyBefore": "false"}
    output_ops = [opB, "(" + opA + "*" + opB + "*" + opC + ")"]  # output for the correlation function. tau=0 has to be extracted separately
    options["output_ops"] = output_ops
    mtos = [op1, op2]
    return _ops_two_time(system, t_axis, *pulses, mtos=mtos, tau_max=tau_max, dt=dt, options=options, debug=debug, workers=workers, n_mto=2, t_start=t_start)

def five_op_two_time(system, t_axis, *pulses, opA="|1><0|_2", opB="|1><0|_2", opC="|1><1|_2", opD="|0><1|_2", opE="|0><1|_2", tau_max=500, dt=0.1, t_start=-500, options={"lindblad": True, "phonons": False}, debug=False, workers=15):
    """
    Description
    -----------
    Calculates <A(0)B(t)C(t+tau)D(t)E(0)> for a system. t > 0, but using t_start < 0 can be reached to reach a stationary state at t=0.
    CAUTION: the value at t=0,tau=0 is not correct, it takes <B(0)C(0)D(0)> instead of <A(0)B(0)C(0)D(0)E(0)>. 
    I'm too busy to fix this right now, should also only matter in edge cases. 
    One could do a case distinction in _ops_two_time for len(mtos)>2.
    The propagation starts at t_start <=0, at t=0 the operators A and E are applied to the system, etc.

    Parameters
    ----------
    system : object
        The system to calculate the correlation function for.
    opA : str
        The first operator to use in the calculation.
    opB : str
        The second operator to use in the calculation.
    opC : str
        The third operator to use in the calculation.
    opD : str
        The fourth operator to use in the calculation.
    opE : str
        The fifth operator to use in the calculation.
    tau_max : float
        The maximum tau for the tau axes.
    dt : float
        The time step for the calculation.  
    t_start : float
        The start time for the calculation. If t_start<0, the system is propagated until from t_start to t=0 before applying the MTO.
    options : dict
        A dictionary of options for the calculation.
    debug : bool
        plot results and debug info.
    workers : int
        The number of workers to use for the calculation.
    Returns
    -------
    tuple
        A tuple (tau1, tau1, G3) containing the tau1 and tau2 arrays, and the correlation function G3.
    """
    op1 = {"operator": opA, "applyFrom": "_right", "applyBefore": "false", "time": 0}
    op2 = {"operator": opB, "applyFrom": "_right", "applyBefore": "false"}
    op3 = {"operator": opD, "applyFrom": "_left", "applyBefore": "false"}
    op4 = {"operator": opE, "applyFrom": "_left", "applyBefore": "false", "time": 0}
    output_ops = [opC, "(" + opA + "*" + opB + "*" + opC + "*" + opD + "*" + opE + ")"]  # output for the correlation function. tau=0 has to be extracted separately
    options["output_ops"] = output_ops
    mtos = [op2, op3, op1, op4]
    return _ops_two_time(system, t_axis, *pulses, mtos=mtos, tau_max=tau_max, dt=dt, options=options, debug=debug, workers=workers, n_mto=2, t_start=t_start)

def get_spectrum(g1, tau, dir="", plot=False):
    """
    Description
    -----------
    Calculate the spectrum continuous wave excitation using the FFT of the G1(tau) function.

    Parameters
    ----------
    g1 : array
        The G1(tau) function.
    tau : array
        The time array.
    dir : str
        The directory to save the plots.
    plot : bool
        Whether to plot the results or not.
    
    Returns
    -------
    tuple
        A tuple (s, omega) containing the spectrum and the frequencies, already fft-shifted.
    """
    g1 = g1.copy()
    # dtau for the FFT
    dtau = np.abs(tau[1] - tau[0])
    g1 = g1 - g1[-1]  # subtract the last value to remove the offset
    # symmetrize the g1 function for the FFT. This makes physicaly sense as negative taus lead to a flip of the operators in the correlation function
    g1 = np.concatenate((np.conj(np.flip(g1[1:])),g1))
    tau = np.concatenate((-np.flip(tau[1:]),tau))

    # do the FFT
    s_omega = np.real(np.fft.fft(g1))
    s_omega = np.fft.fftshift(s_omega)
    fft_freqs = 2*np.pi * hbar * np.fft.fftfreq(len(g1),d=dtau)  # energies in meV
    fft_freqs = np.fft.fftshift(fft_freqs)
    if plot:
        plt.clf()
        plt.plot(tau,np.abs(g1))
        plt.xlim(-1,1)
        plt.xlabel("Time (ps)")
        plt.ylabel("|G1(t)|")
        plt.savefig(dir+"g1_tendsymm.png")

        plt.clf()
        plt.plot((fft_freqs),np.log(np.abs(s_omega)))
        plt.xlim(-3,3)
        plt.ylim(-10,10)
        plt.xlabel("Frequency (meV)")
        plt.ylabel("S(omega)")
        plt.title("Spectrum")
        plt.savefig(dir+"spectrum_log.png")

        plt.clf()
        plt.plot((fft_freqs),np.abs(s_omega))
        plt.xlim(-3,3)
        plt.xlabel("Frequency (meV)")
        plt.ylabel("S(omega)")
        plt.title("Spectrum")
        plt.savefig(dir+"spectrum_nolog.png")
        plt.clf()
    return s_omega, fft_freqs

                                             
# # example usage
# p1 = CWLaser(e0=0.05, e_start=0)
# # G1 and spectrum of tls under CW excitation
# tau, G1 = two_op_one_time(tls, p1, opA="|1><0|_2", opB="|0><1|_2", t0=-500, t_MTO=0, tend=500, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0}, debug=True)
# get_spectrum(G1,tau,dir="pyaceqd/tests/", plot=True)
# # G2 of a tls under CW excitation
# tau, G2 = three_op_one_time(tls, p1, opA="|1><0|_2", opB="|1><1|_2", opC="|0><1|_2", t0=-500, t_MTO=0, tend=500, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0}, debug=True)


# p1 = ChirpedPulse(tau_0=5,e_start=0,e0=3,t0=4*5)
# t_axis = simple_t_gaussian(0, 8*5, 50, 0.1, 1, p1)

# t1,t2,G1 = two_op_two_time(tls, t_axis, p1, opA="|1><0|_2", opB="|0><1|_2", tau_max=50, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0}, debug=True)
# plt.clf()
# plt.pcolormesh(t1,t2,np.abs(G1).T,shading="gouraud")
# plt.colorbar()
# plt.xlabel("t1")
# plt.ylabel("t2")
# plt.title("G1(t1,t2)")
# plt.savefig("pyaceqd/tests/g1_t1t2.png")
# plt.clf()

# # test against old G1 function
from pyaceqd.two_time.G1 import G1_twols
# t,tau,g1 = G1_twols(0,50,0,50,0.1,0.5,p1, gamma_e=2/100, phonons=False)
# plt.clf()
# plt.pcolormesh(t,tau, np.abs(g1).T,shading="gouraud")
# plt.colorbar()
# plt.xlabel("t")
# plt.ylabel("tau")
# plt.title("G1(t,tau)")
# plt.savefig("pyaceqd/tests/g1_t_tau.png")
# plt.clf()

# t1,t2,G2 = three_op_two_time(tls, t_axis, p1,opA="|1><0|_2", opB="|1><1|_2", opC="|0><1|_2", tau_max=50, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0}, debug=True)
# plt.clf()
# plt.pcolormesh(t1,t2,np.abs(G2).T,shading="gouraud")
# plt.colorbar()
# plt.xlabel("t1")
# plt.ylabel("t2")
# plt.title("G2(t1,t2)")
# plt.savefig("pyaceqd/tests/g2_t1t2.png")
# plt.clf()

# p1 = CWLaser(e0=0.01, e_start=0)
# t_axis = np.arange(0, 500, 5)
# t1,t2,G3 = five_op_two_time(tls, t_axis, p1, opA="|1><0|_2", opB="|1><0|_2", opC="|1><1|_2", opD="|0><1|_2", opE="|0><1|_2", tau_max=500, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0, "temperature": 4}, debug=True)
# plt.clf()
# plt.pcolormesh(t1,t2,np.abs(G3).T,shading="gouraud")
# plt.colorbar()
# plt.xlabel("t1")
# plt.ylabel("t2")
# plt.title("G3(t1,t2)")
# plt.savefig("pyaceqd/tests/g3_t1t2.png")
# plt.clf()

# cProfile.run('five_op_two_time(tls, t_axis, p1, opA="|1><0|_2", opB="|1><0|_2", opC="|1><1|_2", opD="|0><1|_2", opE="|0><1|_2", tau_max=500, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0, "temperature": 4}, debug=True)', 'restats')
# p = pstats.Stats('restats')
# p.strip_dirs().sort_stats('cumulative').print_stats(10)

def chain_apply(dm_slice, state):
    for mat in reversed(dm_slice):
        state = mat @ state
    return state

def tl_two_op_two_time(system, t_axis, *pulses, t_mem=10, opA="|1><0|_2", opB="|0><1|_2", tau_max=500, dt=0.1, rho0=np.array([[1,0],[0,0]],dtype=complex), options={"lindblad": True, "phonons": False}, debug=False, workers=15, use_dm=False, fortran_only=False):
    """
    Description
    -----------
    Calculate the two-time correlation function <A(t+tau)B(t)>, for ex. G1(t,t+tau), for a system.
    The propagation starts at time 0, and up until t the system is propagated regularly.
    At t, operator B is applied to the system from the left, and the system is propagated until t+tau_end.
    The result is <A(_t)> after _t with tau = _t-t.
    0---------t--------->|<--------tau-------->|   _t axis = time axis of simulation: t_start,...,0...,t+tau_end
    
    Parameters
    ----------
    system : object
        The system to calculate the correlation function for.
    pulses : list
        The list of pulses to use in the calculation.
    t_mem : float
        memory time of the environment. 
    opA : str
        The first operator to use in the calculation.
    opB : str
        The second operator to use in the calculation.
    options : dict
        A dictionary of options for the calculation.
    debug : bool
        plot results and debug info.
    workers : int
        The number of workers to use for the calculation.

    Returns
    -------
    tuple
        A tuple (tau, G1) containing the tau array, and correlation function 
    """

    # check if all values of t_axis can be divided by dt without remainder
    # if not np.all(np.isclose(np.round(t_axis,6) % dt, 0)):
    #     print("t_axis", t_axis%dt)
    #     raise ValueError("t_axis values must be divisible by dt without remainder.")
    if not t_axis[0] == 0:
        raise ValueError("t_axis must start at 0.")

    # get matrix representation of the operators
    opA_mat = op_to_matrix(opA)
    opB_mat = op_to_matrix(opB)

    # first we need to get the dynamic maps
    mto = {"operator": opB, "applyFrom": "_left", "applyBefore": "false", "time": 2*t_mem}
    result, dm = system(0,4*t_mem, *pulses, dt=dt, rho0=rho0, multitime_op=[mto], calc_dynmap=True, **options)
    _t = result[0]  # time axis for getting the dynamic maps
    _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
    dm_tl = calc_tl_dynmap_pseudo(dm, _t)
    tl_map, dms = extract_dms(dm_tl, _t, t_mem,[2*t_mem])
    n_tau = int(tau_max / dt)
    tau = np.linspace(0, tau_max, n_tau + 1)
    G1 = np.zeros((len(t_axis), len(tau)), dtype=complex)

    dim = len(rho0[0])

    if use_dm:
        tend = t_axis[-1] + tau_max
        result, dm = system(0, tend, *pulses, dt=dt, rho0=rho0, multitime_op=[], calc_dynmap=True, **options)
        _t = result[0]  # time axis for getting the dynamic maps
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        if debug:
            rho_t = np.zeros((len(_t), dim, dim), dtype=complex)
            rho_t[0] = rho0.copy().reshape(dim, dim)
            for i in tqdm.trange(1, len(_t)):
                rho_t[i] = (dm_tl[i-1] @ rho_t[i-1].reshape(dim**2)).reshape(dim, dim)
            plt.clf()
            plt.plot(_t, np.abs(rho_t[:,0,0]), label="rho_00")
            plt.plot(_t, np.abs(rho_t[:,1,1]), label="rho_11")
            plt.plot(_t, result[1], dashes=[2,2], label="rho_00_true")
            plt.plot(_t, result[2], dashes=[2,2], label="rho_11_true")
            plt.legend()
            plt.savefig("pyaceqd/tests/rho_tl.png")
            plt.clf()
        
        # start time for profiling
        time_start = time.time()

        if fortran_only:
            dm_tl_f = np.asfortranarray(dm_tl.transpose(1, 2, 0))
            G1 = propagate_tau_module.calc_onetime_parallel(dm_tl_f, rho0.reshape(dim**2), n_tau, dim, np.identity(dim), opA_mat, opB_mat, _t, t_axis)
            end_time = time.time()
            print(f"Time taken for tl_two_op_two_time with dm: {end_time - time_start:.2f} seconds")
            return t_axis, tau, G1
        # we use dm_tl at every t, not using the time-locality.
        G1[0,0] = np.trace(opA_mat @ opB_mat @ rho0)
        # calculate rho(t) for all t. at every t in t_axis, we have to apply the MTO and then
        # calculate up to t+tau_max
        rho_t = rho0.copy().reshape(dim**2)
        j = 0
        for i in tqdm.trange(len(t_axis)):
            t = t_axis[i]
            # propagate to the next t

            # if i == 0: # special case t=0
            #     n_j = 0
            # else:
            #     n_j = int((t - t_axis[i-1]) / dt) 

            # if n_j > 1:
            #     rho_t = chain_apply(dm_tl[j:j+n_j], rho_t)
            #     # rho_t = reduce(np.matmul, reversed(dm_tl[j:j+n_j])) @ rho_t
            # else:
            #     rho_t = dm_tl[j] @ rho_t
            # j += n_j

            while _t[j] < t:
                # ex: t=1, dt=0.25, _t=[0,0.25,0.5,0.75,1,1.25,...], j=5
                # we need to use 4 time steps to get to from t=0 to t=1
                rho_t = dm_tl[j] @ rho_t
                j += 1
            # print(j)
            G1[i,0] = np.trace(opA_mat @ opB_mat @ rho_t.copy().reshape(dim,dim))
            # now we have to apply the MTO
            rho_t_mto = opB_mat@rho_t.copy().reshape(dim,dim)
            rho_t_mto = rho_t_mto.reshape(dim**2)
            # propagate from tau=0 to tau_max
            # rho_tau = np.zeros((n_tau+1, dim**2), dtype=complex)
            # rho_tau[0] = rho_t_mto
            # for k in range(j, j+n_tau):
            #     rho_tau[k-j+1] = dm_tl[k] @ rho_tau[k-j]
                # rho_t_mto = dm_tl[k] @ rho_t_mto
                # G1[i,k-j+1] = np.trace(opA_mat @ rho_t_mto.reshape(dim,dim))
            # G1[i,1:] = np.einsum('ij,tji->t', opA_mat, rho_tau[1:].reshape(n_tau, dim, dim))

            dm_tl_f = np.asfortranarray(dm_tl.transpose(1, 2, 0))
            rho_init = rho_t_mto.astype(np.complex128)  # dim²
            # rho_out = np.empty((dim**2, n_tau + 1), dtype=np.complex128, order='F')

            rho_out = propagate_tau_module.propagate_tau(dm_tl_f, rho_init, n_tau, dim, j)
            # Optional reshape for further use:
            rho_tau = rho_out.T
            
            G1[i,1:] = np.trace(opA_mat @ rho_tau.reshape(n_tau+1,dim,dim), axis1=1, axis2=2)[1:]
        end_time = time.time()
        print(f"Time taken for tl_two_op_two_time with dm: {end_time - time_start:.2f} seconds")
        return t_axis, tau, G1

    if "phonons" not in options or not options["phonons"]:
        # we can just use the tl_map for everything, using the QRT
        # special case t,tau=0
        # G1[0,0] = np.trace(opA_mat @ opB_mat @ rho0)
        # calculate rho(t) for all t. at every t in t_axis, we have to apply the MTO and then 
        # calculate up to t+tau_max
        rho_t = rho0.copy().reshape(dim**2)
        for i in tqdm.trange(0,len(t_axis)):
            t = t_axis[i]
            if i == 0: # special case t=0
                n_steps = 0
            else:
                # we need to use t_axis[i]-t_axis[i-1] time steps to get to from t=t_axis[i-1] to t=t_axis[i]
                n_steps = int((t - t_axis[i-1]) / dt)
            # use tl_map^n_steps
            rho_t = np.linalg.matrix_power(tl_map, n_steps) @ rho_t
            G1[i,0] = np.trace(opA_mat @ opB_mat @ rho_t.reshape(dim,dim))
            # now we have to apply the MTO
            rho_t_mto = opB_mat@rho_t.reshape(dim,dim)
            rho_tau = tl_pad_stationary_nsteps(tl_map, n_tau, rho_t_mto)
            G1[i,1:] = np.trace(opA_mat @ rho_tau, axis1=1, axis2=2)
    else:
        print("phonons not implemented yet")
    return t_axis, tau, G1

# p1 = CWLaser(e0=0.05, e_start=0)
# dt = 1
# t_end = 100
# n_t = int(t_end / dt)
# t_axis = np.linspace(0, t_end, n_t+1)


# t1,t2,G1 = two_op_two_time(tls, t_axis, p1, opA="|1><0|_2", opB="|0><1|_2", tau_max=100, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0}, debug=True)
# plt.clf()
# plt.pcolormesh(t1,t2,np.abs(G1).T,shading="gouraud")
# plt.colorbar()
# plt.xlabel("t1")
# plt.ylabel("t2")
# plt.title("G1(t1,t2)")
# plt.savefig("pyaceqd/tests/g1_t1t2_cw.png")
# plt.clf()

# # test against old G1 function
# from pyaceqd.two_time.G1 import G1_twols
# t,tau,g1 = G1_twols(0,100,0,100,0.1,0.5,p1, gamma_e=2/100, phonons=False)
# plt.clf()
# plt.pcolormesh(t,tau, np.abs(g1).T,shading="gouraud")
# plt.colorbar()
# plt.xlabel("t")
# plt.ylabel("tau")
# plt.title("G1(t,tau)")
# plt.savefig("pyaceqd/tests/g1_t_tau_cw.png")
# plt.clf()

# test time local 
# t1,t2,G1 = tl_two_op_two_time(tls, t_axis, p1, t_mem=10, opA="|1><0|_2", opB="|0><1|_2", tau_max=100, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0}, debug=True, use_dm=False)
# plt.clf()
# plt.pcolormesh(t1,t2,np.abs(G1).T,shading="gouraud")
# plt.colorbar()
# plt.xlabel("t1")
# plt.ylabel("t2")
# plt.title("G1(t1,t2) tl")
# plt.savefig("pyaceqd/tests/g1_t1t2_tl.png")
# plt.clf()

# p1 = ChirpedPulse(tau_0=5,e_start=0,e0=3,t0=4*5)
# t_axis = simple_t_gaussian(0, 8*5, 100, 0.1, 1, p1, decimals=1)
# # print("t_axis",t_axis)
# # print("time-dependent")
# t1,t2,G1 = tl_two_op_two_time(tls, t_axis, p1, t_mem=10, opA="|1><0|_2", opB="|0><1|_2", tau_max=50, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0}, debug=False, use_dm=True, fortran_only=True)

# plt.clf()
# plt.pcolormesh(t1,t2,np.abs(G1).T,shading="gouraud")
# plt.colorbar()
# plt.xlabel("t1")
# plt.ylabel("t2")
# plt.title("G1(t1,t2)")
# plt.savefig("g1_t1t2_pulsed_tl.png")
# plt.clf()

# t,tau,G1_2 = tl_two_op_two_time(tls, t_axis, p1, t_mem=10, opA="|1><0|_2", opB="|0><1|_2", tau_max=50, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0}, debug=False, use_dm=True, fortran_only=False)

# # t1,t2,G1 = two_op_two_time(tls, t_axis, p1, opA="|1><0|_2", opB="|0><1|_2", tau_max=50, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0, "pulse_file": "/mnt/temp_data/tls_G1_pulse.dat"}, debug=True)
# # t,tau,G1_2 = G1_twols(0,100,0,50,0.1,0.1,p1, gamma_e=2/100, phonons=False,coarse_t=True,simple_exp=True,gaussian_t=8*5)
# # print(np.allclose(t1,t))
# # print(np.allclose(t2,tau))
# plt.close()
# plt.clf()
# plt.pcolormesh(t1,t2,np.abs(G1_2-G1).T,shading="gouraud")
# # plt.plot(tau,np.abs(G1[50]), label="G1")
# # plt.plot(t,np.abs(G1_2[:,1]-np.roll(G1[:,1],0)), dashes=[2,2], label="G1_2")
# # plt.ylim(1.5e-5,2.5e-5)
# # find max of difference and index of it
# # max_diff = np.max(np.abs(G1_2-G1))
# # max_diff_index = np.unravel_index(np.argmax(np.abs(G1_2-G1)), G1_2.shape)
# # print("max difference", max_diff)
# # print("max difference index", max_diff_index)
# plt.colorbar()
# plt.xlabel("t1")
# plt.ylabel("t2")
# plt.title("G1(t1,t2)")
# plt.savefig("pyaceqd/tests/g1_t1t2.png")
# plt.clf()

def tl_three_op_two_time(system, t_axis, *pulses, t_mem=10, opA="|1><0|_2", opB="|1><1|_2", opC="|0><1|_2", tau_max=500, dt=0.1, rho0=np.array([[1,0],[0,0]],dtype=complex), options={"lindblad": True, "phonons": False}, debug=False, workers=15, use_dm=False, fortran_only=False):
    """
    Description
    -----------
    Calculate the two-time correlation function <A(t+tau)B(t)>, for ex. G1(t,t+tau), for a system.
    The propagation starts at time 0, and up until t the system is propagated regularly.
    At t, operator B is applied to the system from the left, and the system is propagated until t+tau_end.
    The result is <A(_t)> after _t with tau = _t-t.
    0---------t--------->|<--------tau-------->|   _t axis = time axis of simulation: t_start,...,0...,t+tau_end
    
    Parameters
    ----------
    system : object
        The system to calculate the correlation function for.
    pulses : list
        The list of pulses to use in the calculation.
    t_mem : float
        memory time of the environment. 
    opA : str
        The first operator to use in the calculation.
    opB : str
        The second operator to use in the calculation.
    options : dict
        A dictionary of options for the calculation.
    debug : bool
        plot results and debug info.
    workers : int
        The number of workers to use for the calculation.

    Returns
    -------
    tuple
        A tuple (tau, G1) containing the tau array, and correlation function 
    """

    # check if all values of t_axis can be divided by dt without remainder
    # if not np.all(np.isclose(np.round(t_axis,6) % dt, 0)):
    #     print("t_axis", t_axis%dt)
    #     raise ValueError("t_axis values must be divisible by dt without remainder.")
    if not t_axis[0] == 0:
        raise ValueError("t_axis must start at 0.")

    # get matrix representation of the operators
    opA_mat = op_to_matrix(opA)
    opB_mat = op_to_matrix(opB)
    opC_mat = op_to_matrix(opC)

    # first we need to get the dynamic maps
    mto = {"operator": opC, "applyFrom": "_left", "applyBefore": "false", "time": 2*t_mem}
    mto2 = {"operator": opA, "applyFrom": "_right", "applyBefore": "false", "time": 2*t_mem}
    result, dm = system(0,4*t_mem, *pulses, dt=dt, rho0=rho0, multitime_op=[mto, mto2], calc_dynmap=True, **options)
    _t = result[0]  # time axis for getting the dynamic maps
    _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
    dm_tl = calc_tl_dynmap_pseudo(dm, _t)
    tl_map, dms = extract_dms(dm_tl, _t, t_mem,[2*t_mem])
    n_tau = int(tau_max / dt)
    tau = np.linspace(0, tau_max, n_tau + 1)
    G1 = np.zeros((len(t_axis), len(tau)), dtype=complex)

    dim = len(rho0[0])

    if use_dm:
        tend = t_axis[-1] + tau_max
        result, dm = system(0, tend, *pulses, dt=dt, rho0=rho0, multitime_op=[], calc_dynmap=True, **options)
        _t = result[0]  # time axis for getting the dynamic maps
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        if debug:
            rho_t = np.zeros((len(_t), dim, dim), dtype=complex)
            rho_t[0] = rho0.copy().reshape(dim, dim)
            for i in tqdm.trange(1, len(_t)):
                rho_t[i] = (dm_tl[i-1] @ rho_t[i-1].reshape(dim**2)).reshape(dim, dim)
            plt.clf()
            plt.plot(_t, np.abs(rho_t[:,0,0]), label="rho_00")
            plt.plot(_t, np.abs(rho_t[:,1,1]), label="rho_11")
            plt.plot(_t, result[1], dashes=[2,2], label="rho_00_true")
            plt.plot(_t, result[2], dashes=[2,2], label="rho_11_true")
            plt.legend()
            plt.savefig("pyaceqd/tests/rho_tl.png")
            plt.clf()
        
        # start time for profiling
        time_start = time.time()

        if fortran_only:
            dm_tl_f = np.asfortranarray(dm_tl.transpose(1, 2, 0))
            G1 = propagate_tau_module.calc_onetime_parallel(dm_tl_f, rho0.reshape(dim**2), n_tau, dim, opA_mat, opB_mat, opC_mat, _t, t_axis)
            end_time = time.time()
            print(f"Time taken for tl_two_op_two_time with dm: {end_time - time_start:.2f} seconds")
            return t_axis, tau, G1
        # we use dm_tl at every t, not using the time-locality.
        G1[0,0] = np.trace(opA_mat @ opB_mat @ opC_mat @ rho0)
        # calculate rho(t) for all t. at every t in t_axis, we have to apply the MTO and then
        # calculate up to t+tau_max
        rho_t = rho0.copy().reshape(dim**2)
        j = 0
        for i in tqdm.trange(len(t_axis)):
            t = t_axis[i]
            # propagate to the next t

            # if i == 0: # special case t=0
            #     n_j = 0
            # else:
            #     n_j = int((t - t_axis[i-1]) / dt) 

            # if n_j > 1:
            #     rho_t = chain_apply(dm_tl[j:j+n_j], rho_t)
            #     # rho_t = reduce(np.matmul, reversed(dm_tl[j:j+n_j])) @ rho_t
            # else:
            #     rho_t = dm_tl[j] @ rho_t
            # j += n_j

            while _t[j] < t:
                # ex: t=1, dt=0.25, _t=[0,0.25,0.5,0.75,1,1.25,...], j=5
                # we need to use 4 time steps to get to from t=0 to t=1
                rho_t = dm_tl[j] @ rho_t
                j += 1
            # print(j)
            G1[i,0] = np.trace(opA_mat @ opB_mat @ opC_mat @ rho_t.copy().reshape(dim,dim))
            # now we have to apply the MTO
            rho_t_mto = opC_mat@rho_t.copy().reshape(dim,dim)@opA_mat
            rho_t_mto = rho_t_mto.reshape(dim**2)
            # propagate from tau=0 to tau_max
            # rho_tau = np.zeros((n_tau+1, dim**2), dtype=complex)
            # rho_tau[0] = rho_t_mto
            # for k in range(j, j+n_tau):
            #     rho_tau[k-j+1] = dm_tl[k] @ rho_tau[k-j]
                # rho_t_mto = dm_tl[k] @ rho_t_mto
                # G1[i,k-j+1] = np.trace(opA_mat @ rho_t_mto.reshape(dim,dim))
            # G1[i,1:] = np.einsum('ij,tji->t', opA_mat, rho_tau[1:].reshape(n_tau, dim, dim))

            dm_tl_f = np.asfortranarray(dm_tl.transpose(1, 2, 0))
            rho_init = rho_t_mto.astype(np.complex128)  # dim²
            # rho_out = np.empty((dim**2, n_tau + 1), dtype=np.complex128, order='F')

            rho_out = propagate_tau_module.propagate_tau(dm_tl_f, rho_init, n_tau, dim, j)
            # Optional reshape for further use:
            rho_tau = rho_out.T
            
            G1[i,1:] = np.trace(opB_mat @ rho_tau.reshape(n_tau+1,dim,dim), axis1=1, axis2=2)[1:]
        end_time = time.time()
        print(f"Time taken for tl_two_op_two_time with dm: {end_time - time_start:.2f} seconds")
        return t_axis, tau, G1

    if "phonons" not in options or not options["phonons"]:
        # we can just use the tl_map for everything, using the QRT
        # special case t,tau=0
        # G1[0,0] = np.trace(opA_mat @ opB_mat @ rho0)
        # calculate rho(t) for all t. at every t in t_axis, we have to apply the MTO and then 
        # calculate up to t+tau_max
        rho_t = rho0.copy().reshape(dim**2)
        for i in tqdm.trange(0,len(t_axis)):
            t = t_axis[i]
            if i == 0: # special case t=0
                n_steps = 0
            else:
                # we need to use t_axis[i]-t_axis[i-1] time steps to get to from t=t_axis[i-1] to t=t_axis[i]
                n_steps = int((t - t_axis[i-1]) / dt)
            # use tl_map^n_steps
            rho_t = np.linalg.matrix_power(tl_map, n_steps) @ rho_t
            G1[i,0] = np.trace(opA_mat @ opB_mat @ rho_t.reshape(dim,dim))
            # now we have to apply the MTO
            rho_t_mto = opB_mat@rho_t.reshape(dim,dim)
            rho_tau = tl_pad_stationary_nsteps(tl_map, n_tau, rho_t_mto)
            G1[i,1:] = np.trace(opA_mat @ rho_tau, axis1=1, axis2=2)
    else:
        print("phonons not implemented yet")
    return t_axis, tau, G1


def tl_three_op_two_time_phonons(system, s, *pulses, t_mem=10, opA="|1><0|_2", opB="|1><1|_2", opC="|0><1|_2", tau_max=500, dt=0.1, rho0=np.array([[1,0],[0,0]],dtype=complex), options={"lindblad": True, "phonons": True}, debug=False, fortran_only=False):
    if not t_axis[0] == 0:
        raise ValueError("t_axis must start at 0.")

    # get matrix representation of the operators
    opA_mat = op_to_matrix(opA)
    opB_mat = op_to_matrix(opB)
    opC_mat = op_to_matrix(opC)

    # first we need to get the dynamic maps
    mto = {"operator": opC, "applyFrom": "_left", "applyBefore": "false", "time": 1.2*t_mem}
    mto2 = {"operator": opA, "applyFrom": "_right", "applyBefore": "false", "time": 1.2*t_mem}
    result, dm = system(0,4*t_mem, *pulses, dt=dt, rho0=rho0, multitime_op=[mto, mto2], calc_dynmap=True, **options)
    _t = result[0].real  # time axis for getting the dynamic maps
    # print(result.shape)
    
    _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
    dm_tl = calc_tl_dynmap_pseudo(dm, _t, debug=True)
    tl_map, dms_separated = extract_dms(dm_tl, _t, t_mem,[1.2*t_mem])
    tl_map2 = dm_tl[-1]  # last dynamic map, used for the second part of the calculation
    dms_separated = np.array(dms_separated, dtype=complex)

    rho_test = np.ones((len(_t), 2,2), dtype=complex)
    rho_test[0] = rho0.copy().reshape(2,2)
    n_steps = int((1.2*t_mem) / dt) +1
    for i in tqdm.trange(1,dms_separated[0].shape[0]):
        rho_test[i] = (dms_separated[0,i-1] @ rho_test[i-1].reshape(2**2)).reshape(2,2)
    for i in range(dms_separated[0].shape[0],n_steps):
        rho_test[i] = (tl_map @ rho_test[i-1].reshape(2*2)).reshape(2,2)
    for i in tqdm.trange(n_steps, n_steps + dms_separated[0].shape[0]):
        rho_test[i] = (dms_separated[1,i-n_steps] @ rho_test[i-1].reshape(2**2)).reshape(2,2)
    plt.clf()
    plt.plot(_t, np.abs(result[1]), label="rho_00")
    plt.plot(_t, np.abs(result[2]), label="rho_11")
    plt.plot(_t, np.abs(rho_test[:,0,0]), dashes=[2,2], label="rho_00_tl")
    plt.plot(_t, np.abs(rho_test[:,1,1]), dashes=[2,2], label="rho_11_tl")
    plt.scatter(_t[:len(dms_separated[0])],[1 for _ in range(len(dms_separated[0]))], label="dms")
    plt.legend()
    # plt.xlim(1.1*t_mem, 1.25*t_mem)
    plt.ylim(0,1.1)
    plt.savefig("pyaceqd/tests/rho_tl_phonons.png")
    # print(np.shape(dms))
    n_tau = int(tau_max / dt)
    tau = np.linspace(0, tau_max, n_tau + 1)
    G = np.zeros((len(t_axis), len(tau)), dtype=complex)

    dim = len(rho0[0])

    # check how many values of t_axis are below t_mem
    t_mem_indices = np.where(t_axis < t_mem)[0]
    # print(f"t_mem_indices: {t_mem_indices}")
    # print(t_axis[t_mem_indices])
    dms_tauc = np.empty((len(t_mem_indices), *np.shape(dms_separated)), dtype=complex)
    for i in t_mem_indices:
        t = t_axis[i]
        # we need to calculate the dynamic map for this time
        mto = {"operator": opC, "applyFrom": "_left", "applyBefore": "false", "time": t}
        mto2 = {"operator": opA, "applyFrom": "_right", "applyBefore": "false", "time": t}
        result, dm = system(0, t+t_mem+10*dt, *pulses, dt=dt, rho0=rho0, multitime_op=[mto, mto2], calc_dynmap=True, **options)
        _t = result[0]  # time axis for getting the dynamic maps
        _t = np.round(_t, 6)  # round to 6 digits to avoid floating point errors
        dm_tl = calc_tl_dynmap_pseudo(dm, _t)
        _,_dms = extract_dms(dm_tl, _t, t_mem,[t])
        dms_tauc[i] = _dms
    # print(np.shape(dms_tauc))

    rho_t = rho0.copy().reshape(dim**2)
    n_tauc = len(dms_tauc[0,0])
    t_ges = tau_max + t_axis[-1]
    t_axis_ges = np.linspace(0, t_ges, int(t_ges / dt) + 1)
    rho_test = np.ones((len(t_axis_ges), dim, dim), dtype=complex)
    i_rho = 1
    i_test = 25
    rho_test[0] = rho0.copy().reshape(dim, dim)
    t_test = t_axis[i_test]
    print(f"t_test: {t_test}, i_test: {i_test}, dms: {len(t_mem_indices)}")
    mto = {"operator": opC, "applyFrom": "_left", "applyBefore": "false", "time": t_test}
    mto2 = {"operator": opA, "applyFrom": "_right", "applyBefore": "false", "time": t_test}
    result, dm = system(0,200, *pulses, dt=dt, rho0=rho0, multitime_op=[mto, mto2], calc_dynmap=True, **options)

    for i in tqdm.trange(len(t_axis)):
        rho_t = rho0.copy().reshape(dim**2)
        t = t_axis[i]
        if i == 0:  # special case t=0
            n_steps = 0
        else:
            # we need to use t_axis[i]-t_axis[i-1] time steps to get to from t=t_axis[i-1] to t=t_axis[i]
            n_steps = int((t) / dt) + 1
        # use n_steps of dms_tauc[-1,0] to propagate rho_t from t=0 to t or tauc
        # whichever is smaller
        for j in range(np.min([n_steps, n_tauc])-1):
            rho_t = dms_separated[0,j] @ rho_t
            if i == i_test:
                rho_test[i_rho] = rho_t.reshape(dim, dim)
                i_rho += 1
        # if n_steps > n_tauc, we need to use the tl_map to propagate rho_t
        for j in range(n_steps - n_tauc):
            rho_t = tl_map @ rho_t
            if i == i_test:
                rho_test[i_rho] = rho_t.reshape(dim, dim)
                i_rho += 1
        G[i,0] = np.trace(opA_mat @ opB_mat @ opC_mat @ rho_t.reshape(dim,dim))
        # now use all steps in dms_tauc[i,1] to calculate the first n_tauc values in G[i,1:]
        rho_t_mto = rho_t.copy()
        # if i < (len(dms_tauc)):
        #     for j in range(n_tauc):
        #         rho_t_mto = dms_tauc[i,1,j] @ rho_t_mto
        #         G[i,j+1] = np.trace(opB_mat @ rho_t_mto.reshape(dim,dim))
        # else:
        #     for j in range(n_tauc):
        #         rho_t_mto = dms_separated[1,j] @ rho_t_mto
        #         G[i,j+1] = np.trace(opB_mat @ rho_t_mto.reshape(dim,dim))
        for j in range(n_tauc):
            if i < (len(t_mem_indices)):
                rho_t_mto = dms_tauc[i,1,j] @ rho_t_mto
                if i == i_test:
                    rho_test[i_rho] = rho_t_mto.reshape(dim, dim)
                    i_rho += 1
            else:
                rho_t_mto = dms_separated[1,j] @ rho_t_mto
                if i == i_test:
                    rho_test[i_rho] = rho_t_mto.reshape(dim, dim)
                    i_rho += 1
            G[i,j+1] = np.trace(opB_mat @ rho_t_mto.reshape(dim,dim))
        # use time-local map for rest
        for j in range(n_tau-n_tauc):
            rho_t_mto = tl_map2 @ rho_t_mto
            if i == i_test:
                rho_test[i_rho] = rho_t_mto.reshape(dim, dim)
                i_rho += 1
            G[i,n_tauc+j+1] = np.trace(opB_mat @ rho_t_mto.reshape(dim,dim))
    # dm_tl_f = np.asfortranarray(dms_tauc.transpose(3, 4, 0, 1, 2))
    # print(np.shape(dm_tl_f))

    plt.clf()
    plt.plot(result[0][:1000*int(0.1/dt)], np.abs(result[1][:1000*int(0.1/dt)]-rho_test[:1000*int(0.1/dt),0,0]), label="rho_00")
    # plt.plot(result[0], np.abs(result[2]), label="rho_11")
    # plt.plot(t_axis_ges, np.abs(rho_test[:,0,0]), dashes=[2,2], label="rho_00_tl")
    # plt.plot(t_axis_ges, np.abs(rho_test[:,1,1]), dashes=[2,2], label="rho_11_tl")
    # plt.xlim(18,22)
    plt.savefig("pyaceqd/tests/rho_tl_phonons_test.png")

    return t_axis, tau, G


p1 = CWLaser(e0=0.08)  #
# p1 = ChirpedPulse(tau_0=5,e_start=0,e0=5,t0=4*5)
t_tau_max = 100
t_axis = simple_t_gaussian(0, 1, t_tau_max, 0.1, 0.1, p1, decimals=1, exp_part=False)
# print("t_axis", t_axis)
t1,t2,G2 = tl_three_op_two_time_phonons(tls, t_axis, p1, t_mem=10, opA="|1><0|_2", opB="|1><1|_2", opC="|0><1|_2", tau_max=t_tau_max, dt=0.05, options={"gamma_e": 2/100,"lindblad": True, "phonons": True, "use_infinite": True, "ae": 5.0}, debug=False, fortran_only=True)

# t1,t2,G2 = tl_three_op_two_time(tls, t_axis, p1, t_mem=10, opA="|1><0|_2", opB="|1><1|_2", opC="|0><1|_2", tau_max=t_tau_max, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0}, debug=False, use_dm=True, fortran_only=True)

plt.clf()
plt.pcolormesh(t1,t2,np.abs(G2).T,shading="gouraud")
plt.colorbar()
plt.xlabel("t1 (t)")
plt.ylabel("t2, (tau)")
plt.title("G2(t1,t2)")
plt.savefig("pyaceqd/tests/g2_t1t2_pulsed_tl.png")
plt.clf()

# # t,tau,G2_2 = tl_three_op_two_time(tls, t_axis, p1, t_mem=10, opA="|1><0|_2", opB="|1><1|_2", opC="|0><1|_2", tau_max=50, dt=0.1, options={"gamma_e": 2/100,"lindblad": True, "phonons": False, "use_infinite": True, "ae": 5.0}, debug=False, use_dm=True, fortran_only=False)

t1,t2,G2_2 = three_op_two_time(tls, t_axis, p1, opA="|1><0|_2", opB="|1><1|_2", opC="|0><1|_2", tau_max=t_tau_max, dt=0.05, options={"gamma_e": 2/100,"lindblad": True, "phonons": True, "use_infinite": True, "ae": 5.0}, debug=True)

plt.clf()
plt.pcolormesh(t1,t2,np.abs(G2_2).T,shading="gouraud")
plt.colorbar()
plt.xlabel("t1 (t)")
plt.ylabel("t2, (tau)")
plt.title("G2(t1,t2)")
plt.savefig("pyaceqd/tests/g2_t1t2_phonons.png")
plt.clf()

# # t,tau,G1_2 = G1_twols(0,100,0,50,0.1,0.1,p1, gamma_e=2/100, phonons=False,coarse_t=True,simple_exp=True,gaussian_t=8*5)
plt.pcolormesh(t1,t2,np.abs(G2_2-G2).T,shading="gouraud")
plt.colorbar()
plt.xlabel("t1")
plt.ylabel("t2")
plt.xlim(0,2)
plt.title("G1(t1,t2)")
plt.savefig("pyaceqd/tests/g2_t1t2.png")
plt.clf()


print(t2[400])
plt.plot(t1, np.abs(G2_2[:,400])-np.abs(G2[:,400]), label="G2")
# plt.plot(t1, np.abs(G2[:,400]), dashes=[2,2], label="G2_2")
# plt.xlim(6.5,7)
# plt.ylim(0,1.1)
plt.legend()
plt.xlabel("t")
plt.ylabel("G2(t,0)")
plt.title("G2(t,0)")
plt.savefig("pyaceqd/tests/g2_t0.png")
plt.clf()