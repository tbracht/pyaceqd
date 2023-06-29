import numpy as np
import os
from pyaceqd.tools import export_csv, construct_t, simple_t_gaussian
import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from pyaceqd.general_system.general_system import system_ace_stream

hbar = 0.6582173  # meV*ps

def darkmodel(t_start, t_end, *pulses, dt=0.5, delta_xd=0, delta_b=4, gamma_e=1/100, gamma_b=None, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_4","|1><1|_4","|2><2|_4","|3><3|_4"], initial="|0><0|_4"):
    system_prefix = "darkmodel_"
    # |0> = G, |1> = X, |2> = D, |3> = B
    system_op = ["{}*|3><3|_4".format(-delta_b),"{}*|2><2|_4".format(-delta_xd)]
    # system_op = ["{}*|1><1|_4".format(delta_b*0.5),"{}*|2><2|_4".format(delta_b*0.5-delta_xd)]
    boson_op = "1*(|1><1|_4 + |2><2|_4) + 2*|3><3|_4"
    # initial = "|0><0|_4"
    lindblad_ops = []
    if lindblad:
        if gamma_b is None:
            gamma_b = gamma_e  # the same rate for x and b, as we only consider one x-state
        lindblad_ops = [["|0><1|_4",gamma_e],["|1><3|_4",gamma_b]]  #  |2> is dark, does not decay 
    # we use 'x'-polar for coupling between G, X and B, while 'y'-polar couples X and D
    interaction_ops = [["|2><0|_4","x"],["|3><2|_4","x"],["|1><0|_4","y"],["|3><1|_4","y"]]
    
    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=20.48, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only)
    return result

def darkmodel_photons(t_start, t_end, *pulses, dt=0.5, delta_xd=0, delta_b=4, delta_cx=-2, rad_loss=1/100, cav_loss=1/20, cav_coupl=1/30, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_4 otimes |0><0|_3","|1><1|_4 otimes |0><0|_3","|2><2|_4 otimes |0><0|_3","|3><3|_4 otimes |0><0|_3"], initial="|0><0|_4 otimes |0><0|_3"):
    system_prefix = "darkmodel_photons"
    # |0> = G, |1> = X, |2> = D, |3> = B
    system_op = ["{}*|3><3|_4 otimes Id_3".format(-delta_b),"{}*|2><2|_4 otimes Id_3".format(-delta_xd)]
    boson_op = "|1><1|_4 otimes Id_3 + |2><2|_4 otimes Id_3 + 2*|3><3|_4 otimes Id_3"
    lindblad_ops = []
    if lindblad:
        # radiative decay of dot, outside the cavity
        lindblad_ops = [["|0><1|_4 otimes Id_3",rad_loss],["|1><3|_4 otimes Id_3",rad_loss]]  #  |2> is dark, does not decay 
    # interaction with laser
    interaction_ops = [["|2><0|_4 otimes Id_3","x"],["|3><2|_4 otimes Id_3","x"],["|1><0|_4 otimes Id_3","y"],["|3><1|_4 otimes Id_3","y"]]
    # cavity decay
    lindblad_ops.append(["Id_4 otimes b_3",cav_loss])
    # cavity detuning
    system_op.append(" {} * (Id_4 otimes n_3)".format(delta_cx))
    # cavity-qd coupling
    system_op.append("{}*(|1><0|_4 otimes b_3 + |3><1|_4 otimes b_3 + |0><1|_4 otimes bdagger_3 + |1><3|_4 otimes bdagger_3)".format(hbar*cav_coupl))
    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=20.48, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only)
    return result


def timebin_integrate(t,tau,f,timebin_width,n_t=1,n_tau=1,debug=False,normalize=False):
    """
    integrates 2d-function f along t and then tau, such that t+tau lie within the same time-bin
    n gives the order of the time-bin, i.e., 1: first timebin, 2: second timebin...
    """
    t_start = timebin_width*(n_t-1)
    t_end = timebin_width*n_t
    tau_start = timebin_width*(n_tau-1)
    tau_end = timebin_width*n_tau
    # first, integrate over tau
    f_t = np.zeros_like(t)
    for i in tqdm.trange(len(t)):
        tau_axis = []
        tau_values = []
        for j in range(len(tau)):
            if t[i] >= t_start and t[i] < t_end:
                if t[i]+tau[j] >= tau_start and t[i]+tau[j] < tau_end:
                    tau_axis.append(tau[j])
                    tau_values.append(f[i,j])
                    if debug:
                        f[i,j] = 1
        tau_axis = np.array(tau_axis)
        tau_values = np.array(tau_values)
        f_t[i] = np.trapz(tau_values, tau_axis)
        if normalize and len(tau_axis>1):
            f_t[i] = f_t[i]/(np.abs(tau_axis[-1]-tau_axis[0]))
    f_complete = np.trapz(f_t, t)
    if normalize:
        f_complete = f_complete / (np.abs(t[-1]-t[0]))
    return t,tau,f,f_t,f_complete


def two_photon_density(t0=0, tend=1600, tau0=0, tauend=1600, dt=0.1, dtau=0.1, *pulses, delta_xd=4, delta_b=4, gamma_e=1/65, timebin=800, workers=15, temp_dir='/mnt/temp_data/', coarse_t=True):
    options = {"delta_xd": delta_xd, "delta_b": delta_b, "gamma_e": gamma_e, "workers": workers, "temp_dir": temp_dir, "coarse_t": coarse_t,
               "t0": t0, "tend": tend, "tau0": tau0, "tauend": tauend, "dt": dt, "dtau": dtau}
    t1,tau1,g2_ee = G2_ee(*pulses, **options)
    _,_,_,_,rho_ee_ee = timebin_integrate(t1,tau1,g2_ee,timebin_width=timebin,n_t=1,n_tau=1)
    _,_,_,_,rho_ll_ll = timebin_integrate(t1,tau1,g2_ee,timebin_width=timebin,n_t=2,n_tau=2)
    _,_,_,_,rho_el_el = timebin_integrate(t1,tau1,g2_ee,timebin_width=timebin,n_t=1,n_tau=2)
    return rho_ee_ee,rho_ll_ll,rho_el_el

def G2_ee(*pulses, t0=0, tend=600, tau0=0, tauend=600, dt=0.1, dtau=0.1, delta_xd=4, delta_b=4, gamma_e=1/65, workers=15, temp_dir='/mnt/temp_data/', coarse_t=True):
    """
    calculates G2 for assuming an XX-emission triggers the coincidence measurment at time t, following an X at time t+tau, i.e.:
    <sigma^dagger_XX(t) sigma^dagger_X(t+tau) sigma_x(t+tau) sigma_xx(t)>
    This can be used to calculate the possibility to obtain:
    (1) two photons early (integrate over first timebin)
    (2) two photons late (integrate over second timebin)
    (3) one XX photon early and one X photon late (integrate t over first and tau over second timebin)
    note that this does not capture the detrimental case where one X photon is emitted during the early time-bin
    and one XX photon follows in the late time-bin due to errors in the excitation protocol. 
    """
    # includes tend
    t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    n_tau = int((tauend-tau0)/dtau)
    tau = np.linspace(tau0, tauend, n_tau + 1)
    multitime_op = {"operator": "|1><3|_4","applyFrom": "", "applyBefore":"false"}
    if coarse_t:
        t = construct_t(t0, tend, dt, 10*dt, *pulses)
    
    # the pulse has to be better resolved, because ACE uses intermediate steps
    # tend + tauend is the maximum simulation length
    _t_pulse = np.arange(t0,(tend+tauend),step=dtau)
    # different polarizations
    pulse_file_x = temp_dir + "G2_pulse_x.dat"
    pulse_file_y = temp_dir + "G2_pulse_y.dat"
    pulse_x = np.zeros_like(_t_pulse, dtype=complex)
    pulse_y = np.zeros_like(_t_pulse, dtype=complex)
    for _p in pulses:
        pulse_x = pulse_x + _p.polar_x*_p.get_total(_t_pulse)
        pulse_y = pulse_y + _p.polar_y*_p.get_total(_t_pulse)
    export_csv(pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
    export_csv(pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')
    # special case tau=0:
    # all 4 operators are applied at the same time.
    # G2(t,0) = Tr(sigma^dagger_XX(t) sigma^dagger_X(t+tau) sigma_x(t+tau) sigma_xx(t) * rho) = |X><X|
    options = {"dt": dtau, "verbose": False, "delta_xd": delta_xd, "delta_b": delta_b, "gamma_e": gamma_e, "lindblad": True,
               "pulse_file_x": pulse_file_x, "pulse_file_y": pulse_file_y, "temp_dir": '/mnt/temp_data/'}
    _G2 = np.zeros([len(t),len(tau)])
    with tqdm.tqdm(total=len(t)) as tq:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(len(t)):
                multitime_op_new = dict(multitime_op)  # must make a copy of the dict
                multitime_op_new["time"] = t[i]
                _e = executor.submit(darkmodel,t0,t[i] + tauend,*pulses,multitime_op=multitime_op_new, suffix=i, **options)
                _e.add_done_callback(lambda f: tq.update())
                futures.append(_e)
            # wait for all futures
            wait(futures)
        for i in range(len(futures)):
            # futures are still 'future' objects
            futures[i] = futures[i].result()
        # futures now contains t,g,x,d,xx for every i
        for i in range(len(t)):
            # special case tau=0:
            # as Tr(sigma^dagger*sigma^dagger*sigma*sigma * rho) = x, G2(t,0) = x(t), which is the value with index [-(n_tau+1)]
            _G2[i,0] = np.real(futures[i][4][-n_tau-1])
            # futures[i][4] are the xx values , [2] are the x values
            _G2[i,1:] = np.real(futures[i][2][-n_tau:])
    os.remove(pulse_file_x)
    os.remove(pulse_file_y)
    return t, tau, _G2

def G2_eeee(*pulses, t0=0, dt=0.1, dtau=0.1, delta_xd=4, delta_b=4, gamma_e=1/65, workers=15, temp_dir='/mnt/temp_data/', simple_exp=False, tb=800, gaussian_t=None, normalize=False):
    """
    calculates G2 for assuming an XX-emission triggers the coincidence measurment at time t, following an X at time t+tau, i.e.:
    <sigma^dagger_XX(t) sigma^dagger_X(t+tau) sigma_x(t+tau) sigma_xx(t)>
    This can be used to calculate the possibility to obtain:
    (1) two photons early (integrate over first timebin)
    (2) two photons late (integrate over second timebin)
    (3) one XX photon early and one X photon late (integrate t over first and tau over second timebin)
    note that this does not capture the detrimental case where one X photon is emitted during the early time-bin
    and one XX photon follows in the late time-bin due to errors in the excitation protocol. 
    """

    multitime_op = {"operator": "|1><3|_4","applyFrom": "", "applyBefore":"false"}

    if gaussian_t is not None:
        t1 = simple_t_gaussian(t0,gaussian_t,tb,dt,10*dt,*pulses)
    else:
        t1 = construct_t(t0, tb, dt, 10*dt, *pulses, simple_exp=simple_exp)

    # tau1 = construct_t(tau0, tb, dt, 10*dt, *pulses)
    n_tau = int((tb)/dtau)
    t2 = np.linspace(0, tb, n_tau + 1)
    # 2*tb is the maximum simulation length
    _t_pulse = np.arange(t0,2.1*tb,step=dtau)
    # different polarizations
    pulse_file_x = temp_dir + "G2_pulse_x.dat"
    pulse_file_y = temp_dir + "G2_pulse_y.dat"
    pulse_x = np.zeros_like(_t_pulse, dtype=complex)
    pulse_y = np.zeros_like(_t_pulse, dtype=complex)
    for _p in pulses:
        pulse_x = pulse_x + _p.polar_x*_p.get_total(_t_pulse)
        pulse_y = pulse_y + _p.polar_y*_p.get_total(_t_pulse)
    export_csv(pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
    export_csv(pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')
    # special case tau=0:
    # all 4 operators are applied at the same time.
    # G2(t,0) = Tr(sigma^dagger_XX(t) sigma^dagger_X(t+tau) sigma_x(t+tau) sigma_xx(t) * rho) = |X><X|
    options = {"dt": dtau, "verbose": False, "delta_xd": delta_xd, "delta_b": delta_b, "gamma_e": gamma_e, "lindblad": True,
               "pulse_file_x": pulse_file_x, "pulse_file_y": pulse_file_y, "temp_dir": '/mnt/temp_data/', "output_ops": ["|0><0|_4","|1><1|_4","|2><2|_4","|3><3|_4","|0><1|_4","|0><3|_4"]}
    _G2 = np.zeros([len(t1)])
    tend = tb
    with tqdm.tqdm(total=len(t1)) as tq:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(len(t1)):
                multitime_op_new = dict(multitime_op)  # must make a copy of the dict
                multitime_op_new["time"] = t1[i]
                _e = executor.submit(darkmodel,t0,tend,*pulses,multitime_op=multitime_op_new, suffix=i, **options)
                _e.add_done_callback(lambda f: tq.update())
                futures.append(_e)
            # wait for all futures
            wait(futures)
        for i in range(len(futures)):
            # futures are still 'future' objects
            futures[i] = futures[i].result()
        # futures now contains t,g,x,d,xx for every i
        for i in range(len(t1)):
            n_t2 = n_tau - int((t1[i])/dtau)
            temp_t2 = np.zeros(n_t2+1)
            # special case tau=0:
            # as Tr(sigma^dagger*sigma^dagger*sigma*sigma * rho) = x, G2(t,0) = x(t), which is the value with index [-(n_tau+1)]
            # _G2[i,0] = np.real(futures[i][4][-n_tau-1])
            temp_t2[0] = np.abs(futures[i][4][-n_t2-1])
            # futures[i][4] are the xx values , [2] are the x values
            #_G2[i,1:] = np.real(futures[i][2][-n_tau:])
            if n_t2 > 0: 
                temp_t2[1:n_t2+1] = np.abs(futures[i][2][-n_t2:])
            t_new = t2[:len(temp_t2)]
            _G2[i] = np.trapz(temp_t2,t_new)
            if normalize and len(t_new) > 1:
                _G2[i] = 1/np.abs(t_new[-1]-t_new[0]) * _G2[i]
    os.remove(pulse_file_x)
    os.remove(pulse_file_y)
    return t1, _G2

def G2_eell(*pulses, t0=0, dt=0.1, dtau=0.1, delta_xd=4, delta_b=4, gamma_e=1/65, workers=15, temp_dir='/mnt/temp_data/', simple_exp=False, tb=800, gaussian_t=None, normalize=False):
    # note that dtau is used for dt in the actual propagation
    # t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    # n_tau = int((tauend-tau0)/dtau)
    # tau = np.linspace(tau0, tauend, n_tau + 1)
    # sigma_x = {"operator": "|0><1|_4","applyFrom": "", "applyBefore":"false"}  # not needed
    # note that the 'right' operators are transposed again, because of a current 
    # 'bug' in ACE
    sigma_xxdag = {"operator": "|3><1|_4","applyFrom": "_right", "applyBefore":"false"}
    sigma_xdag = {"operator": "|1><0|_4","applyFrom": "_right", "applyBefore":"false"}
    sigma_xx = {"operator": "|1><3|_4","applyFrom": "_left", "applyBefore":"false"}
    
    if simple_t_gaussian is not None:
        t1 = simple_t_gaussian(t0,gaussian_t,tb,dt,10*dt,*pulses)
        t3 = simple_t_gaussian(tb,tb+gaussian_t,2*tb,dt,10*dt,*pulses)
    else:
        t1 = construct_t(t0, tb, dt, 10*dt, *pulses, simple_exp=simple_exp)
        # t2 is same as t1
        t3 = construct_t(tb, 2*tb, dt, 10*dt, *pulses, simple_exp=simple_exp)

    # tau1 = construct_t(tau0, tb, dt, 10*dt, *pulses)
    n_tau = int((tb)/dtau)
    t4 = np.linspace(0, tb, n_tau + 1)
    # 2*tb is the maximum simulation length
    _t_pulse = np.arange(t0,2.1*tb,step=dtau)
    # different polarizations
    pulse_file_x = temp_dir + "G2_pulse_x.dat"
    pulse_file_y = temp_dir + "G2_pulse_y.dat"
    pulse_x = np.zeros_like(_t_pulse, dtype=complex)
    pulse_y = np.zeros_like(_t_pulse, dtype=complex)
    for _p in pulses:
        pulse_x = pulse_x + _p.polar_x*_p.get_total(_t_pulse)
        pulse_y = pulse_y + _p.polar_y*_p.get_total(_t_pulse)
    export_csv(pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
    export_csv(pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')
    # special case tau=0:
    # all 4 operators are applied at the same time.
    # G2(t,0) = Tr(sigma^dagger_XX(t) sigma^dagger_X(t+tau) sigma_x(t+tau) sigma_xx(t) * rho) = |X><X|
    options = {"dt": dtau, "verbose": False, "delta_xd": delta_xd, "delta_b": delta_b, "gamma_e": gamma_e, "lindblad": True,
               "pulse_file_x": pulse_file_x, "pulse_file_y": pulse_file_y, "temp_dir": '/mnt/temp_data/', "output_ops": ["|0><0|_4","|1><1|_4","|2><2|_4","|3><3|_4","|0><1|_4","|0><3|_4"]}
    _G2 = np.zeros([len(t1),len(t1),len(t3)], dtype=complex)  # G2(t1,t2,t3)  # G2(t1,t2,tau1,tau2)
    for i in tqdm.trange(len(t1),leave=None):
        # tau1: use the interval 0,...,tb-t1
        # i.e., if t1 = 0,...,tx then tau1 expands to absolute times of tx,...,tb
        _t1 = t1[i]
        for j in tqdm.trange(len(t1)-i,leave=None):
            # j=0 is a special case that has to be addressed: here, |XX><G| has to be applied at t=t1
            # this is addressed by taking care to use the correct order of operators in the parameter file
            # if the time is the same, the order in the param file is the order that is used to apply the operators
            _t2 = t1[j+i]
            futures = []
            with tqdm.tqdm(total=len(t3), leave=None) as tq:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for k in range(len(t3)):
                        _t3 = t3[k]
                        sigma_xdag_new = dict(sigma_xdag)
                        sigma_xx_new = dict(sigma_xx)
                        sigma_xxdag_new = dict(sigma_xxdag)
                        # add correct times
                        sigma_xxdag_new["time"] = _t1
                        sigma_xdag_new["time"] = _t2
                        sigma_xx_new["time"] = _t3
                        _t4_end = 2*tb  # this is always the same
                        # the order of the operators is important to catch the special case where t1=t2
                        # because then ACE applies the operator first, that is first in the parameter file
                        multitime_op_new = [sigma_xdag_new,sigma_xxdag_new,sigma_xx_new]
                        _e = executor.submit(darkmodel,t0,_t4_end,*pulses,multitime_op=multitime_op_new, suffix=k, **options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
                    wait(futures)
            for k in range(len(futures)):
                # futures are still 'future' objects
                futures[k] = futures[k].result()
            # futures now contains t,g,x,d,xx,pgx,pgb for every i
            #_G2temp = np.zeros([len(t3),len(t4)])
            for k in range(len(t3)):
                # carfeul, not like this:
                # for k=0 : t3[0] = tb -> t4 = tb,...,2tb -> n_t4 = n_tau - 0
                # for k=1 : t3[1] = tb+dt -> t4 = tb+dt,...,2tb -> n_t4 = n_tau - 1
                # as t3 is unequally spaced, we have to consider the following, where dtau is the
                # timestep during the simulation:
                # for k=0: t3[0] = tb -> t4 = tb,...,2tb -> n_t4 = n_tau - 0
                # for k=i : t3[i] != tb+i*dtau -> t4 = t3[i],...,2tb -> n_t4 = n_tau - int((t3[i]-tb)/dtau)
                n_t4 = n_tau - int((t3[k]-tb)/dtau)  # = len(t4)-1 - k
                temp_t4 = np.zeros(n_t4+1)
                # pgb
                temp_t4[0] = np.abs(futures[k][6][-n_t4-1])
                # _G2[i,j,k,0] = np.real(futures[k][6][-n_t4-1])
                # pgx
                if n_t4 > 0: 
                    temp_t4[1:n_t4+1] = np.abs(futures[k][5][-n_t4:])
                # _G2[i,j,k,1:n_t4+1] = np.real(futures[k][5][-n_t4:])
                #_G2temp[k] = 
                t_new = t4[:len(temp_t4)]  # time axis for integration
                _G2[i,j,k] = np.trapz(temp_t4,t_new)
                if normalize and len(t_new) > 1:
                    _G2[i,j,k] = 1/np.abs(t_new[-1]-t_new[0]) * _G2[i,j,k]
    # _G2 now has to be integrated along all axes correctly, meaning with the correct time-axes
    # note that especially t2 is special, as it is 'folded' from t1
    os.remove(pulse_file_x)
    os.remove(pulse_file_y)
    return t1,t3,_G2


def G2_eell_easy(*pulses, t0=0, dt=0.1, dtau=0.1, delta_xd=4, delta_b=4, gamma_e=1/65, workers=15, temp_dir='/mnt/temp_data/', simple_exp=False, t_sep=800, tb=800, gaussian_t=None):
    # note that dtau is used for dt in the actual propagation
    # t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    # n_tau = int((tauend-tau0)/dtau)
    # tau = np.linspace(tau0, tauend, n_tau + 1)
    # sigma_x = {"operator": "|0><1|_4","applyFrom": "", "applyBefore":"false"}  # not needed
    sigma_xxdag = {"operator": "|3><1|_4","applyFrom": "_right", "applyBefore":"false"}
    sigma_xdag = {"operator": "|1><0|_4","applyFrom": "_right", "applyBefore":"false"}
    sigma_xx = {"operator": "|1><3|_4","applyFrom": "_left", "applyBefore":"false"}
    
    if simple_t_gaussian is not None:
        t1 = simple_t_gaussian(t0, gaussian_t, tb, dt, 10*dt, *pulses)
    else:
        t1 = construct_t(t0, tb, dt, 10*dt, *pulses, simple_exp=simple_exp)

    # 2*tb is the maximum simulation length
    _t_pulse = np.arange(t0,2.1*tb,step=dtau)
    # different polarizations
    pulse_file_x = temp_dir + "G2_pulse_x.dat"
    pulse_file_y = temp_dir + "G2_pulse_y.dat"
    pulse_x = np.zeros_like(_t_pulse, dtype=complex)
    pulse_y = np.zeros_like(_t_pulse, dtype=complex)
    for _p in pulses:
        pulse_x = pulse_x + _p.polar_x*_p.get_total(_t_pulse)
        pulse_y = pulse_y + _p.polar_y*_p.get_total(_t_pulse)
    export_csv(pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
    export_csv(pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')
    options = {"dt": dtau, "verbose": False, "delta_xd": delta_xd, "delta_b": delta_b, "gamma_e": gamma_e, "lindblad": True,
               "pulse_file_x": pulse_file_x, "pulse_file_y": pulse_file_y, "temp_dir": '/mnt/temp_data/', "output_ops": ["|0><0|_4","|1><1|_4","|2><2|_4","|3><3|_4","|0><1|_4","|0><3|_4"]}
    _G2 = np.zeros([len(t1),len(t1)], dtype=complex)  # G2(t1,t2)
    for i in tqdm.trange(len(t1),leave=None):
        # tau1: use the interval 0,...,tb-t1
        # i.e., if t1 = 0,...,tx then tau1 expands to absolute times of tx,...,tb
        _t1 = t1[i]
        futures = []
        with tqdm.tqdm(total=len(t1)-i, leave=None) as tq:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for j in range(len(t1)-i):
                    # j=0 is a special case that has to be addressed: here, |XX><G| has to be applied at t=t1
                    # this is addressed by taking care to use the correct order of operators in the parameter file
                    # if the time is the same, the order in the param file is the order that is used to apply the operators
                    _t2 = t1[j+i]
                    #if j == 0:  # special case
                    #    _t2 += 2 * dtau
                    _t3 = _t1 + t_sep
                    _t4_end = _t2 + t_sep
                    #if j == 0:
                    #    _t4_end += dtau
                    sigma_xdag_new = dict(sigma_xdag)
                    sigma_xx_new = dict(sigma_xx)
                    sigma_xxdag_new = dict(sigma_xxdag)
                    # add correct times
                    sigma_xxdag_new["time"] = _t1
                    sigma_xdag_new["time"] = _t2
                    sigma_xx_new["time"] = _t3
                    # the order of the operators is important to catch the special case where t1=t2
                    # because then ACE applies the operator first, that is first in the parameter file
                    multitime_op_new = [sigma_xdag_new,sigma_xxdag_new,sigma_xx_new]
                    if _t3 >= tb and _t4_end <= 2*tb:
                        _e = executor.submit(darkmodel,t0,_t4_end,*pulses,multitime_op=multitime_op_new, suffix=j, **options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
                    else:
                        # print("_t3:{:.4f}, _t4_end:{:.4f}".format(_t3, _t4_end))
                        _e = executor.submit(lambda: np.zeros([7,1]))
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
                wait(futures)
        for k in range(len(futures)):
            # futures are still 'future' objects
            futures[k] = futures[k].result()
        # futures now contains t,g,x,d,xx,pgx,pgb for every i
        # pgb [not anymore!, just take the abs before integration  
        # , but we take pgx two timesteps later. 
        # this does not make a huge difference, but avoids artifacts
        # due to oscillating pgb's ]
        _G2[i,0] = futures[0][6][-1]
        for k in range(1,len(t1)-i):
            # pgx
            _G2[i,k] = futures[k][5][-1]
    # _G2 now has to be integrated along all axes correctly, meaning with the correct time-axes
    # note that especially t2 is special, as it is 'folded' from t1
    os.remove(pulse_file_x)
    os.remove(pulse_file_y)
    return t1,_G2

def G2_lele(*pulses, t0=0, dt=0.1, dtau=0.1, delta_xd=4, delta_b=4, gamma_e=1/65, workers=15, temp_dir='/mnt/temp_data/', simple_exp=False, tb=800, gaussian_t=None):
    # note that dtau is used for dt in the actual propagation
    # t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    # n_tau = int((tauend-tau0)/dtau)
    # tau = np.linspace(tau0, tauend, n_tau + 1)
    # sigma_x = {"operator": "|0><1|_4","applyFrom": "", "applyBefore":"false"}  # not needed
    sigma_xdag = {"operator": "|1><0|_4","applyFrom": "_right", "applyBefore":"false"}
    sigma_x = {"operator": "|0><1|_4","applyFrom": "_left", "applyBefore":"false"}
    
    if simple_t_gaussian is not None:
        t1 = simple_t_gaussian(t0, gaussian_t, tb, dt, 10*dt, *pulses)
    else:
        t1 = construct_t(t0, tb, dt, 10*dt, *pulses, simple_exp=simple_exp)

    # 2*tb is the maximum simulation length
    _t_pulse = np.arange(t0,2.1*tb,step=dtau)
    # different polarizations
    pulse_file_x = temp_dir + "G2_pulse_x.dat"
    pulse_file_y = temp_dir + "G2_pulse_y.dat"
    pulse_x = np.zeros_like(_t_pulse, dtype=complex)
    pulse_y = np.zeros_like(_t_pulse, dtype=complex)
    for _p in pulses:
        pulse_x = pulse_x + _p.polar_x*_p.get_total(_t_pulse)
        pulse_y = pulse_y + _p.polar_y*_p.get_total(_t_pulse)
    export_csv(pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
    export_csv(pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')
    options = {"dt": dtau, "verbose": False, "delta_xd": delta_xd, "delta_b": delta_b, "gamma_e": gamma_e, "lindblad": True,
               "pulse_file_x": pulse_file_x, "pulse_file_y": pulse_file_y, "temp_dir": '/mnt/temp_data/', "output_ops": ["|3><3|_4"]}
    
    futures = []
    n_tau = int(tb/dtau)
    t2 = np.linspace(tb, 2*tb, n_tau + 1)
    _G2 = np.zeros([len(t1),len(t2)], dtype=complex)  # G2(t1,t2)
    with tqdm.tqdm(total=len(t1), leave=None) as tq:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for i in range(len(t1)):
                    # t1 = 0,...,tb
                    _t1 = t1[i]
                    sigma_xdag_new = dict(sigma_xdag)
                    sigma_x_new = dict(sigma_x)
                    # add correct times
                    sigma_xdag_new["time"] = _t1
                    sigma_x_new["time"] = _t1
                    # the order of the operators is important to catch the special case where t1=t2
                    # because then ACE applies the operator first, that is first in the parameter file
                    multitime_op_new = [sigma_xdag_new,sigma_x_new]
                    _e = executor.submit(darkmodel,t0,2*tb,*pulses,multitime_op=multitime_op_new, suffix=i, **options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                wait(futures)
    for k in range(len(futures)):
        # futures are still 'future' objects
        futures[k] = futures[k].result()
    # futures now contains t,xx for every i
    # _G2[i,0] = futures[0][5][-1]
    for i in range(len(t1)):
        _G2[i] = np.real(futures[i][1][-(n_tau+1):])
    # _G2 now has to be integrated along all axes correctly, meaning with the correct time-axes
    # note that especially t2 is special, as it is 'folded' from t1
    os.remove(pulse_file_x)
    os.remove(pulse_file_y)
    return t1,t2,_G2    


def G2_b(t0=0, tend=600, tau0=0, tauend=600, dt=0.1, dtau=0.1, *pulses, delta_xd=0, delta_b=4, gamma_e=1/100, workers=15, temp_dir='/mnt/temp_data/', coarse_t=True):
    """
    calculates G2 for the xx->g emission. This is not a 'real' G2 function, as it directly uses 2-photon emission.
    for every t1 in t, propagate to t1, then
    apply sigma = |g><xx| from left and sigma^dagger from the right to the density matrix
    propagate from t1 to t1+tau_max
    use results to calculate G2(t1,tau=0,..,tau_max) by applying sigma^dagger*sigma = |xx><xx| from the left to the density matrix
    and then taking the trace of the dens. matrix = rho_xx

    dtau is used as dt in calculations, dt just defines the t-grid discretization of G2
    dtau is the tau grid discretization.
    coarse_t uses dt during the pulse and 10*dt outside the pulse, i.e, -4*tau,...,4*tau
    """
    # includes tend
    t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    n_tau = int((tauend-tau0)/dtau)
    tau = np.linspace(tau0, tauend, n_tau + 1)
    multitime_op = {"operator": "|0><1|_4+|1><3|_4","applyFrom": "", "applyBefore":"false"}
    if coarse_t:
        t = construct_t(t0, tend, dt, 10*dt, *pulses)
    
    # the pulse has to be better resolved, because ACE uses intermediate steps
    # tend + tauend is the maximum simulation length
    _t_pulse = np.arange(t0,(tend+tauend),step=dtau)
    # different polarizations
    pulse_file_x = temp_dir + "G2_pulse_x.dat"
    pulse_file_y = temp_dir + "G2_pulse_y.dat"
    pulse_x = np.zeros_like(_t_pulse, dtype=complex)
    pulse_y = np.zeros_like(_t_pulse, dtype=complex)
    for _p in pulses:
        pulse_x = pulse_x + _p.polar_x*_p.get_total(_t_pulse)
        pulse_y = pulse_y + _p.polar_y*_p.get_total(_t_pulse)
    export_csv(pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
    export_csv(pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')

    # special case tau=0:
    # all 4 operators are applied at the same time.
    # G2(t,0) = Tr(sigma^dagger * sigma * sigma * rho(t) * sigma^dagger) = 0, as is sigma*sigma always zero.
    options = {"dt": dtau, "verbose": False, "delta_xd": delta_xd, "delta_b": delta_b, "gamma_e": gamma_e, "lindblad": True,
               "pulse_file_x": pulse_file_x, "pulse_file_y": pulse_file_y, "temp_dir": '/mnt/temp_data/'}
    _G2 = np.zeros([len(t),len(tau)])
    with tqdm.tqdm(total=len(t)) as tq:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(len(t)):
                multitime_op_new = dict(multitime_op)  # must make a copy of the dict
                multitime_op_new["time"] = t[i]
                _e = executor.submit(darkmodel,t0,t[i] + tauend,*pulses,multitime_op=multitime_op_new, suffix=i, **options)
                _e.add_done_callback(lambda f: tq.update())
                futures.append(_e)
            # wait for all futures
            wait(futures)
        for i in range(len(futures)):
            # futures are still 'future' objects
            futures[i] = futures[i].result()
        # futures now contains t,g,x,d,xx for every i
        for i in range(len(t)):
            # special case tau=0:
            # as Tr(sigma^dagger*sigma^dagger*sigma*sigma * rho) = xx, G2(t,0) = xx(t), which is the value with index [-(n_tau+1)]
            _G2[i,0] = np.real(futures[i][4][-n_tau-1])
            # futures[i][4] are the xx values , [2] are the x values
            _G2[i,1:] = np.real(futures[i][4][-n_tau:]+futures[i][2][-n_tau:])
    os.remove(pulse_file_x)
    os.remove(pulse_file_y)
    return t, tau, _G2
