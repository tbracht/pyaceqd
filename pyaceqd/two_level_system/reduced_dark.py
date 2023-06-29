import numpy as np
import os
from pyaceqd.tools import export_csv, construct_t, simple_t_gaussian
import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from pyaceqd.general_system.general_system import system_ace_stream

hbar = 0.6582173  # meV*ps

def darkmodel(t_start, t_end, *pulses, dt=0.5, delta_xd=0, gamma_e=1/65, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_3","|1><1|_3","|2><2|_3"], initial="|0><0|_3"):
    system_prefix = "tls_dark"
    # |0> = G, |1> = X, |2> = D
    system_op = ["{}*|2><2|_3".format(-delta_xd)]
    # system_op = ["{}*|1><1|_4".format(delta_b*0.5),"{}*|2><2|_4".format(delta_b*0.5-delta_xd)]
    boson_op = "|1><1|_3 + |2><2|_3"
    # initial = "|0><0|_4"
    lindblad_ops = []
    if lindblad:
        lindblad_ops = [["|0><1|_3",gamma_e]]  #  |2> is dark, does not decay 
    # we use 'x'-polar for coupling between G and D, as well as X and D.
    interaction_ops = [["|2><0|_3","x"],["|1><2|_3","x"],["|1><0|_3","y"]]
    
    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=20.48, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only)
    return result

def darkmodel_photons(t_start, t_end, *pulses, dt=0.1, delta_xd=0, delta_cx=-2, rad_loss=1/100, cav_loss=1/20, cav_coupl=1/30, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_3 otimes |0><0|_3","|1><1|_3 otimes |0><0|_3","|2><2|_3 otimes |0><0|_3"], initial="|0><0|_3 otimes |0><0|_3"):
    system_prefix = "darkmodel_tls_photons"
    # |0> = G, |1> = X, |2> = D
    system_op = ["{}*|2><2|_3 otimes Id_3".format(-delta_xd)]
    boson_op = "|1><1|_3 otimes Id_3 + |2><2|_3 otimes Id_3"
    lindblad_ops = []
    if lindblad:
        # radiative decay of dot, outside the cavity
        lindblad_ops = [["|0><1|_3 otimes Id_3",rad_loss]]  #  |2> is dark, does not decay 
    # interaction with laser
    interaction_ops = [["|2><0|_3 otimes Id_3","x"],["|1><2|_3 otimes Id_3","x"],["|1><0|_3 otimes Id_3","y"]]
    # cavity decay
    lindblad_ops.append(["Id_3 otimes b_3",cav_loss])
    # cavity detuning
    system_op.append(" {} * (Id_3 otimes n_3)".format(delta_cx))
    # cavity-qd coupling
    system_op.append("{}*(|1><0|_3 otimes b_3 + |0><1|_3 otimes bdagger_3 )".format(hbar*cav_coupl))
    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=20.48, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only)
    return result

def G1_ee(*pulses, t0=0, dt=0.05, delta_xd=4, gamma_e=1/65, temp_dir='/mnt/temp_data/', tb=800, normalize=False, phonons=False, pt_file=None, prepare_only=False):
    t,g,x,d = darkmodel(t0,tb,*pulses,dt=dt,delta_xd=delta_xd,gamma_e=gamma_e,lindblad=True,temp_dir=temp_dir,phonons=phonons, pt_file=pt_file,prepare_only=prepare_only)
    x = np.real(x)
    t = np.real(t)
    rho_ee = np.trapz(x,t)
    if normalize:
        return rho_ee/gamma_e
    return rho_ee

def G1_ll(*pulses, t0=0, dt=0.05, delta_xd=4, gamma_e=1/65, temp_dir='/mnt/temp_data/', tb=800, normalize=False, phonons=False, pt_file=None):
    t,g,x,d = darkmodel(t0,2*tb,*pulses,dt=dt,delta_xd=delta_xd,gamma_e=gamma_e,lindblad=True,temp_dir=temp_dir,phonons=phonons, pt_file=pt_file)
    x = np.real(x)
    t = np.real(t)
    n_t = int(tb/dt)
    relevant_x = x[-n_t:]
    relevant_t = t[-n_t:]
    rho_ee = np.trapz(relevant_x,relevant_t)
    if normalize:
        return rho_ee/gamma_e
    return rho_ee

def G1_el(*pulses, t0=0, dt=0.1, dtau=0.05, delta_xd=4, gamma_e=1/65, temp_dir='/mnt/temp_data/', tb=800, workers=15, normalize=False, simple_exp=False, gaussian_t=None, phonons=False, pt_file=None):
    multitime_op = {"operator": "|1><0|_3","applyFrom": "_right", "applyBefore":"false"}

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
    options = {"dt": dtau, "verbose": False, "delta_xd": delta_xd, "gamma_e": gamma_e, "lindblad": True,
               "pulse_file_x": pulse_file_x, "pulse_file_y": pulse_file_y, "temp_dir": '/mnt/temp_data/', "output_ops": ["|0><0|_3","|1><1|_3","|2><2|_3","|0><1|_3"],
               "phonons": phonons, "pt_file": pt_file}
    _G1 = np.zeros([len(t1),len(t2)],dtype=complex)
    tend = 2*tb
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
        # futures now contains t,g,x,d,pgx for every i
        for i in range(len(t1)):
            # x
            _G1[i,0] = futures[i][2][-n_tau-1]
            # pgx
            _G1[i,1:] = futures[i][4][-n_tau:]
    os.remove(pulse_file_x)
    os.remove(pulse_file_y)
    return t1,t2,_G1

def G1_easy_el(*pulses, t0=0, dt=0.1, dtau=0.05, delta_xd=4, gamma_e=1/65, temp_dir='/mnt/temp_data/', tb=800, t_offset=0, workers=15, normalize=False, simple_exp=False, gaussian_t=None, phonons=False, pt_file=None):
    multitime_op = {"operator": "|1><0|_3","applyFrom": "_right", "applyBefore":"false"}

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
    options = {"dt": dtau, "verbose": False, "delta_xd": delta_xd, "gamma_e": gamma_e, "lindblad": True,
               "pulse_file_x": pulse_file_x, "pulse_file_y": pulse_file_y, "temp_dir": '/mnt/temp_data/', "output_ops": ["|0><0|_3","|1><1|_3","|2><2|_3","|0><1|_3"],
               "phonons": phonons, "pt_file": pt_file}
    _G1 = np.zeros([len(t1)],dtype=complex)
    with tqdm.tqdm(total=len(t1)) as tq:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(len(t1)):
                multitime_op_new = dict(multitime_op)  # must make a copy of the dict
                multitime_op_new["time"] = t1[i]
                tend = t1[i] + tb + t_offset
                _e = executor.submit(darkmodel,t0,tend,*pulses,multitime_op=multitime_op_new, suffix=i, **options)
                _e.add_done_callback(lambda f: tq.update())
                futures.append(_e)
            # wait for all futures
            wait(futures)
        for i in range(len(futures)):
            # futures are still 'future' objects
            futures[i] = futures[i].result()
        # futures now contains t,g,x,d,pgx for every i
        for i in range(len(t1)):
            # pgx
            _G1[i] = futures[i][4][-1]
    os.remove(pulse_file_x)
    os.remove(pulse_file_y)
    return t1,_G1