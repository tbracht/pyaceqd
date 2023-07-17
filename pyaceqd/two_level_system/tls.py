from asyncio import futures
import subprocess
import numpy as np
import os
from pyaceqd.tools import export_csv, construct_t
import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from pyaceqd.general_system.general_system import system_ace_stream
from pyaceqd.general_system.general_dressed_states import dressed_states
hbar = 0.6582173  # meV*ps

def tls_(t_start, t_end, *pulses, dt=0.1, gamma_e=1/100, phonons=False, t_mem=10, ae=3.0, temperature=1,verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
         multitime_op=None, pulse_file=None, prepare_only=False, output_ops=["|0><0|_2","|1><1|_2","|0><1|_2","|1><0|_2"], LO_params=None, dressedstates=False, rf=False, rf_file=None, firstonly=False):
    system_prefix = "tls"
    system_op = None
    boson_op = "1*|1><1|_2"
    initial = "|0><0|_2"
    lindblad_ops = []
    if lindblad:
        lindblad_ops = [["|0><1|_2",gamma_e]]
    # note that the TLS uses x-polar
    interaction_ops = [["|1><0|_2","x"]]
    # rotating frame of pulse
    rf_op = None
    if rf:
        rf_op = "|1><1|_2"
    # multitime: for ex. ["|1><0|_2",0,"left"] applies |1><0|_2 at t=0 from the left
    # invoke_dict = {"dt": dt, "phonons": phonons, "generate_pt": generate_pt, "t_mem": t_mem, "ae": ae, "temperature": temperature}
    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, pulse_file_x=pulse_file, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only, LO_params=LO_params, dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file,
                  firstonly=firstonly)
    return result

def tls_dressed_states(t_start, t_end, *pulses, plot=True, filename="tls_dressed", firstonly=False, **options):
    # dim = 2 for TLS
    colors=["#0000FF", "#FF0000"]
    dim = 2
    return dressed_states(tls_, dim, t_start, t_end, *pulses, filename=filename, plot=plot, firstonly=firstonly, colors=colors, **options)

def tls_photons(t_start, t_end, *pulses, dt=0.1, gamma_e=1/100, cav_coupl1=0.06, cav_loss1=0.12/hbar, delta_cx1=-2, cav_coupl2=None, cav_loss2=None, delta_cx2=-2, phonons=False, t_mem=10, ae=5.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
         multitime_op=None, n_phot1=2, n_phot2=2, pulse_file=None, prepare_only=False, output_ops=["|0><0|_2 otimes Id_2 otimes Id_2","|1><1|_2 otimes Id_2 otimes Id_2"], dressedstates=False, rf=False, rf_file=None, firstonly=False):
    n1 = n_phot1 + 1
    n2 = n_phot2 + 1
    system_prefix = "tls_cavity"
    system_op = []
    boson_op = "|1><1|_2 otimes Id_{} otimes Id_{}".format(n1,n2)
    initial = "|0><0|_2 otimes |0><0|_{} otimes |0><0|_{}".format(n1,n2)
    lindblad_ops = []
    if lindblad:
        lindblad_ops = [["|0><1|_2 otimes Id_{} otimes Id_{}".format(n1,n2), gamma_e]]
    # note that the TLS uses x-polar
    interaction_ops = [["|1><0|_2 otimes Id_{} otimes Id_{}".format(n1,n2),"x"]]
    # rotating frame of pulse
    rf_op = None
    if rf:
        rf_op = "|1><1|_2 otimes Id_{} otimes Id_{}".format(n1,n2)
        rf_op = rf_op + " + Id_2 otimes n_{} otimes Id_{}".format(n1,n2)
        rf_op = rf_op + " + Id_2 otimes Id_{} otimes n_{}".format(n1,n2)
        if pulse_file is not None and rf_file is None:
            print("Error: pulse file is given, but no file for rotating frame")
            return 0

    if cav_coupl2 is None:
        cav_coupl2 = cav_coupl1
    if cav_loss2 is None:
        cav_loss2 = cav_loss1
    # cavity detuning
    system_op.append(" {} * (Id_2 otimes n_{} otimes Id_{})".format(delta_cx1, n1, n2))
    system_op.append(" {} * (Id_2 otimes Id_{} otimes n_{})".format(delta_cx2, n1, n2))
    # cavity coupling
    system_op.append(" {} * (|1><0|_2 otimes b_{} otimes Id_{} + |0><1|_2 otimes bdagger_{} otimes Id_{})".format(cav_coupl1, n1, n2, n1, n2))
    system_op.append(" {} * (|1><0|_2 otimes Id_{} otimes b_{} + |0><1|_2 otimes Id_{} otimes bdagger_{})".format(cav_coupl2, n1, n2, n1, n2))
    # cavity loss
    lindblad_ops.append(["Id_2 otimes b_{} otimes Id_{}".format(n1, n2), cav_loss1])
    lindblad_ops.append(["Id_2 otimes Id_{} otimes b_{}".format(n1, n2), cav_loss2])

    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, pulse_file_x=pulse_file, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only, dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file,
                  firstonly=firstonly)
    return result

def tls_photons_dressed_states(t_start, t_end, *pulses, plot=True, filename="tls_photons_dressed", firstonly=False, **options):
    # dim = 2 for TLS
    n1 = options["n_phot1"] + 1
    n2 = options["n_phot2"] + 1
    dim = [2,n1,n2]
    return dressed_states(tls_photons, dim, t_start, t_end, *pulses, filename=filename, plot=plot, firstonly=firstonly, colors=None, **options)

def tls_ace(t_start, t_end, *pulses, dt=0.1, gamma_e=1/100, phonons=False, generate_pt=False, t_mem=10, ae=3.0, temperature=1,verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
                  apply_op=None, apply_op_t=0, apply="", ninterm=10, pulse_file=None):
    tmp_file = temp_dir + "tls{}.param".format(suffix)
    out_file = temp_dir + "tls{}.out".format(suffix)
    duration = np.abs(t_end)+np.abs(t_start)
    if pt_file is None:
        pt_file = "tls_generate_{}ps_{}K_{}nm.pt".format(duration,temperature,ae)
    if phonons:
        if not os.path.exists(pt_file):
            print("{} not found. Calculating...".format(pt_file))
            generate_pt = True
            verbose = True
    multitime = False
    if apply_op is not None:
        multitime = True
    t = np.arange(1.1*t_start,1.1*t_end,step=0.1*dt)
    _remove_pulse_file = False
    if pulse_file is None:
        _remove_pulse_file = True
        pulse_file = temp_dir + "tls_pulse{}.dat".format(suffix)
        pulse = np.zeros_like(t, dtype=complex)
        for _p in pulses:
            pulse = pulse + _p.get_total(t)
        export_csv(pulse_file, t, pulse.real, pulse.imag, precision=8, delimit=' ')
    try:
        t,g,x,p = 0,0,0,0
        with open(tmp_file,'w') as f:
            f.write("ta    {}\n".format(t_start))
            f.write("te    {}\n".format(t_end))
            f.write("dt    {}\n".format(dt))
            if generate_pt:
                f.write("t_mem    {}\n".format(t_mem))
                f.write("threshold 1e-7\n")
                f.write("use_Gaussian true\n")
                f.write("Boson_SysOp    { |1><1|_2 }\n")
                f.write("Boson_J_type         QDPhonon\n")
                f.write("Boson_J_a_e    {}\n".format(ae))
                f.write("Boson_temperature    {}\n".format(temperature))
                f.write("Boson_subtract_polaron_shift       true\n")
            else:
                f.write("Nintermediate    {}\n".format(ninterm))
                f.write("use_symmetric_Trotter true\n")
            if phonons and not generate_pt:
                # process tensor path has to be given or in current dir!
                f.write("read_PT    {}\n".format(pt_file))
                f.write("Boson_subtract_polaron_shift       true\n")
            f.write("initial    {}\n".format("{|0><0|_2}"))
            if lindblad:
                f.write("add_Lindblad {:.5f}  {{|0><1|_2}}\n".format(gamma_e))  # x->g
            # pulse
            f.write("add_Pulse file {}  {{-{}*(|1><0|_2)}}\n".format(pulse_file,np.pi*hbar/2))
            if multitime:
                # apply_Operator 20 {|0><1|_2} would apply the operator |0><1|_2 at t=20 from the left and the h.c. on the right on the density matrix
                # note the Operator is applied at time t, i.e., in this example at t=20, so its effect is only visible at t=20+dt
                if apply == "left":
                    f.write("apply_Operator_left {} {{ {} }}\n".format(apply_op_t, apply_op))
                elif apply == "right":
                    f.write("apply_Operator_right {} {{ {} }}\n".format(apply_op_t, apply_op))
                else:
                    f.write("apply_Operator {} {{ {} }}\n".format(apply_op_t, apply_op))
            # output 
            f.write("add_Output {|0><0|_2}\n")
            f.write("add_Output {|1><1|_2}\n")
            f.write("add_Output {|0><1|_2}\n")
            f.write("add_Output {|1><0|_2}\n")
            if generate_pt:
                f.write("write_PT {}\n".format(pt_file))
            f.write("outfile {}\n".format(out_file))
        if not verbose:
            subprocess.check_output(["ACE",tmp_file])
        else:
            subprocess.check_call(["ACE",tmp_file])
        data = np.genfromtxt(out_file)
        t = data[:,0]  # note that the 't' of ACE is used in the end
        g = data[:,1]
        x = data[:,3]
        pgx = data[:,5] + 1j*data[:,6]
        pxg = data[:,7] + 1j*data[:,8]

    finally:
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        os.remove(tmp_file)
        if _remove_pulse_file:
            os.remove(pulse_file)
    return t,g,x,pgx,pxg


def G2(t0=0, tend=600, tau0=0, tauend=600, dt=0.1, dtau=0.1, *pulses, ae=5.0, gamma_e=1/100, phonons=False, pt_file="g2_tensor.pt", thread=False, workers=15, ninterm=100, temp_dir='/mnt/temp_data/', coarse_t=False):
    """
    calculates G2 for the x->g emission
    for every t1 in t, propagate to t1, then
    apply sigma = |g><x| from left and sigma^dagger from the right to the density matrix
    propagate from t1 to t1+tau_max
    use results to calculate G2(t1,tau=0,..,tau_max) by applying sigma^dagger*sigma from the left to the density matrix
    and then taking the trace of the dens. matrix

    dtau is used as dt in calculations, dt just defines the t-grid discretization of G2
    dtau is the tau grid discretization.
    coarse_t uses dt during the pulse and 10*dt outside the pulse, i.e, -4*tau,...,4*tau
    """
    # includes tend
    t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    n_tau = int((tauend-tau0)/dtau)
    tau = np.linspace(tau0, tauend, n_tau + 1)

    if coarse_t:
        t = construct_t(t0, tend, dt, 10*dt, *pulses)
    
    # the pulse has to be better resolved, because ACE uses intermediate steps
    # tend + tauend is the maximum simulation length
    _t_pulse = np.arange(1.1*t0,1.1*(tend+tauend),step=0.01*dtau)
    pulse_file = temp_dir + "G2_pulse.dat"
    pulse = np.zeros_like(_t_pulse, dtype=complex)
    for _p in pulses:
        pulse = pulse + _p.get_total(_t_pulse)
    export_csv(pulse_file, _t_pulse, pulse.real, pulse.imag, precision=8, delimit=' ')

    # calculate process tensor for longest time tend+tauend. this can then be re-used for every following phonon calculation
    if phonons:
        tls_ace(t0,tend+tauend,*pulses,dt=dtau,ae=ae,verbose=True,phonons=phonons, pt_file=pt_file,ninterm=ninterm)

    # special case tau=0:
    # all 4 operators are applied at the same time.
    # G2(t,0) = Tr(sigma^dagger * sigma * sigma * rho(t) * sigma^dagger) = 0, as is sigma*sigma always zero.
    options = {"dt": dtau, "ae": ae, "ninterm": ninterm,"verbose": False, "phonons": phonons, "gamma_e": gamma_e, "lindblad": True,
                "apply_op": "|0><1|_2", "pt_file": pt_file, "pulse_file": pulse_file, "temp_dir": '/mnt/temp_data/'}
    _G2 = np.zeros([len(t),len(tau)])
    if thread:
        with tqdm.tqdm(total=len(t)) as tq:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for i in range(len(t)):
                    _e = executor.submit(tls_ace,t0,t[i] + tauend,*pulses,apply_op_t=t[i], suffix=i, **options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                futures[i] = futures[i].result()
            # futures now contains t,g,x,p for every i
            for i in range(len(t)):
                # futures[i][2] are the x values 
                _G2[i,1:] = futures[i][2][-n_tau:]
    else:
        for i in tqdm.trange(len(t)):
            _tend = t[i] + tauend
            t,g,x,p = tls_ace(t0,_tend,*pulses,apply_op_t=t[i], suffix=i, **options)
            # use, that Tr(sigma_x^dagger*sigma_x*rho) = x
            # for the last n_tau elements, not including tau=0, which stays zero
            _G2[i,1:] = x[-n_tau:]
    os.remove(pulse_file)
    return t, tau, _G2


def G2hom(t0=0, tend=600, tau0=0, tauend=600, dt=0.1, dtau=0.1, *pulses, ae=5.0, gamma_e=1/100, phonons=False, pt_file="g2_tensor.pt", thread=True, workers=15, ninterm=100, temp_dir='/mnt/temp_data/'):
    """
    calculates G2 for the x->g emission
    for every t1 in t, propagate to t1, then
    apply sigma = |g><x| from left and sigma^dagger from the right to the density matrix
    propagate from t1 to t1+tau_max
    use results to calculate G2(t1,tau=0,..,tau_max) by applying sigma^dagger*sigma from the left to the density matrix
    and then taking the trace of the dens. matrix (this results in the occupation at that point)

    dtau is used as dt in calculations, dt just defines the t-grid discretization of G2hom. Note, that for a pulse train, this also has to be well-resolved
    dtau is the tau grid discretization.
    """
    # includes tend
    t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    n_tau = int((tauend-tau0)/dtau)
    tau = np.linspace(tau0, tauend, n_tau + 1)

    # the pulse has to be better resolved, because ACE uses intermediate steps
    # tend + tauend is the maximum simulation length
    _t_pulse = np.arange(1.1*t0,1.1*(tend+tauend),step=0.01*dtau)
    pulse_file = temp_dir + "G2_pulse.dat"
    pulse = np.zeros_like(_t_pulse, dtype=complex)
    for _p in pulses:
        pulse = pulse + _p.get_total(_t_pulse)
    export_csv(pulse_file, _t_pulse, pulse.real, pulse.imag, precision=8, delimit=' ')

    # calculate process tensor for longest time tend+tauend. this can then be re-used for every following phonon calculation
    if phonons:
        tls_ace(t0,tend+tauend,*pulses,dt=dtau,ae=ae,verbose=True,phonons=phonons, pt_file=pt_file,ninterm=ninterm,pulse_file=pulse_file)

    # special case tau=0:
    # all 4 operators are applied at the same time.
    # G2(t,0) = Tr(sigma^dagger * sigma * sigma * rho(t) * sigma^dagger) = 0, as is sigma*sigma always zero.
    options = {"dt": dtau, "ae": ae, "ninterm": ninterm,"verbose": False, "phonons": phonons, "gamma_e": gamma_e, "lindblad": True,
                "apply_op": "|0><1|_2", "pt_file": pt_file, "pulse_file": pulse_file, "temp_dir": '/mnt/temp_data/'}
    _G2hom = np.zeros([len(t),len(tau)])
    # occupation part
    _t,_,_x,_ = tls_ace(t0, tend+tauend, *pulses, dt=dtau, ae=ae, verbose=False, phonons=phonons, pt_file=pt_file, ninterm=ninterm, pulse_file=pulse_file, gamma_e=gamma_e, lindblad=True)
    for i in range(len(t)):
        # add G2hom(t,tau) += x(t)x(t+tau)
        # note, that dt can be different from the dt of the calculation
        i_dt = i*int(dt/dtau)
        _G2hom[i] += _x[i_dt]*_x[i_dt:i_dt+n_tau+1]
    # G2 part
    with tqdm.tqdm(total=len(t)) as tq:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(len(t)):
                _e = executor.submit(tls_ace,t0,t[i] + tauend, *pulses, apply_op_t=t[i], suffix=i, **options)
                _e.add_done_callback(lambda f: tq.update())
                futures.append(_e)
            # wait for all futures
            wait(futures)
        for i in range(len(futures)):
            # futures are still 'future' objects
            futures[i] = futures[i].result()
        # futures now contains t,g,x,p for every i
        for i in range(len(t)):
            # futures[i][2] are the x values 
            _G2hom[i,1:] += futures[i][2][-n_tau:]
    # G1 part
    with tqdm.tqdm(total=len(t)) as tq:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(len(t)):
                # remember to only apply sigma from the left for G1
                _e = executor.submit(tls_ace,t0,t[i] + tauend, *pulses, apply_op_t=t[i], apply="left", suffix=i, **options)
                _e.add_done_callback(lambda f: tq.update())
                futures.append(_e)
            # wait for all futures
            wait(futures)
        for i in range(len(futures)):
            # futures are still 'future' objects
            futures[i] = futures[i].result()
        # futures now contains [t,g,x,pgx,pxg] for every i
        for i in range(len(t)):
            # futurs[i] is [t,g,x,pgx,pxg]
            # futures[i][3] are the pgx values
            # futures[i][2] are the x values
            # special case tau=0:
            # as Tr(sigma^dagger*sigma * rho) = x, G1(t,0) = x(t), which is the value with index [-(n_tau+1)]
            _G2hom[i,0] -= np.abs(futures[i][2][-n_tau-1])**2
            # as Tr(sigma^dagger * rho) =  <|x><g|> = pxg
            _G2hom[i,1:] -= np.abs(futures[i][4][-n_tau:])**2
    os.remove(pulse_file)
    return t, tau, _G2hom

def G1(t0=0, tend=600, tau0=0, tauend=600, dt=0.1, dtau=0.5, *pulses, ae=5.0, temp=0.0, gamma_e=1/100, phonons=False, pt_file="g1_tensor.pt", thread=True, workers=15, ninterm=100, temp_dir='/mnt/temp_data/', coarse_t=False):
    """
    calculates G1 for the x->g emission
    for every t1 in t, propagate to t1, then
    apply sigma = |g><x| from the left to the density matrix
    propagate from t1 to t1+tau_max
    use results to calculate G1(t1,tau=0,..,tau_max) by applying sigma^dagger from the left to the density matrix
    and then taking the trace of the dens. matrix (this results in some polarization at that point)

    dtau is used as dt in calculations, dt just defines the t-grid discretization of G1. Note, that for a pulse train, this also has to be well-resolved
    dtau is also the tau grid discretization.
    """
    # includes tend
    t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    n_tau = int((tauend-tau0)/dtau)
    tau = np.linspace(tau0, tauend, n_tau + 1)

    if coarse_t:
        t = construct_t(t0, tend, dt, 10*dt, *pulses)

    # the pulse has to be better resolved, because ACE uses intermediate steps
    # tend + tauend is the maximum simulation length
    _t_pulse = np.arange(1.1*t0,1.1*(tend+tauend),step=dtau/(10*ninterm))
    pulse_file = temp_dir + "G1_pulse.dat"
    pulse = np.zeros_like(_t_pulse, dtype=complex)
    for _p in pulses:
        pulse = pulse + _p.get_total(_t_pulse)
    export_csv(pulse_file, _t_pulse, pulse.real, pulse.imag, precision=8, delimit=' ')

    # calculate process tensor for longest time tend+tauend. this can then be re-used for every following phonon calculation
    if phonons and not os.path.exists(pt_file):
        print("calculating pt file for G1")
        tls_ace(t0,tend+tauend,*pulses,dt=dtau,ae=ae,verbose=True,phonons=phonons, pt_file=pt_file,ninterm=ninterm,pulse_file=pulse_file,temperature=temp)
    if phonons:
        print("using pt file {}".format(pt_file))
    options = {"dt": dtau, "ae": ae, "ninterm": ninterm,"verbose": False, "phonons": phonons, "gamma_e": gamma_e, "lindblad": True,
                "apply_op": "|0><1|_2", "pt_file": pt_file, "pulse_file": pulse_file, "temp_dir": '/mnt/temp_data/', "temperature": temp}
    _G1 = np.zeros([len(t),len(tau)],dtype=complex)
    # G1 part
    with tqdm.tqdm(total=len(t)) as tq:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(len(t)):
                # remember to only apply sigma from the left for G1
                _e = executor.submit(tls_ace,t0,t[i] + tauend, *pulses, apply_op_t=t[i], apply="left", suffix=i, **options)
                _e.add_done_callback(lambda f: tq.update())
                futures.append(_e)
            # wait for all futures
            wait(futures)
        for i in range(len(futures)):
            # futures are still 'future' objects
            futures[i] = futures[i].result()
        # futures now contains [t,g,x,pgx,pxg] for every i
        for i in range(len(t)):
            # futurs[i] is [t,g,x,pgx,pxg]
            # futures[i][3] are the pgx values
            # futures[i][2] are the x values
            # special case tau=0:
            # as Tr(sigma^dagger*sigma * rho) = x, G1(t,0) = x(t), which is the value with index [-(n_tau+1)]
            _G1[i,0] = futures[i][2][-n_tau-1]
            # as Tr(sigma^dagger * rho) =  <|x><g|> = pxg
            _G1[i,1:] = futures[i][4][-n_tau:]
    os.remove(pulse_file)
    return t, tau, _G1

