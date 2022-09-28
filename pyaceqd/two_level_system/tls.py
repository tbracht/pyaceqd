from asyncio import futures
import subprocess
import numpy as np
import os
from pyaceqd.tools import export_csv
import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

hbar = 0.6582173  # meV*ps

def tls_ace(t_start, t_end, *pulses, dt=0.1, gamma_e=1/100, phonons=False, generate_pt=False, t_mem=10, ae=3.0, temperature=1,verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
                  apply_op_l=None, apply_op_t=0):
    tmp_file = temp_dir + "tls{}.param".format(suffix)
    out_file = temp_dir + "tls{}.out".format(suffix)
    duration = np.abs(t_end)+np.abs(t_start)
    if pt_file is None:
        pt_file = "tls_generate_{}ps_{}K_{}nm.pt".format(duration,temperature,ae)
    pulse_file = temp_dir + "tls_pulse{}.dat".format(suffix)
    t,g,x,p = 0,0,0,0
    if phonons:
        if not os.path.exists(pt_file):
            print("{} not found. Calculating...".format(pt_file))
            generate_pt = True
            verbose = True
    multitime = False
    if apply_op_l is not None:
        multitime = True
    try:
        t = np.arange(1.1*t_start,1.1*t_end,step=0.5*dt)
        pulse = np.zeros_like(t, dtype=complex)
        for _p in pulses:
            pulse = pulse + _p.get_total(t)
        export_csv(pulse_file, t, pulse.real, pulse.imag, precision=8, delimit=' ')
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
                f.write("Nintermediate    10\n")
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
                # apply_Operator 20 {|1><0|_2} would apply the operator |1><0|_2 at t=20 from the left and the h.c. on the right on the density matrix
                # note the Operator is applied at time t, i.e., in this example at t=20, so its effect is only visible at t=20+dt
                f.write("apply_Operator {} {{ {} }}\n".format(apply_op_t, apply_op_l))
            # output 
            f.write("add_Output {|0><0|_2}\n")
            f.write("add_Output {|1><1|_2}\n")
            f.write("add_Output {|0><1|_2}\n")
            if generate_pt:
                f.write("write_PT {}\n".format(pt_file))
            f.write("outfile {}\n".format(out_file))
        if not verbose:
            subprocess.check_output(["ACE",tmp_file])
        else:
            subprocess.check_call(["ACE",tmp_file])
        data = np.genfromtxt(out_file)
        t = data[:,0]
        g = data[:,1]
        x = data[:,3]
        p = data[:,5] + 1j*data[:,6]

    finally:
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        os.remove(tmp_file)
        os.remove(pulse_file)
    return t,g,x,p


def G2(t0=0, tend=600, tau0=0, tauend=600, dt=0.1, *pulses, ae=5.0, gamma_e=1/100, phonons=False, pt_file="g2_tensor.pt", thread=False, workers=15):
    """
    calculates G2 for the x->g emission
    for every t1 in t, propagate to t1, then
    apply sigma = |g><x| from left and sigma^dagger from the right to the density matrix
    propagate from t1 to t1+tau_max
    use results to calculate G2(t1,tau=0,..,tau_max) by applying sigma^dagger*sigma from the left to the density matrix
    and then taking the trace of the dens. matrix
    """
    t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    n_tau = int((tauend-tau0)/dt)
    tau = np.linspace(tau0, tauend, n_tau + 1)

    # calculate process tensor for longest time tend+tauend. this can then be re-used for every following phonon calculation
    if phonons:
        tls_ace(t0,tend+tauend,*pulses,dt=dt,ae=ae,verbose=True,phonons=phonons, delta_b=4, pt_file=pt_file)

    # special case tau=0:
    # all 4 operators are applied at the same time.
    # G2(t,0) = Tr(sigma^dagger * sigma * sigma * rho(t) * sigma^dagger) = 0, as is sigma*sigma always zero.
    options = {"dt": dt, "ae": ae, "verbose": False, "phonons": phonons, "gamma_e": gamma_e, "lindblad": True, "apply_op_l": "|0><1|_2", "pt_file": pt_file}
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
    return _G2
