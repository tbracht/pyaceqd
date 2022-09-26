import subprocess
import numpy as np
import os
from pyaceqd.tools import export_csv
import tqdm

hbar = 0.6582173  # meV*ps

def biexciton_ace(t_start, t_end, *pulses, dt=0.1, delta_xy=0, delta_b=4, gamma_e=1/100, gamma_b=2/100, phonons=False, generate_pt=False, t_mem=10, ae=3.0, temperature=1,verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
                  apply_op_l=None, apply_op_t=0):
    tmp_file = temp_dir + "biex{}.param".format(suffix)
    out_file = temp_dir + "biex{}.out".format(suffix)
    duration = np.abs(t_end)+np.abs(t_start)
    if pt_file is None:
        pt_file = "sixls_linear_generate_{}ps_{}K_{}nm.pt".format(duration,temperature,ae)
    pulse_file_x = temp_dir + "sixls_linear_pulse_x{}.dat".format(suffix)
    pulse_file_y = temp_dir + "sixls_linear_pulse_y{}.dat".format(suffix)
    t,g,x,y,b = 0,0,0,0,0
    gamma_b = gamma_b / 2  # both X and Y decay. the input gamma_b is 1/tau_b, where tau_b is the lifetime of the biexciton
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
        pulse_x = np.zeros_like(t, dtype=complex)
        pulse_y = np.zeros_like(t, dtype=complex)
        for _p in pulses:
            pulse_x = pulse_x + _p.polar_x * _p.get_total(t)
            pulse_y = pulse_y + _p.polar_y * _p.get_total(t)
        export_csv(pulse_file_x, t, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
        export_csv(pulse_file_y, t, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')
        with open(tmp_file,'w') as f:
            f.write("ta    {}\n".format(t_start))
            f.write("te    {}\n".format(t_end))
            f.write("dt    {}\n".format(dt))
            if generate_pt:
                f.write("t_mem    {}\n".format(t_mem))
                f.write("threshold 1e-7\n")
                f.write("use_Gaussian true\n")
                f.write("Boson_SysOp    { 1*(|1><1|_4 + |2><2|_4) + 2*|3><3|_4}\n")
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
            f.write("initial    {}\n".format("{|0><0|_4}"))
            if lindblad:
                f.write("add_Lindblad {:.5f}  {{|0><1|_4}}\n".format(gamma_e))  # x->g
                f.write("add_Lindblad {:.5f}  {{|0><2|_4}}\n".format(gamma_e))  # y->g
                f.write("add_Lindblad {:.5f}  {{|1><3|_4}}\n".format(gamma_b))  # b->x
                f.write("add_Lindblad {:.5f}  {{|2><3|_4}}\n".format(gamma_b))  # b->y
            # energies
            f.write("add_Hamiltonian  {{ -{}*|3><3|_4}}\n".format(delta_b))
            f.write("add_Hamiltonian  {{ -{}*|2><2|_4}}\n".format(delta_xy))
            # pulse
            f.write("add_Pulse file {}  {{-{}*(|1><0|_4+|3><1|_4)}}\n".format(pulse_file_x,np.pi*hbar/2))
            f.write("add_Pulse file {}  {{-{}*(|2><0|_4+|3><2|_4)}}\n".format(pulse_file_y,np.pi*hbar/2))
            if multitime:
                # apply_Operator 20 {|1><0|_2} would apply the operator |1><0|_2 at t=20 from the left and the h.c. on the right on the density matrix
                # note the Operator is applied at time t, i.e., in this example at t=20, so its effect is only visible at t=20+dt
                f.write("apply_Operator {} {{ {} }}\n".format(apply_op_t, apply_op_l))
            # output 
            f.write("add_Output {|0><0|_4}\n")
            f.write("add_Output {|1><1|_4}\n")
            f.write("add_Output {|2><2|_4}\n")
            f.write("add_Output {|3><3|_4}\n")
            # if multitime:
            #     f.write("add_Output {|0><1|_4}\n")
            #     f.write("add_Output {|0><2|_4}\n")
            #     f.write("add_Output {|0><3|_4}\n")
            #     f.write("add_Output {|1><2|_4}\n")
            #     f.write("add_Output {|1><3|_4}\n")
            #     f.write("add_Output {|2><3|_4}\n")

                # f.write("add_Output {|1><0|_4}\n")
                # f.write("add_Output {|2><0|_4}\n")
                # f.write("add_Output {|3><0|_4}\n")
                # f.write("add_Output {|2><1|_4}\n")
                # f.write("add_Output {|3><1|_4}\n")
                # f.write("add_Output {|3><2|_4}\n")
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
        y = data[:,5]
        b = data[:,7]
        # if multitime:
        #     p_gx = data[:,9] + 1j*data[:,10]
        #     p_gy = data[:,11] + 1j*data[:,12]
        #     p_gb = data[:,13] + 1j*data[:,14]
        #     p_xy = data[:,15] + 1j*data[:,16]
        #     p_xb = data[:,17] + 1j*data[:,18]
        #     p_yb = data[:,19] + 1j*data[:,20]

            # p_xg = data[:,21] + 1j*data[:,22]
            # p_yg = data[:,23] + 1j*data[:,24]
            # p_bg = data[:,25] + 1j*data[:,26]
            # p_yx = data[:,27] + 1j*data[:,28]
            # p_bx = data[:,29] + 1j*data[:,30]
            # p_by = data[:,31] + 1j*data[:,32]
    finally:
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        os.remove(tmp_file)
        os.remove(pulse_file_x)
        os.remove(pulse_file_y)
    # if multitime:
    #     return t,g,x,y,b,p_gx,p_gy,p_gb,p_xy,p_xb,p_yb
    return t,g,x,y,b


def G2(t0=0, tend=600, tau0=0, tauend=600, dt=0.1, *pulses):
    """
    for every t1 in t, propagate to t1, then
    apply sigma = |g><x| from left and sigma^dagger from the right to the density matrix
    propagate from t1 to t1+tau_max
    use results to calculate G2(t1,tau=0,..,tau_max) by applying sigma^dagger*sigma from the left to the density matrix
    and then taking the trace of the dens. matrix
    """
    t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    n_tau = int((tauend-tau0)/dt)
    tau = np.linspace(tau0, tauend, n_tau + 1)

    # special case tau=0:
    # all 4 operators are applied at the same time.
    # G2(t,0) = Tr(sigma^dagger * sigma * sigma * rho(t) * sigma^dagger) = 0, as is sigma*sigma always zero.

    _G2 = np.zeros([len(t),len(tau)])
    for i in tqdm.trange(len(t)):
        _tend = t[i] + tauend
        t,g,x,y,b = biexciton_ace(t0,_tend,*pulses,dt=0.1,ae=5.0,verbose=False,phonons=False, delta_b=4,gamma_e=1/100, gamma_b=2/100, lindblad=True, apply_op_l="|0><1|_4", apply_op_t=t[i])
        # use, that Tr(sigma_x^dagger*sigma_x*rho) = x
        # for the last n_tau elements, not including tau=0, which stays zero
        _G2[i,1:] = x[-n_tau:]

    return _G2
