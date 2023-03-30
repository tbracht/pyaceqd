import subprocess
import numpy as np
import os
from pyaceqd.tools import export_csv
from pyaceqd.general_system.general_system import system_ace_stream

d0 = 0.25  # meV
d1 = 0.12
d2 = 0.05
mu_b = 5.7882818012e-2   # meV/T
g_ex = -0.65  # in plane electron g factor
g_ez = -0.8  # out of plane electron g factor
g_hx = -0.35  # in plane hole g factor
g_hz = -2.2  # out of plane hole g factor
hbar = 0.6582173  # meV*ps

def energies_linear(d0=0.25, d1=0.12, d2=0.05, delta_B=4, delta_E=0.0):
    E_X = delta_E + (d0 + d1)/2.0 
    E_Y = delta_E + (d0 - d1)/2.0 
    E_S = delta_E - (d0 - d2)/2.0 
    E_F = delta_E - (d0 + d2)/2.0 
    E_B = 2.*delta_E - delta_B
    return E_X, E_Y, E_S, E_F, E_B

def sixls_linear(t_start, t_end, *pulses, dt=0.5, delta_b=4, gamma_e=1/100, gamma_b=None, bx=0, bz=0, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_4","|1><1|_4","|2><2|_4","|3><3|_4"], initial="|0><0|_4"):
    system_prefix = "sixls_linear"
    # |0> = G, |1> = X, |2> = Y, |3> = S, |4> = F, |5> = B
    E_X, E_Y, E_S, E_F, E_B = energies_linear(delta_B=delta_b)
    system_op = ["{}*|1><1|_6 + {}*|2><2|_6 + {}*|3><3|_6 + {}*|4><4|_6 + {}*|5><5|_6".format(E_X,E_Y,E_S,E_F,E_B)]
    # bright-dark coupling depending on Bx
    if bx != 0:
        system_op.append("{}*(|1><3|_6 + |3><1|_6 )".format(-0.5*mu_b*bx*(g_ex+g_hx)))
        system_op.append("{}*(|2><4|_6 + |4><2|_6 )".format(-0.5*mu_b*bx*(g_ex-g_hx)))
    # bright-bright and dark-dark coupling depending on Bz
    if bz != 0.0:
        system_op.append("-i*{}*(|2><1|_6 - |1><2|_6 )".format(-0.5*mu_b*bz*(g_ez-3*g_hz)))
        system_op.append("-i*{}*(|4><3|_6 - |3><4|_6 )".format(+0.5*mu_b*bz*(g_ez+3*g_hz)))
    boson_op = "1*(|1><1|_6+|2><2|_6+|3><3|_6+|4><4|_6) + 2*|5><5|_6"
    lindblad_ops = []
    if lindblad:
        if gamma_b is None:
            gamma_b = 2*gamma_e
        lindblad_ops = [["|0><1|_6",gamma_e],["|0><2|_6",gamma_e],
                        ["|1><5|_6",gamma_b],["|2><5|_6",gamma_b]]
    interaction_ops = [["|1><0|_4+|5><1|_4","x"],["|2><0|_4+|5><2|_4","y"]]
    
    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=20.48, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only)
    return result

def sixls_linear_general(t_start, t_end, *pulses, dt=0.1, delta_b=4, bx=0, bz=0, pulse_file_x=None, pulse_file_y=None, phonons=False, generate_pt=False, t_mem=10, ae=3, temperature=1, verbose=False, d0=0.25, d1=0.12, d2=0.05, temp_dir="/mnt/temp_data/"):
    # print(pulses)
    tmp_file = temp_dir + "sixls_linear.param"
    out_file = temp_dir + "sixls_linear.out"
    duration = np.abs(t_end)+np.abs(t_start)
    pt_file = "sixls_linear_generate_{}ps_{}K_{}nm.pt".format(duration,temperature,ae)
    t,g,x,y,s,f,b = 0,0,0,0,0,0,0
    # E_x, E_y, E_s, E_f, E_b = energies_linear(d0, d1, d2, delta_b, delta_E=0)
    E_x = 0.5*(d0+d1)
    E_xnew = E_x # - E_x  # if a different rotating frame is used, i.e., with respect to E_X
    E_y = 0.5*(d0-d1) # - E_x
    E_s = -0.5*(d0-d2) # - E_x
    E_f = -0.5*(d0+d2) # - E_x
    E_b = -delta_b # - 2*E_x
    # polar_y = np.sqrt(1-polar_x**2)
    # nr_scans = 1
    # mapping: |0>=G, |1>=X, |2>=Y, |3>=S, |4>=F, |5>=B
    if phonons:
        if not os.path.exists(pt_file):
            print("{} not found. Calculating...".format(pt_file))
            generate_pt = True
            verbose = True
    if verbose:
        print("E_x:{:.4f}, E_y:{:.4f}, E_s:{:.4f}, E_f:{:.4f}, E_b:{:.4f}".format(E_xnew,E_y,E_s,E_f,E_b))
    try:
        t = np.arange(1.1*t_start,1.1*t_end,step=0.5*dt)
        if pulse_file_x is None:
            pulse_file_x = temp_dir + "sixls_linear_pulse_x.dat"
            pulse_file_y = temp_dir + "sixls_linear_pulse_y.dat"
            pulse_x = np.zeros_like(t, dtype=complex)
            pulse_y = np.zeros_like(t, dtype=complex)
            for _p in pulses:
                pulse_x = pulse_x + _p.polar_x * _p.get_total(t)
                pulse_y = pulse_y + _p.polar_y * _p.get_total(t)
            export_csv(pulse_file_x, t, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
            export_csv(pulse_file_y, t, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')
        if pulse_file_y is None:
            print("supply pulse_file_x and pulse_file_y")
            exit(0)
        with open(tmp_file,'w') as f:
            f.write("ta    {}\n".format(t_start))
            f.write("te    {}\n".format(t_end))
            f.write("dt    {}\n".format(dt))
            # f.write("nr_scans    {}\n".format(nr_scans))
            if generate_pt:
                f.write("t_mem    {}\n".format(t_mem))
                f.write("threshold 1e-7\n")   # be careful when changing dt and threshold
                # a smaller dt may need a significantly smaller threshold, so there is a trade-off
                f.write("use_Gaussian true\n")
                f.write("Boson_SysOp    { 1*(|1><1|_6+|2><2|_6+|3><3|_6+|4><4|_6) + 2*|5><5|_6}\n")
                f.write("Boson_J_type         QDPhonon\n")
                f.write("Boson_J_a_e    {}\n".format(ae))
                f.write("Boson_temperature    {}\n".format(temperature))
                f.write("Boson_subtract_polaron_shift       true\n")
            else:
                f.write("Nintermediate    10\n")
                f.write("use_symmetric_Trotter true\n")
            if phonons and not generate_pt:
                # process tensor has to be present in current dir!
                f.write("read_PT    {}\n".format(pt_file))
                f.write("Boson_subtract_polaron_shift       true\n")
            f.write("initial    {}\n".format("{|0><0|_6}"))
            # energies: E_x is set to zero, i.e., substract it above
            f.write("add_Hamiltonian  {{ {}*|1><1|_6 + {}*|2><2|_6 + {}*|3><3|_6 + {}*|4><4|_6 + {}*|5><5|_6}}\n".format(E_xnew,E_y,E_s,E_f,E_b))
            # bright-dark coupling depending on Bx
            if bx != 0:
                f.write("add_Hamiltonian {{ {}*(|1><3|_6 + |3><1|_6 ) }}\n".format(-0.5*mu_b*bx*(g_ex+g_hx)))
                f.write("add_Hamiltonian {{ {}*(|2><4|_6 + |4><2|_6 ) }}\n".format(-0.5*mu_b*bx*(g_ex-g_hx)))
            # bright-bright and dark-dark coupling depending on Bz
            if bz != 0.0:
                f.write("add_Hamiltonian {{ -i*{}*(|2><1|_6 - |1><2|_6 ) }}\n".format(-0.5*mu_b*bz*(g_ez-3*g_hz)))
                f.write("add_Hamiltonian {{ -i*{}*(|4><3|_6 - |3><4|_6 ) }}\n".format(+0.5*mu_b*bz*(g_ez+3*g_hz)))
            # pulse: couple G<>X<>B and G<>Y<>B
            # if nr_scans == 1:
            f.write("add_Pulse file {}  {{-{}*(|1><0|_6 + |5><1|_6)}}\n".format(pulse_file_x,0.5*np.pi*hbar))
            f.write("add_Pulse file {}  {{-{}*(|2><0|_6 + |5><2|_6)}}\n".format(pulse_file_y,0.5*np.pi*hbar))
            #else:
            #    for i in range(nr_scans):
            #        f.write("scan{}_add_Pulse file {}  {{-pi*hbar/2*(|1><0|_6 + |5><1|_6)}}\n".format(i,pulse_file_x+"_scan{}".format(i)))
            #        f.write("scan{}_add_Pulse file {}  {{-pi*hbar/2*(|2><0|_6 + |5><2|_6)}}\n".format(i,pulse_file_y+"_scan{}".format(i)))
            # output 
            f.write("add_Output {|0><0|_6}\n")
            f.write("add_Output {|1><1|_6}\n")
            f.write("add_Output {|2><2|_6}\n")
            f.write("add_Output {|3><3|_6}\n")
            f.write("add_Output {|4><4|_6}\n")
            f.write("add_Output {|5><5|_6}\n")
            if generate_pt:
                f.write("write_PT {}\n".format(pt_file))
            f.write("outfile {}\n".format(out_file))
        if not verbose:
            subprocess.check_output(["ACE",tmp_file])
        else:
            subprocess.check_call(["ACE",tmp_file])
            #os.system("ACE /tmp/biex.param")
        # if nr_scans == 1:
        data = np.genfromtxt(out_file)
        # else:
        #     data = np.genfromtxt(out_file+"_scan0")
        t = data[:,0]
        g = data[:,1]
        x = data[:,3]
        y = data[:,5]
        s = data[:,7]
        f = data[:,9]
        b = data[:,11]
    finally:
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        # if nr_scans == 1:
        
        os.remove(tmp_file)
        os.remove(pulse_file_x)
        os.remove(pulse_file_y)
        # else:
        #     for i in range(nr_scans):
        #         os.remove(out_file+"_scan{}".format(i))
        #         os.remove(pulse_file+"_scan{}".format(i))
    return t,g,x,y,s,f,b
