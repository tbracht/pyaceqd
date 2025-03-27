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
import pyaceqd.constants as constants

hbar = constants.hbar  # meV*ps

def tls(t_start, t_end, *pulses, dt=0.1, gamma_e=1/100, phonons=False, t_mem=6.4, ae=3.0, temperature=1,verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
         multitime_op=None, pulse_file=None, pulse_file_x=None, prepare_only=False, output_ops=["|0><0|_2","|1><1|_2","|0><1|_2","|1><0|_2"], phonon_factor=1.0, LO_params=None, dressedstates=False, rf=False, rf_file=None, firstonly=False,\
             J_to_file=None, J_file=None, factor_ah=None, use_infinite=False, threshold=10, **options):
    """
    t_start: time at which the simulation starts in ps
    t_end: time at which the simulation ends in ps
    *pulses: pulse objects that are applied to the system
    dt: time step of the simulation. should divide t_end-t_start
    gamma_e: decay rate of the excited state in 1/ps
    phonons: if True, the system is coupled to a phonon bath
    t_mem: memory time of the phonon bath in ps. optimally it is (2^n)*dt and sufficiently large
    ae: electron confinementh length in nm. 3 o4 5 nm is a good value for a quantum dot. the smaller, the stronger the e-ph coupling
    temperature: temperature of the phonon bath in K
    verbose: if True, prints various information during the simulation
    lindblad: if True, the rate of the decay is included in the simulation
    temp_dir: directory where the temporary files are stored (param files for ACE and the output files)
    pt_file: if not None, the calculation uses this process tensor file for phonons. if None, it generates a new one
    suffix: suffix for the temporary files, useful for parallel calculations
    multitime_op: if not None, the operator is applied at multiple times. for example 
                  {"operator": "|0><1|_2","applyFrom": "_left", "applyBefore": "false", "time": 0} 
                  applies |0><1|_2 at t=0 from the left AFTER the system has been propagated to t=0
    pulse_file: if not None, the pulse is read from this file in the format of ACE: time, REAL, IMAG
    pulse_file_x: same as pulse_file, compatibility
    prepare_only: if True, the function only prepares the ACE input files and does not run the simulation
    output_ops: list of expectation values that are calculated
    phonon_factor: factor that scales the phonon coupling
    LO_params: can add LO phonons as a single mode. mostly not usable though
    dressedstates: if True, the dressed states are calculated. should never be used standalone, but only by the dressed state functions
    rf: if True, a rotating frame with respect to the laser is used
    rf_file: if not None, the rotating frame is read from this file
    firstonly: if True, only the first pulse is applied for the dressed state calculation
    J_to_file: saves phonon spectral density to file
    J_file: reads phonon spectral density from file
    factor_ah: factor that scales the hole localization length
    **options: not passed, but can catch any additional keyword arguments
    """
    system_prefix = "tls"
    system_op = None
    boson_op = "{:.3f}*|1><1|_2".format(phonon_factor)
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
    if pulse_file is None and pulse_file_x is not None:
        pulse_file = pulse_file_x

    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, pulse_file_x=pulse_file, system_prefix=system_prefix, threshold=str(int(threshold)), threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only, LO_params=LO_params, dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file,
                  firstonly=firstonly, J_to_file=J_to_file, J_file=J_file, factor_ah=factor_ah, use_infinite=use_infinite)
    return result

def tls_dressed_states(t_start, t_end, *pulses, plot=True, t_lim=None, e_lim=None, filename="tls_dressed", firstonly=False, colors=["#0000FF", "#FF0000"], visible_states=None, return_eigenvectors=False, **options):
    """
    e_lim limits the energy range of the plot
    visible_states is a list of states that are plotted. if None, all states are plotted
    can also pass RGBA colors, to achieve the same effect as visible_states: for example ["#0000FF00", "#FF0000FF"]
    """
    # dim = 2 for TLS
    dim = 2
    return dressed_states(tls, dim, t_start, t_end, *pulses, filename=filename, plot=plot, t_lim=t_lim, e_lim=e_lim, firstonly=firstonly, colors=colors, visible_states=visible_states, return_eigenvectors=return_eigenvectors, **options)

def tls_two_sensor(t_start, t_end, *pulses, dt=0.1, gamma_e=1/100, phonons=False, t_mem=10, ae=3.0, delta_s1=0, delta_s2=0, epsilon=0.0001, linewidth1=0.01, linewidth2=None, temperature=1,verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
         multitime_op=None, pulse_file=None, prepare_only=False, output_ops=["|0><0|_2 otimes Id_2 otimes Id_2","|1><1|_2 otimes Id_2 otimes Id_2"], initial=None, dressedstates=False, rf=False, rf_file=None, firstonly=False, use_infinite=False):
    system_prefix = "tls_two_sensor"
    system_op = []
    boson_op = "|1><1|_2 otimes Id_2 otimes Id_2"
    if initial is None:
        initial = "|0><0|_2 otimes |0><0|_2 otimes |0><0|_2"
    lindblad_ops = []
    if lindblad:
        lindblad_ops = [["|0><1|_2 otimes Id_2 otimes Id_2",gamma_e]]
    # note that the TLS uses x-polar
    interaction_ops = [["|1><0|_2 otimes Id_2 otimes Id_2","x"]]
    # rotating frame of pulse
    rf_op = None
    if rf:
        rf_op = "|1><1|_2 otimes Id_2 otimes Id_2"

    # sensor hamiltonian
    system_op.append("{} * (Id_2 otimes |1><1|_2 otimes Id_2)".format(delta_s1))
    system_op.append("{} * (Id_2 otimes Id_2 otimes |1><1|_2)".format(delta_s2))
    # sensor coupling
    system_op.append("{} * (|1><0|_2 otimes |0><1|_2 otimes Id_2 + |0><1|_2 otimes |1><0|_2 otimes Id_2)".format(epsilon))
    system_op.append("{} * (|1><0|_2 otimes Id_2 otimes |0><1|_2 + |0><1|_2 otimes Id_2 otimes |1><0|_2)".format(epsilon))
    # sensor loss
    if linewidth2 is None:
        linewidth2 = linewidth1
    lindblad_ops.append(["Id_2 otimes |0><1|_2 otimes Id_2", linewidth1])
    lindblad_ops.append(["Id_2 otimes Id_2 otimes |0><1|_2", linewidth2])

    # multitime: for ex. ["|1><0|_2",0,"left"] applies |1><0|_2 at t=0 from the left
    # invoke_dict = {"dt": dt, "phonons": phonons, "generate_pt": generate_pt, "t_mem": t_mem, "ae": ae, "temperature": temperature}
    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, pulse_file_x=pulse_file, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only, dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file,
                  firstonly=firstonly, use_infinite=use_infinite)
    return result

def tls_photons(t_start, t_end, *pulses, dt=0.1, gamma_e=1/100, cav_coupl1=0.06, cav_loss1=0.12/hbar, delta_cx1=-2, cav_coupl2=None, cav_loss2=None, delta_cx2=-2, phonons=False, t_mem=10, ae=5.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
         multitime_op=None, n_phot1=2, n_phot2=2, laser_cav_coupl=None, pulse_file=None, prepare_only=False, output_ops=None, dressedstates=False, rf=False, rf_file=None, firstonly=False, initial=None):
    n1 = n_phot1 + 1
    n2 = n_phot2 + 1
    system_prefix = "tls_cavity"
    system_op = []
    boson_op = "|1><1|_2 otimes Id_{} otimes Id_{}".format(n1,n2)
    if initial is None:
        initial = "|0><0|_2 otimes |0><0|_{} otimes |0><0|_{}".format(n1,n2)
    if output_ops is None:
        output_ops=["|0><0|_2 otimes Id_{} otimes Id_{}".format(n1,n2),"|1><1|_2 otimes Id_{} otimes Id_{}".format(n1,n2)]
    lindblad_ops = []
    if lindblad:
        lindblad_ops = [["|0><1|_2 otimes Id_{} otimes Id_{}".format(n1,n2), gamma_e]]
    # note that the TLS uses x-polar
    interaction_ops = [["|1><0|_2 otimes Id_{} otimes Id_{}".format(n1,n2),"x"]]
    if laser_cav_coupl is not None:
        interaction_ops.append(["{}*(Id_2 otimes bdagger_{} otimes Id_{})".format(laser_cav_coupl,n1,n2),"x"])
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

def tls_photons_dressed_states(t_start, t_end, *pulses, plot=True, t_lim=None, e_lim=None, filename="tls_photons_dressed", firstonly=False, visible_states=None,print_states=None, **options):
    # dim = 2 for TLS
    n1 = options["n_phot1"] + 1
    n2 = options["n_phot2"] + 1
    dim = [2,n1,n2]
    return dressed_states(tls_photons, dim, t_start, t_end, *pulses, filename=filename, plot=plot, t_lim=t_lim, e_lim=e_lim, firstonly=firstonly, colors=None, visible_states=visible_states,print_states=print_states, **options)
