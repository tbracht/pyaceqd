from pyaceqd.general_system.general_system import system_ace_stream
from pyaceqd.general_system.general_dressed_states import dressed_states
import pyaceqd.constants as constants

hbar = constants.hbar  # meV*ps

def biexciton(t_start, t_end, *pulses, dt=0.5, delta_xy=0, shift_x=True, coupl_xy=0, delta_b=4, gamma_e=1/100, gamma_b=None, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_4","|1><1|_4","|2><2|_4","|3><3|_4"], initial="|0><0|_4", t_mem=20.48, dressedstates=False, rf=False, rf_file=None, firstonly=False,
               use_infinite=False, calc_dynmap=False):
    system_prefix = "b_linear"
    # |0> = G, |1> = X, |2> = Y, |3> = B
    if shift_x:
        system_op = ["{}*|3><3|_4".format(-delta_b),"{}*|1><1|_4".format(-delta_xy/2),"{}*|2><2|_4".format(delta_xy/2)]
    else:
        system_op = ["{}*|3><3|_4".format(-delta_b),"{}*|2><2|_4".format(delta_xy)]
    boson_op = "1*(|1><1|_4 + |2><2|_4) + 2*|3><3|_4"
    lindblad_ops = []
    if lindblad:
        if gamma_b is None:
            gamma_b = gamma_e
        lindblad_ops = [["|0><1|_4",gamma_e],["|0><2|_4",gamma_e],
                        ["|1><3|_4",gamma_b],["|2><3|_4",gamma_b]]
    interaction_ops = [["|1><0|_4+|3><1|_4","x"],["|2><0|_4+|3><2|_4","y"]]

    if coupl_xy != 0:
        system_op.append("{}*|1><2|_4".format(coupl_xy))
        system_op.append("{}*|2><1|_4".format(coupl_xy))
    
    rf_op = None
    if rf:
        # 2*|3><3|_4 because B contains 2 excitons
        rf_op = "|1><1|_4 + |2><2|_4 + 2*|3><3|_4" 

    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only,
                  dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file, firstonly=firstonly, use_infinite=use_infinite, calc_dynmap=calc_dynmap)
    return result

def biexciton_dressed_states(t_start, t_end, *pulses, plot=True, t_lim=None, e_lim=None, colors=["#0000FF", "#00CC33", "#F9A627", "#FF0000"], filename="biexciton_dressed", firstonly=False, visible_states=None, return_eigenvectors=False, **options):
    dim = 4
    return dressed_states(biexciton, dim, t_start, t_end, *pulses, filename=filename, t_lim=t_lim, e_lim=e_lim, plot=plot, firstonly=firstonly, colors=colors, visible_states=visible_states, return_eigenvectors=return_eigenvectors, **options)

def biexciton_photons(t_start, t_end, *pulses, dt=0.5, delta_xy=0, delta_b=4, gamma_e=1/100, cav_coupl=0.06, cav_loss=0.12/hbar, delta_cx=-2, gamma_b=None, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=None, initial=None,
                t_mem=20.48, dressedstates=False, rf=False, rf_file=None, firstonly=False, n_phot1=1, n_phot2=1):
    n1 = n_phot1 + 1
    n2 = n_phot2 + 1
    # initial state and output operators with correct number of photons
    if initial is None:
        initial = "|0><0|_4 otimes |0><0|_{} otimes |0><0|_{}".format(n1,n2)
    if output_ops is None:
        output_ops = ["|0><0|_4 otimes Id_{} otimes Id_{}".format(n1,n2),
                      "|1><1|_4 otimes Id_{} otimes Id_{}".format(n1,n2),
                      "|2><2|_4 otimes Id_{} otimes Id_{}".format(n1,n2),
                      "|3><3|_4 otimes Id_{} otimes Id_{}".format(n1,n2)]
    system_prefix = "b_linear_cavity"
    # |0> = G, |1> = X, |2> = Y, |3> = B
    system_op = ["-{}*|3><3|_4 otimes Id_{} otimes Id_{}".format(delta_b,n1,n2),
                 "-{}*|1><1|_4 otimes Id_{} otimes Id_{}".format(delta_xy/2,n1,n2),
                 "{}*|2><2|_4 otimes Id_{} otimes Id_{}".format(delta_xy/2,n1,n2)]
    boson_op = "|1><1|_4 otimes Id_{} otimes Id_{} + |2><2|_4 otimes Id_{} otimes Id_{} + 2*|3><3|_4 otimes Id_{} otimes Id_{}".format(n1,n2,n1,n2,n1,n2)
    lindblad_ops = []
    # QD decay outside of the cavity
    if lindblad:
        if gamma_b is None:
            gamma_b = gamma_e
        lindblad_ops = [["|0><1|_4 otimes Id_{} otimes Id_{}".format(n1,n2),gamma_e],
                        ["|0><2|_4 otimes Id_{} otimes Id_{}".format(n1,n2),gamma_e],
                        ["|1><3|_4 otimes Id_{} otimes Id_{}".format(n1,n2),gamma_b],
                        ["|2><3|_4 otimes Id_{} otimes Id_{}".format(n1,n2),gamma_b]]
    # interaction with laser
    interaction_ops = [["|1><0|_4 otimes Id_{} otimes Id_{} +|3><1|_4 otimes Id_{} otimes Id_{} ".format(n1,n2,n1,n2),"x"],
                       ["|2><0|_4 otimes Id_{} otimes Id_{} +|3><2|_4 otimes Id_{} otimes Id_{} ".format(n1,n2,n1,n2),"y"]]
    # cavity decay
    lindblad_ops.append(["Id_4 otimes b_{} otimes Id_{}".format(n1,n2),cav_loss])
    lindblad_ops.append(["Id_4 otimes Id_{} otimes b_{}".format(n1,n2),cav_loss])
    # cavity detuning
    system_op.append(" {} * (Id_4 otimes n_{} otimes Id_{})".format(delta_cx,n1,n2))
    system_op.append(" {} * (Id_4 otimes Id_{} otimes n_{})".format(delta_cx,n1,n2))
    # cavity-qd coupling
    # X-cavity
    system_op.append("{} * (|1><0|_4 otimes b_{} otimes Id_{} + |0><1|_4 otimes bdagger_{} otimes Id_{})".format(cav_coupl,n1,n2,n1,n2))
    system_op.append("{} * (|3><1|_4 otimes b_{} otimes Id_{} + |1><3|_4 otimes bdagger_{} otimes Id_{})".format(cav_coupl,n1,n2,n1,n2))
    # Y-cavity
    system_op.append("{} * (|2><0|_4 otimes Id_{} otimes b_{} + |0><2|_4 otimes Id_{} otimes bdagger_{})".format(cav_coupl,n1,n2,n1,n2))
    system_op.append("{} * (|3><2|_4 otimes Id_{} otimes b_{} + |2><3|_4 otimes Id_{} otimes bdagger_{})".format(cav_coupl,n1,n2,n1,n2))
    
    rf_op = None
    if rf:
        rf_op = "|1><1|_4 otimes Id_{} otimes Id_{}".format(n1,n2)
        rf_op = rf_op + " + Id_4 otimes n_{} otimes Id_{}".format(n1,n2)
        rf_op = rf_op + " + Id_4 otimes Id_{} otimes n_{}".format(n1,n2)
        if pulse_file_x is not None and rf_file is None:
            print("Error: pulse file is given, but no file for rotating frame")
            return 0

    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only,
                  dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file, firstonly=firstonly)
    return result

def biexciton_photons_dressed_states(t_start, t_end, *pulses, plot=True, t_lim=None, e_lim=None, filename="biexciton_photons_dressed", firstonly=False, visible_states=None, **options):
    n1 = options["n_phot1"] + 1 
    n2 = options["n_phot2"] + 1
    dim = [4,n1,n2]
    return dressed_states(biexciton_photons, dim, t_start, t_end, *pulses, filename=filename, plot=plot, t_lim=t_lim, e_lim=e_lim, firstonly=firstonly, colors=None, visible_states=visible_states, **options)

def biexciton_photons_extended(t_start, t_end, *pulses, dt=0.5, delta_xy=0, delta_b=4, gamma_e=1/100, cav_coupl=0.06, cav_loss=0.12/hbar, delta_cx=-2, gamma_b=None, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_18 + |1><1|_18 + |2><2|_18 + |3><3|_18 + |4><4|_18 + |5><5|_18","|6><6|_18 + |7><7|_18 + |8><8|_18 + |9><9|_18","|10><10|_18 + |11><11|_18 + |12><12|_18 + |13><13|_18","|14><14|_18 + |15><15|_18 + |16><16|_18 + |17><17|_18"], initial="|0><0|_18",
               t_mem=20.48, dressedstates=False, rf=False, rf_file=None, firstonly=False):
    system_prefix = "b_linear_cavity_extended"
    # state mapping: |G,0,0> 0, |G,1,0> 1, |G,0,1> 2, |G,1,1> 3, |G,2,0> 4, |G,0,2> 5, |X,0,0> 6, |X,1,0> 7, |X,0,1> 8, |X,1,1> 9, |Y,0,0> 10, |Y,1,0> 11, |Y,0,1> 12, |Y,1,1> 13, |B,0,0> 14, |B,1,0> 15, |B,0,1> 16, |B,1,1> 17
    # this system accounts for two excitations in total, i.e., G,2,0 and G,0,2 are also considered.
    # it shows that this is enough in most cases.
    d_C = delta_cx
    d_0 = delta_xy
    d_B = delta_b
    system_op = ["{}*|1><1|_18".format(d_C), "{}*|2><2|_18".format(d_C), "{}*|3><3|_18".format(2*d_C), "{}*|4><4|_18".format(2*d_C), "{}*|5><5|_18".format(2*d_C), "{}*|6><6|_18".format(-d_0/2), "{}*|7><7|_18".format(-d_0/2 + d_C),
                 "{}*|8><8|_18".format(-d_0/2 + d_C), "{}*|9><9|_18".format(-d_0/2 + 2*d_C), "{}*|10><10|_18".format(d_0/2), "{}*|11><11|_18".format(d_0/2 + d_C), "{}*|12><12|_18".format(d_0/2 + d_C), "{}*|13><13|_18".format(d_0/2 + 2*d_C),
                 "{}*|14><14|_18".format(-d_B), "{}*|15><15|_18".format(-d_B + d_C), "{}*|16><16|_18".format(-d_B + d_C), "{}*|17><17|_18".format(-d_B + 2*d_C)]
    boson_op = "|6><6|_18 + |7><7|_18 + |8><8|_18 + |9><9|_18 + |10><10|_18 + |11><11|_18 + |12><12|_18 + |13><13|_18 + 2 * ( |14><14|_18 + |15><15|_18 + |16><16|_18 + |17><17|_18)"
    lindblad_ops = []
    # QD decay outside of the cavity
    if lindblad:
        if gamma_b is None:
            gamma_b = gamma_e
        lindblad_ops = [["|0><6|_18 + |1><7|_18 + |2><8|_18 + |3><9|_18",gamma_e],["|0><10|_18 + |1><11|_18 + |2><12|_18 + |3><13|_18",gamma_e],
                        ["|6><14|_18 + |7><15|_18 + |8><16|_18 + |9><17|_18",gamma_b],["|10><14|_18 + |11><15|_18 + |12><16|_18 + |13><17|_18",gamma_b]]
    # interaction with laser
    interaction_ops = [["|6><0|_18 + |7><1|_18 + |8><2|_18 + |9><3|_18 + |14><6|_18 + |15><7|_18 + |16><8|_18 + |17><9|_18","x"],
                       [" |10><0|_18 + |11><1|_18 + |12><2|_18 + |13><3|_18 + |14><10|_18 + |15><11|_18 + |16><12|_18 + |17><13|_18","y"]]
    # cavity decay
    # cavity loss x
    lindblad_ops.append(["|0><1|_18 + sqrt(2)*|1><4|_18 + |2><3|_18 + |6><7|_18 + |8><9|_18 + |10><11|_18 + |12><13|_18 + |14><15|_18 + |16><17|_18",cav_loss])
    # cavity loss y
    lindblad_ops.append(["|0><2|_18 + |1><3|_18 + sqrt(2)*|2><5|_18 + |6><8|_18 + |7><9|_18 + |10><12|_18 + |11><13|_18 + |14><16|_18 + |15><17|_18",cav_loss])
    # cavity-qd coupling
    # X-cavity
    system_op.append("{} * ( |1><6|_18 + |3><8|_18 + sqrt(2)*|4><7|_18 + |6><1|_18 + sqrt(2)*|7><4|_18 + |7><14|_18 + |8><3|_18 + |9><16|_18 + |14><7|_18 + |16><9|_18)".format(cav_coupl))
    # Y-cavity
    system_op.append("{} * ( |2><10|_18 + |3><11|_18 + sqrt(2)*|5><12|_18 + |10><2|_18 + |11><3|_18 + sqrt(2)*|12><5|_18 + |12><14|_18 + |13><15|_18 + |14><12|_18 + |15><13|_18)".format(cav_coupl))
    
    rf_op = None
    if rf:
        # the factors correpond to the number of excitations (QD&photons) in the respective state
        rf_op = "|1><1|_18 + |2><2|_18 + 2*|3><3|_18 + 2*|4><4|_18 + 2*|5><5|_18 + |6><6|_18 + 2*|7><7|_18 + 2*|8><8|_18 + 3*|9><9|_18 + |10><10|_18 + 2*|11><11|_18 + 2*|12><12|_18 + 3*|13><13|_18 + 2*|14><14|_18 + 3*|15><15|_18 + 3*|16><16|_18 + 4*|17><17|_18"

    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only, 
                  dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file, firstonly=firstonly)
    return result

def biexciton_photons_extended_dressed_states(t_start, t_end, *pulses, plot=True, t_lim=None, e_lim=None, filename="biexciton_photons_extended_dressed", firstonly=False, visible_states=None, **options):
    dim = 18
    return dressed_states(biexciton_photons_extended, dim, t_start, t_end, *pulses, filename=filename, t_lim=t_lim, e_lim=e_lim, plot=plot, firstonly=firstonly, colors=None, visible_states=visible_states, **options)

def biexciton_sensors(t_start, t_end, *pulses, dt=0.1, delta_xy=0, shift_x=True, delta_s1=0, delta_s2=0, epsilon=0.0001, linewidth1=0.01, linewidth2=None, delta_b=4, gamma_e=1/100, gamma_b=None, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_4 otimes Id_2 otimes Id_2","|1><1|_4 otimes Id_2 otimes Id_2","|2><2|_4 otimes Id_2 otimes Id_2","|3><3|_4 otimes Id_2 otimes Id_2"], initial="|0><0|_4 otimes |0><0|_2 otimes |0><0|_2", t_mem=12.8, dressedstates=False, rf=False, rf_file=None, firstonly=False):
    system_prefix = "b_linear_sensor"
    # |0> = G, |1> = X, |2> = Y, |3> = B
    if shift_x:
        system_op = ["{}*|3><3|_4 otimes Id_2 otimes Id_2".format(-delta_b),"{}*|1><1|_4 otimes Id_2 otimes Id_2".format(-delta_xy/2),"{}*|2><2|_4 otimes Id_2 otimes Id_2".format(delta_xy/2)]
    else:
        system_op = ["{}*|3><3|_4 otimes Id_2 otimes Id_2".format(-delta_b),"{}*|2><2|_4 otimes Id_2 otimes Id_2".format(delta_xy)]
    boson_op = "1*(|1><1|_4 otimes Id_2 otimes Id_2 + |2><2|_4 otimes Id_2 otimes Id_2) + 2*(|3><3|_4 otimes Id_2 otimes Id_2) "
    lindblad_ops = []
    if lindblad:
        if gamma_b is None:
            gamma_b = gamma_e
        lindblad_ops = [["|0><1|_4 otimes Id_2 otimes Id_2",gamma_e],["|0><2|_4 otimes Id_2 otimes Id_2",gamma_e],
                        ["|1><3|_4 otimes Id_2 otimes Id_2",gamma_b],["|2><3|_4 otimes Id_2 otimes Id_2",gamma_b]]
    interaction_ops = [["|1><0|_4 otimes Id_2 otimes Id_2 +|3><1|_4 otimes Id_2 otimes Id_2","x"],
                       ["|2><0|_4 otimes Id_2 otimes Id_2 +|3><2|_4 otimes Id_2 otimes Id_2","y"]]
    
    rf_op = None
    if rf:
        # 2*|3><3|_4 because B contains 2 excitons
        rf_op = "|1><1|_4 otimes Id_2 otimes Id_2 + |2><2|_4 otimes Id_2 otimes Id_2 + 2*(|3><3|_4 otimes Id_2 otimes Id_2)" 

    # sensor hamiltonian
    system_op.append("{} * (Id_4 otimes |1><1|_2 otimes Id_2)".format(delta_s1))
    system_op.append("{} * (Id_4 otimes Id_2 otimes |1><1|_2)".format(delta_s2))
    # sensor coupling
    # coupling to G-Y
    system_op.append("{} * (|2><0|_4 otimes |0><1|_2 otimes Id_2 + |0><2|_4 otimes |1><0|_2 otimes Id_2)".format(epsilon))
    # coupling to Y-B
    system_op.append("{} * (|3><2|_4 otimes |0><1|_2 otimes Id_2 + |2><3|_4 otimes |1><0|_2 otimes Id_2)".format(epsilon))
    # coupling to G-X
    system_op.append("{} * (|1><0|_4 otimes Id_2 otimes |0><1|_2 + |0><1|_4 otimes Id_2 otimes |1><0|_2)".format(epsilon))    
    # coupling to X-B
    system_op.append("{} * (|3><1|_4 otimes Id_2 otimes |0><1|_2 + |1><3|_4 otimes Id_2 otimes |1><0|_2)".format(epsilon))

    # sensor loss
    if linewidth2 is None:
        linewidth2 = linewidth1
    lindblad_ops.append(["Id_4 otimes |0><1|_2 otimes Id_2", linewidth1])
    lindblad_ops.append(["Id_4 otimes Id_2 otimes |0><1|_2", linewidth2])

    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only,
                  dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file, firstonly=firstonly)
    return result