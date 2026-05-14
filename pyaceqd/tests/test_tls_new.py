import numpy as np
import matplotlib.pyplot as plt
from pyaceqd.pulses import ChirpedPulse, SmoothRectangle
from pyaceqd.two_level_system.tls import TLS, TLSTwoSensor
from pyaceqd.two_level_system.legacy.legacy_tls import tls_new, tls_dressed_states_new
from pyaceqd.tools import ketbra
from pyaceqd.helpers.ace_operators import op_to_matrix
from pyaceqd.constants import hbar

p1 = ChirpedPulse(tau_0=4, e_start=0, t0=4*4, e0=1, polarization="x")
# p1 = SmoothRectangle(tau=60, t0=40, e0=3.87*hbar, polarization="x", e_start=-6, alpha_onoff=1.)
# print(p1)
output_ops=["|0><0|_2","|1><1|_2"]
# output_ops = [ketbra(0,0, 2), ketbra(1,1, 2)]
# print(output_ops)
mto = {'operator': "|1><0|_2", 'applyFrom': '_left', 'time': 25, 'applyBefore': False}

t,g,x= tls_new(0, 10*4, p1, dt=0.1, phonons=True,multitime_op=[mto], lindblad=True,
                ae=5.0, temperature=4, verbose=True, prepare_only=False, output_ops=output_ops, pt_file="tls_test.pt")
t = t.real
a = TLS(phonons=True, dt=0.1, ae=5.0, temperature=4, lindblad=True, verbose=True)
mto_op = "|1><0|_2"
mto = {'operator': op_to_matrix(mto_op), 'applyFrom': 'left', 'time': 25, 'applyBefore': False}
t2,(g2,x2) = a.run(0, 10*4, p1, output_ops=output_ops, multitime_op=[mto])
# print(result[1].shape)
# reshaped = np.vstack([result[0][np.newaxis, :], result[1].T])
# print(reshaped.shape)
plt.plot(t, g.real, label="Ground State")
plt.plot(t, x.real, label="Excited State")
plt.plot(t2, g2.real, "--", label="Ground State (GeneralSystemACE)")
plt.plot(t2, x2.real, "--", label="Excited State (GeneralSystemACE)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.savefig("test_tls_new.png")

exit()
a.dressed_states(0, 10*4, p1, print_states_t=4*4)

plt.clf()
p1 = ChirpedPulse(tau_0=5, e_start=-2*hbar, t0=4*5, e0=14, polarization="x")
p2 = ChirpedPulse(tau_0=5, e_start=2*hbar, t0=4*5, e0=14, polarization="x")
# p1 = ChirpedPulse(tau_0=2.4, e_start=-8, alpha=0, e0=22.65, polar_x=1.0, t0=2*4, polarization="x")
# p2 = ChirpedPulse(tau_0=3, e_start=-19.163, alpha=0, e0=19.29, polar_x=1.0, t0=2*4, polarization="x")

a = TLS(phonons=False, dt=0.02, ae=5.0, temperature=4, lindblad=False, verbose=True)
t2,(g2,x2) = a.run(0, 10*4, p1,p2, output_ops=output_ops)
plt.plot(t2, p1.get_envelope(t2), label="Pulse 1 Envelope")
plt.plot(t2, g2.real, label="Ground State (GeneralSystemACE)")
plt.plot(t2, x2.real, label="Excited State (GeneralSystemACE)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.savefig("test_tls_new.png")
# exit()
# import time
# a = TLSTwoSensor(dt=0.1, gamma_e=0.01, phonons=True, delta_s1=1, delta_s2=-1, propagate_Taylor=None)
# b = TLSTwoSensor(dt=0.1, gamma_e=0.01, phonons=True, delta_s1=1, delta_s2=-1, propagate_Taylor=3)
# out_ops = ["Id_2 otimes |1><1|_2 otimes Id_2", "Id_2 otimes Id_2 otimes |1><1|_2"]
# multi_time_op = {'operator': "Id_2 otimes |1><0|_2 otimes Id_2", 'applyFrom': 'left', 'time': 25, 'applyBefore': False}
# start = time.time()
# t, (g,x) = a.run(0, 1e3, p1, output_ops=out_ops)
# end1 = time.time()
# print("Time taken for two sensor TLS simulation: {:.2f} seconds".format(end1 - start))
# t, (g2,x2) = b.run(0, 1e3, p1, output_ops=out_ops)
# end2 = time.time()
# print("Time taken for two sensor TLS simulation: {:.2f} seconds".format(end2 - end1))
# plt.clf()
# plt.plot(t, g.real, label="Ground State")
# plt.plot(t, x.real, label="Excited State")
# plt.plot(t, g2.real, "--", label="Ground State (Taylor)")
# plt.plot(t, x2.real, "--", label="Excited State (Taylor)")
# plt.xlabel("Time")
# plt.ylabel("Population")
# plt.legend()
# plt.savefig("test_tls_two_sensor.png")

# result, dm = tls_new(0, 10*4, p1, dt=0.1, phonons=True, ae=5.0, temperature=4, verbose=False, prepare_only=False, output_ops=output_ops, calc_dynmap=True)
# print(dm.shape)
# print(t.shape)

# a.dressed_states(0, 10*4, p1, print_states_t=4*4)

# t,H = tls_new(0, 10*4, p1, dt=0.1, phonons=True, ae=5.0, temperature=4, verbose=False, prepare_only=False, print_H=True, output_ops=output_ops)
# print(H.shape)
# eigvals = np.zeros((len(t),2))
# for i in range(len(t)):
#     eigvals[i], _ = np.linalg.eigh(H[i])

# plt.clf()
# plt.plot(t, eigvals[:,0], label="Dressed State 1")
# plt.plot(t, eigvals[:,1], label="Dressed State 2")
# plt.xlabel("Time")
# plt.ylabel("Energy")
# plt.legend()
# plt.savefig("test_tls_new_dressed_energies.png")
# tls_dressed_states_new(0, 10*4, p1, dt=0.1, phonons=False, ae=5.0, temperature=4, verbose=True, firstonly=False, rf=True)
# result = tls_new(0, 10*4, p1, dt=0.1, phonons=True, ae=5.0, temperature=4, verbose=False, prepare_only=False, print_H=False, output_ops=[])
# print(result.shape)

p1 = ChirpedPulse(tau_0=2.4, e_start=-8, alpha=0, e0=22.65, polar_x=1.0, t0=2*4, polarization="x")
p2 = ChirpedPulse(tau_0=3, e_start=-19.163, alpha=0, e0=19.29, polar_x=1.0, t0=2*4, polarization="x")
# pulses = [p1, p2]
# tls_dressed_states_new(0, 20, *pulses, dt=0.02, phonons=False, ae=5.0, temperature=4, verbose=True, firstonly=True, rf=True)


# a = Pulse(.....)

# class Pulse_object():
#     def __init__(self):
#         self.polarization = "x"

#     def get_total(self, t):
#         pass

#     def get_rf_array(self, t)

# b = a.exportObject()