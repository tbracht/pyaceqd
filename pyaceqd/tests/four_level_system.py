import numpy as np
import matplotlib.pyplot as plt
from pyaceqd.pulses import ChirpedPulse
from pyaceqd.four_level_system.linear import Biexciton, BiexcitonSensors, BiexcitonPhotons, BiexcitonFourSensors
from pyaceqd.tools import ketbra
from pyaceqd.helpers.ace_operators import op_to_matrix
import time

p1 = ChirpedPulse(tau_0=4, e_start=0, t0=4*4, e0=1, polarization="x")
a = Biexciton(dt=0.1, phonons=False, ae=5.0, temperature=4)
out_ops = ["|0><0|_4", "|1><1|_4", "|2><2|_4", "|3><3|_4"]
t,(g,x,y,b) =a.run(0, 40, p1, output_ops=out_ops)

plt.plot(t, g.real, label="Ground State")
plt.plot(t, x.real, label="X State")
plt.plot(t, y.real, label="Y State")
plt.plot(t, b.real, label="Biexciton State")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.savefig("test_biexciton.png")

# b = BiexcitonSensors(dt=0.1, phonons=False, ae=5.0, temperature=4, propagate_Taylor=3)
# out_ops = ["|0><0|_4 otimes Id_2 otimes Id_2", "|1><1|_4 otimes Id_2 otimes Id_2", "|2><2|_4 otimes Id_2 otimes Id_2", "|3><3|_4 otimes Id_2 otimes Id_2"]
# start = time.time()
# t,(g2,x2,y2,b2) =b.run(0, 40, p1, output_ops=out_ops)
# end = time.time()
# print("Time taken for biexciton sensors simulation: {:.2f} seconds".format(end - start))
# plt.plot(t, g2.real, "--", label="Ground State (Sensors)")
# plt.plot(t, x2.real, "--", label="X State (Sensors)")
# plt.plot(t, y2.real, "--", label="Y State (Sensors)")
# plt.plot(t, b2.real, "--", label="Biexciton State (Sensors)")
# plt.xlabel("Time")
# plt.ylabel("Population")
# plt.legend()
# plt.savefig("test_biexciton_sensors.png")

n=2
p1 = ChirpedPulse(tau_0=4, e_start=-2, t0=4*4, e0=5, polarization="x")
c = BiexcitonPhotons(dt=0.1, phonons=True, ae=5.0, temperature=4, n_phot1=n-1, n_phot2=n-1, propagate_Taylor=3, cav_coupl=0.01)
out_ops = [f"Id_4 otimes n_{n} otimes Id_{n}",
           f"Id_4 otimes Id_{n} otimes n_{n}",
           f"|1><1|_4 otimes Id_{n} otimes Id_{n}",
           f"|2><2|_4 otimes Id_{n} otimes Id_{n}",
           f"|3><3|_4 otimes Id_{n} otimes Id_{n}"]
start = time.time()
t,(n1,n2,x2,y2,b2) =c.run(0, 400, p1, output_ops=out_ops)
end = time.time()
print("Time taken for biexciton photon simulation: {:.2f} seconds".format(end - start))
plt.clf()
plt.plot(t, n1.real, label="N1 cavity")
plt.plot(t, n2.real, label="N2 cavity")
plt.plot(t, x2.real, label="X State")
plt.plot(t, y2.real, label="Y State")
plt.plot(t, b2.real, label="Biexciton State")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.savefig("test_biexciton_photons.png")

# d = BiexcitonFourSensors(dt=0.1, sensor_detunings=[-1,1,-1,1], phonons=True, ae=5.0, temperature=4, propagate_Taylor=3)
# out_ops = ["|0><0|_4 otimes Id_2 otimes Id_2 otimes Id_2 otimes Id_2",
#            "|1><1|_4 otimes Id_2 otimes Id_2 otimes Id_2 otimes Id_2",
#            "|2><2|_4 otimes Id_2 otimes Id_2 otimes Id_2 otimes Id_2",
#            "|3><3|_4 otimes Id_2 otimes Id_2 otimes Id_2 otimes Id_2"]
# start = time.time()
# t,(g2,x2,y2,b2) = d.run(0, 40, p1, output_ops=out_ops)
# end = time.time()
# print("Time taken for biexciton four sensors simulation: {:.2f} seconds".format(end - start))
# plt.plot(t, g2.real, "--", label="Ground State (Sensors)")
# plt.plot(t, x2.real, "--", label="X State (Sensors)")
# plt.plot(t, y2.real, "--", label="Y State (Sensors)")
# plt.plot(t, b2.real, "--", label="Biexciton State (Sensors)")
# plt.xlabel("Time")
# plt.ylabel("Population")
# plt.legend()
# plt.savefig("test_biexciton_sensors.png")