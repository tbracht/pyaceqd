import numpy as np
import matplotlib.pyplot as plt
from pyaceqd.pulses import AsymmetricPulse
from pyaceqd.two_level_system.tls import tls

t = np.linspace(-20, 20, 1000)
pulse = AsymmetricPulse(tau1=5, tau2=0.8, e_start=0, t0=0, e0=2)
plt.plot(t, pulse.get_envelope(t))
plt.savefig("pyaceqd/tests/asymmetric_pulse.png")
plt.clf()

t,g,x,p,_ = tls(-20,20,pulse,dt=0.1)
t,g,x = t.real, g.real, x.real
plt.plot(t, g)
plt.plot(t, x)
plt.savefig("pyaceqd/tests/asymmetric_pulse_tls.png")
