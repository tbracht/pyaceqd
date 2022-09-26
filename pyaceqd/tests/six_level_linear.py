import numpy as np
from pyaceqd.pulses import ChirpedPulse
import matplotlib.pyplot as plt
from pyaceqd.six_level_system.linear import energies_linear,sixls_linear_general

E_X, E_Y, E_S, E_F, E_B = energies_linear(delta_B=4.0, d0=0.25, d1=0.2, d2=0.05)
p1 = ChirpedPulse(tau_0=2.7, e_start=E_X, alpha=40, e0=5.3, polar_x=1.0, t0=0)
p2 = ChirpedPulse(tau_0=2.7, e_start=(E_B-E_X), alpha=40, e0=4.06, polar_x=1.0, t0=2*60)

t,g,x,y,s,f,b = sixls_linear_general(-60,3*60,p1, p2,dt=0.1,ae=5.0,verbose=True,phonons=True,d0=0.25, d1=0.2,d2=0.05,delta_b=4.0,bx=2)
plt.plot(t,g,label='g')
plt.plot(t,x,label='x')
plt.plot(t,y,label='y')
plt.plot(t,s,label='s')
plt.plot(t,f,label='f')
plt.plot(t,b,label='b')
plt.legend()
plt.savefig("sixls_compare_.png")
