import numpy as np
from pyaceqd.pulses import ChirpedPulse
import matplotlib.pyplot as plt
from pyaceqd.six_level_system.legacy.linear import energies_linear,sixls_linear,sixls_linear_dressed_states
from pyaceqd.six_level_system.linear import SixLevelLinearSystem
from pyaceqd.helpers.ace_operators import ketbra

E_X, E_Y, E_S, E_F, E_B = energies_linear(delta_B=4.0, d0=0.25, d1=0.2, d2=0.05)
p1 = ChirpedPulse(tau_0=2.7, e_start=E_X, alpha=40, e0=5.3, polar_x=1.0, t0=0, polarization="x")
p2 = ChirpedPulse(tau_0=2.7, e_start=(E_B-E_X), alpha=40, e0=4.06, polar_x=1.0, t0=2*60, polarization="x")

a = SixLevelLinearSystem(dt=0.1, phonons=False, ae=5.0, temperature=4, lindblad=False, verbose=True, delta_b=4.0, d0=0.25, d1=0.2, d2=0.05, bx=2.0)
output_ops = ["|0><0|_6","|1><1|_6","|2><2|_6","|3><3|_6","|4><4|_6","|5><5|_6"]
output_ops = [ketbra(i,i,6) for i in range(6)]
t2, (g2, x2, y2, s2, f2, b2) = a.run(-60,3*60,p1, p2, output_ops=output_ops, prepare_only=False)
t,g,x,y,s,f,b = sixls_linear(-60,3*60,p1, p2,dt=0.1,ae=5.0,verbose=True,phonons=True,delta_b=4.0,bx=2,use_infinite=True, pt_file="sixls_linear_5.0nm_4k_th8_dt0.1.pt")
plt.plot(t.real,g.real,label='g')
plt.plot(t.real,x.real,label='x')
plt.plot(t.real,y.real,label='y')
plt.plot(t.real,s.real,label='s')
plt.plot(t.real,f.real,label='f')
plt.plot(t.real,b.real,label='b')
plt.plot(t2,g2.real,"--",label='g (GeneralSystemACE)')
plt.plot(t2,x2.real,"--",label='x (GeneralSystemACE)')
plt.plot(t2,y2.real,"--",label='y (GeneralSystemACE)')
plt.plot(t2,s2.real,"--",label='s (GeneralSystemACE)')
plt.plot(t2,f2.real,"--",label='f (GeneralSystemACE)')
plt.plot(t2,b2.real,"--",label='b (GeneralSystemACE)')
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.savefig("sixls_compare_.png")

sixls_linear_dressed_states(-60,3*60,p1, p2,dt=0.1,ae=5.0,verbose=True,phonons=True,delta_b=4.0,bx=2,use_infinite=True, rf=True, firstonly=True)
a.dressed_states(-60,3*60,p1, p2, rf=True, no_pulse=False, firstonly=True, e_lim=(-1,1), t_lim=(-50,50), fix_order=True)
# t,res = a.run(-60,3*60,p1, p2, output_ops=[], prepare_only=False)
# print(res.shape)
