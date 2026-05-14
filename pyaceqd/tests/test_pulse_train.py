from pyaceqd.pulses import ChirpedPulse, Pulse, PulseTrain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

p = ChirpedPulse(1,0,t0=4)
pt = PulseTrain(50,4,p,t_shift=0)

t = np.linspace(0,180,300)
y = pt.get_total(t)

plt.plot(t,y.real)
plt.xlabel("time in ps")
left, bottom, width, height = (25+4, 0, 49, 1.3)
rect1=ptch.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="red",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect1)
left, bottom, width, height = (75+4, 0, 49, 1.3)
rect2=ptch.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="blue",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect2)
plt.savefig("train.png")
plt.clf()

p3 = ChirpedPulse(3,1,t0=15, repeat_tb=50,phase_option="random")#,phase=0.5*np.pi)
plt.plot(t,p3.get_total(t).real)
plt.xlabel("time in ps")
plt.savefig("train_repeated.png")  
plt.clf()

p3 = ChirpedPulse(3,1,t0=15, repeat_tb=50,phase_option="same")#,phase=0.5*np.pi)
plt.plot(t,p3.get_total(t).real)
plt.plot(t,p3.get_total(t-50).real, dashes=[5,5])
plt.xlabel("time in ps")
plt.xlim(0,50)
plt.savefig("train_repeated_same_phase.png")

p2 = ChirpedPulse(2,0,alpha=40,t0=75)
plt.plot(t,p2.get_total(t).real)
plt.xlabel("time in ps")
plt.savefig("single_chirped.png")
plt.clf()

t = np.linspace(0,400,1200)

p2 = ChirpedPulse(2,0,alpha=40,t0=75, repeat_tb=150, phase_option="same")
plt.plot(t,p2.get_total(t).real)
plt.xlabel("time in ps")
plt.savefig("train_chirped.png")
plt.clf()