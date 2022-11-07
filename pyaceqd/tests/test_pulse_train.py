import imp
from pyaceqd.pulses import ChirpedPulse, Pulse, PulseTrain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

p = ChirpedPulse(1,0,t0=4)
pt = PulseTrain(50,4,p,start=0)

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
