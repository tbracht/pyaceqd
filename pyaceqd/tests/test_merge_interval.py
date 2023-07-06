from pyaceqd.tools import _merge_intervals, construct_t
from pyaceqd.pulses import ChirpedPulse


interv1 = [[-1,1],[1,2],[5,8]]
print(_merge_intervals(interv1))
print("expect:" + str(interv1))

interv1 = [[-1,2],[1,3],[4,8]]
print(_merge_intervals(interv1))
print("expect:" + str([[-1,3],[4,8]]))

interv1 = [[-1,3],[1,3],[4,8]]
print(_merge_intervals(interv1))
print("expect:" + str([[-1,3],[4,8]]))

interv1 = [[-1,1],[1,3],[4,8]]
print(_merge_intervals(interv1))
print("expect:" + str([[-1,3],[4,8]]))

interv1 = [[-1,7],[1,3],[4,8]]
print(_merge_intervals(interv1))
print("expect:" + str([[-1,8]]))

p1 = ChirpedPulse(tau_0=1,e_start=0, t0=1*4)
p2 = ChirpedPulse(tau_0=1,e_start=0, t0=20)

print(construct_t(0,80,0.1,1.0,p1,p2,simple_exp=False))

p1 = ChirpedPulse(tau_0=1,e_start=0, t0=1*4)
p2 = ChirpedPulse(tau_0=1,e_start=0, t0=5)

print(construct_t(0,80,0.1,1.0,p1,p2,simple_exp=True))
