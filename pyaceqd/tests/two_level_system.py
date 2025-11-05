import numpy as np
from pyaceqd.pulses import ChirpedPulse, PulseTrain
import matplotlib.pyplot as plt
from pyaceqd.two_level_system.tls import tls, tls_dressed_states
from pyaceqd.two_time.G1 import G1_twols
# from photonprops.two_level_system.tls import twolevel_system
from pyaceqd.constants import hbar
import cProfile
import pstats

p1 = ChirpedPulse(tau_0=2.4, e_start=-8, alpha=0, e0=22.65, polar_x=1.0, t0=2*4)
p2 = ChirpedPulse(tau_0=3, e_start=-19.163, alpha=0, e0=19.29, polar_x=1.0, t0=2*4)

t2,g2,x2,pgx2,pxg2 = tls(0,40,p1,dt=0.02,phonons=True,ae=5.0,temperature=4,prepare_only=True,threshold=8,use_infinite=True)

# # tls_(0,20,p1,temp_dir="/home/t_brac02/Dokumente/repos/pyaceqd/", phonons=True, prepare_only=True)
# t,g,x,pgx,pxg = tls_(0,40,p1,p2,dt=0.02,factor_ah=1.15,phonons=True,J_to_file=True,ae=7.0,temperature=4,prepare_only=False, temp_dir='pyaceqd/tests/')#,multitime_op={"operator": "|1><0|_2","time": 8})
# t2,g2,x2,pgx2,pxg2 = tls_(0,40,p1,p2,dt=0.02,phonons=True,ae=3.0,temperature=4,prepare_only=False, temp_dir='pyaceqd/tests/')#,multitime_op={"operator": "|1><0|_2","time": 8})

# compare both results 
# plt.clf()
# plt.plot(t.real,np.real(x),'g-',label="x")
# plt.plot(t2.real,np.real(x2),dashes=[6, 2],label="x2")
# plt.legend()
# plt.xlim(0,15)
# plt.savefig("tls.png")
p3 = ChirpedPulse(tau_0=5, e_start=0, alpha=0, e0=15, polar_x=1.0, t0=5*4)
t,g,x,pgx,pxg = tls(0,40,p3,dt=0.02,phonons=True,ae=5.0,temperature=4,prepare_only=False,threshold=8)#,multitime_op={"operator": "|1><0|_2","time": 8})
t2,g2,x2,pgx2,pxg2 = tls(0,40,p3,dt=0.02,phonons=True,ae=5.0,temperature=4,prepare_only=False,threshold=8,use_infinite=True)#,multitime_op={"operator": "|1><0|_2","time": 8})

tls_dressed_states(0,40,p3,dt=0.02,phonons=False,rf=True)
# t2,g2,x2,pgx2,pxg2 =tls_(-0.01,600,p1,dt=0.2,ninterm=10,stream=False,phonons=True,temperature=4,prepare_only=False)#,multitime_op={"operator": "|1><0|_2","time": 8}) # tls_ace(0,20,p1,dt=0.5,phonons=True,temperature=4)# apply_op="|1><0|_2", apply_op_t=8,apply='')

plt.plot(t.real,np.real(x),label="x")
plt.plot(t2.real,np.real(x2),dashes=[4,4], label="x2")
plt.plot(t.real,np.abs(pgx),label="pgx")
plt.plot(t2.real,np.abs(pgx2),dashes=[4,4],label="pgx2")
# plt.plot(t.real,np.real(g),label="g")
# plt.plot(t2.real,np.real(x2),dashes=[6, 2],label="x2")
# plt.plot(t2.real,np.abs(pgx2),dashes=[6, 2],label="pgx2")
# make grid for plot
plt.grid()
plt.legend()
plt.savefig("tls.png")

# plot phonon spectral density for 5nm dot
# data = np.genfromtxt("J_omega.dat")
# plt.clf()
# plt.plot(hbar*data[:,0],data[:,1])
# plt.xlabel("hbar*w in meV")
# plt.ylabel("J(w) in meV")
# plt.savefig("J.png")

# create 

# dt = 0.1
# tauend = 5
# n_tau = int(tauend/dt)
# t_apply = 10
# tend = t_apply + tauend
# p1 = ChirpedPulse(tau_0=1, e_start=0, alpha=0, e0=1, t0=1*4)
# t,g,x,p = tls_ace(0,20,p1,dt=dt,ae=5.0,verbose=False,phonons=False,gamma_e=1/250, lindblad=True, apply_op="|0><1|_2", apply_op_t=10)
# print(x[-100])
# print(x[-101])
# print(x[100])
# t2,g2,x2,p = tls_ace(0,20,p1,dt=0.01,ae=5.0,verbose=False,phonons=False,gamma_e=1/250, lindblad=True, apply_op_l="|0><1|_2", apply_op_t=t_apply)
# p1 = ChirpedPulse(tau_0=1, e_start=0, alpha=0, e0=1, t0=1*4)
# pt = PulseTrain(150,4,p1,start=0)
# t,g,x,p = tls_ace(0,600,pt,dt=dt,ae=5.0,verbose=False,phonons=False,gamma_e=1/100, lindblad=True)
# brightness = np.trapz(x,t)
# x2,p2,t2 = twolevel_system(tau1=2.4, area1=2.7*np.pi, alpha1=0, det1=0, tau2=1,alpha2=0, area2=0*np.pi, det2=0, gamma_e=1/250, delay=0, mode="pop", tend=tend)

# print(len(t))
# for _t,_g,_x in zip(t,g,x):
#     print(_t,_g,_x)

# plt.plot(t,g,label='g')
# plt.plot(t2,x2,label="x2")
# plt.plot(t,x-x2[0::10],label='x, dt=0.1-dt=0.01')
# plt.plot(t,x,label='x, dt=0.1')
# plt.plot(t2,x2,label='x, dt=0.01')
# plt.plot(t2,np.abs(p2),label='|p2|')
# plt.plot(t,np.abs(p),label='|p|')
# plt.legend()
# plt.savefig("tls.png")
# tend = 20
# pr = cProfile.Profile()
# pr.enable()


p1 = ChirpedPulse(tau_0=2, e_start=0, alpha=0, e0=3, polar_x=1.0, t0=2*4)
# t_axis, tau_axis, g1 = G1(0,20,0,20,0.1,0.1,p1)
# plt.pcolormesh(t_axis,tau_axis,np.abs(g1.transpose()),shading='auto')
# plt.xlabel("t in ps")
# plt.ylabel("tau in ps")
# plt.colorbar()
# plt.savefig("g1_old.png")

# tend = 500
# tend = 20
# t_axis, tau_axis, g1 = G1_twols(0,tend,0,tend,0.1,0.02,p1,simple_exp=False,phonons=False,workers=15,pt_file="tls_3.0nm_4k_th10_tmem20.48_dt0.02.ptr", gamma_e=1/100, coarse_t=True)
# plt.pcolormesh(t_axis,tau_axis,np.abs(g1.transpose()),shading='auto')
# print(np.trapz(np.trapz(g1.transpose(),t_axis),tau_axis))
# plt.xlabel("t in ps")
# plt.ylabel("tau in ps")
# plt.colorbar()
# plt.savefig("g1_new2_2.png")
# # stats = pstats.Stats(pr)
# # stats.sort_stats('time')
# # stats.print_stats(10)
# tend = 20
# t_axis, tau_axis, f = G2(0,tend,0,tend,0.1,0.1,p1, thread=True,gamma_e=1/250,ninterm=100)
# # t_axis, tau_axis, f = G2hom(0,400,0,400,0.4,0.1,pt, thread=True,gamma_e=1/100,ninterm=100)#,coarse_t=False)
# #pr.disable()
# # np.save("g2.npy",f)
# # t = np.linspace(0, 100, int((100)/0.2)+1)
# # n_tau = int((600)/0.1)
# # tau = np.linspace(0, 600, n_tau + 1)
# # f = np.load("g2.npy")
# plt.clf()
# plt.pcolormesh(t_axis,tau_axis,np.abs(f.transpose()),shading='auto')
# plt.xlabel("t in ps")
# plt.ylabel("tau in ps")
# plt.colorbar()
# plt.savefig("g2_tls_2.png")
# G2_tau = np.abs(np.trapz(f.transpose(), t_axis))
# g2 = 2 * np.trapz(G2_tau, tau_axis) / brightness**2
# print(1/100*brightness)
# print(g2)
#stats = pstats.Stats(pr)
#stats.sort_stats('time')
#stats.print_stats(10)
# np.save("g2_dt01.npy",f)
# f2 = G2(0,20,0,20,0.01,p1, thread=True,gamma_e=1/250)
# np.save("g2_dt001.npy",f2)
# f3 = G2(0,20,0,20,0.05,p1, thread=True,gamma_e=1/250)
# np.save("g2_dt005.npy",f3)
# gamma_e=1/250
# # brightness = np.trapz(x,t)
# # print(gamma_e*brightness)
# f = np.load("g2_dt01.npy")
# f2 = np.load("g2_dt001.npy")
# f3 = np.load("g2_dt005.npy")
# # print(f.shape)
# t_axis = np.linspace(0, tend, int(tend/0.1)+1)
# tau_axis = np.linspace(0, tend, int(tend/0.1)+1)
# tend = 20
# plt.clf()
# plt.pcolormesh(t_axis, tau_axis,
#                   (f-f2[0::10,0::10]).transpose(),shading='auto')
# plt.pcolormesh(t_axis, tau_axis,
#                   (f2[0::10,0::10]-f5).transpose(),shading='auto')

# plt.pcolormesh(t_axis, tau_axis,
#                   f.transpose(),shading='auto')
# plt.clf()
# plt.pcolormesh(t_axis,tau_axis,np.abs(f.transpose()),shading='auto')
# plt.xlabel("t in ps")
# plt.ylabel("tau in ps")
# plt.colorbar()
# plt.savefig("g2_diff.png")
