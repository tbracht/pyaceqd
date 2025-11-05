from pyaceqd.pulses import ChirpedPulse
from pyaceqd.two_level_system.tls import tls
from pyaceqd.two_time.purity import Indistinguishability
from pyaceqd.tools import op_to_matrix
import numpy as np
import matplotlib.pyplot as plt

tau = 5
t0_n = 4.5
p1 = ChirpedPulse(tau_0=tau, e_start=0, alpha=0, t0=t0_n*tau, e0=9, polar_x=1)
options = {"verbose": False, "gamma_e": 1/100, "lindblad": True,
 "temp_dir": '/mnt/temp_data/', "phonons": True, "use_infinite": True, "ae": 5, "temperature": 4}

sigma_x = op_to_matrix("|0><1|_2")
sigma_xdag = op_to_matrix("|1><0|_2")
a = Indistinguishability(tls, "|0><1|_2", "|1><0|_2", p1, dt=0.1, dt_small=0.1, tb=2000, simple_exp=False, gaussian_t=2*t0_n*tau, verbose=False, workers=15, t_simul=None, options=options, dm=True,
                         sigma_x_mat=sigma_x, sigma_xdag_mat=sigma_xdag, t_mem=10)

a.factor_tau = 4
t,rho = a.calc_timedynamics_tl_phonons()
plt.clf()
plt.plot(t.real, rho[:,1,1].real, label="X")
plt.xlabel("Time (a.u.)")
plt.ylabel("Population of |1>")
plt.legend()
plt.savefig("x_train_tl.png")



# tau2, t2, G12 = a.G1_tl_phonons()
a.factor_tau = 2
# print(a.calc_indistinguishability())
# plt.clf()
t2,g2_tl = a.G2_tl_phonons()
plt.clf()
plt.plot(t2,np.abs(g2_tl), label="G2_tl_new")
plt.xlabel("tau")
plt.ylabel("G2")
plt.legend()
plt.savefig("g2.png")

# tau2, G12 = a.G1_tl_phonons()
# print("")
# print(np.trapz(np.abs(G12), tau2))

# a = Indistinguishability(tls, "|0><1|_2", "|1><0|_2", p1, dt=0.1, tb=2000, simple_exp=False, gaussian_t=2*t0_n*tau, verbose=False, workers=15, t_simul=None, options=options, dm=False,
#                          sigma_x_mat=sigma_x, sigma_xdag_mat=sigma_xdag, t_mem=10)


# data = np.genfromtxt("/home/t_brac02/repos/pyaceqd/pyaceqd/two_time/tls_gammae_0.0058_ae8.0_ahfactor2.30.txt",skip_header=1)
# temp = data[:,0]
# indist = data[:,1]
# purity = data[:,2]



# tau = 3
# t0_n = 4.5
# p1 = ChirpedPulse(tau_0=tau, e_start=0, alpha=0, t0=t0_n*tau, e0=1, polar_x=1)
# options = {"verbose": False, "gamma_e": 1./172., "lindblad": True,
#  "temp_dir": '/mnt/temp_data/', "phonons": True, "use_infinite": True, "ae": 8.0, "factor_ah": 2.3, "temperature": 20, "threshold": 11}
# data = np.zeros((len(temp), 2))

# for i, t in enumerate(temp):
#     print(f"Calculating for temperature {t} K, {i+1}/{len(temp)}")
#     options["temperature"] = t
#     a = Indistinguishability(tls, "|0><1|_2", "|1><0|_2", p1, dt=0.1, tb=4000, simple_exp=False, gaussian_t=2*t0_n*tau, verbose=False, workers=15, t_simul=None, options=options, dm=True,
#                              sigma_x_mat=sigma_x, sigma_xdag_mat=sigma_xdag, t_mem=10)
#     data[i] = a.calc_indistinguishability()
#     # data[i, 1] = a.calc_purity()

# np.savez("indistinguishability_purity_data.npz", temp=temp, indist=data[:,0], purity=data[:,1])
# data2 = np.load("indistinguishability_purity_data.npz")
# temp2 = data2['temp']
# indist2 = data2['indist']
# purity2 = data2['purity']

# plt.clf()
# plt.plot(temp, np.abs(indist), label="Indistinguishability")
# plt.plot(temp2, np.abs(indist2), linestyle='dashed', label="Indistinguishability (new)")
# plt.plot(temp, np.abs(purity), label="Purity")
# plt.plot(temp2, np.abs(purity2), linestyle='dashed', label="Purity (new)")
# plt.xlabel("Temperature (K)")
# plt.ylabel("Indistinguishability / Purity")
# plt.legend()
# # plt.xlim(0, 100)
# plt.savefig("indistinguishability_purity.png")


# print(a.calc_indistinguishability())

# tau, t, G1 = a.G1()

# np.savez("g1_tl_phonons.npz", tau=tau, t=t, g1=G1, 
#          tau2=tau2, t2=t2, g12=G12)

# data = np.load("g1_tl_phonons.npz")
# tau = data['tau']
# t = data['t']
# G1 = data['g1']
# tau2 = data['tau2']
# t2 = data['t2']
# G12 = data['g12']

# plt.clf()
# plt.plot(tau, np.abs(G1[40,:]), label="G1")
# plt.plot(tau2, np.abs(G12[40,:]), linestyle='dashed', label="G1_tl")
# plt.xlabel("tau")
# plt.ylabel("G1")
# plt.legend()
# plt.xlim(0, 3000)
# plt.savefig("g1_tl_phonons.png")    


# print(a.calc_indistinguishability())
# t,g2 = a.G2()

# np.savez("g2_tl_phonons.npz", tau=t, tau_tl=t2, g2=g2, g2_tl=g2_tl)

# load 
# data = np.load("g2_tl_phonons.npz")
# t = data['tau']
# # t2 = data['tau_tl']
# g2 = data['g2']
# # g2_tl = data['g2_tl']

# plt.clf()
# plt.plot(t,np.abs(g2), label="G2")
# plt.plot(t2,np.abs(g2_tl), dashes=[2,2], label="G2_tl")
# plt.xlabel("tau")
# plt.ylabel("G2")
# plt.title("G2 Comparison with phonons")
# plt.legend()
# # plt.xlim(1900, 2100)
# plt.savefig("g2_comparison.png")
# print("")
# print("integrated g2    ",np.trapz(np.abs(g2), t))
# print("integrated g2_tl ",np.trapz(np.abs(g2_tl), t2))

# t,x = a.calc_timedynamics(output_ops=["|1><1|_2"])
# t2, rho = a.calc_timedynamics_tl_phonons()
# x2 = rho[:,1,1]  # extract the population of state |1>
# plt.clf()
# plt.plot(t.real,x.real)
# plt.plot(t2.real,x2.real, linestyle='dashed')
# # plt.xlim(0,100)
# plt.savefig("x_train.png")

# t,g0 = a.simple_propagation()
# t2,g02 = a.simple_propagation_tl_phonons()
# plt.clf()   
# plt.plot(t,np.abs(g0), label="G0")
# plt.plot(t2,np.abs(g02), linestyle='dashed', label="G0_tl")
# plt.xlabel("t")
# plt.ylabel("G0")
# # plt.xlim(1500-10, 1500+10)
# plt.legend()
# plt.savefig("g0.png")

# t12,g12 = a.G1_tl()
# t12,g12= a.G1_tl_phonons()
# plt.clf()
# plt.plot(t12,np.abs(g12), label="G1_tl")
# plt.xlabel("tau")
# plt.ylabel("G1")
# plt.legend()
# plt.savefig("g1.png")

# # tau, t, G1 = a.G1_tl_phonons()
# # np.savez("g1_tl_phonons.npz", tau=tau, t=t, g1=G1)

# # data = np.load("g1_tl_phonons.npz")
# # tau = data['tau']
# # t = data['t']
# # G1 = data['g1']

# # _tau, _result, temp_res, G1 = a.G1_tl_phonons()
# # np.savez("g1_tl_phonons.npz", tau=_tau, g1=G1)

# data = np.load("g1_tl_phonons.npz")
# tau = data['tau']
# t = data['t']
# G1 = data['g1']

# # print(f"tau: {tau.shape}, t: {t.shape}, G1: {G1.shape}")
# # print(t[:59])
# plt.clf()
# plt.plot(tau, np.abs(G1[40,:]), label="G1_tl_phonons")
# plt.plot(t, np.abs(G1[:,0]), linestyle='dashed', label="G1_tl_phonons_new")
# # plt.plot(_tau, np.abs(_result), linestyle='dashed', label="G1_tl_phonons_new")
# # plt.plot(_tau, np.imag(result_rho[:,0,1]), 'b-', label="im 01")
# # plt.plot(_tau, np.real(result_rho[:,0,1]), 'r-', label="re 01")
# # plt.plot(_tau, np.abs(result_rho[:,1,0]), label="im 10")
# # plt.plot(_tau, np.real(result_rho[:,1,0]), label="re 10")

# plt.xlabel("tau")
# plt.ylabel("G1")
# plt.legend()
# plt.xlim(0, 3000)
# plt.savefig("g1_tl_phonons.png")

# plt.clf()
# t23,g23 = a.G2_tl_phonons()
# plt.clf()
# plt.plot(t23,np.abs(g23), label="G2_tl_new")
# plt.xlabel("tau")
# plt.ylabel("G2")
# plt.legend()
# plt.savefig("g2.png")