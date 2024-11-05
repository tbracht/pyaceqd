# calculate purity of single-photon source
# this bascically compares the peaks of the two-time correlation function
# G2(tau=0) with the peak of the two-time correlation function G2(tau=T_pulse)
# where T_pulse is the separation of pulses in the pulse train
# this means that the simulation needs to span at least 2*T_pulse, i.e., 3 pulses in the pulse train
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from concurrent.futures import ThreadPoolExecutor, wait
from pyaceqd.tools import construct_t, simple_t_gaussian, export_csv
from pyaceqd.timebin.timebin import TimeBin
from pyaceqd.pulses import PulseTrain, ChirpedPulse
from pyaceqd.two_level_system.tls import tls
import matplotlib.pyplot as plt

class Purity(TimeBin):
    def __init__(self, system, sigma_x, sigma_xdag, *pulses, dt=0.1, tb=800, dt_small=0.1, simple_exp=True, gaussian_t=None, verbose=False, workers=15, t_simul=None, options={}, factor_t=1, factor_tau=2) -> None:
        pulse = PulseTrain(tb, 5, *pulses)
        self.factor_t = factor_t
        self.factor_tau = factor_tau
        super().__init__(system, pulse, dt=dt, tb=tb, simple_exp=simple_exp, gaussian_t=gaussian_t, verbose=verbose, workers=workers, t_simul=t_simul, options=options)
        self.sigma_x = "(" + sigma_x + ")"
        self.sigma_xdag = "(" + sigma_xdag + ")"
        
        try:
            self.gamma_e = options["gamma_e"]
        except KeyError:
            print("gamma_e not included in options, setting to 100")
            self.options["gamma_e"] = 100
            self.gamma_e = self.options["gamma_e"]
        if self.gaussian_t is not None:
            self.t1 = simple_t_gaussian(0,self.gaussian_t,self.tb,dt_small,10*dt_small,*pulses,decimals=1)
            # _t = np.concatenate((_t,self.tb-_t))
            # sort and remove duplicates
            # _t = np.sort(np.unique(_t))
            # self.t1 = _t
            # plt.clf()
            # plt.scatter(self.t1, np.zeros_like(self.t1))
            # plt.savefig("t1.png")
            # plt.clf()
        else:
            self.t1 = construct_t(0, self.tb, dt_small, 10*dt_small, *pulses, simple_exp=self.simple_exp)
        # complete t-axis, when t1 is repeated for factor_t > 1
        t_axis_complete = np.array([])
        for i in range(factor_t):
            t_axis_complete = np.concatenate((t_axis_complete, self.t1 + i*self.tb))
        self.t_axis_complete = t_axis_complete
        # compatibility with tls, which needs no polarization
        self.options["pulse_file_x"] = self.pulse_file_x
        self.options["pulse_file_y"] = self.pulse_file_y
        # print(self.options)

    def prepare_pulsefile(self, verbose=False, t_simul=None, plot=False):
        # override prepare_pulsefile from TimeBin
        # because we need a different t_end, and also use a PulseTrain
        t_end = (self.factor_t + self.factor_tau + 1.1)*self.tb
        if t_simul is not None:
            t_end = t_simul
        _t_pulse = np.arange(0,t_end,step=self.dt/5)
        # different polarizations
        self.pulse_file_x = self.temp_dir + "twotime_pulse_x_{}.dat".format(id(self))
        self.pulse_file_y = self.temp_dir + "twotime_pulse_y_{}.dat".format(id(self))
        pulse_x = np.zeros_like(_t_pulse, dtype=complex)
        pulse_y = np.zeros_like(_t_pulse, dtype=complex)
        pulse_x, pulse_y = self.pulses[0].get_total_xy(_t_pulse)
        export_csv(self.pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ', verbose=verbose)
        export_csv(self.pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ', verbose=verbose)
        if plot:
            plt.clf()
            plt.plot(_t_pulse, pulse_x.real)
            plt.xlabel("t")
            plt.ylabel("E_x")
            plt.savefig("pulsetrain.png")
            plt.clf()

    def calc_timedynamics(self, output_ops=None):
        new_options = dict(self.options)
        if output_ops is not None:
            new_options["output_ops"] = output_ops
        t_end = (self.factor_t + self.factor_tau + 1.1)*self.tb
        return self.system(0, t_end, *self.pulses, **new_options)

    def G2(self, return_whole=False):
        sigma_left = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}
        sigma_right = {"operator": self.sigma_xdag, "applyFrom": "_right", "applyBefore":"false"}
        
        out_op1 = self.sigma_xdag + "*" + self.sigma_x
        out_op_tau0 = self.sigma_xdag + "*" + self.sigma_xdag + "*" + self.sigma_x + "*" + self.sigma_x
        output_ops = [out_op1, out_op_tau0]
        t1 = self.t1
        factor_t = self.factor_t
        t_axis_complete = self.t_axis_complete
        factor_tau = self.factor_tau
        n_tau = factor_tau*int(self.tb/self.dt)
        t2 = np.linspace(0, factor_tau*self.tb, n_tau + 1)
        _G2 = np.zeros([factor_t*len(t1), len(t2)])
        with tqdm.tqdm(total=factor_t*len(t1), leave=None) as tq:
            for i in range(factor_t):
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    futures = []
                    for j in range(len(t1)):
                        tend = i*self.tb + t1[j] + factor_tau*self.tb
                        sigma_X_new = dict(sigma_left)
                        sigma_Xdag_new = dict(sigma_right)
                        sigma_X_new["time"] = i*self.tb + t1[j]
                        sigma_Xdag_new["time"] = i*self.tb + t1[ j]
                        multitime_ops = [sigma_X_new, sigma_Xdag_new]
                        _e = executor.submit(self.system, 0, tend, multitime_op=multitime_ops, suffix=j, output_ops=output_ops, **self.options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
                    wait(futures)
                for j in range(len(t1)):
                    futures[j] = futures[j].result()
                for j in range(len(t1)):
                    _G2[j+i*len(t1),1:] = np.abs(futures[j][1][-(n_tau):])
                    # special case tau=0:
                    _G2[j+i*len(t1),0] = np.abs(futures[j][2][-(n_tau+1)])
        # integrate over t1
        G2 = np.trapz(_G2, t_axis_complete, axis=0)
        if return_whole:
            return t1, t2, _G2
        return t2, G2
    
    def calc_purity(self):
        t,g2 = self.G2()
        dt = self.dt
        tb = self.tb
        n_1 = int(0.5*tb/dt)
        G21 = 2*np.trapz(g2[:n_1], t[:n_1])
        G22 = np.trapz(g2[n_1:3*n_1], t[n_1:3*n_1])
        return 1-G21/G22
    
class Indistinguishability(Purity):
    def __init__(self, system, sigma_x, sigma_xdag, *pulses, dt=0.1, tb=800, dt_small=0.1, simple_exp=True, gaussian_t=None, verbose=False, workers=15, t_simul=None, options={}) -> None:
        super().__init__(system, sigma_x, sigma_xdag, *pulses, dt=dt, tb=tb, dt_small=dt_small, simple_exp=simple_exp, gaussian_t=gaussian_t, verbose=verbose, workers=workers, t_simul=t_simul, options=options)

    def G1(self):
        sigma_x = {"operator": self.sigma_x, "applyFrom": "_left", "applyBefore":"false"}
 
        out_op1 = self.sigma_xdag
        out_op_tau0 = self.sigma_xdag + "*" + self.sigma_x
        output_ops = [out_op1, out_op_tau0]
        t1 = self.t1
        factor_t = self.factor_t
        t_axis_complete = self.t_axis_complete
        factor_tau = self.factor_tau
        n_tau = factor_tau*int(self.tb/self.dt)
        t2 = np.linspace(0, factor_tau*self.tb, n_tau + 1)
        _G1 = np.zeros([factor_t*len(t1), len(t2)], dtype=complex)
        with tqdm.tqdm(total=factor_t*len(t1), leave=None) as tq:
            for i in range(factor_t):
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    futures = []
                    for j in range(len(t1)):
                        tend = i*self.tb + t1[j] + factor_tau*self.tb
                        sigma_X_new = dict(sigma_x)
                        sigma_X_new["time"] = i*self.tb + t1[j]
                        multitime_ops = [sigma_X_new]
                        _e = executor.submit(self.system, 0, tend, multitime_op=multitime_ops, suffix=j, output_ops=output_ops, **self.options)
                        _e.add_done_callback(lambda f: tq.update())
                        futures.append(_e)
                    wait(futures)
                for j in range(len(t1)):
                    futures[j] = futures[j].result()
                for j in range(len(t1)):
                    _G1[j+i*len(t1),1:] = futures[j][1][-(n_tau):]
                    # special case tau=0:
                    _G1[j+i*len(t1),0] = futures[j][2][-(n_tau+1)]
        # plot _G1
        # plt.clf()
        # plt.pcolormesh(t2, t_axis_complete, np.abs(_G1)**2)
        # plt.xlabel("tau")
        # plt.ylabel("t1")
        # plt.savefig("G1.png")
        # plt.clf()
        # integrate over t1
        G1 = np.trapz(np.abs(_G1)**2, t_axis_complete, axis=0)
        return t2, G1
    
    def simple_propagation(self, return_whole=False):
        # most importantly, in all calculations, the same factor_t, factor_tau and tb must be used
        output_ops = [self.sigma_xdag + "*" + self.sigma_x]
        factor_tau = self.factor_tau
        # print(self.t_axis_complete[-1])
        tend = (self.factor_t + factor_tau)*self.tb
        n_tau = factor_tau*int(self.tb/self.dt)
        t2 = np.linspace(0, factor_tau*self.tb, n_tau + 1)
        t, val = self.system(0, tend, suffix=-1, output_ops=output_ops, **self.options)
        val = np.abs(val)
        # <x(t)>*<x(t+tau)>
        t1 = np.linspace(0, self.factor_t*self.tb, int((self.factor_t*self.tb)/self.dt) + 1)
        G0 = np.zeros([len(t1), len(t2)])
        # the following loop can efficiently implemented using numpy
        # for i in tqdm.trange(len(t1)):
        #     for j in range(len(t2)):
        #         G0[i,j] = val[i]*val[i+j]
        # efficient implementation
        i_indices, j_indices = np.ogrid[:len(t1), :len(t2)]
        G0 = val[i_indices] * val[i_indices + j_indices]
        # integrate over t1
        G0_tau = np.trapz(G0, t1, axis=0)
        if return_whole:
            return t1, t2, G0
        return t2, G0_tau

    def calc_indistinguishability(self):
        """
        returns indistinguishability,single-photon purity
        """
        # calculate G0, G1 and G2
        # and integrate over tau=0,...,tb/2 and tb/2,...,3tb/2
        t,g1 = self.G1()
        dt = self.dt
        tb = self.tb
        n_1 = int(0.5*tb/dt)
        G11 = 2*np.trapz(g1[:n_1], t[:n_1])
        G12 = np.trapz(g1[n_1:3*n_1], t[n_1:3*n_1])
        # print("G11", G11, "G12", G12)

        t2,g2 = self.G2()
        G21 = 2*np.trapz(g2[:n_1], t2[:n_1])
        G22 = np.trapz(g2[n_1:3*n_1], t2[n_1:3*n_1])
        # print("G21", G21, "G22", G22)

        t0,g0 = self.simple_propagation()
        # special, integrate 0,...,tb and tb,...,2tb
        # n_2 = int(tb/dt)
        G01 = 2*np.trapz(g0[:n_1], t0[:n_1])
        G02 = np.trapz(g0[n_1:3*n_1], t0[n_1:3*n_1])
        # print("G01", G01, "G02", G02)

        result = (G01-G11+G21)/(G02-G12+G22)
        return 1 - result, 1-G21/G22

# tau=3
# p1 = ChirpedPulse(tau_0=tau, e_start=0, alpha=0, t0=4*tau, e0=1, polar_x=1)
# options = {"verbose": False, "gamma_e": 1/100, "lindblad": True,
#  "temp_dir": '/mnt/temp_data/', "phonons": False}

def resample(x, y, z, s_x, s_y):
    x_new = np.zeros(int((len(x))/s_x))
    y_new = np.zeros(int((len(y))/s_y))
    z_new = np.zeros((len(y_new),len(x_new)))
    for i in range(len(x_new)):
        for j in range(len(y_new)):
            x_new[i] = x[int(i*s_x)]
            y_new[j] = y[int(j*s_y)]
            z_new[j,i] = z[int(j*s_y),int(i*s_x)]
    return x_new, y_new, z_new

# a = Purity(tls, "|0><1|_2", "|1><0|_2", p1, dt=0.1, tb=100, simple_exp=False, gaussian_t=None, verbose=False, workers=15, t_simul=None, options=options, factor_t=1, factor_tau=1)
# t1,t2,g2 = a.G2(return_whole=True)
# print(t1)
# print(t2)
# plt.clf()
# plt.pcolormesh(t2, t1, np.abs(g2)**2)
# plt.xlabel("tau")
# plt.ylabel("t1")
# plt.colorbar()
# plt.savefig("g2.png")
# plt.clf()
# a = Indistinguishability(tls, "|0><1|_2", "|1><0|_2", p1, dt=0.1, tb=2000, simple_exp=False, gaussian_t=None, verbose=False, workers=15, t_simul=None, options=options)
# print(a.calc_indistinguishability())
# t,x = a.simple_propagation()
# t,x = a.calc_timedynamics(output_ops=["|1><1|_2"])
# plt.clf()
# plt.plot(t.real,x.real)
# # plt.plot(t[int(2000/0.1):]-2000,x[int(2000/0.1):])
# plt.savefig("x_train.png")
# plt.clf()
# t,g2 = a.G2()
# t2,xtau = a.simple_propagation(return_whole=False)
# t1,t2,g0 = a.simple_propagation(return_whole=True)
# print(t1.shape, t2.shape, g0.shape)
# plt.clf()
# plt.pcolormesh(*resample(t2, t1, g0, 10, 20))
# plt.xlim(0,150)
# plt.ylim(0,150)
# plt.savefig("xx_g0_1pi.png")
# tau,g0 = a.simple_propagation()
# np.save("g0_train_1pi.npy", g0)
# np.save("t_g0_train_1pi.npy", tau)
# t1,t2,g2 = a.G2(return_whole=True)
# tau,g1 = a.G1()
# np.save("g1_train_1pi.npy", g1)
# np.save("t1_g1_train_1pi.npy", tau)
# g1 = np.load("g1_train_1pi.npy")
# tau = np.load("t1_g1_train_1pi.npy")
# plt.clf()
# plt.plot(tau,g1)
# plt.xlabel("tau")
# plt.savefig("g1_train_1pi.png")

# plt.clf()
# plt.plot(t,g2)
# plt.savefig("g2_train_1pi.png")
# np.save("g2_train_1pi.npy", g2)
# np.save("t_train_1pi.npy", t)
# plt.clf()
# plt.plot(t,g2)
# plt.xlim(1900,2100)
# plt.savefig("g2_train_zoom_1pi.png")
# plt.clf()
# plt.pcolormesh(tau, t, np.abs(g2)**2)
# plt.xlabel("tau")
# plt.ylabel("t1")
# plt.colorbar()
# plt.savefig("g2_2d.png")
# np.save("g2_tau_1pi.npy", g2)
# np.save("t_tau_1pi.npy", t)
# # dt = 0.1
# # tb = 2000
# # n_1 = int(0.5*tb/dt)

# t = np.load("t_tau_1pi.npy")
# g2 = np.load("g2_tau_1pi.npy")

# # print(t[:n_1])
# # print(t[n_1:3*n_1])
# G21 = 2*np.trapz(g2[:n_1], t[:n_1])
# G22 = np.trapz(g2[n_1:3*n_1], t[n_1:3*n_1])
# print(G21)
# print(G22)
# print(1-G21/G22)
# plt.clf()
# plt.plot(t,g2,"b-")
# plt.plot(-t,g2,"b-")
# # plt.xlim(-200,200)
# plt.xlabel("tau")
# plt.ylabel("G2")
# plt.savefig("g2_tau_1pi.png")

# t,g1 = a.G1()
# np.save("g1_train2.npy", g1)
# np.save("tg1_train2.npy", t)

# g1 = np.load("g1_train2.npy")
# t = np.load("tg1_train2.npy")
# plt.clf()
# plt.plot(t,np.abs(g1))
# # plt.plot(t,np.real(g1),linestyle='dashed', label="real")
# # plt.plot(t,np.imag(g1),linestyle='dashed', label="imag")
# plt.legend()
# plt.xlabel("tau")
# plt.ylabel("G1")
# plt.savefig("g1_train2.png")