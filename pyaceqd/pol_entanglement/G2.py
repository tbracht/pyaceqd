import numpy as np
import os
from pyaceqd.tools import export_csv, construct_t, concurrence, simple_t_gaussian
import tqdm
from concurrent.futures import ThreadPoolExecutor, wait
import matplotlib.pyplot as plt

class PolarizatzionEntanglement():
    def __init__(self, system, sigma_x, sigma_y, sigma_xdag, sigma_ydag, *pulses, dt=0.1, tend=400, time_intervals=None, simple_exp=True, dt_small=0.1, gaussian_t=None, verbose=False, workers=2, options={}) -> None:
        self.system = system  # system that is used for the simulation
        self.dt = dt  # timestep during simulation
        self.options = dict(options)
        self.options["dt"] = dt  # also save it in the options dict
        self.tend = tend  # timebin width
        self.simple_exp = simple_exp  # use exponential timestepping
        self.gaussian_t = gaussian_t  # use gaussian timestepping during pulse
        self.pulses = pulses
        self.workers = workers  # number of threads spawned by ThreadPoolExecutor
        self.ax = "(" + sigma_x + ")"
        self.ay = "(" + sigma_y + ")"
        self.axdag = "(" + sigma_xdag + ")"
        self.aydag = "(" + sigma_ydag + ")"
        try:
            self.temp_dir = options["temp_dir"]
        except KeyError:
            print("temp_dir not included in options, setting to /mnt/temp_data/")
            self.options["temp_dir"] = "/mnt/temp_data/"
            self.temp_dir = self.options["temp_dir"]
        self.prepare_pulsefile(verbose=verbose)
        self.options["pulse_file_x"] = self.pulse_file_x  # put pulse files in options dict
        self.options["pulse_file_y"] = self.pulse_file_y
        self.gamma_e = options["gamma_e"]
        if time_intervals is not None:
            if len(time_intervals) != 2:
                return ValueError("time_intervals must be a list of length 2")
            ts = []
            ts.append(np.arange(0,time_intervals[0],dt_small))
            ts.append(np.arange(time_intervals[0],time_intervals[1],10*dt_small))
            _exp_part = np.exp(np.arange(np.log(time_intervals[1]),np.log(tend),dt_small))
            ts.append(np.round(_exp_part))
            ts.append(np.array([tend]))
            self.t1 = np.concatenate(ts, axis=0)
        elif self.gaussian_t is not None:
            self.t1 = simple_t_gaussian(0,self.gaussian_t,self.tend,dt_small,10*dt_small,*self.pulses,decimals=1, exp_part=self.simple_exp)
            # print(self.t1)
            # print(len(self.t1))
        else:
            self.t1 = construct_t(0, self.tend, dt_small, 10*dt_small, *self.pulses, simple_exp=self.simple_exp)

    def prepare_pulsefile(self, verbose=False):
        # 2*tb is the maximum simulation length, 0 is the start of the simulation
        _t_pulse = np.arange(0,self.tend,step=self.dt/5)  # notice that for usual propagation, dt/10 is used
        # different polarizations
        self.pulse_file_x = self.temp_dir + "polar_ent_pulse_x_{}.dat".format(id(self))  # add object id, otherwise sometimes the wrong file is used
        self.pulse_file_y = self.temp_dir + "polar_ent_pulse_y_{}.dat".format(id(self))  # probably because the destructor is called after the next object is created
        pulse_x = np.zeros_like(_t_pulse, dtype=complex)
        pulse_y = np.zeros_like(_t_pulse, dtype=complex)
        for _p in self.pulses:
            pulse_x = pulse_x + _p.polar_x*_p.get_total(_t_pulse)
            pulse_y = pulse_y + _p.polar_y*_p.get_total(_t_pulse)
        export_csv(self.pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ', verbose=verbose)
        export_csv(self.pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ', verbose=verbose)

    def __del__(self):
        os.remove(self.pulse_file_x)
        os.remove(self.pulse_file_y)

    def calc_densitymatrix(self):
        density_matrix = np.zeros([4,4], dtype=complex)
        with tqdm.tqdm(total=10, leave=None) as tq:
            _,_,density_matrix[0,0] = self.G2(self.axdag, self.axdag, self.ax, self.ax)  # xx,xx
            tq.update()
            _,_,density_matrix[3,3] = self.G2(self.aydag, self.aydag, self.ay, self.ay)  # yy,yy
            tq.update()
            _,_,density_matrix[1,1] = self.G2(self.axdag, self.aydag, self.ay, self.ax)  # xy,xy
            tq.update()
            _,_,density_matrix[2,2] = self.G2(self.aydag, self.axdag, self.ax, self.ay)  # yx,yx
            tq.update()

            _,_,density_matrix[0,1] = self.G2(self.axdag, self.axdag, self.ay, self.ax)  # xx,xy
            tq.update()
            density_matrix[1,0] = np.conj(density_matrix[0,1])
            _,_,density_matrix[0,2] = self.G2(self.axdag, self.axdag, self.ax, self.ay)  # xx,yx
            tq.update()
            density_matrix[2,0] = np.conj(density_matrix[0,2])
            _,_,density_matrix[0,3] = self.G2(self.axdag, self.axdag, self.ay, self.ay)  # xx,yy
            tq.update()
            density_matrix[3,0] = np.conj(density_matrix[0,3])

            _,_,density_matrix[1,2] = self.G2(self.axdag, self.aydag, self.ax, self.ay)  # xy,yx
            tq.update()
            density_matrix[2,1] = np.conj(density_matrix[1,2])
            _,_,density_matrix[1,3] = self.G2(self.axdag, self.aydag, self.ay, self.ay)  # xy,yy
            tq.update()
            density_matrix[3,1] = np.conj(density_matrix[1,3])

            _,_,density_matrix[2,3] = self.G2(self.aydag, self.axdag, self.ay, self.ay)  # yx,yy
            tq.update()
            density_matrix[3,2] = np.conj(density_matrix[2,3])

        norm = np.trace(density_matrix)
        density_matrix = density_matrix / norm
        return concurrence(density_matrix)

    def G2(self, op1_t, op2_ttau, op3_ttau, op4_t):
        """
        calculates G2 for four operators:
        <op1(t1) op2(t1+tau) op3(t1+tau) op4(t1)>
        returns the integral of G2 over t1 and tau
        """
        op23_ttau = op2_ttau + " * " + op3_ttau
        tau0_op = op1_t + " * " + op23_ttau + " * " + op4_t
        output_ops = [op23_ttau, tau0_op]
        # at t1, apply op4 from left and op1 from right
        op_1 = {"operator": op1_t, "applyFrom": "_right", "applyBefore":"false"}
        op_4 = {"operator": op4_t, "applyFrom": "_left", "applyBefore":"false"}

        t1 = self.t1
        n_tau = int((self.tend)/self.dt)
        # simulation time-axis
        t2 = np.linspace(0, self.tend, n_tau + 1)
        _G2 = np.zeros([len(t1)], dtype=complex)
        tend = self.tend  # always the same
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    op_1_new = dict(op_1)  # must make a copy of the dict
                    op_4_new = dict(op_4)
                    op_1_new["time"] = t1[i]
                    op_4_new["time"] = t1[i]
                    # apply op4 from left and sigma_bbdag from right
                    multitme_ops = [op_1_new, op_4_new]
                    _e = executor.submit(self.system,0,tend,multitime_op=multitme_ops, suffix=i, output_ops=output_ops, **self.options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                futures[i] = futures[i].result()
            # futures now contains [t,<op2*op3>,<op1*op2*op3*op4>] for every i
            for i in range(len(t1)):
                # t2 = t1,...,tend
                n_t2 = n_tau - int((t1[i])/self.dt)
                temp_t2 = np.zeros(n_t2+1, dtype=complex)
                # special case tau=0:
                # as then, Tr(op1*op2*op3*op4 * rho) = G2(t,0), which is the value with index [2][-(n_t2+1)]
                temp_t2[0] = futures[i][2][-(n_t2+1)]
                # futures[i][2] are the corresponding values, [1] are the values for tau>0, when the operators are applied separately
                # here, we want the <op2*op3>-values for every t2=t1,..,tend
                if n_t2 > 0: 
                    temp_t2[1:n_t2+1] = futures[i][1][-n_t2:]
                t_new = t2[:len(temp_t2)]
                # plt.clf()
                # plt.plot(t_new,np.real(temp_t2),'r-')
                # plt.plot(t_new,np.imag(temp_t2),'b-')
                # plt.savefig("aa_tests/plot_{}.png".format(i))
                # integrate over t_new
                _G2[i] = np.trapz(temp_t2,t_new)
        return t1, _G2, np.trapz(_G2,t1)
    
    def calc_densitymatrix_reuse(self, plot_G2=None, return_counts=False, return_rho=False):
        density_matrix = np.zeros([4,4], dtype=complex)
        with tqdm.tqdm(total=3, leave=None) as tq:
            # XX,XX; XX,XY; XY,XY
            op23s = [self.axdag + " * " + self.ax, self.axdag + " * " + self.ay, self.aydag + " * " + self.ay]
            t1, G2_1_t, G2_1 = self.G2_reuse(self.axdag, op23s, self.ax)
            tq.update()
            # XX,YX; XX,YY; XY,YX; XY,YY
            op23s = [self.axdag + " * " + self.ax, self.axdag + " * " + self.ay, self.aydag + " * " + self.ax,self.aydag + " * " + self.ay]
            t2, G2_2_t, G2_2 = self.G2_reuse(self.axdag, op23s, self.ay)
            tq.update()
            # YX,YX; YX,YY; YY,YY
            op23s = [self.axdag + " * " + self.ax, self.axdag + " * " + self.ay, self.aydag + " * " + self.ay]
            t3, G2_3_t, G2_3 = self.G2_reuse(self.aydag, op23s, self.ay)
            tq.update()

            density_matrix[0,0] = np.abs(G2_1[0])  # xx,xx
            density_matrix[3,3] = np.abs(G2_3[2])  # yy,yy
            density_matrix[1,1] = np.abs(G2_1[2])  # xy,xy
            density_matrix[2,2] = np.abs(G2_3[0])  # yx,yx

            density_matrix[0,1] = G2_1[1]  # xx,xy
            density_matrix[1,0] = np.conj(density_matrix[0,1])
            density_matrix[0,2] = G2_2[0]  # xx,yx
            density_matrix[2,0] = np.conj(density_matrix[0,2])
            density_matrix[0,3] = G2_2[1]  # xx,yy
            density_matrix[3,0] = np.conj(density_matrix[0,3])

            density_matrix[1,2] = G2_2[2]  # xy,yx
            density_matrix[2,1] = np.conj(density_matrix[1,2])
            density_matrix[1,3] = G2_2[3]  # xy,yy
            density_matrix[3,1] = np.conj(density_matrix[1,3])

            density_matrix[2,3] = G2_3[1]  # yx,yy
            density_matrix[3,2] = np.conj(density_matrix[2,3])

        norm = np.trace(density_matrix)

        if plot_G2 is not None:
            plt.clf()
            plt.plot(t1, np.abs(G2_1_t[0]), label="xx,xx")
            plt.plot(t1, np.abs(G2_1_t[2]), label="xy,xy")
            plt.plot(t2, np.abs(G2_2_t[1]), label="xx,yy")
            plt.plot(t3, np.abs(G2_3_t[0]), dashes=[4,4],label="yx,yx")
            plt.plot(t3, np.abs(G2_3_t[2]), dashes=[4,4],label="yy,yy")
            plt.xlabel("t (ps)")
            plt.ylabel("G2(t)")
            plt.legend()
            plt.savefig("{}.png".format(plot_G2))
            np.save("{}.npy".format(plot_G2), np.array([t1, G2_1_t[0], G2_1_t[1], G2_1_t[2], G2_2_t[0], G2_2_t[1], G2_2_t[2], G2_2_t[3], G2_3_t[0], G2_3_t[1], G2_3_t[2]]))
        if return_rho:
            return concurrence(density_matrix/norm), density_matrix
        if return_counts:
            return concurrence(density_matrix/norm), density_matrix[0,0], density_matrix[1,1], density_matrix[2,2], density_matrix[3,3], density_matrix[0,3]
        
        return concurrence(density_matrix/norm)
    
    def G2_reuse(self, op1_t, op23s_ttau, op4_t):
        """
        re-uses the same simulation for different output operators,
        which are given in op23s_ttau
        """
        tau0_ops = []  # operators for tau=0
        for op23_ttau in op23s_ttau:
            tau0_ops.append(op1_t + " * " + op23_ttau + " * " + op4_t)
        output_ops = op23s_ttau + tau0_ops  # concatenate lists
        # at t1, apply op4 from left and op1 from right
        op_1 = {"operator": op1_t, "applyFrom": "_right", "applyBefore":"false"}
        op_4 = {"operator": op4_t, "applyFrom": "_left", "applyBefore":"false"}

        t1 = self.t1
        n_tau = int((self.tend)/self.dt)
        # simulation time-axis
        t2 = np.linspace(0, self.tend, n_tau + 1)
        _G2 = np.zeros([len(op23s_ttau),len(t1)], dtype=complex)
        tend = self.tend  # always the same, note that for pol.-ent. we might want to change this
        with tqdm.tqdm(total=len(t1), leave=None) as tq:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for i in range(len(t1)):
                    op_1_new = dict(op_1)  # must make a copy of the dict
                    op_4_new = dict(op_4)
                    op_1_new["time"] = t1[i]
                    op_4_new["time"] = t1[i]
                    # apply op4 from left and sigma_bbdag from right
                    multitme_ops = [op_1_new, op_4_new]
                    _e = executor.submit(self.system,0,tend,multitime_op=multitme_ops, suffix=i, output_ops=output_ops, **self.options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                # wait for all futures
                wait(futures)
            for i in range(len(futures)):
                # futures are still 'future' objects
                futures[i] = futures[i].result()
            # futures now contains [t,<op2*op3> for all op23s,<op1*op2*op3*op4> for all op23s] for every i
            # so if op_23s has length 2, futures[i] has length 5
            # and futures[i][1,...,len(op23s)] are the <op2*op3>-values
            # and futures[i][len(op23s)+1,...,2*len(op23s)] are the <op1*op2*op3*op4>-values
            for i in range(len(t1)):
                # t2 = t1,...,tend
                n_t2 = n_tau - int((t1[i])/self.dt)
                temp_t2 = np.zeros([len(op23s_ttau),n_t2+1], dtype=complex)
                for j in range(len(op23s_ttau)):
                    # special case tau=0:
                    # as then, Tr(op1*op2*op3*op4 * rho) = G2(t,0), which is the value with index [1+len(op23s)+j][-(n_t2+1)]
                    temp_t2[j,0] = futures[i][1+len(op23s_ttau) + j][-(n_t2+1)]
                    # futures[i][1+len(op23s)+j] are the corresponding values, [1+j] are the values for tau>0, when the operators are applied separately
                    # here, we want the <op2*op3>-values for every t2=t1,..,tend
                    if n_t2 > 0: 
                        temp_t2[j,1:n_t2+1] = futures[i][1+j][-n_t2:]
                t_new = t2[:n_t2+1]
                # integrate over t_new
                for j in range(len(op23s_ttau)):
                    _G2[j,i] = np.trapz(temp_t2[j],t_new)
        return t1, _G2, np.trapz(_G2,t1,axis=1)
