from pyaceqd.two_level_system.tls import tls
import numpy as np
import matplotlib.pyplot as plt
from pyaceqd.pulses import ChirpedPulse
from pyaceqd.tools import export_csv
from concurrent.futures import ThreadPoolExecutor, wait
import tqdm
import os
import subprocess
import time
from pyaceqd.constants import hbar
import pyaceqd.pulsegenerator as pg

class RabiRotations():
    def __init__(self, dt=0.1, tau=5, area_max=30, n_area=150, gamma_e=1/100, phonons=False, temperature=4, ae=5, ah_ratio=1.15, J_from_file=None, phonon_factor=1, t_mem=10, temp_dir="/mnt/temp_data/") -> None:
        self.dt = dt
        self.tau = tau
        self.areas = np.linspace(0, area_max, n_area)
        self.gamma_e = gamma_e
        self.phonons = phonons
        self.temperature = temperature
        self.ae = ae
        self.ah_ratio = ah_ratio
        self.J_from_file = J_from_file
        self.phonon_factor = phonon_factor
        self.t_mem=t_mem
        if J_from_file is not None:
            self.pt_name = J_from_file.split(".")[0]+".ptr"
        else:
            self.pt_name = "pt_T{:.1f}K_AE{:.1f}_AHratio{:.2f}_coupl{:.1f}_dt{:.2f}_tmem{:.1f}.ptr".format(self.temperature,self.ae,self.ah_ratio,self.phonon_factor,self.dt,self.t_mem)
        self.full_names = [self.pt_name+"_initial",self.pt_name+"_initial_0", self.pt_name+"_repeated", self.pt_name+"_repeated_0"]
        self.options = dict({"gamma_e": self.gamma_e,"dt": self.dt,"phonons": self.phonons, "temp_dir": temp_dir, "pt_file": self.pt_name})  # "factor_ah": ah_ratio, "ae": self.ae,
        if os.path.exists(self.full_names[0]):
            print("Warning: pt files already exist")

    def delete_pt_files(self):
        for name in self.full_names:
            if os.path.exists(name):
                os.remove(name)

    def get_J_omega(self, plot=False):
        """
        Returns the spectral density J(omega) of the environment for the given parameters
        """
        # some pulse, doesnt really matter
        p = ChirpedPulse(4, 0, t0=20)
        tls(0,40,p, dt=self.dt, prepare_only=True, phonons=True, ae=self.ae, temperature=self.temperature, verbose=False, lindblad=True, temp_dir='/mnt/temp_data/', J_to_file="J_omega.dat", factor_ah=self.ah_ratio)
        data = np.loadtxt("J_omega.dat")
        omega = data[:,0]
        J = data[:,1]
        max_omega = self.areas/np.sqrt(2*np.pi*self.tau**2)
        if plot: 
            _J = np.zeros([len(omega), len(self.areas)])
            for i in range(len(self.areas)):
                _J[:,i] = J
            plt.pcolormesh(self.areas,omega,_J,cmap="Greens")
            plt.plot(self.areas,2*np.pi*max_omega, label='pulse peak Rabi frequency')
            plt.legend()
            plt.xlabel("pulse area / pi")
            plt.ylabel("omega (1/s)")
            plt.colorbar()
            plt.savefig("J_omega.png")
        return omega, J
    
    def generate_pt(self):
        """
        generates process tensors for environment with given parameters
        """
        # some pulse, doesnt really matter
        p1 = ChirpedPulse(tau_0=self.tau, e_start=0, alpha=0, e0=1, polar_x=1.0, t0=4*self.tau)
        tls(0,8*self.tau,p1,dt=self.dt ,t_mem=self.t_mem, lindblad=False, phonons=True, factor_ah=self.ah_ratio, ae=self.ae,temperature=self.temperature,prepare_only=False, phonon_factor=self.phonon_factor, pt_file=self.pt_name, J_file=self.J_from_file)
        # change permission of pt files to read-only
        for name in self.full_names:
            subprocess.run(["chmod", "444", name])
        time.sleep(1)
        return

    def calc_timedynamics(self, tau, area, path="", save=False, plot_pulse=False, detuning=0, tend=None, plot=False, plotlims=None, lindblad=True, carve_pulse=False, pulse_args={"width_t": 4, "central_f": 0}, filter_width=0.14):
        """
        calculates the time dynamics for a given pulse area and tau
        saves the plot as path+timedynamics_<tau>ps_<area>pi.png
        and the data as path+timedynamics_<tau>ps_<area>pi.csv
        """
        p1 = ChirpedPulse(tau_0=tau, e_start=detuning, alpha=0, e0=area, polar_x=1.0, t0=4*tau)
        if tend is None:
            tend = np.round(10/self.gamma_e)+100

        # first generate process tensors
        if self.phonons and not os.path.exists(self.pt_name+"_initial"):
            self.generate_pt()
        # time from 0 to 10/gamma_e to catch the whole decay process
        if carve_pulse:
            pulse = pg.PulseGenerator(0,np.round(10/self.gamma_e),0.02)
            pulse.add_gaussian_time(t0=100, sig_or_fwhm='fwhm', field_or_intesity='int',area_time=area,**pulse_args)
            pulse.add_filter_double_erf(central_f=0,width_f=filter_width,rise_f=0.01)
            pulse.apply_frequency_filter()
            pulse_file, _ = pulse.generate_pulsefiles(suffix="timedynamics",temp_dir=self.options["temp_dir"])
            t,g,x,pgx,pxg = tls(0,tend,p1,lindblad=lindblad, pulse_file=pulse_file, **self.options)
            if plot_pulse:
                pulse.plot_pulses(t_0=100,t_end=400,frequ_0=-0.5,frequ_end=0.5,save_name=path+"pulse_{:.2f}ps_{:.2f}pi".format(tau,area),save=True)
        else: 
            t,g,x,pgx,pxg = tls(0,tend,p1,lindblad=lindblad, **self.options)
        if plot:
            plt.clf()
            plt.plot(t.real,np.real(x),label="x")
            plt.plot(t.real,np.abs(pgx),label="|p_gx|")
            if plotlims is not None:
                plt.xlim(plotlims[0],plotlims[1])
            plt.xlabel("time (ps)")
            plt.ylabel("population")
            plt.legend()
            plt.savefig(path+"timedynamics_{:.2f}ps_{:.2f}pi.png".format(tau,area))
            plt.clf()
        if save:
            export_csv(path+"timedynamics_{:.2f}ps_{:.2f}pi.csv".format(tau,area), t.real, x.real)
        return t.real,g,x,pgx,pxg
    
    def get_rabi_rotations(self,detuning=0, integrate=True, plot=False, delete_pt=True, path="", workers=15, carve_pulse=False, pulse_args={"width_t": 4, "central_f": 0}, filter_width=0.14, rise_f=0.01, exp_data=None, plot_dynamic=False):
        """
        returns the rabi rotations for the given parameters
        integrate: if True, the rabi rotations are calculated by integrating the population of the excited state over time
        plot: None or string, if not None, the rabi rotations are plotted and saved as a png with the given name
        """
        filename = path
        filename += "rabi_"
        if carve_pulse:
            # pulse width is different if pulse is carved
            filename += "carve_{:.2f}ps_{:.3f}nm_".format(pulse_args["width_t"],filter_width)
        if self.phonons:
            filename += "{:.1f}K_tau_{:.1f}ps_ae_{:.1f}_ah_{:.2f}_coupl_{:.1f}".format(self.temperature,self.tau,self.ae,self.ah_ratio,self.phonon_factor)

        def plot_data(areas, results, exp_data=None):
            # plotting of the results, is used with data that is already present or calculated
            plt.clf()
            plt.plot(areas, results)
            if exp_data is not None:
                exp_x = exp_data[0]
                exp_y = exp_data[1]
                exp_offset = exp_data[2]
                exp_y = np.max(results)*exp_y/np.max(exp_y) + exp_offset
                plt.plot(exp_x,exp_y,label="Experiment")
                plt.legend()
            plt.xlabel("pulse area / pi")
            plt.ylabel("Counts")
            if self.phonons:
                plt.title("T={:.1f}K, tau={:.1f}ps, ae={:.1f}, ah_ratio={:.2f}, coupl={:.1f}".format(self.temperature,self.tau,self.ae,self.ah_ratio,self.phonon_factor))
                plt.savefig(filename+".png")
            else:
                plt.title("tau={:.1f}ps".format(self.tau))
                plt.savefig(path+"rabi.png")
        
        # check if data already exists
        if os.path.exists(filename+".csv"):
            data = np.loadtxt(filename+".csv", delimiter=",")
            areas = data[:,0]
            results = data[:,1]
            if plot:
                plot_data(areas, results, exp_data=exp_data)
            return areas, results
        
        # first generate process tensors
        if self.phonons and not os.path.exists(self.pt_name+"_initial"):
            self.generate_pt()
        results = np.zeros_like(self.areas)
        # calculate rabi rotations
        futures = []
        pulse_file = None
        pulse_files = []
        t_end_add = 0
        with tqdm.tqdm(total=len(self.areas),leave=None) as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for i in range(len(self.areas)):
                    p1 = ChirpedPulse(tau_0=self.tau, e_start=detuning, alpha=0, e0=self.areas[i], polar_x=1.0, t0=4*self.tau)
                    if carve_pulse:
                        pulse = pg.PulseGenerator(0,np.round(10/self.gamma_e),0.02)
                        t_end_add = 400
                        pulse.add_gaussian_time(t0=200, sig_or_fwhm='fwhm', field_or_intesity='int',area_time=self.areas[i],**pulse_args)
                        pulse.add_filter_double_erf(central_f=0,width_f=filter_width,rise_f=rise_f)
                        pulse.apply_frequency_filter()
                        pulse_file, _ = pulse.generate_pulsefiles(suffix=str(i),temp_dir=self.options["temp_dir"])
                        pulse_files.append(pulse_file)
                        # pulse area changes after filtering
                        self.areas[i] = np.sqrt(pulse.pulse_power) # square root of power that is left after filtering
                        # print("area after filtering: ", self.areas[i])
                        if plot_dynamic:
                            pulse.plot_pulses(t_0=0,t_end=400,frequ_0=-0.5,frequ_end=0.5,save_name=path+"pulse_{:.2f}ps_{:.2f}pi".format(self.tau,self.areas[i]),save=True)
                    if integrate:
                        # time from 0 to 10/gamma_e to catch the whole decay process

                        _e = executor.submit(tls,0,np.round(11/self.gamma_e)+t_end_add,p1,lindblad=True, suffix=i, pulse_file=pulse_file, **self.options)
                    else: 
                        # use last time step as result otherwise
                        _e = executor.submit(tls,0,8*self.tau+t_end_add,p1,lindblad=False, suffix=i, pulse_file=pulse_file, **self.options)
                    _e.add_done_callback(lambda p: pbar.update(1))
                    futures.append(_e)
            wait(futures)
            if carve_pulse:
                for file in pulse_files:
                    os.remove(file)
        for i in range(len(self.areas)):
            t,g,x,pgx,pxg = futures[i].result()
            path_dynamics = path+"dynamics/"
            if not os.path.exists(path_dynamics):
                os.makedirs(path_dynamics)
            if plot_dynamic:
                plt.clf()
                # plt.xlim(0,200)
                plt.plot(t.real,np.real(x),label="x")
                plt.xlabel("time (ps)")
                plt.ylabel("population")
                plt.legend()
                plt.savefig(path_dynamics+"timedynamics_{:.2f}ps_{:.2f}pi.png".format(self.tau,self.areas[i]))
                export_csv(path_dynamics+"timedynamics_{:.2f}ps_{:.2f}pi.csv".format(self.tau,self.areas[i]), t.real, x.real)
            if integrate:
                results[i] = self.gamma_e*np.trapz(np.real(x),np.real(t))
            else:
                results[i] = np.real(x[-1])
        #if self.phonons:
        export_csv(filename+".csv", self.areas, results)
        #else:
        #    export_csv(path+"rabi.csv", self.areas, results)
        if plot:
            plot_data(self.areas, results, exp_data=exp_data)
        if delete_pt:
            self.delete_pt_files()
        return self.areas, results
