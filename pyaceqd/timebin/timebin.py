import numpy as np
from pyaceqd.tools import export_csv
import os

class TimeBin():
    def __init__(self, system, *pulses, dt=0.02, tb=800, simple_exp=True, gaussian_t=None, verbose=False, workers=15, t_simul=None, options={}) -> None:
        self.system = system  # system that is used for the simulation
        self.dt = dt  # timestep during simulation
        self.options = dict(options)
        self.options["dt"] = dt  # also save it in the options dict
        self.tb = tb  # timebin width
        self.simple_exp = simple_exp  # use exponential timestepping
        self.gaussian_t = gaussian_t  # use gaussian timestepping during pulse
        self.pulses = pulses
        self.workers = workers  # number of threads spawned by ThreadPoolExecutor
        try:
            self.temp_dir = options["temp_dir"]
        except KeyError:
            print("temp_dir not included in options, setting to /mnt/temp_data/")
            self.options["temp_dir"] = "/mnt/temp_data/"
            self.temp_dir = self.options["temp_dir"]
        if not "pulse_file_x" in self.options or not "pulse_file_y" in self.options or self.options["pulse_file_x"] is None and self.options["pulse_file_y"] is None:
            self.prepare_pulsefile(verbose=verbose, t_simul=t_simul)
            self.options["pulse_file_x"] = self.pulse_file_x  # put pulse files in options dict
            self.options["pulse_file_y"] = self.pulse_file_y
        else:
            self.pulse_file_x = self.options["pulse_file_x"]
            self.pulse_file_y = self.options["pulse_file_y"]

    def prepare_pulsefile(self, verbose=False, t_simul=None):
        # 2*tb is the maximum simulation length, 0 is the start of the simulation
        t_end = 2.1*self.tb
        if t_simul is not None:
            t_end = t_simul
        _t_pulse = np.arange(0,t_end,step=self.dt/5)  # notice that for usual propagation, dt/10 is used
        # different polarizations
        self.pulse_file_x = self.temp_dir + "timebin_pulse_x_{}.dat".format(id(self))  # add object id, otherwise sometimes the wrong file is used
        self.pulse_file_y = self.temp_dir + "timebin_pulse_y_{}.dat".format(id(self))  # probably because the destructor is called after the next object is created
        pulse_x = np.zeros_like(_t_pulse, dtype=complex)
        pulse_y = np.zeros_like(_t_pulse, dtype=complex)
        for _p in self.pulses:
            pulse_x = pulse_x + _p.polar_x*_p.get_total(_t_pulse)
            pulse_y = pulse_y + _p.polar_y*_p.get_total(_t_pulse)
        export_csv(self.pulse_file_x, _t_pulse, pulse_x.real, pulse_x.imag, precision=8, delimit=' ', verbose=verbose)
        export_csv(self.pulse_file_y, _t_pulse, pulse_y.real, pulse_y.imag, precision=8, delimit=' ', verbose=verbose)

    def prepare_puslefile_tls(self, verbose=False):
        """
        prepare pulse files for use with time-local dynamical maps
        """
        _t_pulse1 = np.arange(0,self.tb,step=self.dt/5)  # notice that for usual propagation, dt/10 is used
        _t_pulse2 = np.arange(self.tb,2*self.tb,step=self.dt/5)  # notice that for usual propagation, dt/10 is used
        pulses_tb1 = []
        pulses_tb2 = []
        for __pulse in self.pulses:
            if __pulse.t0 < self.tb:
                pulses_tb1.append(__pulse)
            else:
                pulses_tb2.append(__pulse)
        
        self.pulse_file_x1 = self.temp_dir + "timebin_pulse_x_tb1_{}.dat".format(id(self))
        self.pulse_file_y1 = self.temp_dir + "timebin_pulse_y_tb1_{}.dat".format(id(self)) 

        self.pulse_file_x2 = self.temp_dir + "timebin_pulse_x_tb2_{}.dat".format(id(self))
        self.pulse_file_y2 = self.temp_dir + "timebin_pulse_y_tb2_{}.dat".format(id(self)) 

        pulse_x1 = np.zeros_like(_t_pulse1, dtype=complex)
        pulse_y1 = np.zeros_like(_t_pulse1, dtype=complex)

        pulse_x2 = np.zeros_like(_t_pulse2, dtype=complex)
        pulse_y2 = np.zeros_like(_t_pulse2, dtype=complex)
        
        for _p in pulses_tb1:
            pulse_x1 = pulse_x1 + _p.polar_x*_p.get_total(_t_pulse1)
            pulse_y1 = pulse_y1 + _p.polar_y*_p.get_total(_t_pulse1)
        for _p in pulses_tb2:
            pulse_x2 = pulse_x2 + _p.polar_x*_p.get_total(_t_pulse2)
            pulse_y2 = pulse_y2 + _p.polar_y*_p.get_total(_t_pulse2)
        export_csv(self.pulse_file_x1, _t_pulse1, pulse_x1.real, pulse_x1.imag, precision=8, delimit=' ', verbose=verbose)
        export_csv(self.pulse_file_y1, _t_pulse1, pulse_y1.real, pulse_y1.imag, precision=8, delimit=' ', verbose=verbose)
        # shift by tb: we use them for a porpagation starting at t=0
        # we do it this way to preserve the phase of the pulse
        export_csv(self.pulse_file_x2, _t_pulse2-self.tb, pulse_x2.real, pulse_x2.imag, precision=8, delimit=' ', verbose=verbose)
        export_csv(self.pulse_file_y2, _t_pulse2-self.tb, pulse_y2.real, pulse_y2.imag, precision=8, delimit=' ', verbose=verbose)


    def __del__(self):
        os.remove(self.pulse_file_x)
        os.remove(self.pulse_file_y)
        try:
            os.remove(self.pulse_file_x1)
            os.remove(self.pulse_file_y1)
            os.remove(self.pulse_file_x2)
            os.remove(self.pulse_file_y2)
        except AttributeError:
            pass  # files not created
