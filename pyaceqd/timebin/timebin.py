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
        self.prepare_pulsefile(verbose=verbose, t_simul=t_simul)
        self.options["pulse_file_x"] = self.pulse_file_x  # put pulse files in options dict
        self.options["pulse_file_y"] = self.pulse_file_y

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

    def __del__(self):
        os.remove(self.pulse_file_x)
        os.remove(self.pulse_file_y)
