import numpy as np

HBAR = 0.6582173  # meV*ps

class Pulse:
    def __init__(self, tau, e_start, w_gain=0, t0=0, e0=1, phase=0, polar_x=1):
        self.tau = tau  # in ps
        self.e_start = e_start  # in meV
        self.w_start = e_start / HBAR  # in 1 / ps
        self.w_gain = float(w_gain)  #  in 1/ps^2
        self.t0 = t0
        self.e0 = e0
        self.phase = phase
        self.freq = None
        self.phase_ = None
        self.polar_x = polar_x
        self.polar_y = np.sqrt(1-polar_x**2)

    def __repr__(self):
        return "%s(tau=%r, e_start=%r, w_gain=%r, t0=%r, e0=%r)" % (
            self.__class__.__name__, self.tau, self.e_start, self.w_gain, self.t0, self.e0
        )

    def get_envelope(self, t):
        return self.e0 * np.exp(-0.5 * ((t - self.t0) / self.tau) ** 2) / (np.sqrt(2 * np.pi) * self.tau)
    
    def set_frequency(self, f):
        """
        use a lambda function f taking a time t to set the time dependent frequency.
        """
        self.freq = f

    def get_frequency(self, t):
        """
        phidot, i.e. the derivation of the phase,
        is the current frequency
        :return: frequency omega for a given time 
        """
        if self.freq is not None:
            return self.freq(t)
        return self.w_start + self.w_gain * (t - self.t0)

    def get_full_phase(self,t):
        return self.w_start * (t - self.t0) + 0.5*self.w_gain * ((t - self.t0) **2) + self.phase
    
    def get_energies(self):
        """
        get energy diff of +- tau for chirped pulse
        E=hbar*w
        if tau and everything is in ps, output in meV
        """
        low = self.get_frequency(-self.tau)
        high = self.get_frequency(self.tau)
        energy_range = np.abs(high-low)*HBAR  # meV
        return energy_range

    def get_total(self, t):
        return self.get_envelope(t) * np.exp(-1j * self.get_full_phase(t))


class ChirpedPulse(Pulse):
    def __init__(self, tau_0, e_start, alpha=0, t0=0, e0=1*np.pi, polar_x=1):
        self.tau_0 = tau_0
        self.alpha = alpha
        super().__init__(tau=np.sqrt(alpha**2 / tau_0**2 + tau_0**2), e_start=e_start, w_gain=alpha/(alpha**2 + tau_0**4), t0=t0, e0=e0, polar_x=polar_x)
    
    def get_parameters(self):
        """
        returns tau and chirp parameter
        """
        return "tau: {:.4f} ps , a: {:.4f} ps^-2".format(self.tau, self.w_gain)

    def get_envelope(self, t):
        return self.e0 * np.exp(-0.5 * ((t - self.t0) / self.tau) ** 2) / (np.sqrt(2 * np.pi * self.tau * self.tau_0))

    def get_ratio(self):
        """
        returns ratio of pulse area chirped/unchirped: tau / sqrt(tau * tau_0)
        """
        return np.sqrt(self.tau / self.tau_0)
