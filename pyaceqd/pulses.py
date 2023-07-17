import numpy as np
from scipy.special import erf

HBAR = 0.6582173  # meV*ps

class Pulse:
    def __init__(self, tau, e_start, w_gain=0, t0=0, e0=1, phase=0, polar_x=1):
        self.tau = tau  # in ps
        self.e_start = e_start  # in meV
        # self.w_start = e_start / HBAR  # in 1 / ps
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
    
    def get_energy(self):
        return self.e_start, self.w_gain
    
    def set_energy(self, e_start, w_gain):
        self.e_start = e_start
        self.w_gain = w_gain

    def get_envelope(self, t):
        return self.e0 * np.exp(-0.5 * ((t - self.t0) / self.tau) ** 2) / (np.sqrt(2 * np.pi) * self.tau)
    
    def get_integral(self, t):
        return self.e0 * 0.5 * (1 - erf((self.t0 - t)/(np.sqrt(2)*self.tau)))

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
        w_start = self.e_start / HBAR  # in 1 / ps
        return w_start + self.w_gain * (t - self.t0)

    def get_full_phase(self,t):
        w_start = self.e_start / HBAR  # in 1 / ps
        return w_start * (t - self.t0) + 0.5*self.w_gain * ((t - self.t0) **2) + self.phase
    
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

    def copy(self):
        return Pulse(self.tau, self.e_start, self.w_gain, self.t0, self.e0, self.phase, self.polar_x)

class ChirpedPulse(Pulse):
    def __init__(self, tau_0, e_start, alpha=0, t0=0, e0=1*np.pi, polar_x=1, phase=0):
        self.tau_0 = tau_0
        self.alpha = alpha
        super().__init__(tau=np.sqrt(alpha**2 / tau_0**2 + tau_0**2), e_start=e_start, w_gain=alpha/(alpha**2 + tau_0**4), t0=t0, e0=e0, polar_x=polar_x, phase=phase)
    
    def get_parameters(self):
        """
        returns tau and chirp parameter
        """
        return "tau: {:.4f} ps , a: {:.4f} ps^-2".format(self.tau, self.w_gain)

    def get_envelope(self, t):
        return self.e0 * np.exp(-0.5 * ((t - self.t0) / self.tau) ** 2) / (np.sqrt(2 * np.pi * self.tau * self.tau_0))

    def get_integral(self, t):
        return self.e0 * 0.5 * np.sqrt(self.tau/self.tau_0) * (1 - erf((self.t0 - t)/(np.sqrt(2)*self.tau)))

    def get_ratio(self):
        """
        returns ratio of pulse area chirped/unchirped: tau / sqrt(tau * tau_0)
        """
        return np.sqrt(self.tau / self.tau_0)
    
    def copy(self):
        return ChirpedPulse(self.tau_0, self.e_start, self.alpha, self.t0, self.e0, self.polar_x, self.phase)


class PulseTrain:
    """
    pulse train, with pulses separated by delta_t. Each occurence can be constituted of multiple pulses, i.e., for 
    multi-pulse schemes.
    """

    def __init__(self, delta_t, n_pulses, *pulses, start=0):
        self.delta_t = delta_t
        self.n_pulses = n_pulses
        self.start = start
        self.pulses = list(pulses)
        self.start = start

    def get_total(self, t):
        field = np.zeros_like(t,dtype=complex)
        for i in range(self.n_pulses):
            for p in self.pulses:
                field += p.get_total(t-self.delta_t*i-self.start)
        return field

    def get_total_xy(self, t):
        field_x = np.zeros_like(t, dtype=complex)
        field_y = np.zeros_like(field_x)
        for i in range(self.n_pulses):
            for p in self.pulses:
                field_x += p.polar_x * p.get_total(t-self.delta_t*i-self.start)
                field_y += p.polar_y * p.get_total(t-self.delta_t*i-self.start)
        return field_x, field_y

class CWLaser(Pulse):
    """
    cw-laser, i.e., it is just on the whole time without any switch-on process
    """

    def __init__(self, e0, e_start=0, polar_x=1):
        super().__init__(tau=5, e_start=e_start, e0=e0, polar_x=polar_x)

    def get_envelope(self, t):
        return self.e0

class SmoothRectangle(Pulse):
    """
    Rectangular pulse that is switched on/off with a sigmoid shape.

    """

    def __init__(self, tau, e_start, w_gain=0, t0=0, e0=1, phase=0, alpha_onoff=0.1, polar_x=1):
        self.alpha_onoff = alpha_onoff
        self.alpha = 1/alpha_onoff  # switch on/off time
        super().__init__(tau, e_start, w_gain=w_gain, t0=t0, e0=e0, phase=phase, polar_x=polar_x)

    def get_envelope_f(self):
        return lambda t: self.e0/( (1+np.exp(-self.alpha*(t+self.tau/2 - self.t0))) * (1+np.exp(-self.alpha*(-t+self.tau/2 + self.t0))) )

    def get_envelope(self, t):
        return self.e0/( (1+np.exp(-self.alpha*(t+self.tau/2 - self.t0))) * (1+np.exp(-self.alpha*(-t+self.tau/2 + self.t0))) )
    
    def copy(self):
        return SmoothRectangle(self.tau, self.e_start, self.w_gain, self.t0, self.e0, self.phase, self.alpha_onoff, self.polar_x)
    