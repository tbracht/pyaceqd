import numpy as np
from scipy.special import erf
import pyaceqd.constants as constants

hbar = constants.hbar  # meV*ps

class Pulse:
    def __init__(self, tau, detuning, w_gain=0, t_center=0, pulse_area=1, phase=0, polar_x=1, polars=None, polarization=None, interaction_op=None, repeat_tb=None,
                 phase_option="same"):
        self.tau = tau  # in ps
        self.detuning = detuning  # in meV
        self.w_gain = float(w_gain)  # chirp parameter, in 1/ps^2
        self.t_center = t_center
        self.pulse_area = pulse_area
        self.phase = phase
        self.freq = None
        self.phase_ = None  # time-dependent phase function, if set. Otherwise automatically calculated from detuning and w_gain
        self.polarization = polarization
        self.interaction_op = interaction_op
        self.polar_x = polar_x
        self.polar_y = np.sqrt(1-polar_x**2)
        self.repeat_tb = repeat_tb  # if not None, the pulse is repeated every repeat_tb ps
        self.phase_option = phase_option  # for repeated pulses
        if phase_option not in ["same", "random", "continuous"]:
            raise ValueError("phase option must be 'same', 'random' or 'continuous'")
        if self.w_gain != 0 and phase_option == "continuous":
            raise ValueError("Continuous phase is not supported for non-zero chirp/gain.")
        if polars is not None:
            norm = np.sqrt(np.abs(polars[0])**2 + np.abs(polars[1])**2)
            self.polar_x = polars[0]/norm
            self.polar_y = polars[1]/norm

    def __repr__(self):
        return "%s(tau=%r, detuning=%r, w_gain=%r, t_center=%r, pulse_area=%r)" % (
            self.__class__.__name__, self.tau, self.detuning, self.w_gain, self.t_center, self.pulse_area
        )

    def set_repeat_tb(self, tb):
        self.repeat_tb = tb
    
    def get_energy(self):
        return self.detuning, self.w_gain
    
    def set_energy(self, detuning, w_gain):
        self.detuning = detuning
        self.w_gain = w_gain

    def get_tcenter(self):
        return self.t_center

    def set_tcenter(self, t_center):
        self.t_center = t_center

    def get_envelope(self, t):
        return self.pulse_area * np.exp(-0.5 * ((t - self.t_center) / self.tau) ** 2) / (np.sqrt(2 * np.pi) * self.tau)
    
    def get_integral(self, t):
        return self.pulse_area * 0.5 * (1 - erf((self.t_center - t)/(np.sqrt(2)*self.tau)))

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
        w_start = self.detuning / hbar  # in 1 / ps
        return w_start + self.w_gain * (t - self.t_center)

    def set_phase(self, f):
        self.phase_ = f

    def get_full_phase(self,t):
        if self.phase_ is not None:
            return self.phase_(t)
        w_start = self.detuning / hbar  # in 1 / ps
        return w_start * (t - self.t_center) + 0.5*self.w_gain * ((t - self.t_center) **2) + self.phase
    

    def get_energies(self):
        """
        get energy diff of +- tau for chirped pulse
        E=hbar*w
        if tau and everything is in ps, output in meV
        """
        low = self.get_frequency(-self.tau)
        high = self.get_frequency(self.tau)
        energy_range = np.abs(high-low)*hbar  # meV
        return energy_range

    def get_total(self, t):
        if self.repeat_tb is not None:
            t_mod = np.mod(t, self.repeat_tb)
            if self.phase_option == "continuous":
                # if continuous phase, calc phase from unmodulated time.
                phase = self.get_full_phase(t)
            elif self.phase_option == "same":
                # if same, calc from time modulo repeat_tb, i.e., the same for each pulse 
                phase = self.get_full_phase(t_mod)
            elif self.phase_option == "random":
                # else, calculate separately for each "timebin"
                phase_parts = []
                dt = np.abs(t[1] - t[0])
                steps_repeat = int(self.repeat_tb / dt)
                n_parts = int(np.ceil(len(t) / steps_repeat))
                random_phase = 0
                for i in range(n_parts):
                    steps_remaining = len(t) - i*steps_repeat
                    steps_this_part = min(steps_repeat, steps_remaining)
                    phase_parts.append(self.get_full_phase(t[0:steps_this_part]) + random_phase)
                    random_phase = np.random.uniform(0, 2*np.pi)  # new random phase for next part, not for i = 0, but for i = 1 and onwards
                phase = np.concatenate(phase_parts)
            else:
                raise ValueError("Invalid phase option")
            return self.get_envelope(t_mod) * np.exp(-1j * phase)
        return self.get_envelope(t) * np.exp(-1j * self.get_full_phase(t))
    
    def get_total_dict(self, t):
        return {
            "time": t,
            "polarization": self.polarization,
            "total": self.get_total(t),
        }

    def copy(self):
        return Pulse(self.tau, self.detuning, self.w_gain, self.t_center, self.pulse_area, self.phase, self.polar_x, None, self.polarization, self.interaction_op, self.repeat_tb)

    def _base_kwargs(self):
        return dict(
            t_center=self.t_center,
            pulse_area=self.pulse_area,
            phase=self.phase,
            polar_x=self.polar_x,
            polarization=self.polarization,
            interaction_op=self.interaction_op,
            repeat_tb=self.repeat_tb,
            phase_option=self.phase_option
        )

class AsymmetricPulse(Pulse):
    def __init__(self, tau1, tau2, detuning, t_center=0, pulse_area=1, phase=0, polar_x=1, polars=None, **pulse_kwargs):
        self.tau1 = tau1
        self.tau2 = tau2
        super().__init__(tau1, detuning, w_gain=0, t_center=t_center, pulse_area=pulse_area, phase=phase, polar_x=polar_x, polars=polars, **pulse_kwargs)

    def get_envelope(self, t):
        # gaussian with tau1 up to t_center and tau2 after t_center
        t_smaller = t[t <= self.t_center]
        t_larger = t[t > self.t_center]
        e_smaller = self.pulse_area * np.exp(-0.5 * ((t_smaller - self.t_center) / self.tau1) ** 2) / (np.sqrt(2 * np.pi) * self.tau1)
        e_larger = self.pulse_area * np.exp(-0.5 * ((t_larger - self.t_center) / self.tau2) ** 2) / (np.sqrt(2 * np.pi) * self.tau1)  #  divide by tau1 to get a smooth transition
        return np.concatenate((e_smaller, e_larger))
    
    def copy(self):
        return AsymmetricPulse(self.tau1, self.tau2, self.detuning, polars=None, **self._base_kwargs())

class ChirpedPulse(Pulse):
    def __init__(self, tau_0, detuning, alpha=0, t_center=0, pulse_area=1*np.pi, polar_x=1, phase=0, polars=None, polarization=None, interaction_op=None, **pulse_kwargs):
        self.tau_0 = tau_0  # transform-limited pulse duration, in ps. The actual pulse duration is given by tau = sqrt(tau_0^2 + alpha^2 / tau_0^2)
        self.alpha = alpha  # chirp parameter in ps^2
        if self.alpha != 0 and pulse_kwargs.get("continuous_phase", False):
            raise ValueError("Continuous phase is not supported for non-zero chirp.")
        super().__init__(tau=np.sqrt(alpha**2 / tau_0**2 + tau_0**2), detuning=detuning, w_gain=alpha/(alpha**2 + tau_0**4), t_center=t_center, pulse_area=pulse_area, polar_x=polar_x, phase=phase, polars=polars, polarization=polarization,
                         interaction_op=interaction_op, **pulse_kwargs)
    
    def get_parameters(self):
        """
        returns tau and chirp parameter
        """
        return "tau: {:.4f} ps , a: {:.4f} ps^-2".format(self.tau, self.w_gain)

    def get_envelope(self, t):
        return self.pulse_area * np.exp(-0.5 * ((t - self.t_center) / self.tau) ** 2) / (np.sqrt(2 * np.pi * self.tau * self.tau_0))

    def get_integral(self, t):
        return self.pulse_area * 0.5 * np.sqrt(self.tau/self.tau_0) * (1 - erf((self.t_center - t)/(np.sqrt(2)*self.tau)))

    def get_ratio(self):
        """
        returns ratio of pulse area chirped/unchirped: tau / sqrt(tau * tau_0)
        """
        return np.sqrt(self.tau / self.tau_0)
    
    def copy(self):
        return ChirpedPulse(self.tau_0, self.detuning, self.alpha, polars=None, **self._base_kwargs())

class QuenchedChirpedPulse(ChirpedPulse):
    """
    Pulse that is set to zero after t_quench.
    Warning: repeat_tb is not compatible with this pulse, as it will return zero even after repeat_tb if t_quench is reached.
    """
    def __init__(self, tau_0, detuning, alpha=0, t_center=0, pulse_area=1*np.pi, t_quench=0, polar_x=1, phase=0, polars=None, polarization=None, interaction_op=None, **pulse_kwargs):
        self.t_quench = t_quench
        if pulse_kwargs.get("repeat_tb", None) is not None:
            print("WARNING: repeat_tb is not compatible with QuenchedChirpedPulse")
        super().__init__(tau_0, detuning, alpha, t_center, pulse_area, polar_x, phase, polars, polarization, interaction_op, **pulse_kwargs)
    
    def get_envelope(self, t):
        env = super().get_envelope(t)
        env[t >= self.t_quench] = 0
        return env
    
    def get_integral(self, t):
        if t < self.t_quench:
            return super().get_integral(t)
        else:
            return super().get_integral(self.t_quench)

    def get_tquench(self):
        return self.t_quench

    def set_tquench(self, t_quench):
        self.t_quench = t_quench
    
    def copy(self):
        return QuenchedChirpedPulse(self.tau_0, self.detuning, self.alpha, t_quench=self.t_quench, polars=None, **self._base_kwargs())

class PulseTrain:
    """
    pulse train, with pulses separated by delta_t. Each occurence can be constituted of multiple pulses, i.e., for 
    multi-pulse schemes.
    """

    def __init__(self, delta_t, n_pulses, *pulses, t_shift=0):
        self.delta_t = delta_t
        self.n_pulses = n_pulses
        self.pulses = list(pulses)
        self.t_shift = t_shift

    def get_total(self, t):
        field = np.zeros_like(t,dtype=complex)
        for i in range(self.n_pulses):
            for p in self.pulses:
                field += p.get_total(t-self.delta_t*i-self.t_shift)
        return field

    def get_total_xy(self, t):
        field_x = np.zeros_like(t, dtype=complex)
        field_y = np.zeros_like(field_x)
        for i in range(self.n_pulses):
            for p in self.pulses:
                field_x += p.polar_x * p.get_total(t-self.delta_t*i-self.t_shift)
                field_y += p.polar_y * p.get_total(t-self.delta_t*i-self.t_shift)
        return field_x, field_y

class CWLaser(Pulse):
    """
    cw-laser, i.e., it is just on the whole time without any switch-on process
    """

    def __init__(self, pulse_area, detuning=0, polar_x=1, phase=0, polars=None, polarization=None, interaction_op=None, **pulse_kwargs):
        super().__init__(tau=5, detuning=detuning, pulse_area=pulse_area, polar_x=polar_x, polars=polars, phase=phase, polarization=polarization, interaction_op=interaction_op, **pulse_kwargs)

    def get_envelope(self, t):
        return self.pulse_area
    
    def copy(self):
        return CWLaser(self.pulse_area, self.detuning, polars=None, **self._base_kwargs())

class SmoothRectangle(Pulse):
    """
    Rectangular pulse that is switched on/off with a sigmoid shape.

    """

    def __init__(self, tau, detuning, w_gain=0, t_center=0, pulse_area=1, phase=0, alpha_onoff=0.1, polar_x=1, polars=None, polarization=None, interaction_op=None, **pulse_kwargs):
        self.alpha_onoff = alpha_onoff
        self.alpha = 1/alpha_onoff  # switch on/off time
        super().__init__(tau, detuning, w_gain=w_gain, t_center=t_center, pulse_area=pulse_area, phase=phase, polar_x=polar_x, polars=polars, polarization=polarization, interaction_op=interaction_op, **pulse_kwargs)

    def get_envelope_f(self):
        return lambda t: self.pulse_area/( (1+np.exp(-self.alpha*(t+self.tau/2 - self.t_center))) * (1+np.exp(-self.alpha*(-t+self.tau/2 + self.t_center))) )

    def get_envelope(self, t):
        return self.pulse_area/( (1+np.exp(-self.alpha*(t+self.tau/2 - self.t_center))) * (1+np.exp(-self.alpha*(-t+self.tau/2 + self.t_center))) )
    
    def copy(self):
        return SmoothRectangle(self.tau, self.detuning, self.w_gain, alpha_onoff=self.alpha_onoff, polars=None, **self._base_kwargs())
    
class SmoothRectangleGaussian(Pulse):
    """
    Rectangular pulse that is switched on/off with a half-Gaussian shape.
    It is possible to have different widths for the switch-on and switch-off process.
    """

    def __init__(self, tau_constant, detuning, t_center=0, pulse_area=1, phase=0, sigma_on=2, sigma_off=None, polarization=None, interaction_op=None, **pulse_kwargs):
        self.sigma_on = sigma_on
        self.sigma_off = sigma_off if sigma_off is not None else sigma_on
        super().__init__(tau_constant, detuning, t_center=t_center, pulse_area=pulse_area, phase=phase, polarization=polarization, interaction_op=interaction_op, **pulse_kwargs)

    def get_envelope(self, t):
        tau_constant = self.tau
        t_on = self.t_center - tau_constant/2
        t_off = self.t_center + tau_constant/2
        env_on = self.pulse_area * np.exp(-0.5 * ((t - t_on) / self.sigma_on) ** 2)
        env_off = self.pulse_area * np.exp(-0.5 * ((t - t_off) / self.sigma_off) ** 2)
        env_middle = self.pulse_area * np.ones_like(t)
        env = np.where(t < t_on, env_on, np.where(t > t_off, env_off, env_middle))
        return env
