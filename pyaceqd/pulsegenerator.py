import numpy as np
import matplotlib.pyplot as plt
import pyaceqd.pulses as pulses
import math as math
from pyaceqd.tools import export_csv
hbar = 0.6582173  # meV*ps

class PulseGenerator:
    def __init__(self, t0, tend, dt,central_wavelgth = 800) -> None: 
        # central_wavelength should match with rotating frame
        # Unit of time is ps 
        # central_f is the shift from the rotationg frame frequency
        # Fouierspace (Energy) is calculatet in THz. Units are transformed to THz -> 'nm' are expected ~linear around central wavelength (carefull with high numbers) 
        self.t0 = t0
        self.tend = tend
        self.dt = dt
        self.central_wavelength = central_wavelgth

        self.time = np.arange(t0,tend,dt)
        self.frequencies = -np.fft.fftshift(np.fft.fftfreq(len(self.time),d=dt))
        self.energies = 2*np.pi*hbar*self.frequencies
        self.temporal_representation_x = np.zeros_like(self.time, dtype=complex)
        self.temporal_representation_y = np.zeros_like(self.time, dtype=complex)
        self.frequency_representation_x = np.zeros_like(self.time, dtype=complex)
        self.frequency_representation_y = np.zeros_like(self.time, dtype=complex)

        self.frequency_filter_x = np.zeros_like(self.time, dtype=complex)
        self.frequency_filter_y = np.zeros_like(self.time, dtype=complex)
        self.temporal_filter_x = np.zeros_like(self.time, dtype=complex)
        self.temporal_filter_y = np.zeros_like(self.time, dtype=complex)
        

    ### Pulse building functions 
        # polar_x is polarisation of pulse [polar_y = sqrt(1-polar_x^2)]
    def add_gaussian_time(self, width_t, central_f = 0, alpha=0, t0=0, area_time=1, polar_x=1, phase=0, field_or_intesity = 'field',sig_or_fwhm = 'sig',unit = 'Hz'):
        # Gaussian pulse in time 
        # sig_or_fwhm expects either sigma of fwhm of the gaussian
        # field_or_intensity can be used if the intesity of a pulse is measured
        # area_time = Transformlimited pulse area in time 
        # alpha is the chirp parameter in ps^2 
        central_f = self._Units(central_f,unit)
        width_t = np.abs(self._sig_fwhm(field_or_intesity,sig_or_fwhm,width_t))

        central_f = central_f*hbar*2*np.pi
      
        pulse = pulses.ChirpedPulse(width_t, central_f, alpha, t0, area_time, polar_x, phase)
        pulse_x = pulse.get_total(self.time) * pulse.polar_x
        pulse_y = pulse.get_total(self.time) * pulse.polar_y

        self._add_time(pulse_x,pulse_y)
        pass

    def add_gaussian_freq(self, width_f, central_f = 0, area_time = 1, polar_x = 1,field_or_intesity = 'field',sig_or_fwhm = 'sig',phase_taylor=[],shift_time = None, unit = 'Hz'):
        # Gaussian pulse in Fourier space 
        # area_time = Transformlimited pulse area in time 
        # sig_or_fwhm expects either sigma of fwhm of the gaussian
        # field_or_intensity can be used if the intesity of a pulse is measured
        # phases (chirps) are handled vie phase_taylor in units ps^n e.g [pi,0,20] -> [(no unit),ps, ps^2]
        central_f = self._Units(central_f,unit)
        width_f = np.abs(self._Units(width_f,unit))

        width_f = self._sig_fwhm(field_or_intesity,sig_or_fwhm,width_f)
        polar_y = np.sqrt(1-polar_x**2) 
        pulse = 1/self.dt*area_time*np.exp(-(self.frequencies-central_f)**2/(2*width_f**2))*np.exp(1j*self._Taylor(self.frequencies*2*np.pi,central_f*2*np.pi,coefficients=phase_taylor))
        pulse *= np.exp(1j*2*np.pi*self.frequencies*shift_time)
        pulse_x = pulse*polar_x
        pulse_y = pulse*polar_y
        self._add_spectral(pulse_x,pulse_y)
        pass

    def add_rectangle_frequ(self,central_f, width_f, hight,phase_taylor=[], polar_x = 1,shift_time = None, unit = 'Hz'):
        # sqare pulse in Fourier space 
        central_f = self._Units(central_f,unit)
        width_f = np.abs(self._Units(width_f,unit))
        polar_y = np.sqrt(1-polar_x**2)

        pulse = np.zeros_like(self.frequencies,dtype=complex)
        pulse[np.abs(self.frequencies-central_f)<=width_f/2] = hight
        pulse *= np.exp(1j*self._Taylor(self.frequencies*2*np.pi,central_f*2*np.pi,coefficients=phase_taylor))
        pulse *= np.exp(1j*2*np.pi*self.frequencies*shift_time)
        pulse_x = pulse*polar_x
        pulse_y = pulse*polar_y
        self._add_spectral(pulse_x,pulse_y)
        pass

    def _add_time(self, pulse_x_time, pulse_y_time):
        # internal function to add pulses defined in time
        self.temporal_representation_x += pulse_x_time
        self.temporal_representation_y += pulse_y_time

        self.frequency_representation_x += np.fft.fftshift(np.fft.fft(pulse_x_time))
        self.frequency_representation_y += np.fft.fftshift(np.fft.fft(pulse_y_time))
        pass

    def _add_spectral(self, pulse_x_freq, pulse_y_freq):
        # internal function to add pulses in Fourier space
        self.frequency_representation_x += pulse_x_freq
        self.frequency_representation_y += pulse_y_freq

        self.temporal_representation_x += np.fft.ifft(np.fft.ifftshift(pulse_x_freq))
        self.temporal_representation_y += np.fft.ifft(np.fft.ifftshift(pulse_y_freq))
        pass

    ### Filter functions
        # Filters that can be applied to Fourier space
        # filters have a transmission that can be invertet (1- transmission) bu setting invert = True
        # different merging techniques can be used merging = 'x' x: + -> adding filters; * -> multiplying filters; m -> Overlaying filters
        # Filters can be applied to either('x' or 'y') or both ('b') pulse polarisations 
    def add_filter_rectangle(self, central_f, width_f, transmission = 1 ,polarisation = 'b', invert = False,merging = '+', unit = 'Hz'):
        # Square filter
        central_f = self._Units(central_f,unit)
        width_f = np.abs(self._Units(width_f,unit))

        filter = np.zeros_like(self.frequencies,dtype=complex)
        filter[np.abs(self.frequencies-central_f)<=width_f/2] = transmission

        if invert:
            filter = (1-filter)

        self._add_filter(filter,polarisation,merging=merging)
        pass

    def add_filter_gaussian(self, central_f, width_f, transmission = 1 ,super_gauss = 1,polarisation = 'b',field_int = 'field',sig_fwhm = 'sig', invert = False,merging = '+',unit = 'Hz'):
        #gaussian filter
        #super_gauss allows for super gaussian functions 
        central_f = self._Units(central_f,unit)
        width_f = np.abs(self._Units(width_f,unit))
        tau = self._sig_fwhm(field_int,sig_fwhm,width_f)

        gauss = np.exp(-((self.frequencies-central_f)**2/(2*tau**2))**super_gauss)*transmission

        if invert:
            gauss = 1- gauss

        self._add_filter(gauss,polarisation,merging=merging)
        pass

    def add_filter_sigmoid(self,central_f,width_f,rise_f,transmission=1,polarisation = 'b',invert = False,merging = '+',unit = 'Hz'):
        # double sigmoid filter
        central_f = self._Units(central_f,unit)
        width_f = np.abs(self._Units(width_f,unit))
        rise_f = np.abs(self._Units(rise_f,unit))

        sigm = self._sigmoid(self.frequencies,central_f,width_f,rise_f)
        sigm = sigm/np.max(sigm)*transmission
        if invert:
            sigm = 1-sigm
        self._add_filter(sigm,polarisation,merging)

    def add_phase_filter(self,central_f = 0, phase_taylor=[], polarisation = 'b',unit = 'Hz'):
        # phase filter via Taylor expansion around central_f
        central_f = self._Units(central_f,unit)
        
        phase = np.exp(1j*self._Taylor(self.frequencies*2*np.pi,central_f*2*np.pi,coefficients=phase_taylor))
        self._add_filter(phase,pol=polarisation,merging='*')

        pass

    def apply_frequency_filter(self,pol = 'b'):
        # applies the filter to the pulse 
        if pol.lower()[0] == 'b' or pol.lower()[0] == 'x':
            self.frequency_representation_x *= self.frequency_filter_x
            self.temporal_representation_x = np.fft.ifft(np.fft.ifftshift(self.frequency_representation_x))
        if pol.lower()[0] == 'b' or pol.lower()[0] == 'y':
            self.frequency_representation_y *= self.frequency_filter_y
            self.temporal_representation_y = np.fft.ifft(np.fft.ifftshift(self.frequency_representation_y))
        pass

    def _add_filter(self,filter,pol='both',merging = '+'):
        # internal function for constructiong filters
        if pol.lower()[0] == 'b' or pol.lower()[0] == 'x':
            if merging == '+':
                self.frequency_filter_x += filter
            elif merging == '*':
                self.frequency_filter_x *= filter
            elif merging.lower()[0] == 'm': 
                for i, value in enumerate(self.frequency_filter_x):
                    self.frequency_filter_x[i] = np.max([value,filter[i]])
        if pol.lower()[0] == 'b' or pol.lower()[0] == 'y':
            if merging == '+':
                self.frequency_filter_y += filter
            elif merging == '*':
                self.frequency_filter_y *= filter
            elif merging.lower()[0] == 'm': 
                for i, value in enumerate(self.frequency_filter_y):
                    self.frequency_filter_y[i] = np.max([value,filter[i]])
        if np.any(np.logical_or(self.frequency_filter_x > 1, self.frequency_filter_y > 1)): 
            print('WARNING: Transmission in filter > 1. Capped to 1.')
            self.frequency_filter_x[self.frequency_filter_x > 1] = 1 
            self.frequency_filter_y[self.frequency_filter_y > 1] = 1 
    ### SLM
    def apply_SLM(self, pixelwidth , pixel_center = 0, N_pixel = 128, unit = 'Hz', kind = 'rectangle',polarisation = 'both', SLM = 'amp',generate_mask = False, calibration_file = None,cal_type = 'r', save_dir = '', mask_name = 'mask_output',loop = 0):
        # applys a discretisation to the filter, simulating pixels of an (for now) amplitude SLM
        # N_pixel = # of pixels/discretisation steps
        # pixel_center is the position of the central pixel / for even N_pixel the central position 
        # setting generate_mask = True generates a driving mask, given that a callibration_file is specified
        # callibration_file is in retardance (transmission) | voltage and should (must?) be non redundant -> type can be set by cal_type = 'r' or 't' 
        pixel_center = self._Units(pixel_center,unit)
        pixelwidth = abs(self._Units(pixelwidth,unit))

        start_f = pixel_center - N_pixel/2*pixelwidth
        end_f = pixel_center + N_pixel/2*pixelwidth

        pixel_transmission = []
        if kind.lower()[0] == 'r':
            if polarisation.lower()[0] == 'b' or polarisation.lower()[0] == 'x':
                self.frequency_filter_x[self.frequencies < start_f] = 0 
                self.frequency_filter_x[self.frequencies >= end_f] = 0 
                pass
            if polarisation.lower()[0] == 'b' or polarisation.lower()[0] == 'y':
                self.frequency_filter_y[self.frequencies < start_f] = 0 
                self.frequency_filter_y[self.frequencies >= end_f] = 0 

            for i in range(N_pixel):
                L_slice = np.where((self.frequencies >= (start_f + i*pixelwidth)) & (self.frequencies < (start_f +  (i+1)*pixelwidth)))
                
                if polarisation.lower()[0] == 'b' or polarisation.lower()[0] == 'x':
                    cur_slice = self.frequency_filter_x[L_slice]
                    self.frequency_filter_x[L_slice] = np.mean(np.abs(cur_slice))*np.exp(1j*np.angle(cur_slice))
             
                if polarisation.lower()[0] == 'b' or polarisation.lower()[0] == 'y':
                    cur_slice = self.frequency_filter_y[L_slice]
                    self.frequency_filter_y[L_slice] = np.mean(np.abs(cur_slice))*np.exp(1j*np.angle(cur_slice))
                pixel_transmission.append(np.mean(np.abs(cur_slice)))

            if generate_mask: 
                mask_name = save_dir + mask_name+str(loop)+'.txt'
                if calibration_file is None: 
                    print('No calibration file!')
                    return
                else: 
                    voltage, transmission = self._calibrate_SLM(calibration_file,cal_type)
                    mask_voltage = np.interp(pixel_transmission,transmission,voltage)
                    with open(mask_name, "w") as txt_file:
                        for line in list(mask_voltage):
                            txt_file.write(str(line) + "\n")
                    txt_file.close()

    def _calibrate_SLM(self,calib_file,type):
        # internal function for reading a calibration file 
        # format should be 2 coulums, no header : retardance (transmission) | voltage
        file = calib_file
        f=open(file,"r")
        lines=f.readlines()
        retardance = []
        voltage = []
        for x in lines:
            retardance.append(float(x.split('\t')[0]))
            voltage.append(float(x.split('\t')[1]))
        f.close()
        retardance = np.array(retardance)
        voltage = np.array(voltage)
        if type.lower()[0] == 'r':
            transmission = np.sin(retardance)**2
        elif type.lower()[0] == 't':
            transmission = retardance
        return voltage, transmission

    ### Additional functions

    def _Units(self,input,unit = 'Hz'): 
        # transforming nm and meV to THz
        if unit.lower()[0] == 'm': 
            output = input/(2*np.pi*hbar) 
        elif unit.lower()[0] == 'n': 
            central_f = 299792.458/self.central_wavelength
            input_f = 299792.458/(self.central_wavelength+input)
            output = central_f-input_f
            output = - output
        else:
            output = input
        return output


    def _Taylor(self,frequency,frequency_0=0,coefficients = []):
        # a Taylor expansion
        phase = np.zeros_like(frequency)
        for n, coeff in enumerate(coefficients):
            phase += coeff/math.factorial(n)*(frequency-frequency_0)**n
        return phase 

    def _sig_fwhm(self,field_int,sig_fwhm,width):
        # transforming fwhm -> sigma for gaussian pulses
        # field (intesity) can be set via field_int = 'f' ('i')
        if field_int.lower()[0] == 'f':
            if sig_fwhm.lower()[0] == 's':
                tau_0 = width
            elif sig_fwhm.lower()[0] == 'f':
                tau_0 =  width / (2 * np.sqrt(np.log(2) * 2))
        elif field_int.lower()[0] == 'i':
            if sig_fwhm.lower()[0] == 's':
                tau_0 = np.sqrt(2)*width
            elif sig_fwhm.lower()[0] == 'f':
                tau_0 = width/(2*np.sqrt(np.log(2)))
        return tau_0

    def _sigmoid(self,x,center,width,rise): 
        c1 = center-width/2
        c2 = center+width/2
        sigm1 = 1/(1+np.exp(-(x-c1)/rise))
        sigm2 = 1/(1+np.exp(-(c2-x)/rise))
        return sigm1*sigm2
    
    ### plotting functions
        # limits can be set in time (t_0,t_end) and in Fourier space (frequ_0, frequ_end)
        # polarisation ('x' , 'y' or 'both') are set via plot_pol
        # plotting in different domains ('Hz' -> THz; 'meV' - > meV; 'nm' -> nm) can be controlled via domain = '' 
        # save = True  saves figures 
    def plot_filter(self,t_0 = None,t_end = None,frequ_0 = None, frequ_end = None ,plot_pol = 'both',domain = 'Hz',save = False, save_name = 'fig'):
        # plotting the current Fourier space filter function 
        if domain == 'meV':
            self.plot_domain = self.energies
            self.domain = domain
        elif domain == 'Hz': 
            self.plot_domain = self.frequencies
            self.domain = 'THz'
        elif domain == 'nm': 
            central_f = 299792.458/self.central_wavelength
            self.plot_domain = 299792.458/(central_f + self.frequencies)
            self.domain = 'nm'

        ### setting default limits 
        if t_0 is None: 
            t_0 = np.min(self.time)
        if t_end is None:
            t_end = np.max(self.time)
        if frequ_0 is None: 
            frequ_0 = np.min(self.plot_domain)
        if frequ_end is None:
            frequ_end = np.max(self.plot_domain)

        plt.figure()
        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'x':
            plt.plot(self.plot_domain, np.abs(self.frequency_filter_x),'b-', label="x_envel")

        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'y':
            plt.plot(self.plot_domain, np.abs(self.frequency_filter_y),'r-', label="y_envel")

        plt.xlim(frequ_0,frequ_end)
        plt.xlabel(self.domain)
        plt.grid()
        plt.legend() 
        plt.ylabel('Transmission')
        plt.title('Filter frequency')
        if save:
            plt.savefig(save_name+'_frequ_filter.png')
        

    def plot_pulses(self,t_0 = None,t_end = None,frequ_0 = None, frequ_end = None ,plot_pol = 'both',domain = 'Hz',save = False,save_name = 'fig_'):
        #plotting the current pulse in both time (abs() and real() are plotted)and Fourier space (only abs() is plotted)
        if domain == 'meV':
            self.plot_domain = self.energies
            self.domain = domain
        elif domain == 'Hz': 
            self.plot_domain = self.frequencies
            self.domain = 'THz'
        elif domain == 'nm': 
            central_f = 299792.458/self.central_wavelength
            self.plot_domain = 299792.458/(central_f + self.frequencies)
            self.domain = 'nm'

        ### setting default limits 
        if t_0 is None: 
            t_0 = np.min(self.time)
        if t_end is None:
            t_end = np.max(self.time)
        if frequ_0 is None: 
            frequ_0 = np.min(self.plot_domain)
        if frequ_end is None:
            frequ_end = np.max(self.plot_domain)

        plt.figure()
        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'x':
            plt.plot(self.time, np.abs(self.temporal_representation_x),'b-', label="x_envel")
            plt.plot(self.time, np.real(self.temporal_representation_x),'b:', label="x_field")

        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'y':
            plt.plot(self.time, np.abs(self.temporal_representation_y),'r-', label="y_envel")
            plt.plot(self.time, np.real(self.temporal_representation_y),'r:', label="y_field")
        plt.xlim(t_0,t_end)
        plt.xlabel('time / ps')
        plt.grid()
        plt.legend() 
        plt.title('Pulses time')
        if save:
            plt.savefig(save_name+"_time.png")

        plt.figure()
        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'x':
            plt.plot(self.plot_domain, np.abs(self.frequency_representation_x),'b-', label="x_envel")

        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'y':
            plt.plot(self.plot_domain, np.abs(self.frequency_representation_y),'r-', label="y_envel")
        plt.xlim(frequ_0,frequ_end)
        plt.xlabel(self.domain)
        plt.grid()
        plt.legend() 
        plt.title('Pulses frequency')
        if save:
            plt.savefig(save_name+"_frequ.png")

    def generate_pulsefiles(self, temp_dir = '', file_name = 'pulse_time', loop = ''):
        #Translating the generated pulse for use with the PYACEQD Quantum Dot simulation enviroment 
        pulse_file_x = temp_dir + file_name + str(loop)+'_x.dat' 
        pulse_file_y = temp_dir + file_name + str(loop)+'_y.dat'

        export_csv(pulse_file_x, self.time, np.real(self.temporal_representation_x), np.imag(self.temporal_representation_x), precision=8, delimit=' ')
        export_csv(pulse_file_y, self.time, np.real(self.temporal_representation_y), np.imag(self.temporal_representation_y), precision=8, delimit=' ')
        return pulse_file_x, pulse_file_y


    ### clear functions
    def clear_all(self):
        self.clear_filter()
        self.clear_pulses()

    def clear_filter(self):
        self.frequency_filter_x = np.zeros_like(self.time, dtype=complex)
        self.frequency_filter_y = np.zeros_like(self.time, dtype=complex)
        self.temporal_filter_x = np.zeros_like(self.time, dtype=complex)
        self.temporal_filter_y = np.zeros_like(self.time, dtype=complex)
    
    def clear_pulses(self):
        self.temporal_representation_x = np.zeros_like(self.time, dtype=complex)
        self.temporal_representation_y = np.zeros_like(self.time, dtype=complex)
        self.frequency_representation_x = np.zeros_like(self.time, dtype=complex)
        self.frequency_representation_y = np.zeros_like(self.time, dtype=complex)
