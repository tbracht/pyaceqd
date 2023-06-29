import numpy as np
import matplotlib.pyplot as plt
import pyaceqd.pulses as pulses
import math as math
from pyaceqd.tools import export_csv
from scipy.io import savemat, loadmat
from scipy import integrate
from scipy import interpolate

hbar = 0.6582173  # meV*ps

class PulseGenerator:
    def __init__(self, t0, tend, dt,central_wavelength = 800) -> None: 
        # central_wavelength should match with rotating frame
        # Unit of time is ps 
        # central_f is the shift from the rotationg frame frequency
        # Fouierspace (Energy) is calculatet in THz. Units are transformed to THz -> 'nm' are expected ~linear around central wavelength (carefull with high numbers) 
        self.t0 = t0
        self.tend = tend
        self.dt = dt
        self.central_wavelength = central_wavelength

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
    def add_gaussian_time(self, width_t, central_f = 0, alpha=0, t0=0, area_time=1, polarisation = [1,0], phase=0, field_or_intesity = 'field',sig_or_fwhm = 'sig',unit = 'Hz'):
        # Gaussian pulse in time 
        # sig_or_fwhm expects either sigma of fwhm of the gaussian
        # field_or_intensity can be used if the intesity of a pulse is measured
        # area_time = Transformlimited pulse area in time 
        # alpha is the chirp parameter in ps^2 
        central_f = self._Units(central_f,unit)
        width_t = np.abs(self._sig_fwhm(field_or_intesity,sig_or_fwhm,width_t))

        central_f = central_f*hbar*2*np.pi

        polar_x, polar_y = self._normalise_polarisation(polarisation) 
        pulse = pulses.ChirpedPulse(width_t, central_f, alpha, t0, area_time, polar_x, phase)
        pulse_x = pulse.get_total(self.time) * polar_x
        pulse_y = pulse.get_total(self.time) * polar_y

        self._add_time(pulse_x,pulse_y)
        pass

    def add_gaussian_freq(self, width_f, central_f = 0, area_time = 1, polarisation = [1,0],field_or_intesity = 'field',sig_or_fwhm = 'sig',phase_taylor=[],shift_time = 0, unit = 'Hz'):
        # Gaussian pulse in Fourier space 
        # area_time = Transformlimited pulse area in time 
        # sig_or_fwhm expects either sigma of fwhm of the gaussian
        # field_or_intensity can be used if the intesity of a pulse is measured
        # phases (chirps) are handled vie phase_taylor in units ps^n e.g [pi,0,20] -> [(no unit),ps, ps^2]
        central_f = self._Units(central_f,unit)
        width_f = np.abs(self._Units(width_f,unit))

        width_f = self._sig_fwhm(field_or_intesity,sig_or_fwhm,width_f)
        
        polar_x,polar_y = self._normalise_polarisation(polarisation)
        pulse = 1/self.dt*area_time*np.exp(-(self.frequencies-central_f)**2/(2*width_f**2))*np.exp(1j*self._Taylor(self.frequencies*2*np.pi,central_f*2*np.pi,coefficients=phase_taylor))
        pulse *= np.exp(1j*2*np.pi*self.frequencies*(shift_time-np.min(self.time)))
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

    def add_filter_gaussian(self, central_f, width_f, transmission = 1 ,super_gauss = 1,polarisation = 'b',field_int = 'field',sig_fwhm = 'sig', invert = False,merging = '+',unit = 'Hz',phase = False):
        #gaussian filter
        #super_gauss allows for super gaussian functions 
        central_f = self._Units(central_f,unit)
        width_f = np.abs(self._Units(width_f,unit))
        tau = self._sig_fwhm(field_int,sig_fwhm,width_f)

        gauss = np.exp(-((self.frequencies-central_f)**2/(2*tau**2))**super_gauss)*transmission

        if invert:
            gauss = 1- gauss

        if phase:
            apply_phase = np.exp(1j*gauss**1*np.pi*2.*transmission)
            self._add_filter(apply_phase,polarisation,merging='*')
        else:
            self._add_filter(gauss,polarisation,merging=merging)
        pass

    def add_filter_make_square(self,T = 1,pol = 'x'):
        frequ = self.frequencies*2*np.pi
        spec_x = np.abs(self.frequency_representation_x**2)
        spec_y = np.abs(self.frequency_representation_y**2)
        
        spec_x_norm = spec_x/integrate.trapz(np.abs(spec_x),frequ)
        spec_y_norm = spec_y/integrate.trapz(np.abs(spec_y),frequ)
        
        spec_cum_x = T*integrate.cumtrapz(spec_x_norm,frequ,initial=0)
        spec_cum_y = T*integrate.cumtrapz(spec_y_norm,frequ,initial=0)
        
        shift = T/2 #??
        spec_cum_cum_x = integrate.cumtrapz(spec_cum_x-shift,frequ,initial=0)
        spec_cum_cum_y = integrate.cumtrapz(spec_cum_y-shift,frequ,initial=0)   
        
        if pol.lower()[0] == 'b' or pol.lower()[0] == 'x':
            self._add_filter(np.exp(1j*spec_cum_cum_x),pol='x',merging='*')
        if pol.lower()[0] == 'b' or pol.lower()[0] == 'y':
            self._add_filter(np.exp(1j*spec_cum_cum_y),pol='y',merging='*')
        
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

    def add_phase_filter(self,central_f = 0, phase_taylor=[], polarisation = 'b',unit = 'Hz',f_start = None, f_end = None):
        # phase filter via Taylor expansion around central_f
        if f_start is None:
            f_start = np.min(self.frequencies)
        else:
            f_start = self._Units(f_start,unit)
        if f_end is None:
            f_end = np.max(self.frequencies)
        else:
            f_end = self._Units(f_end,unit)
        
        central_f = self._Units(central_f,unit)
        
        phase = self._Taylor(self.frequencies*2*np.pi,central_f*2*np.pi,coefficients=phase_taylor)
        phase[self.frequencies < f_start] = 0
        phase[self.frequencies > f_end] = 0
        
        phase = np.exp(1j*phase)
        
        
        
        self._add_filter(phase,pol=polarisation,merging='*')

        pass
    
    
    
    def add_phase_wedge(self, time_shift, central_f = 0, shift_time = True, polarisation = 'b', unit = 'Hz',kind = 'double'):
        # phase wedge for shifting in time
        central_f = self._Units(central_f,unit)

        if shift_time:
            time_shift = 2*np.pi*time_shift
        else:
            time_shift = self._Units(time_shift,unit)

        if unit == 'nm':
            time_shift *= -1 
        
        if kind.lower()[0] == 'd':
            wedge = np.exp(1j*time_shift*np.abs((self.frequencies-central_f)))
        elif kind.lower()[0] == 'r':
            phase_vec = np.zeros_like(self.frequencies)
            phase_vec[self.frequencies >= central_f] = np.abs(self.frequencies[self.frequencies >= central_f]-central_f)
            wedge = np.exp(1j*time_shift*phase_vec)
        elif kind.lower()[0] == 'l':    
            phase_vec = np.zeros_like(self.frequencies)
            phase_vec[self.frequencies <= central_f] = np.abs(self.frequencies[self.frequencies <= central_f]-central_f)
            wedge = np.exp(1j*time_shift*phase_vec)
        self._add_filter(wedge,pol=polarisation,merging='*')
        pass

    def apply_frequency_filter(self,pol = 'b'): #you changed the and part here 
        # applies the filter to the pulse 
        if pol.lower()[0] == 'b' or pol.lower()[0] == 'x' and np.any(self.frequency_representation_x != 0):
            self.frequency_representation_x *= self.frequency_filter_x
            self.temporal_representation_x = np.fft.ifft(np.fft.ifftshift(self.frequency_representation_x))
        if pol.lower()[0] == 'b' or pol.lower()[0] == 'y' and np.any(self.frequency_representation_y != 0):
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
        if np.any(np.logical_or(np.abs(self.frequency_filter_x) > 1, np.abs(self.frequency_filter_y) > 1)): 
            print('WARNING: Transmission in filter > 1. Capped to 1.')
            self.frequency_filter_x[self.frequency_filter_x > 1] = 1 
            self.frequency_filter_y[self.frequency_filter_y > 1] = 1 
    ### SLM
    
                
    def apply_SLM(self, pixelwidth = None, pixel_center = 0, N_pixel = 128, unit = 'Hz', kind = 'rectangle',polarisation = 'both',
                   SLM = 'amp',generate_mask = False, save_dir = '', mask_name = 'mask_output',
                   suffix = 0,psf_width = None,psf_sig_fwhm = 'fwhm',calibration_file = None, orientation = 'rising',
                   pixel_transmission_mask = None, pixel_binning = 1):
        # applys a discretisation to the filter, simulating pixels of an (for now) amplitude SLM
        # N_pixel = # of pixels/discretisation steps
        # pixel_center is the position of the central pixel / for even N_pixel the central position 
        # setting generate_mask = True generates a driving mask, given that a callibration_file is specified
        # callibration_file is in retardance (transmission) | voltage and should (must?) be non redundant -> type can be set by cal_type = 'r' or 't' 
        
        if np.mod(N_pixel,pixel_binning) != 0: 
            print('N_pixel / pixel_binning is no integer! No binning applied.')
            pixel_binning = 1
        else: 
            N_pixel = int(N_pixel/pixel_binning)
        
        
        if calibration_file is not None:
            pixel_center, pixelwidth = self._calibrate_SLM(calibration_file)
            print('Calibrated to center_wavelength: ' +str(pixel_center)+'nm and pixelwidth: '+str(pixelwidth)+'nm.')
            pixel_center = self._Units(pixel_center,'nm')
            pixelwidth = abs(self._Units(pixelwidth,'nm'))*pixel_binning
        else:
            pixel_center = self._Units(pixel_center,unit)
            pixelwidth = abs(self._Units(pixelwidth,unit))*pixel_binning

        if pixel_transmission_mask is not None:
            if len(pixel_transmission_mask) != N_pixel:
                print('Mask file does not agree with pixel number!')
                return
        
        
        
        start_f = pixel_center - N_pixel/2*pixelwidth
        end_f = pixel_center + N_pixel/2*pixelwidth

        pixel_transmission_x = []
        pixel_transmission_y = []
       
            
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
                if pixel_transmission_mask is None:
                    cur_slice = self.frequency_filter_x[L_slice]
                else:
                    cur_slice = pixel_transmission_mask[N_pixel -1 - i]
                    
                if SLM.lower() == 'ap':
                    self.frequency_filter_x[L_slice] = np.mean(np.abs(cur_slice))*np.exp(1j*np.mean(np.angle(cur_slice)))
                elif SLM.lower()[0] == 'p':
                    self.frequency_filter_x[L_slice] = np.abs(cur_slice)*np.exp(1j*np.mean(np.angle(cur_slice)))
                    pixel_transmission_x.append(np.mean(np.angle(cur_slice)))
                elif SLM.lower()[0] == 'a':
                    self.frequency_filter_x[L_slice] = np.mean(np.abs(cur_slice))*np.exp(1j*np.angle(cur_slice))
                    pixel_transmission_x.append(np.mean(np.abs(cur_slice))) # <-- carefull
            if polarisation.lower()[0] == 'b' or polarisation.lower()[0] == 'y':
                if pixel_transmission_mask is None:
                    cur_slice = self.frequency_filter_y[L_slice]
                else:
                    cur_slice = pixel_transmission_mask[N_pixel -1 -i]
                if SLM.lower() == 'ap':
                    self.frequency_filter_y[L_slice] = np.mean(np.abs(cur_slice))*np.exp(1j*np.mean(np.angle(cur_slice)))
                elif SLM.lower()[0] == 'p':
                    self.frequency_filter_y[L_slice] = np.abs(cur_slice)*np.exp(1j*np.mean(np.angle(cur_slice)))
                    pixel_transmission_y.append(np.mean(np.angle(cur_slice)))
                elif SLM.lower()[0] == 'a':
                    self.frequency_filter_y[L_slice] = np.mean(np.abs(cur_slice))*np.exp(1j*np.angle(cur_slice))
                    pixel_transmission_y.append(np.mean(np.abs(cur_slice)))
        if orientation.lower()[0] == 'r':    
            pixel_transmission_x = np.flipud(np.array(pixel_transmission_x))
            pixel_transmission_y = np.flipud(np.array(pixel_transmission_y))
        elif orientation.lower()[0] == 'f':
            pixel_transmission_x = np.array(pixel_transmission_x)
            pixel_transmission_y = np.array(pixel_transmission_y)

        if kind.lower()[0] == 'p':
            if psf_width is None:
                psf_width = pixelwidth*0.25
            else:
                psf_width = self._sig_fwhm(field_int='field',sig_fwhm=psf_sig_fwhm,width=psf_width)
                psf_width = self._Units(psf_width,unit=unit)
            psf = np.exp(-0.5*(self.frequencies/psf_width)**2)*1/np.sqrt(2*np.pi*psf_width**2)
            

            if polarisation.lower()[0] == 'b' or polarisation.lower()[0] == 'x':
            
                if SLM.lower() == 'ap':
                    self.frequency_filter_x = self._convolve_normalise(np.abs(self.frequency_filter_x),np.abs(psf)) * \
                            np.exp(1j*self._convolve_normalise(np.angle(self.frequency_filter_x),np.abs(psf)))
                elif SLM.lower()[0] == 'p':
                    self.frequency_filter_x = np.abs(self.frequency_filter_x)*np.exp(1j*self._convolve_normalise(np.angle(self.frequency_filter_x),np.abs(psf)))
                elif SLM.lower()[0] == 'a':
                    self.frequency_filter_x = self._convolve_normalise(np.abs(self.frequency_filter_x),np.abs(psf)) *np.exp(1j*np.angle(self.frequency_filter_x))
            
            
            if polarisation.lower()[0] == 'b' or polarisation.lower()[0] == 'y':
            
                if SLM.lower() == 'ap':
                    self.frequency_filter_y = self._convolve_normalise(np.abs(self.frequency_filter_y),np.abs(psf)) * \
                            np.exp(1j*self._convolve_normalise(np.angle(self.frequency_filter_y),np.abs(psf)))
                elif SLM.lower()[0] == 'p':
                    self.frequency_filter_y = np.abs(self.frequency_filter_y)*np.exp(1j*self._convolve_normalise(np.angle(self.frequency_filter_y),np.abs(psf)))
                elif SLM.lower()[0] == 'a':
                    self.frequency_filter_y = self._convolve_normalise(np.abs(self.frequency_filter_y),np.abs(psf)) *np.exp(1j*np.angle(self.frequency_filter_y))
      
                
        if generate_mask: 
            mask_name_x = save_dir + mask_name+str(suffix)+'_x.txt'
            mask_name_y = save_dir + mask_name+str(suffix)+'_y.txt'
    
  
            with open(mask_name_x, "w") as txt_file:
                for line in list(pixel_transmission_x):
                    txt_file.write(str(line) + "\n")
            txt_file.close()
            
            with open(mask_name_y, "w") as txt_file:
                for line in list(pixel_transmission_y):
                    txt_file.write(str(line) + "\n")
            txt_file.close()
            
            return mask_name_x, mask_name_y


    def _calibrate_SLM(self,calib_file):
      container = loadmat(calib_file)
      center_pixel = float(container['slm_calibration']['center_pixel'][0,0])
      pixel_width = float(container['slm_calibration']['pixel_width'][0,0])
      
      return center_pixel,pixel_width

    ### Additional functions

    def _Units(self,input,unit = 'Hz'): 
        # transforming nm and meV to THz
        if unit.lower()[0] == 'm': 
            output = input/(2*np.pi*hbar) 
        elif unit.lower()[0] == 'n': 
            central_f = 299792.458/self.central_wavelength

            if np.abs(input - self.central_wavelength) < np.abs(input):
                input = input - self.central_wavelength

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
    
    def _fft_convolve(self,a,b):
        ft_a = np.fft.fft(a)
        ft_b = np.fft.fft(b)
        return np.fft.ifft(ft_a*ft_b)
    
    def _convolve_normalise(self,orig,psf):

        orig_height = np.max(orig)

        conv = np.convolve(orig,psf,mode='same')
        conv /= np.max(conv)

        return conv*orig_height 
    
    def _normalise_polarisation(self,pol):
        pol = np.array(pol,dtype=complex)
        norm = np.sqrt(np.abs(pol[0]**2)+np.abs(pol[1]**2))
        pol_x = pol[0]/norm
        pol_y = pol[1]/norm
        return pol_x, pol_y
    ### plotting functions
        # limits can be set in time (t_0,t_end) and in Fourier space (frequ_0, frequ_end)
        # polarisation ('x' , 'y' or 'both') are set via plot_pol
        # plotting in different domains ('Hz' -> THz; 'meV' - > meV; 'nm' -> nm) can be controlled via domain = '' 
        # save = True  saves figures 
    def plot_filter(self,t_0 = None,t_end = None,frequ_0 = None, frequ_end = None ,plot_pol = 'both',
                    domain = 'Hz',save = False, save_name = 'fig',save_dir = '',plot_phase = True):
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

        plot_phase_x = np.empty_like(self.frequencies)
        plot_phase_x[:] = np.nan
        plot_phase_y = np.empty_like(self.frequencies)
        plot_phase_y[:] = np.nan
        

        plot_limit = 1e-3
        plot_phase_x[np.abs(self.frequency_filter_x)>plot_limit] = np.angle(self.frequency_filter_x[np.abs(self.frequency_filter_x)>plot_limit])
        plot_phase_y[np.abs(self.frequency_filter_y)>plot_limit] = np.angle(self.frequency_filter_y[np.abs(self.frequency_filter_y)>plot_limit])


        fig,ax = plt.subplots()
        ax2=ax.twinx()
        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'x':
            ax.plot(self.plot_domain, np.abs(self.frequency_filter_x),'b-', label="T_x")
            if plot_phase:
                ax2.plot(self.plot_domain,plot_phase_x/np.pi)
                
        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'y':
            ax.plot(self.plot_domain, np.abs(self.frequency_filter_y),'r-', label="T_y")
            if plot_phase:
                ax2.plot(self.plot_domain,plot_phase_y/np.pi)       
        ax.set_xlim([frequ_0,frequ_end])
        ax.set_xlabel(self.domain)
        ax.grid()
        ax.legend() 
        ax.set_ylabel('Transmission')
        ax2.set_ylabel('Phase / pi')
        ax.set_title('Filter frequency')
        if save:
            fig.savefig(save_dir+save_name+'_frequ_filter.png')
            
        

    def plot_pulses(self,t_0 = None,t_end = None,frequ_0 = None, frequ_end = None ,plot_pol = 'both',
                    plot_phase = False, phase_time_shift = 0,domain = 'Hz',save = False,save_name = 'fig_',save_dir = '',
                    sim_input = None,sim_label = [],plot_frequ_intensity = False):
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


        fig_t, ax_t = plt.subplots()
        ax_2 = ax_t.twinx()
        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'x':
            ax_t.plot(self.time, np.abs(self.temporal_representation_x),'b-', label="x_envel")
            ax_t.plot(self.time, np.real(self.temporal_representation_x),'b:', label="x_field")

        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'y':
            ax_t.plot(self.time, np.abs(self.temporal_representation_y),'r-', label="y_envel")
            ax_t.plot(self.time, np.real(self.temporal_representation_y),'r:', label="y_field")
        if sim_input is not None:
            time_sim = np.real(sim_input[0])
            ax_2.set_ylabel('rho_QD')
            for i in range(len(sim_input)-1):
                if i > len(sim_label)-1:
                    rho_label = str(i)
                else:
                    rho_label=sim_label[i]
                ax_2.plot(time_sim,np.abs(sim_input[i+1]),label=rho_label)
            ax_2.legend(loc = 'upper right')
            ax_2.set_ylim([0,1])
        
        ax_t.set_xlabel('time / ps')
        ax_t.set_ylabel('Pulse')
        ax_t.set_xlim([t_0,t_end])
        ax_t.legend(loc = 'upper left')

        if save:
            fig_t.savefig(save_dir+save_name+"_time.png")
            
        plot_phase_x = np.empty_like(self.frequencies,dtype=complex)
        plot_phase_x[:] = np.nan
        plot_phase_y = np.empty_like(self.frequencies,dtype=complex)
        plot_phase_y[:] = np.nan

        plot_limit = 1e-3
        L_plot_phase_x = np.abs(self.frequency_representation_x)>plot_limit
        L_plot_phase_y = np.abs(self.frequency_representation_y)>plot_limit
        plot_phase_x[L_plot_phase_x] = self.frequency_representation_x[L_plot_phase_x]*np.exp(1j*2*np.pi*self.frequencies[L_plot_phase_x]*phase_time_shift) 
        plot_phase_y[L_plot_phase_y] = self.frequency_representation_y[L_plot_phase_y]*np.exp(1j*2*np.pi*self.frequencies[L_plot_phase_y]*phase_time_shift) 
        plot_phase_x[L_plot_phase_x] = np.angle(plot_phase_x[L_plot_phase_x])
        plot_phase_y[L_plot_phase_y] = np.angle(plot_phase_x[L_plot_phase_y])
        #plot_phase_x *= np.exp(-1j*2*np.pi*self.frequencies*phase_time_shift)
        #plot_phase_y *= np.exp(-1j*2*np.pi*self.frequencies*phase_time_shift)
        fig,ax = plt.subplots()
        ax2=ax.twinx()
        if plot_frequ_intensity: 
            plot_frequency_x = np.abs(self.frequency_representation_x)**2
            plot_frequency_y = np.abs(self.frequency_representation_y)**2
        else:
            plot_frequency_x = np.abs(self.frequency_representation_x)
            plot_frequency_y = np.abs(self.frequency_representation_y)
            
        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'x':
            ax.plot(self.plot_domain, plot_frequency_x,'b-', label="x_envel")
            if plot_phase:
                ax2.plot(self.plot_domain,plot_phase_x/np.pi)
                
        if plot_pol.lower()[0] == 'b' or plot_pol.lower()[0] == 'y':
            ax.plot(self.plot_domain, plot_frequency_y,'r-', label="y_envel")
            if plot_phase:
                ax2.plot(self.plot_domain,plot_phase_y/np.pi)       
        ax.set_xlim([frequ_0,frequ_end])
        ax.set_xlabel(self.domain)
        ax.grid()
        ax.legend() 
        ax.set_ylabel('|FT|')
        ax2.set_ylabel('Phase / pi')
        ax.set_title('Pulses frequency')
        if save:
            fig.savefig(save_dir+save_name+'_frequ.png')
            

    def generate_pulsefiles(self, temp_dir = '', file_name = 'pulse_time', suffix = ''):
        #Translating the generated pulse for use with the PYACEQD Quantum Dot simulation enviroment 
        pulse_file_x = temp_dir + file_name + str(suffix)+'_x.dat' 
        pulse_file_y = temp_dir + file_name + str(suffix)+'_y.dat'

        export_csv(pulse_file_x, self.time, np.real(self.temporal_representation_x), np.imag(self.temporal_representation_x), precision=8, delimit=' ')
        export_csv(pulse_file_y, self.time, np.real(self.temporal_representation_y), np.imag(self.temporal_representation_y), precision=8, delimit=' ')
        return pulse_file_x, pulse_file_y


    #merging with other pulses
    def merge_pulses(self,other_pulse):
        # checks 
        if other_pulse.central_wavelength != self.central_wavelength:
            print('ERROR MERGING: Central wavelength of pulses do not agree!')
            return
        if other_pulse.dt != self.dt:
            print('CAUTION MERGING: Time steps of pulses do not agree!')
            
        other_pulse_real_x = interpolate.interp1d(other_pulse.time, np.real(other_pulse.temporal_representation_x),
                                                  kind='linear', fill_value=0,bounds_error=False)
        other_pulse_imag_x = interpolate.interp1d(other_pulse.time, np.imag(other_pulse.temporal_representation_x),
                                                  kind='linear', fill_value=0,bounds_error=False)
        
        other_pulse_real_y = interpolate.interp1d(other_pulse.time, np.real(other_pulse.temporal_representation_y),
                                                  kind='linear', fill_value=0,bounds_error=False)
        other_pulse_imag_y = interpolate.interp1d(other_pulse.time, np.imag(other_pulse.temporal_representation_y),
                                                  kind='linear', fill_value=0,bounds_error=False)
        
        self._add_time(other_pulse_real_x(self.time)+1j*other_pulse_imag_x(self.time),
                       other_pulse_real_y(self.time)+1j*other_pulse_imag_y(self.time))
        
            
    
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
