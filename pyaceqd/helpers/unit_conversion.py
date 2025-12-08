import numpy as np
from pyaceqd.constants import hbar, c_light

# hbar in meV*ps
# c_light in nm/ps

h = 2*np.pi * hbar  # Planck constant in meV*ps

def nm_to_mev(lambda_light):
    """
    E = h * f = h * c / lambda
    :param lambda_light: wavelength in nm
    """
    return h * c_light / lambda_light

def mev_to_nm(energy_light):
    """
    lambda = h * c / E
    :param energy_light: energy in meV
    """
    return h * c_light / energy_light

def ghz_to_mev(ghz):
    """
    Convert frequency in GHz to energy in meV.
    E = h * f
    Parameters:
    ghz (float): Frequency in GHz.
    
    Returns:
    float: Energy in meV.
    """
    return ghz * h * 1e-3  # Convert GHz to meV using h

def mev_to_ghz(mev):
    """
    Convert energy in meV to frequency in GHz.
    
    Parameters:
    mev (float): Energy in meV.
    
    Returns:
    float: Frequency in GHz.
    """
    return mev / (h * 1e-3)  # Convert meV to GHz using h