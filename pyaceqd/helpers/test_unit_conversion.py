import pytest
import numpy as np
from pyaceqd.helpers.unit_conversion import nm_to_mev, mev_to_nm, ghz_to_mev, mev_to_ghz

class TestWavelengthEnergyConversion:
    """Test conversions between wavelength (nm) and energy (meV)."""
    
    def test_nm_to_mev_930nm(self):
        """Test conversion of 930 nm (typical QD emission)."""
        energy = nm_to_mev(930.0)
        assert np.isclose(energy, 1333.16138, rtol=1e-3)
    
    def test_nm_to_mev_780nm(self):
        """Test conversion of 780 nm (Rb D2 line)."""
        energy = nm_to_mev(780.0)
        assert np.isclose(energy, 1589.53858, rtol=1e-3)
    
    def test_mev_to_nm_1p5mev(self):
        """Test conversion of 1.5 meV."""
        wavelength = mev_to_nm(1500)
        assert np.isclose(wavelength, 826.56006, rtol=1e-3)
    
    def test_roundtrip_nm_to_mev_to_nm(self):
        """Test that converting back and forth preserves the value."""
        original = 920.0
        energy = nm_to_mev(original)
        result = mev_to_nm(energy)
        assert np.isclose(original, result, rtol=1e-10)
    
    def test_roundtrip_mev_to_nm_to_mev(self):
        """Test that converting back and forth preserves the value."""
        original = 1.4
        wavelength = mev_to_nm(original)
        result = nm_to_mev(wavelength)
        assert np.isclose(original, result, rtol=1e-10)
    
    def test_nm_to_mev_array(self):
        """Test conversion with numpy arrays."""
        wavelengths = np.array([780.0, 930.0, 1000.0])
        energies = nm_to_mev(wavelengths)
        assert energies.shape == wavelengths.shape
        assert np.isclose(energies[0], 1589.53857, rtol=1e-3)
        assert np.isclose(energies[1], 1333.16138, rtol=1e-3)
        assert np.isclose(energies[2], 1239.84009, rtol=1e-3)


class TestFrequencyEnergyConversion:
    """Test conversions between frequency (GHz) and energy (meV)."""
    
    def test_ghz_to_mev_100ghz(self):
        """Test conversion of 100 GHz (typical fine structure splitting)."""
        energy = ghz_to_mev(100.0)
        assert np.isclose(energy, 0.413566, rtol=1e-3)
    
    def test_mev_to_ghz_0p1mev(self):
        """Test conversion of 0.1 meV."""
        freq = mev_to_ghz(0.1)
        assert np.isclose(freq, 24.17989, rtol=1e-3)
    
    def test_roundtrip_ghz_to_mev_to_ghz(self):
        """Test that converting back and forth preserves the value."""
        original = 50.0
        energy = ghz_to_mev(original)
        result = mev_to_ghz(energy)
        assert np.isclose(original, result, rtol=1e-10)
    
    def test_roundtrip_mev_to_ghz_to_mev(self):
        """Test that converting back and forth preserves the value."""
        original = 0.5
        freq = mev_to_ghz(original)
        result = ghz_to_mev(freq)
        assert np.isclose(original, result, rtol=1e-10)
    
    def test_ghz_to_mev_array(self):
        """Test conversion with numpy arrays."""
        frequencies = np.array([1.0, 10.0, 100.0])
        energies = ghz_to_mev(frequencies)
        assert energies.shape == frequencies.shape
        assert np.isclose(energies[0], 0.0041356, rtol=1e-4)
        assert np.isclose(energies[1], 0.0413566, rtol=1e-4)
        assert np.isclose(energies[2], 0.4135667, rtol=1e-4)


class TestEdgeCases:
    def test_inverse_relationship(self):
        """Test that energy increases as wavelength decreases."""
        wavelengths = np.array([500.0, 750.0, 1000.0])
        energies = nm_to_mev(wavelengths)
        assert energies[0] > energies[1] > energies[2]
    
    def test_linear_relationship_freq_energy(self):
        """Test that energy scales linearly with frequency."""
        freq1 = ghz_to_mev(10.0)
        freq2 = ghz_to_mev(20.0)
        assert np.isclose(freq2, 2 * freq1, rtol=1e-10)

if __name__ == "__main__":
    # Allow running with: python test_unit_conversion.py
    pytest.main([__file__, "-v"])
