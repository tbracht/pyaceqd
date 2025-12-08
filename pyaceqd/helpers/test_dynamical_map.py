import pytest
import numpy as np
import matplotlib.pyplot as plt
from pyaceqd.pulses import ChirpedPulse
from pyaceqd.two_level_system.tls import tls_new
from pyaceqd.helpers.dynamical_map import calc_tl_dynmap_pseudo, extract_dms
from pyaceqd.helpers.time_axes import round_to_dt
from scipy.linalg import ishermitian


@pytest.fixture(scope="module")
def setup_tls_system():
    """
    Fixture that sets up the TLS system with dynamical maps.
    This expensive calculation is done once per test module.
    
    Returns:
        dict: Contains pulse, output_ops, times, ground/excited states, 
              dynamical map, and dt.
    """
    p1 = ChirpedPulse(tau_0=4, e_start=0, t0=4*4, e0=1, polarization="x")
    output_ops = ["|0><0|_2", "|1><1|_2"]
    tend = 100
    dt = 0.1


    
    # Calculate without dynamical map (for reference)
    t_test, g_test, x_test = tls_new(
        0, tend, p1, dt=dt, 
        phonons=True, lindblad=True, 
        ae=5.0, temperature=4, 
        verbose=False, prepare_only=False, 
        output_ops=output_ops
    )
    t_test = t_test.real
    
    # Calculate with dynamical map
    result, dm = tls_new(
        0, tend, p1, dt=dt,
        phonons=True, ae=5.0, temperature=4,
        verbose=False, prepare_only=False,
        lindblad=True,
        output_ops=output_ops, calc_dynmap=True
    )
    t = result[0].real
    
    return {
        'pulse': p1,
        'output_ops': output_ops,
        'times': t,
        'times_test': t_test,
        'ground_state': g_test,
        'excited_state': x_test,
        'dynmap': dm,
        'dt': dt,
        'tend': tend
    }

@pytest.fixture(scope="module")
def setup_tls_system_mto():
    """
    Fixture that sets up the TLS system with dynamical maps.
    This expensive calculation is done once per test module.
    
    Returns:
        dict: Contains pulse, output_ops, times, ground/excited states, 
              dynamical map, and dt.
    """
    p1 = ChirpedPulse(tau_0=4, e_start=0, t0=4*4, e0=1, polarization="x")
    output_ops = ["|0><0|_2", "|1><1|_2"]
    tend = 100
    dt = 0.1

    mto = {"operator": "|0><1|_2", "applyFrom": "_left", "applyBefore": "false", "time": 10}
    multitime_operators = [mto]
    
    # Calculate without dynamical map (for reference)
    t_test, g_test, x_test = tls_new(
        0, tend, p1, dt=dt, 
        phonons=True, lindblad=True, 
        ae=5.0, temperature=4, 
        verbose=False, prepare_only=False, 
        output_ops=output_ops,
        multitime_op=multitime_operators
    )
    t_test = t_test.real
    
    # Calculate with dynamical map
    result, dm = tls_new(
        0, tend, p1, dt=dt,
        phonons=True, ae=5.0, temperature=4,
        verbose=False, prepare_only=False,
        lindblad=True,
        output_ops=output_ops, calc_dynmap=True,
        multitime_op=multitime_operators
    )
    t = result[0].real
    
    return {
        'pulse': p1,
        'output_ops': output_ops,
        'times': t,
        'times_test': t_test,
        'ground_state': g_test,
        'excited_state': x_test,
        'dynmap': dm,
        'dt': dt,
        'tend': tend
    }


class TestCalcTlDynmapPseudo:
    """Tests for calc_tl_dynmap_pseudo function."""
    
    def test_output_shape(self, setup_tls_system):
        """Test that output has correct shape (n_t-1, n_h^2, n_h^2)."""
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        
        tl_dm = calc_tl_dynmap_pseudo(dm, times)
        
        assert tl_dm.shape[0] == len(times) - 1
        assert tl_dm.shape[1] == dm.shape[1]
        assert tl_dm.shape[2] == dm.shape[2]
    
    def test_first_element(self, setup_tls_system):
        """Test that first element equals dm[0]."""
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        
        tl_dm = calc_tl_dynmap_pseudo(dm, times)
        
        assert np.allclose(tl_dm[0], dm[0])
    
    def test_composition_property(self, setup_tls_system):
        """Test that E_t2,t0 ≈ E_t2,t1 @ E_t1,t0."""
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        
        tl_dm = calc_tl_dynmap_pseudo(dm, times)
        
        # Check at a few indices
        for i in [10, 50, 100]:
            if i < len(dm):
                # E_ti+1,t0 should equal E_ti+1,ti @ E_ti,t0
                reconstructed = np.dot(tl_dm[i], dm[i-1])
                assert np.allclose(reconstructed, dm[i], rtol=1e-8, atol=1e-10)
    
    def test_dtype_complex(self, setup_tls_system):
        """Test that output is complex dtype."""
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        
        tl_dm = calc_tl_dynmap_pseudo(dm, times)
        
        assert tl_dm.dtype == complex


class TestExtractDms:
    """Tests for extract_dms function."""
    
    def test_output_structure(self, setup_tls_system):
        """Test that output has correct structure."""
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        tau_c = 10.0
        t_MTOs = []
        dm_tl = calc_tl_dynmap_pseudo(dm, times)
        tl_map, tl_dms = extract_dms(dm_tl, times, tau_c, t_MTOs)
        
        # tl_map should be a single 2D array
        assert tl_map.ndim == 2
        assert tl_map.shape == (dm.shape[1], dm.shape[2])
        
        # tl_dms should be a list with len(t_MTOs) + 1 elements
        assert len(tl_dms) == len(t_MTOs) + 1
    
    def test_tl_map_equals_dm_at_tau_c(self, setup_tls_system):
        """Test that tl_map equals dm at tau_c index."""
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        tau_c = 10.0
        t_MTOs = []
        dm_tl = calc_tl_dynmap_pseudo(dm, times)
        tl_map, tl_dms = extract_dms(dm_tl, times, tau_c, t_MTOs)
        
        i_timelocal = np.where(times > times[0] + tau_c)[0][0]
        assert np.allclose(tl_map, dm_tl[i_timelocal])
    
    def test_dm1_length(self, setup_tls_system):
        """Test that first dm block has correct length."""
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        tau_c = 10.0
        t_MTOs = []
        dm_tl = calc_tl_dynmap_pseudo(dm, times)
        tl_map, tl_dms = extract_dms(dm_tl, times, tau_c, t_MTOs)
        
        i_timelocal = np.where(times > times[0] + tau_c)[0][0]
        dm_1 = tl_dms[0]
        
        assert len(dm_1) == i_timelocal
    
    def test_dm2_length(self, setup_tls_system_mto):
        """Test that second dm block (after MTO) has correct length."""
        dm = setup_tls_system_mto['dynmap']
        times = setup_tls_system_mto['times']
        tau_c = 10.0
        t_MTOs = [10.0]
        dm_tl = calc_tl_dynmap_pseudo(dm, times)
        tl_map, tl_dms = extract_dms(dm_tl, times, tau_c, t_MTOs)
        
        i_timelocal = np.where(times > times[0] + tau_c)[0][0]
        dm_2 = tl_dms[1]
        
        assert len(dm_2) == i_timelocal
        
    def test_tl_map_consistency(self, setup_tls_system):
        """Test that tl_map from first block equals expected value."""
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        tau_c = 40  # important that this is larger than pulse duration
        t_MTOs = []
        dm_tl = calc_tl_dynmap_pseudo(dm, times)
        tl_map, tl_dms = extract_dms(dm_tl, times, tau_c, t_MTOs)
        dm_1 = tl_dms[0]
        assert len(dm_1) == np.where(times > times[0] + tau_c)[0][0]
        assert np.allclose(tl_map, dm_tl[len(dm_1)], rtol=1e-8, atol=1e-8)
        assert np.allclose(tl_map, dm_1[-1], rtol=1e-8, atol=1e-8)

    def test_hermiticity_no_mto(self, setup_tls_system):
        """Test that extracted dynamical maps are Hermitian."""
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        tau_c = 10.0
        t_MTOs = []
        dm_tl = calc_tl_dynmap_pseudo(dm, times)
        tl_map, tl_dms = extract_dms(dm_tl, times, tau_c, t_MTOs)
        
        # Check tl_map
        column1 = tl_map[:, 0].reshape(int(np.sqrt(tl_map.shape[0])), -1)
        assert ishermitian(column1, atol=1e-10)
        
        #Check each block in tl_dms
        for dm_block in tl_dms:
            for E in dm_block:
                column1 = E[:, 0].reshape(int(np.sqrt(E.shape[0])), -1)
                assert ishermitian(column1, atol=1e-10)

    def test_hermiticity_with_mto(self, setup_tls_system_mto):
        """Test that extracted dynamical maps are Hermitian,
        and that after the MTO the maps become non-hermitian.
        Also ensures that the correct map is extracted at the MTO time."""
        dm = setup_tls_system_mto['dynmap']
        times = setup_tls_system_mto['times']
        tau_c = 10.0
        t_MTOs = [10.0]
        dm_tl = calc_tl_dynmap_pseudo(dm, times)
        tl_map, tl_dms = extract_dms(dm_tl, times, tau_c, t_MTOs)
        i_tmto = np.where(np.isclose(times, t_MTOs[0]))[0][0]
        # tl_dms[1] should start with applied MTO
        assert np.allclose(tl_dms[1][0], dm_tl[i_tmto])
        # this one should be hermitian, before MTO
        col1 = dm_tl[i_tmto-1][:, 0].reshape(int(np.sqrt(dm_tl[i_tmto].shape[0])), -1)
        assert ishermitian(col1, atol=1e-10)
        # this one should NOT be hermitian, after MTO
        col1 = dm_tl[i_tmto][:, 0].reshape(int(np.sqrt(dm_tl[i_tmto].shape[0])), -1)
        assert not ishermitian(col1, atol=1e-10)


class TestPropagation:
    """Tests for propagation using extracted dynamical maps."""
    
    def test_propagation_without_mto(self, setup_tls_system):
        """Test that propagation using extracted maps matches original results."""
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        g_ref = setup_tls_system['ground_state']
        x_ref = setup_tls_system['excited_state']
        tau_c = 10.0
        t_MTOs = []
        output_ops = setup_tls_system['output_ops']
        
        dm_tl = calc_tl_dynmap_pseudo(dm, times)
        tl_map, tl_dms = extract_dms(dm_tl, times, tau_c, t_MTOs)
        
        n_h = 2
        rho0 = np.zeros((n_h, n_h), dtype=complex)
        rho0[0, 0] = 1.0  # Start in ground state
        
        # Propagate using extracted maps
        g_prop = []
        x_prop = []
        g_prop.append(np.real(rho0[0, 0]))
        x_prop.append(np.real(rho0[1, 1]))
        for i in range(len(times)-1):
            rho = np.reshape(rho0, (n_h**2))
            rho = dm_tl[i] @ rho
            rho0 = np.reshape(rho, (n_h, n_h))
            g_prop.append(np.real(rho0[0, 0]))
            x_prop.append(np.real(rho0[1, 1]))
        
        g_prop = np.array(g_prop)
        x_prop = np.array(x_prop)
        assert np.allclose(g_prop, g_ref, rtol=1e-5, atol=1e-7)
        assert np.allclose(x_prop, x_ref, rtol=1e-5, atol=1e-7)

        # plt.plot(times, g_ref, label="Ground State Ref")
        # plt.plot(times, g_prop, '--', label="Ground State Prop")
        # plt.plot(times, x_ref, label="Excited State Ref")
        # plt.plot(times, x_prop, '--', label="Excited State Prop")
        # plt.xlabel("Time")
        # plt.ylabel("Population")
        # plt.legend()
        # plt.savefig("test_propagation_without_mto.png")
    
    def test_propagation_without_mto_dmtl(self, setup_tls_system):
        dm = setup_tls_system['dynmap']
        times = setup_tls_system['times']
        g_ref = setup_tls_system['ground_state']
        x_ref = setup_tls_system['excited_state']
        tau_c = 40
        t_MTOs = []
        output_ops = setup_tls_system['output_ops']
        
        dm_tl = calc_tl_dynmap_pseudo(dm, times)
        tl_map, tl_dms = extract_dms(dm_tl, times, tau_c, t_MTOs)
        
        n_h = 2
        g_prop = []
        x_prop = []
        rho0 = np.zeros((n_h, n_h), dtype=complex)
        rho0[0, 0] = 1.0  # Start in ground state
        g_prop.append(np.real(rho0[0, 0]))
        x_prop.append(np.real(rho0[1, 1]))
        for i in range(len(times)-1):
            rho = np.reshape(rho0, (n_h**2))
            if i < len(tl_dms[0]):
                rho = tl_dms[0][i] @ rho
            else:
                rho = tl_map @ rho
            rho0 = np.reshape(rho, (n_h, n_h))
            g_prop.append(np.real(rho0[0, 0]))
            x_prop.append(np.real(rho0[1, 1]))

        # plt.plot(times, g_ref, label="Ground State Ref")
        # plt.plot(times, g_prop, '--', label="Ground State Prop")
        # plt.plot(times, x_ref, label="Excited State Ref")
        # plt.plot(times, x_prop, '--', label="Excited State Prop")
        # plt.xlabel("Time")
        # plt.ylabel("Population")
        # plt.legend()
        # plt.savefig("test_propagation_without_mto.png")

        g_prop = np.array(g_prop)
        x_prop = np.array(x_prop)
        assert np.allclose(g_prop, g_ref, rtol=1e-5, atol=1e-7)
        assert np.allclose(x_prop, x_ref, rtol=1e-5, atol=1e-7)

    def test_propagation_with_mto(self, setup_tls_system_mto):
        """Test that propagation using extracted maps with MTO matches original results."""
        dm = setup_tls_system_mto['dynmap']
        times = setup_tls_system_mto['times']
        g_ref = setup_tls_system_mto['ground_state'].real
        x_ref = setup_tls_system_mto['excited_state'].real
        tau_c = 40
        t_MTOs = [10.0]
        output_ops = setup_tls_system_mto['output_ops']
        
        dm_tl = calc_tl_dynmap_pseudo(dm, times)
        tl_map, tl_dms = extract_dms(dm_tl, times, tau_c, t_MTOs)
        
        n_h = 2
        rho0 = np.zeros((n_h, n_h), dtype=complex)
        rho0[0, 0] = 1.0  # Start in ground state
        
        # Propagate using extracted maps
        g_prop = []
        x_prop = []
        g_prop.append(np.real(rho0[0, 0]))
        x_prop.append(np.real(rho0[1, 1]))
        # use tl_dms[0] from 0 to t_MTO, then tl_dms[1] after t_MTO
        # after that, use tl_map
        i_tmto = np.where(np.isclose(times, t_MTOs[0]))[0][0]
        for i in range(i_tmto):
            rho = np.reshape(rho0, (n_h**2))
            rho = tl_dms[0][i] @ rho
            rho0 = np.reshape(rho, (n_h, n_h))
            g_prop.append(np.real(rho0[0, 0]))
            x_prop.append(np.real(rho0[1, 1]))
        for i in range(i_tmto, i_tmto + len(tl_dms[1])):
            rho = np.reshape(rho0, (n_h**2))
            rho = tl_dms[1][i - i_tmto] @ rho
            rho0 = np.reshape(rho, (n_h, n_h))
            g_prop.append(np.real(rho0[0, 0]))
            x_prop.append(np.real(rho0[1, 1]))
        for i in range(i_tmto + len(tl_dms[1]), len(times)-1):
            rho = np.reshape(rho0, (n_h**2))
            rho = tl_map @ rho
            rho0 = np.reshape(rho, (n_h, n_h))
            g_prop.append(np.real(rho0[0, 0]))
            x_prop.append(np.real(rho0[1, 1]))


        # plt.plot(times, g_ref, label="Ground State Ref")
        # plt.plot(times, g_prop, '--', label="Ground State Prop")
        # plt.plot(times, x_ref, label="Excited State Ref")
        # plt.plot(times, x_prop, '--', label="Excited State Prop")
        # plt.xlabel("Time")
        # plt.ylabel("Population")
        # plt.legend()
        # # plt.ylim(0.9,1)
        # # plt.xlim(9,12)
        # plt.savefig("test_propagation_with_mto.png")
        
        g_prop = np.array(g_prop)
        x_prop = np.array(x_prop)
        assert np.allclose(g_prop, g_ref, rtol=1e-10, atol=1e-10)
        assert np.allclose(x_prop, x_ref, rtol=1e-10, atol=1e-10)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

