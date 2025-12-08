import pytest
import numpy as np
from pyaceqd.helpers.time_axes import (
    _merge_intervals, 
    round_to_dt,
    time_axis_to_ndt,
    time_axis_to_ndiff_dt,
    n_dt_to_time_axis,
    ndiff_dt_to_time_axis
)


class TestMergeIntervals:
    """Test the _merge_intervals function for merging overlapping time intervals."""
    
    def test_no_overlap(self):
        """Test intervals with no overlap remain separate."""
        intervals = [[0, 1], [2, 3], [4, 5]]
        result = _merge_intervals(intervals)
        assert result == [[0, 1], [2, 3], [4, 5]]
    
    def test_adjacent_intervals(self):
        """Test that adjacent intervals (touching boundaries) are merged."""
        intervals = [[0, 1], [1, 2]]
        result = _merge_intervals(intervals)
        assert result == [[0, 2]]
    
    def test_overlapping_intervals(self):
        """Test that overlapping intervals are merged."""
        intervals = [[0, 2], [1, 3]]
        result = _merge_intervals(intervals)
        assert result == [[0, 3]]
    
    def test_contained_interval(self):
        """Test that a fully contained interval is absorbed."""
        intervals = [[0, 5], [1, 3]]
        result = _merge_intervals(intervals)
        assert result == [[0, 5]]
    
    def test_multiple_overlaps(self):
        """Test merging multiple overlapping intervals."""
        intervals = [[0, 2], [1, 4], [3, 6]]
        result = _merge_intervals(intervals)
        assert result == [[0, 6]]
    
    def test_three_adjacent(self):
        """Test merging three adjacent intervals."""
        intervals = [[0, 10], [10, 20], [20, 30]]
        result = _merge_intervals(intervals)
        assert result == [[0, 30]]
    
    def test_mixed_overlap_and_gap(self):
        """Test intervals with some overlapping and some separated."""
        intervals = [[0, 5], [4, 10], [15, 20], [18, 25]]
        result = _merge_intervals(intervals)
        assert result == [[0, 10], [15, 25]]
    
    def test_single_interval(self):
        """Test that a single interval remains unchanged."""
        intervals = [[5, 10]]
        result = _merge_intervals(intervals)
        assert result == [[5, 10]]
    
    def test_empty_list(self):
        """Test that an empty list remains empty."""
        intervals = []
        result = _merge_intervals(intervals)
        assert result == []
    
    def test_pulse_scenario(self):
        """Test realistic pulse overlap scenario."""
        # Two pulses with tau=5 and factor_tau=4
        # Pulse 1 at t0=20: interval [0, 40]
        # Pulse 2 at t0=30: interval [10, 50]
        intervals = [[0, 40], [10, 50]]
        result = _merge_intervals(intervals)
        assert result == [[0, 50]]
    
    def test_floating_point_intervals(self):
        """Test with floating point boundaries."""
        intervals = [[0.0, 1.5], [1.5, 3.0], [4.0, 5.5]]
        result = _merge_intervals(intervals)
        assert result == [[0.0, 3.0], [4.0, 5.5]]
    
    def test_reverse_contains(self):
        """Test that larger interval after smaller one still merges."""
        intervals = [[1, 3], [0, 5]]
        result = _merge_intervals(intervals)
        # Note: function usually gets sorted input, so this tests the robustness
        assert result == [[0, 5]]


class TestRoundToDt:
    """Test the round_to_dt function for rounding time arrays to timestep."""
    
    def test_scalar_exact_multiple(self):
        """Test scalar value that is already an exact multiple."""
        result = round_to_dt(1.0, 0.1)
        assert np.isclose(result, 1.0)
        assert np.isscalar(result)
    
    def test_scalar_rounding_up(self):
        """Test scalar value that needs rounding up."""
        result = round_to_dt(1.06, 0.1)
        assert np.isclose(result, 1.1)
        assert np.isscalar(result)
    
    def test_scalar_rounding_down(self):
        """Test scalar value that needs rounding down."""
        result = round_to_dt(1.04, 0.1)
        assert np.isclose(result, 1.0)
        assert np.isscalar(result)
    
    def test_array_simple(self):
        """Test array with simple values."""
        t = np.array([0.0, 0.5, 1.0, 1.5])
        result = round_to_dt(t, 0.5)
        expected = np.array([0.0, 0.5, 1.0, 1.5])
        assert np.allclose(result, expected)
    
    def test_array_with_rounding(self):
        """Test array with values needing rounding."""
        t = np.array([0.03, 0.52, 1.07, 1.48])
        result = round_to_dt(t, 0.1)
        expected = np.array([0.0, 0.5, 1.1, 1.5])
        assert np.allclose(result, expected)
    
    def test_removes_duplicates(self):
        """Test that duplicates created by rounding are removed."""
        t = np.array([0.98, 1.02, 1.48, 1.52])
        result = round_to_dt(t, 0.5)
        # All should round to 1.0, 1.0, 1.5, 1.5 -> deduplicate to [1.0, 1.5]
        expected = np.array([1.0, 1.5])
        assert np.allclose(result, expected)
        assert len(result) == 2
    
    def test_preserves_order_after_deduplication(self):
        """Test that deduplication preserves original order."""
        t = np.array([0.01, 1.01, 0.99, 2.01])  # Will round to [0, 1, 1, 2]
        result = round_to_dt(t, 1.0)
        # Should remove second '1' but keep first occurrence order
        expected = np.array([0.0, 1.0, 2.0])
        assert np.allclose(result, expected)
    
    def test_small_dt(self):
        """Test with smaller timestep."""
        t = np.array([0.001, 0.0103, 0.0197])
        result = round_to_dt(t, 0.01)
        expected = np.array([0.0, 0.01, 0.02])
        assert np.allclose(result, expected)
    
    def test_large_dt(self):
        """
        Test with large timestep.
        np.round rounds to nearest even number for values exactly halfway.
        """
        t = np.array([0, 5, 12, 18, 25])
        result = round_to_dt(t, 10.0)
        expected = np.array([0, 10, 20])  # Deduplicated: [0, 0, 10, 20, 20]
        assert np.allclose(result, expected)
    
    def test_medium_dt(self):
        """Test with large timestep."""
        t = np.array([0, 0.5, 1.2, 1.8, 2.5])
        result = round_to_dt(t, 1.0)
        expected = np.array([0, 1.0, 2.0])
        assert np.allclose(result, expected)

    def test_negative_times(self):
        """Test with negative time values."""
        t = np.array([-1.03, -0.52, 0.03, 0.52])
        result = round_to_dt(t, 0.5)
        expected = np.array([-1.0, -0.5, 0.0, 0.5])
        assert np.allclose(result, expected)
    
    def test_dt_equals_one(self):
        """Test with dt=1.0."""
        t = np.array([0.4, 1.6, 2.3, 3.7])
        result = round_to_dt(t, 1.0)
        expected = np.array([0.0, 2.0, 4.0])
        assert np.allclose(result, expected)
    
    def test_realistic_pulse_times(self):
        """Test with realistic pulse simulation times."""
        # Typical pulse times with dt_small=0.1
        t = np.array([0.0, 0.097, 0.203, 0.298, 0.401])
        result = round_to_dt(t, 0.1)
        expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        assert np.allclose(result, expected)
    
    def test_floating_point_precision(self):
        """Test that floating point precision issues don't cause problems."""
        t = np.array([0.1 * i for i in range(11)])  # 0.0, 0.1, ..., 1.0
        result = round_to_dt(t, 0.1)
        expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        assert np.allclose(result, expected)
    
    def test_empty_array(self):
        """Test with empty array."""
        t = np.array([])
        result = round_to_dt(t, 0.1)
        assert len(result) == 0
    
    def test_single_element_array(self):
        """Test with single element array."""
        t = np.array([1.23])
        result = round_to_dt(t, 0.1)
        assert len(result) == 1
        assert np.isclose(result[0], 1.2)


class TestEdgeCases:
    """Test edge cases and interactions between functions."""
    
    def test_round_to_dt_with_merged_interval_boundaries(self):
        """Test rounding times at merged interval boundaries."""
        # Simulate merged interval [0, 50] with dt=0.1
        boundaries = np.array([0.0, 50.0])
        result = round_to_dt(boundaries, 0.1)
        assert np.allclose(result, [0.0, 50.0])
    
    def test_merge_intervals_with_floating_point_precision(self):
        """Test merge with intervals that might have FP precision issues."""
        # Intervals that should touch but might have FP errors
        intervals = [[0.0, 0.1 * 10], [1.0, 2.0]]  # 0.1*10 might not be exactly 1.0
        result = _merge_intervals(intervals)
        # Should merge if boundaries are equal within FP tolerance
        assert len(result) == 1 # or len(result) == 2  # Depends on exact comparison
    
    def test_round_to_dt_scalar_zero(self):
        """Test rounding zero as scalar."""
        result = round_to_dt(0.0, 0.1)
        assert result == 0.0
        assert np.isscalar(result)
    
    def test_merge_intervals_identical(self):
        """Test merging identical intervals."""
        intervals = [[0, 10], [0, 10]]
        result = _merge_intervals(intervals)
        assert result == [[0, 10]]


class TestTimeAxisToNdt:
    """Test time_axis_to_ndt function for converting time arrays to integer indices."""
    
    def test_basic_conversion(self):
        """Test basic conversion from time to integer indices."""
        t_array = np.array([0.0, 0.1, 0.2, 0.3])
        result = time_axis_to_ndt(t_array, 0.1)
        expected = np.array([0, 1, 2, 3])
        assert np.array_equal(result, expected)
        assert result.dtype == int
    
    def test_non_uniform_spacing(self):
        """Test with non-uniformly spaced time array."""
        t_array = np.array([0.0, 0.2, 0.5, 0.9])
        result = time_axis_to_ndt(t_array, 0.1)
        expected = np.array([0, 2, 5, 9])
        assert np.array_equal(result, expected)
    
    def test_rounding_behavior(self):
        """Test that values are properly rounded to nearest integer."""
        t_array = np.array([0.04, 0.16, 0.24])
        res1 = round_to_dt(t_array, 0.1)
        assert np.allclose(res1, [0.0, 0.2])
        result = time_axis_to_ndt(t_array, 0.1)
        expected = np.array([0, 2])  # 0.04->0, (0.16->2, 0.24->2) duplicate removed
        assert np.array_equal(result, expected)
    
    def test_large_dt(self):
        """Test with larger timestep."""
        t_array = np.array([0.0, 1.0, 2.0, 3.0])
        result = time_axis_to_ndt(t_array, 1.0)
        expected = np.array([0, 1, 2, 3])
        assert np.array_equal(result, expected)
    
    def test_small_dt(self):
        """Test with very small timestep."""
        t_array = np.array([0.0, 0.01, 0.02, 0.03])
        result = time_axis_to_ndt(t_array, 0.01)
        expected = np.array([0, 1, 2, 3])
        assert np.array_equal(result, expected)
    
    def test_negative_times(self):
        """Test with negative time values."""
        t_array = np.array([-0.2, -0.1, 0.0, 0.1])
        result = time_axis_to_ndt(t_array, 0.1)
        expected = np.array([-2, -1, 0, 1])
        assert np.array_equal(result, expected)


class TestTimeAxisToNdiffDt:
    """Test time_axis_to_ndiff_dt for converting time arrays to difference indices."""
    
    def test_basic_uniform_spacing(self):
        """Test with uniformly spaced time array."""
        t_array = np.array([0.0, 0.1, 0.2, 0.3])
        result = time_axis_to_ndiff_dt(t_array, 0.1)
        expected = np.array([0, 1, 1, 1])
        assert np.array_equal(result, expected)
    
    def test_non_uniform_spacing(self):
        """Test with non-uniformly spaced time array (from docstring example)."""
        t_array = np.array([0.0, 0.2, 0.5, 0.9])
        result = time_axis_to_ndiff_dt(t_array, 0.1)
        expected = np.array([0, 2, 3, 4])
        assert np.array_equal(result, expected)
    
    def test_varying_gaps(self):
        """Test with varying gaps between time points."""
        t_array = np.array([0.0, 0.1, 0.4, 1.0])
        result = time_axis_to_ndiff_dt(t_array, 0.1)
        expected = np.array([0, 1, 3, 6])
        assert np.array_equal(result, expected)
    
    def test_single_element(self):
        """Test with single element array."""
        t_array = np.array([0.5])
        res1 = round_to_dt(t_array, 0.1)
        assert np.allclose(res1, [0.5])
        res2 = time_axis_to_ndt(t_array, 0.1)
        assert np.array_equal(res2, [5])
        result = time_axis_to_ndiff_dt(t_array, 0.1)
        expected = np.array([5])
        assert np.array_equal(result, expected)
    
    def test_first_element_nonzero(self):
        """Test that first element defines starting index."""
        t_array = np.array([1.0, 1.1, 1.2])
        res1 = round_to_dt(t_array, 0.1)
        assert np.allclose(res1, [1.0, 1.1, 1.2])
        res2 = time_axis_to_ndt(t_array, 0.1)
        assert np.array_equal(res2, [10, 11, 12])
        result = time_axis_to_ndiff_dt(t_array, 0.1)
        expected = np.array([0, 1, 1])
        assert np.array_equal(result, expected)


class TestNDtToTimeAxis:
    """Test n_dt_to_time_axis for converting integer indices to time arrays."""
    
    def test_basic_conversion(self):
        """Test basic conversion from integer indices to time (from docstring)."""
        ndt_array = np.array([0, 1, 2, 3])
        result = n_dt_to_time_axis(ndt_array, 0.1)
        expected = np.array([0.0, 0.1, 0.2, 0.3])
        assert np.allclose(result, expected)
        assert result.dtype == float
    
    def test_non_uniform_indices(self):
        """Test with non-uniformly spaced indices."""
        ndt_array = np.array([0, 2, 5, 9])
        result = n_dt_to_time_axis(ndt_array, 0.1)
        expected = np.array([0.0, 0.2, 0.5, 0.9])
        assert np.allclose(result, expected)
    
    def test_large_dt(self):
        """Test with larger timestep."""
        ndt_array = np.array([0, 1, 2, 3])
        result = n_dt_to_time_axis(ndt_array, 1.0)
        expected = np.array([0.0, 1.0, 2.0, 3.0])
        assert np.allclose(result, expected)
    
    def test_negative_indices(self):
        """Test with negative indices."""
        ndt_array = np.array([-2, -1, 0, 1])
        result = n_dt_to_time_axis(ndt_array, 0.1)
        expected = np.array([-0.2, -0.1, 0.0, 0.1])
        assert np.allclose(result, expected)
    
    def test_roundtrip_with_time_axis_to_ndt(self):
        """Test roundtrip: time -> ndt -> time."""
        original = np.array([0.0, 0.2, 0.5, 0.9])
        ndt = time_axis_to_ndt(original, 0.1)
        result = n_dt_to_time_axis(ndt, 0.1)
        assert np.allclose(result, original)


class TestNdiffDtToTimeAxis:
    """Test ndiff_dt_to_time_axis for converting difference indices to time arrays."""
    
    def test_basic_conversion(self):
        """Test basic conversion from difference indices to time (from docstring)."""
        ndiff_dt_array = np.array([0, 1, 1, 1])
        result = ndiff_dt_to_time_axis(ndiff_dt_array, 0.1)
        expected = np.array([0.0, 0.1, 0.2, 0.3])
        assert np.allclose(result, expected)
    
    def test_non_uniform_differences(self):
        """Test with non-uniform differences (from docstring example)."""
        ndiff_dt_array = np.array([0, 2, 3, 4])
        result = ndiff_dt_to_time_axis(ndiff_dt_array, 0.1)
        expected = np.array([0.0, 0.2, 0.5, 0.9])
        assert np.allclose(result, expected)
    
    def test_varying_gaps(self):
        """Test with varying gap sizes."""
        ndiff_dt_array = np.array([0, 1, 3, 6])
        result = ndiff_dt_to_time_axis(ndiff_dt_array, 0.1)
        expected = np.array([0.0, 0.1, 0.4, 1.0])
        assert np.allclose(result, expected)
    
    def test_single_element(self):
        """Test with single element."""
        ndiff_dt_array = np.array([5])
        result = ndiff_dt_to_time_axis(ndiff_dt_array, 0.1)
        expected = np.array([0.5])
        assert np.allclose(result, expected)
    
    def test_first_element_nonzero(self):
        """Test that first element defines starting time."""
        ndiff_dt_array = np.array([10, 1, 1])
        result = ndiff_dt_to_time_axis(ndiff_dt_array, 0.1)
        expected = np.array([1.0, 1.1, 1.2])
        assert np.allclose(result, expected)
    
    def test_roundtrip_with_time_axis_to_ndiff_dt(self):
        """Test roundtrip: time -> ndiff_dt -> time."""
        original = np.array([0.0, 0.2, 0.5, 0.9])
        ndiff_dt = time_axis_to_ndiff_dt(original, 0.1)
        result = ndiff_dt_to_time_axis(ndiff_dt, 0.1)
        assert np.allclose(result, original)


class TestTimeConversionRoundtrips:
    """Test roundtrip conversions between different time representations."""
    
    def test_full_roundtrip_uniform(self):
        """Test full roundtrip with uniformly spaced times."""
        original = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        dt = 0.1
        
        # time -> ndt -> time
        ndt = time_axis_to_ndt(original, dt)
        back1 = n_dt_to_time_axis(ndt, dt)
        assert np.allclose(original, back1)
        
        # time -> ndiff_dt -> time
        ndiff_dt = time_axis_to_ndiff_dt(original, dt)
        back2 = ndiff_dt_to_time_axis(ndiff_dt, dt)
        assert np.allclose(original, back2)
    
    def test_full_roundtrip_nonuniform(self):
        """Test full roundtrip with non-uniformly spaced times."""
        original = np.array([0.0, 0.2, 0.5, 1.0, 2.0])
        dt = 0.1
        
        # time -> ndt -> time
        ndt = time_axis_to_ndt(original, dt)
        back1 = n_dt_to_time_axis(ndt, dt)
        assert np.allclose(original, back1)
        
        # time -> ndiff_dt -> time
        ndiff_dt = time_axis_to_ndiff_dt(original, dt)
        back2 = ndiff_dt_to_time_axis(ndiff_dt, dt)
        assert np.allclose(original, back2)
    
    def test_consistency_ndt_ndiff_dt(self):
        """Test that ndt and ndiff_dt representations are consistent."""
        original = np.array([0.0, 0.1, 0.3, 0.7])
        dt = 0.1
        
        ndt = time_axis_to_ndt(original, dt)
        ndiff_dt = time_axis_to_ndiff_dt(original, dt)
        
        # ndt should be cumsum of ndiff_dt
        assert np.array_equal(ndt, np.cumsum(ndiff_dt))


if __name__ == "__main__":
    # Allow running with: python test_time_axes.py
    pytest.main([__file__, "-v"])
