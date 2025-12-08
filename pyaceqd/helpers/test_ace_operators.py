import pytest
import numpy as np
from pyaceqd.helpers.ace_operators import ketbra, cron, op_to_matrix


class TestKetBra:
    """Tests for ketbra(i, j, dim)."""
    
    def test_basic_2d(self):
        m = ketbra(1, 0, 2)
        assert m.shape == (2, 2)
        assert m.dtype == float
        assert np.allclose(m, np.array([[0.0, 0.0], [1.0, 0.0]]))
    
    def test_diagonal_element(self):
        m = ketbra(2, 2, 4)
        assert m[2, 2] == 1.0
        assert np.sum(m) == 1.0


class TestCron:
    """Tests for cron(a, b) (Kronecker product)."""
    
    def test_shapes_and_values(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[0, 5], [6, 7]])
        k = cron(a, b)
        assert k.shape == (4, 4)
        # Validate against numpy.kron
        assert np.allclose(k, np.kron(a, b))
    
    def test_identity_kron(self):
        a = np.eye(2)
        b = np.array([[0, 1], [2, 3]])
        k = cron(a, b)
        # I ⊗ B should be block-diagonal with B twice
        expected = np.block([[b, np.zeros_like(b)], [np.zeros_like(b), b]])
        assert np.allclose(k, expected)


class TestOpToMatrix:
    """Tests for op_to_matrix(op_str)."""
    
    def test_single_operator_2d(self):
        M = op_to_matrix("|1><0|_2")
        assert M.shape == (2, 2)
        assert M.dtype == complex
        expected = np.array([[0, 0], [1, 0]], dtype=complex)
        assert np.allclose(M, expected)
    
    def test_sum_of_operators(self):
        M = op_to_matrix("|1><0|_2 + |0><1|_2")
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        assert np.allclose(M, expected)
        # Should be Hermitian
        assert np.allclose(M.conj().T, M)
    
    def test_multiple_terms_with_spaces(self):
        M = op_to_matrix("  |1><0|_3  +  |0><2|_3 +|2><1|_3 ")
        expected = np.zeros((3, 3), dtype=complex)
        expected[1, 0] = 1
        expected[0, 2] = 1
        expected[2, 1] = 1
        assert np.allclose(M, expected)
    
    def test_parentheses_variant(self):
        M = op_to_matrix("(|1><0|_2)")
        expected = np.array([[0, 0], [1, 0]], dtype=complex)
        assert np.allclose(M, expected)
    
    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            _ = op_to_matrix("|1><0|")  # Missing dimension
    
    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            _ = op_to_matrix("|2><0|_2")  # ket index >= dim
        with pytest.raises(ValueError):
            _ = op_to_matrix("|1><3|_3")  # bra index >= dim
    
    def test_large_dimension(self):
        M = op_to_matrix("|10><7|_16")
        assert M.shape == (16, 16)
        assert M[10, 7] == 1.0
        assert np.sum(np.abs(M)) == 1.0
    
    def test_complex_dtype(self):
        M = op_to_matrix("|1><0|_2")
        assert M.dtype == complex
    
    def test_hilbert_space_projection(self):
        # |1><1|_3 should be a projector onto state 1
        M = op_to_matrix("|1><1|_3")
        assert np.allclose(M, np.diag([0, 1, 0]).astype(complex))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
