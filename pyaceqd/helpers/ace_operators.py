import re
import numpy as np

def ketbra(i,j, dim):
    """
    returns the operator |i><j| for a system with dimension dim as matrix
    """
    op = np.zeros((dim,dim))
    op[i,j] = 1.0
    return op

def cron(a,b):
    """
    kronecker product of two arrays a and b
    """
    return np.kron(a,b)

def op_to_matrix(op):
    op_parts = op.split("+")
    op_matrix = _op_to_matrix(op_parts[0].strip())
    if len(op_parts) > 1:
        for part in op_parts[1:]:
            op_matrix += _op_to_matrix(part.strip())
    return op_matrix

def _op_to_matrix(op):
    """
    Description:
        Converts a string representation of an operator (e.g., |1><0|_2) into a matrix.
        The operator is assumed to be in the form |n><m|_dim, where n and m are indices
        and dim is the dimension of the Hilbert space.
    Args:
        op (str): The operator string in the form |n><m|_dim.
    Returns:
        np.ndarray: The matrix representation of the operator.
    Raises:
        ValueError: If the operator string is not in the expected format or if the indices are out of bounds.
    Example:
        op = "|1><0|_2"
        matrix = op_to_matrix(op)
        print(matrix)
    """
    dim_pattern = r"_(\d+)(?:\[.*\])?"
    dim_match = re.search(dim_pattern, op)
    if not dim_match:
        raise ValueError(f"Invalid dimension format in operator: {op}")
    dim = int(dim_match.group(1))

    pattern = r"[(]*\|(\d+)><(\d+)\|_[\d)]*"
    match = re.match(pattern, op)
    # print(f"op: {op}, dim: {dim}, match: {match}")
    if match:
        ket_idx = int(match.group(1))  # number in |n>
        bra_idx = int(match.group(2))  # number in <m|
        
        if ket_idx >= dim or bra_idx >= dim:
            raise ValueError(f"Index out of bounds: ket_idx={ket_idx}, bra_idx={bra_idx}, dim={dim}")

        # Create ket as column vector |n>
        ket = np.zeros((dim, 1), dtype=complex)
        ket[ket_idx, 0] = 1.0
        
        # Create bra as row vector <m|
        bra = np.zeros((1, dim), dtype=complex)
        bra[0, bra_idx] = 1.0
        
        # Outer product |n><m| creates dim × dim matrix
        op_matrix = ket @ bra
        
        return op_matrix