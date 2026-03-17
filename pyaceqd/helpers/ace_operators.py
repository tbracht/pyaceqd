import re
import numpy as np

def id(dim):
    """
    returns the identity operator for a system with dimension dim as matrix
    """
    return np.eye(dim)

def zeros(dim):
    """
    returns the zero operator for a system with dimension dim as matrix
    """
    return np.zeros((dim,dim))

def ketbra(i,j, dim):
    """
    returns the operator |i><j| for a system with dimension dim as matrix
    """
    op = np.zeros((dim,dim))
    op[i,j] = 1.0
    return op

def kron(*ops):
    """
    kronecker product of a list of operators, e.g. cron(A,B,C) = A (x) B (x) C
    -----------
    Args:
        ops: list of operators as numpy arrays
    Returns:
        np.ndarray: Kronecker product of the input operators
    -----------
    Example:
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[0, 1], [1, 0]])
        C = np.array([[1, 0], [0, -1]])
        result = cron(A, B, C)
        print(result)
    -----------
    """
    n = len(ops)
    res = ops[0]
    for k in range(1,n):
        res = np.kron(res, ops[k])
    return res

def b_op(dim):
    """
    returns the annihilation operator for a bosonic mode with dimension dim as matrix
    """
    op = np.zeros((dim,dim))
    for i in range(1,dim):
        op[i-1,i] = np.sqrt(i)
    return op

def bdag_op(dim):
    """
    returns the creation operator for a bosonic mode with dimension dim as matrix
    """
    op = np.zeros((dim,dim))
    for i in range(1,dim):
        op[i,i-1] = np.sqrt(i)
    return op

def n(dim):
    """
    returns the number operator for a bosonic mode with dimension dim as matrix
    """
    op = np.zeros((dim,dim))
    for i in range(dim):
        op[i,i] = i
    return op

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
    
def matrix_to_op(mat, precision=5):
  threshold=10**(-precision)/10.*0.999
  if mat.shape[0] != mat.shape[1]:
    raise ValueError(f'mat.shape[0] != mat.shape[1]')   
  d = mat.shape[0]
  str = ''
  for i in range(d):
    for j in range(d):
      if np.abs(mat[i,j])>threshold:
        if str != '':
          str += '+'
        if mat[i,j].imag>threshold:
          str += f'({mat[i,j].real:.{precision}f}+{mat[i,j].imag:.{precision}f}*i)'
        elif mat[i,j].imag<-threshold:
          str += f'({mat[i,j].real:.{precision}f}{mat[i,j].imag:.{precision}f}*i)'
        else:
          str += f'({mat[i,j].real:.{precision}f})'
        str += f'*|{i}><{j}|_{d}'
  if str == '':
    str = f'0*|0><0|_{d}'
  return str
