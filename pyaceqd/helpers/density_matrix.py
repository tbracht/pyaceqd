import numpy as np
import itertools
"""
contains multiple functions to check if a matrix is a density matrix, calculate Concurrence, 
fidelity with respect to a pure state, construct a valid DM from a given hermitian, trace one matrix etc.
"""

def concurrence(rho: np.ndarray) -> float:
    """
    Concurrence as given in James, Phys. Rev. A 64, 052312 (2001) and Seidelmann, Phys. Rev. Lett. 129, 193604 (2022),
    see https://arxiv.org/pdf/quant-ph/0103121 and supplement of https://arxiv.org/abs/2205.03390
    """    
    T_matrix = np.flip(np.diag([-1.,1.,1.,-1.]),axis=1)  # antidiagonal matrix
    M_matrix = np.dot(rho,np.dot(T_matrix,np.dot(np.conjugate(rho),T_matrix)))
    _eigvals = np.real(np.linalg.eigvals(M_matrix))
    _eigvals = np.sqrt(np.sort(_eigvals))
    return np.max([0.0,_eigvals[-1]-np.sum(_eigvals[:-1])])

def check_density_matrix(rho: np.ndarray, tol: float=1e-10, print_info: bool=False) -> dict:
    """
    Check how well a matrix satisfies the density matrix axioms:
    1. Hermitian: rho == rho.conj().T
    2. Trace one: np.trace(rho) == 1
    3. Positive semidefinite: all eigenvalues of rho are >= 0
    Returns a dict of diagnostics for before/after comparison.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    hermitian_error = np.max(np.abs(rho - rho.conj().T))
    result = {
        "trace":              np.trace(rho).real,
        "trace_error":        abs(np.trace(rho).real - 1.0),
        "min_eigenvalue":     eigenvalues.min(),
        "hermitian_error":    hermitian_error,
        "is_valid":           (
            hermitian_error < tol
            and abs(np.trace(rho).real - 1.0) < tol
            and eigenvalues.min() >= -tol
        ),
    }
    if print_info:
        print(f"Trace: {result['trace']:.6f} (error: {result['trace_error']:.2e})")
        print(f"Min eigenvalue: {result['min_eigenvalue']:.2e}")
        print(f"Hermitian error: {result['hermitian_error']:.2e}")
        print(f"Is valid density matrix: {result['is_valid']}")
    return result

def make_valid_density_matrix(rho_imperfect: np.ndarray, print_warnings: bool = True) -> np.ndarray:
    """
    assumes a matrix that 
    1) is hermitian
    2) has trace one
    but can have negative eigenvalues.
    Fixes the matrix by using algorithm given in doi.org/10.1103/PhysRevLett.108.070502
    or https://arxiv.org/pdf/1106.5458
    By this the eigenbasis stays the same, but the eigenvalues are corrected to be non-negative and sum to one.
    """
    # check if hermitian and trace one, if not print warning and return original matrix
    diagnostics = check_density_matrix(rho_imperfect)
    if diagnostics["hermitian_error"] > 1e-10:
        if print_warnings:
            print("Warning: Input matrix is not hermitian. using rho_new = (rho + rho.conj().T)/2 to make it hermitian.")
        rho_imperfect = (rho_imperfect + rho_imperfect.conj().T)/2
    if abs(diagnostics["trace"] - 1.0) > 1e-10:
        if print_warnings:
            print("Warning: Input matrix does not have trace one. using rho_new = rho / np.trace(rho) to normalize it.")
        rho_imperfect = rho_imperfect / np.trace(rho_imperfect)

    dim = rho_imperfect.shape[0]
    # Step 1: Eigen-decomposition, arrange eigenvalues/eigenvectors in descending order
    eigenvalues, eigenvectors = np.linalg.eigh(rho_imperfect)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues_corrected = np.zeros_like(eigenvalues)  # to store the corrected eigenvalues
    # Step 2: 
    a = 0  # accumulator
    for i in range(dim, 0, -1):  # 
        _temp = eigenvalues[i-1] + a/i
        if _temp < 0:  # if the corrected eigenvalue is still negative, set it to zero and accumulate the negative part
            eigenvalues_corrected[i-1] = 0
            a += eigenvalues[i-1]
        # if _temp >= 0:  # if the corrected eigenvalue is non-negative, we can set the remaining eigenvalues
        #     eigenvalues_corrected[i-1] = eigenvalues[i-1] + a/i
        if _temp >= 0:  # if the corrected eigenvalue is non-negative, we can stop and set the remaining eigenvalues
            for j in range(i):
                eigenvalues_corrected[j] = eigenvalues[j] + a/i
            break
    # Step 3: Reconstruct the density matrix with the corrected eigenvalues
    # rho_corrected = np.zeros_like(rho_imperfect, dtype=np.complex128)
    # for i in range(dim):
        # rho_corrected += eigenvalues_corrected[i] * np.outer(eigenvectors[:, i], np.conjugate(eigenvectors[:, i]))
    rho_corrected = eigenvectors @ np.diag(eigenvalues_corrected) @ np.conjugate(eigenvectors).T
    # Final check
    diagnostics_corrected = check_density_matrix(rho_corrected)
    if not diagnostics_corrected["is_valid"]:
        print("Warning: The corrected matrix is still not a valid density matrix. Check diagnostics:")
        print(diagnostics_corrected)
    return rho_corrected


def project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a real vector v onto the "probabilistic simplex":
        { x : x_i >= 0, sum(x) = 1 }
 
    This is the O(n log n) algorithm of Duchi et al. (ICML 2008).
    https://dl.acm.org/doi/epdf/10.1145/1390156.1390191
    The result is the closest point to v in L2 norm that is a valid
    probability distribution.
    """
    n = len(v)
    u = np.sort(v)[::-1]                          # sort descending
    cssv = np.cumsum(u)                            # cumulative sum
    # find largest k s.t. u[k] - (cumsum[k] - 1)/(k+1) > 0
    rho = np.nonzero(u > (cssv - 1.0) / np.arange(1, n + 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)         # Lagrange multiplier
    return np.maximum(v - theta, 0.0)
 
 
def closest_density_matrix(rho_raw: np.ndarray) -> np.ndarray:
    """
    Find the density matrix closest to rho_raw in Frobenius (Hilbert-Schmidt)
    norm, by analytically projecting onto the convex set of valid density
    matrices. Returns same result as make_valid_density_matrix, but faster.
 
    Steps
    ------
    1. Symmetrise to enforce Hermiticity.
    2. Eigendecompose (real eigenvalues guaranteed for Hermitian matrices).
    3. Project eigenvalue vector onto the probability simplex.
    4. Reconstruct with the original eigenbasis.
 
    Parameters
    ----------
    rho_raw : ndarray, shape (n, n)
        A matrix close to a density matrix (possibly with small negative
        eigenvalues or trace ≠ 1 due to numerical noise).
 
    Returns
    -------
    rho_valid : ndarray, shape (n, n)
        The nearest valid density matrix in Frobenius norm.
    """
    # 1. Enforce Hermiticity (symmetrise), normalise trace to one
    rho_h = (rho_raw + rho_raw.conj().T) / 2.0
    rho_h /= np.trace(rho_h)
 
    # 2. Eigendecompose — eigh exploits Hermitian structure for speed & stability
    eigenvalues, eigenvectors = np.linalg.eigh(rho_h)
 
    # 3. Project eigenvalues onto the probability simplex
    eigenvalues_proj = project_simplex(eigenvalues)
 
    # 4. Reconstruct: U * diag(λ') * U†
    rho_valid = eigenvectors @ np.diag(eigenvalues_proj) @ eigenvectors.conj().T
 
    return rho_valid

# dm = np.array([[0.9, 0.5], [0.5, 0.1]])
# dm2 = np.array([[0.8, 0.8], [.5, 0.2]])
# # # check_density_matrix(dm, print_info=True)
# print(make_valid_density_matrix(dm2))
# print(closest_density_matrix(dm2))
# print(make_valid_density_matrix(dm2))


def rotate_density_matrix(rho, U):
    """
    rotates a density matrix rho with a unitary U, given by rho' = U rho U^\dagger
    """
    return np.dot(U, np.dot(rho, U.conj().T))

def rotate_HHVV(rho):
    """
    Takes a density matrix in the HH,HV,VH,VV basis and applies a rotation
    such that HH,VV and VV,HH are real and positive.
    Also accepts DMs of higher dimension, then rotates the outmost coherence elements.

    For this, it calculates the phase of the HH,VV element and rotates the density matrix by that phase.
    
    :param rho: density matrix
    """
    dim = rho.shape[0]
    # check if rho is a square matrix and has even dimension
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("Input density matrix must be square.")
    if dim % 2 != 0:
        raise ValueError("Input density matrix must have even dimension.")
    phase = np.angle(rho[0,dim-1])
    dimhalf = int(dim/2)
    U = np.eye(dimhalf, dtype=np.complex128)
    U[dimhalf-1,dimhalf-1] = np.exp(1j*phase)
    rotated = rotate_density_matrix(rho, np.kron(np.eye(dimhalf),U))
    # take real of HH,VV and VV,HH elements to avoid numerical errors
    rotated[0,dim-1] = np.real(rotated[0,dim-1])
    rotated[dim-1,0] = np.real(rotated[dim-1,0])
    return rotated

# rho_unrotated = np.array([[0.5, 0, 0, -0.5j], [0, 0, 0, 0], [0, 0, 0, 0], [0.5j, 0, 0, 0.5]])
# rho_rotated = rotate_HHVV(rho_unrotated)
# print("Unrotated density matrix:")
# print(rho_unrotated)
# print("Rotated density matrix:")
# print(rho_rotated)

def fidelity_pure(rho, psi):
    """
    calculates the fidelity of a density matrix rho with respect to a pure state psi, given by F = <psi|rho|psi>
    """
    return np.real(np.dot(psi.conj().T, np.dot(rho, psi)))


"""
functions below were previously used when ACE required string-operators. 
Now that ACE can work with numpy arrays, these functions are not strictly necessary anymore, but they can still be useful for plotting etc.
"""

def serialize_dm(rho):
    """
    serializes a density matrix into a vector, splitting real and imag parts
    """
    return np.concatenate((np.real(rho).flatten(),np.imag(rho).flatten()))

def deserialize_dm(rho):
    """
    deserializes a density matrix from a vector
    """
    dim = int(np.sqrt(len(rho)/2))
    return rho[:dim**2].reshape((dim,dim)) + 1j*rho[dim**2:].reshape((dim,dim))


def compose_dm(outputs, dim=2):
    """
    composes a density matrix from the output of ACE, with every output-array being the time dynamics for the corresponding output operator
    """
    # dim is the dimension of the system
    rho = np.zeros((len(outputs[0]),dim,dim),dtype=np.complex128)
    n = 1  # start at 1, as the zeroth output is the time axis
    for j in range(dim):
        for k in range(j,dim):
            rho[:,j,k] = outputs[n]
            rho[:,k,j] = np.conjugate(outputs[n])
            n += 1
    t = np.real(outputs[0])
    return t, rho

def generate_basis_states(dim):
        basis_states = []
        indices_range = [range(d) for d in dim]
        # from itertools.product documentation:
        # Cartesian product of input iterables.
        # The nested loops cycle like an odometer with the rightmost element advancing on every iteration. 
        for indices in itertools.product(*indices_range):
            basis_states.append(indices)
        return basis_states

def basis_states(dim):
    # generates readable basis state representation, for use in plotting etc.
    # if dim is no list, make it one
    if not isinstance(dim, list):
        dim = [dim]
    basis_states = generate_basis_states(dim)
    _basis_states = []
    for basis_state in basis_states:
        basis_state_str = '|'
        for index in basis_state:
            basis_state_str += f'{index},'
        basis_state_str = basis_state_str.rstrip(',')
        basis_state_str += '⟩'
        _basis_states.append(basis_state_str)
    return _basis_states

def matrix_element_operators(basis_states, dim, readable=False):
        operators = []
        for i in range(len(basis_states)):
            bra_state = basis_states[i]
            for j in range(i,len(basis_states)):
                ket_state = basis_states[j]
                operator_str = ''
                for k, (bra_index, ket_index) in enumerate(zip(bra_state, ket_state)):
                    if readable:
                        operator_str += f'|{bra_index}⟩⟨{ket_index}|_{dim[k]} ⊗ '
                    else:
                        operator_str += f'|{bra_index}><{ket_index}|_{dim[k]} otimes '
                if readable:
                    operator_str = operator_str.rstrip(' ⊗ ')
                else:
                    operator_str = operator_str.rstrip('otimes ')
                operators.append(operator_str)
        return operators

def output_ops_dm(dim=[2,2], readable=False):
    """
    returns the output operators for a system with n1*n2*n3... levels
    to turn this into a density matrix, use:
    compose_dm(outputs, dim=np.prod(dim))
    can also be used instead of output_ops_dm
    """
    if not isinstance(dim, list) and not isinstance(dim, tuple):
        dim = [dim]
    basis_states = generate_basis_states(dim)
    return matrix_element_operators(basis_states, dim, readable=readable)