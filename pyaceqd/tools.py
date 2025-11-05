import numpy as np
import itertools
import configparser 
import re
from functools import wraps
from typing import Optional
import matplotlib.pyplot as plt

def _merge_intervals(intervals):
    """
    assumes intervals sorted by their respective start-value.
    merges them, also merging if the bounds of one interval lie on the next, i.e.,
    [[0,1],[1,2]] -> [[0,2]]
    """
    if len(intervals) > 1:
        for i in range(len(intervals)-1):
            if intervals[i][1] >= intervals[i+1][0]:
                # catch the case, that the whole interval i+1 is contained in i
                _max_t = np.max([intervals[i][1],intervals[i+1][1]])
                intervals[i][1] = _max_t
                del intervals[i+1]
                # not super efficient, because it always loops over every interval
                # instead of using something like i_start = i-1
                _merge_intervals(intervals)
                break
    return intervals

def get_gaussian_t(t0,tend,*pulses,dt_max=1.0,dt_min=0.01,interval_per_step=0.05):
    t_array = [t0]
    t_test = np.arange(t0,tend,dt_min)  # from this it implicitly follows that also the output has this dt_min
    counter = 0
    n_max = int(dt_max/dt_min)
    interval_contains = 0
    intfunc = lambda t: np.sum([p.get_integral(t) for p in pulses])
    for i in range(1,len(t_test)):
        t_now = t_test[i]
        t_prev = t_test[i-1]
        interval_contains += intfunc(t_now) - intfunc(t_prev)
        counter += 1
        if interval_contains >= interval_per_step or counter == n_max:
            t_array.append(t_now)
            counter = 0
            interval_contains = 0
    return np.array(t_array)

def construct_t(t0, tend, dt_small=0.1, dt_big=1.0, dt_exp=None, *pulses, factor_tau=4, simple_exp=False, gaussian_t=False, add_tend=True):
    """
    constructs t-axis that has dt_small discretization during the pulses and dt_big otherwise.
    standard factor is 4, i.e., -4*tau_pulse,..,4*tau_pulse
    """
    if dt_exp is None:
        dt_exp = dt_small
    # put t0 and tau in arrays to sort them
    t0s = []
    taus = []
    for _p in pulses:
        if _p.t0 < tend and _p.t0 > t0:
            t0s.append(_p.t0)
            taus.append(_p.tau)
        else:
            if _p.t0 > tend:
                print("WARNING: tend is smaller than the end of a pulse")
            if _p.t0 < t0:
                print("WARNING: t0 is greater than the start of a pulse")
    # sort taus with respect to t0s
    t0s = np.array(t0s)
    taus = np.array(taus)
    start_v = t0s - factor_tau*taus  # values, where the intervals with small timestep start
    end_v = t0s + factor_tau*taus  # values, where the intervals with small timestep end
    _temp = list(sorted(zip(start_v,end_v)))  # sort them with respect to the start value
    start_v,end_v = zip(*_temp)
    start_v = list(start_v)
    end_v = list(end_v)
    intervals = []  # set up the intervals
    for _ts,_te in zip(start_v,end_v):
        intervals.append([_ts,_te])
    intervals = _merge_intervals(intervals)  # merges intervals if they overlap
    if intervals[0][0] < t0:
        print("WARNING: t0 is greater than the start of the first pulse")
    if intervals[-1][1] > tend:
        print("WARNING: tend is smaller than the end of the last pulse")
    ts = []  # array where the time-axes are stored
    # use, that arange:
    # 1) gives an empty array, if tstart=tend
    # 2) does not include the final value
    ts.append(np.arange(t0,intervals[0][0],dt_big))
    if simple_exp and len(intervals) == 1 and intervals[0][1] != 0:
        if gaussian_t: 
            # interval_per_step: "1/steps per pi pulse area"
            t_gaussian = get_gaussian_t(intervals[0][0],intervals[0][1],*pulses,dt_max=dt_big,dt_min=dt_small,interval_per_step=0.05)
            ts.append(t_gaussian)
        else:
            ts.append(np.arange(intervals[0][0],intervals[0][1],dt_small))
        _exp_part = np.exp(np.arange(np.log(intervals[0][1]),np.log(tend),dt_exp))
        # make sure that there are no crazy numbers, round the exponentially spaced part to 2 decimals. somehow this seems to work best
        # even better than rounding to the step-size dt_small
        ts.append(np.round(_exp_part))
        ts.append(np.array([tend]))
        return np.concatenate(ts,axis=0)  # np.round(np.concatenate(ts,axis=0), decimals=2)  
    for i in range(len(intervals)):
        if i > 0:
            ts.append(np.arange(intervals[i-1][1],intervals[i][0],dt_big))
        ts.append(np.arange(intervals[i][0],intervals[i][1],dt_small))
    ts.append(np.arange(intervals[-1][1],tend,dt_big))
    if add_tend:
        ts.append(np.array([tend]))
    return np.concatenate(ts,axis=0)

def round_to_dt(t, dt):
    """
    rounds the time array t to the nearest multiple of dt
    """
    result = np.round(t/dt)*dt
    # remove duplicates that can occur due to rounding
    _, idx = np.unique(result, return_index=True)
    return result[np.sort(idx)]
    # return np.round(t/dt)*dt

def simple_t_gaussian(t0, texp, tend, dt_small=0.1, dt_big=1.0, *pulses, decimals=2, exp_part=True, add_tend=True):
    """
    uses gaussian timespacing from t0,...,texp, then exponential timespacing from
    texp,...,tend
    """
    ts = []
    t_gaussian = get_gaussian_t(t0,texp,*pulses,dt_max=dt_big,dt_min=dt_small,interval_per_step=0.05)
    ts.append(t_gaussian)
    if exp_part:
        t_exp = np.exp(np.arange(np.log(texp-t0),np.log(tend-t0),dt_small))+t0
        ts.append(t_exp)
    else:
        ts.append(np.arange(texp,tend,dt_big))
    if add_tend:
        ts.append(np.array([tend]))
    return round_to_dt(np.concatenate(ts,axis=0), dt_small)
    # return np.round(np.concatenate(ts,axis=0), decimals=decimals)  

def export_csv(filename, *arg, precision=4, delimit=',', verbose=False):
    """
    Exportiert Arrays als .csv Datei
    :param delimit: delimiter 
    :param filename: filename
    :param precision: number of decimal places after which the number is truncated
    :return: null
    """
    p = '%.{k}f'.format(k=precision)
    ps = []
    for arguments in arg:
        ps.append(p)
    try:
        np.savetxt(
            filename,
            np.c_[arg],
            fmt=ps,
            delimiter=delimit,
            newline='\n',
            # footer='end of file',
            # comments='# ',
            # header='X , MW'
        )
        if verbose:
            print("[i] csv saved to {}".format(filename))
    except TypeError:
        print("TypeError occured")
        for arguments in arg:
            print(arguments)

def concurrence(rho):
        T_matrix = np.flip(np.diag([-1.,1.,1.,-1.]),axis=1)  # antidiagonal matrix
        M_matrix = np.dot(rho,np.dot(T_matrix,np.dot(np.conjugate(rho),T_matrix)))
        _eigvals = np.real(np.linalg.eigvals(M_matrix))
        _eigvals = np.sqrt(np.sort(_eigvals))
        return np.max([0.0,_eigvals[-1]-np.sum(_eigvals[:-1])])

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

def op_to_matrix(op):
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

# print(op_to_matrix("(|0><1|_2)"))  # Example usage, should return a 2x2 matrix with the appropriate values

def read_calibration_file(calibration_file):
    
    # reads in experimentally aquired quantum dot parameters 
    config = configparser.ConfigParser()
    config.read(calibration_file)

    # read the calibration file
    central_wavelength = float(config['EMISSION']['exciton_wavelength']) #nm
    biexciton_wavelength = float(config['EMISSION']['biexciton_wavelength'])
    dark_wavelength = float(config['EMISSION']['dark_wavelength']) 

    fss_bright = float(config['SPLITTING']['fss_bright'])*1e-3 #meV
    fss_dark = float(config['SPLITTING']['fss_dark']) *1e-3 # meV 

    lifetime_exciton = float(config['LIFETIMES']['exciton']) #ps
    lifetime_biexciton = float(config['LIFETIMES']['biexciton'])
    #lifetime_dark = float(config['LIFETIMES']['dark']) 

    g_ex = float(config['G_FACTORS']['g_ex'])
    g_hx = float(config['G_FACTORS']['g_hx'])
    g_ez = float(config['G_FACTORS']['g_ez'])
    g_hz = float(config['G_FACTORS']['g_hz'])

    exciton_meV = 1239.8*1e3/central_wavelength #meV
    biexciton_meV = 1239.8*1e3/biexciton_wavelength
    dark_meV = 1239.8*1e3/dark_wavelength

    exciton_x_energy = fss_bright/2
    exciton_y_energy = -fss_bright/2
    binding_energy = -(exciton_meV - biexciton_meV) # negatively defined
    dark_energy = (dark_meV - exciton_meV)
    dark_x_energy = dark_energy + fss_dark/2
    dark_y_energy = dark_energy - fss_dark/2 

    gamma_e = 1/lifetime_exciton
    gamma_b = 1/(lifetime_biexciton*2)
    gamma_d = 0 #1/lifetime_dark

    return exciton_x_energy, exciton_y_energy, dark_x_energy, dark_y_energy, binding_energy, gamma_e, gamma_b, gamma_d, g_ex, g_hx, g_ez, g_hz



    return exciton_x_energy, exciton_y_energy, dark_x_energy, dark_y_energy, binding_energy, gamma_e, gamma_b, gamma_d, g_ex, g_hx, g_ez, g_hz

def resample(x, y, z, s_x, s_y):
    """Resample a 2D array with different sampling rates for x and y dimensions.
    
    Args:
        x (np.ndarray): X-axis values
        y (np.ndarray): Y-axis values
        z (np.ndarray): 2D array of values
        s_x (int): Sampling rate for x-axis
        s_y (int): Sampling rate for y-axis
        
    Returns:
        tuple: (x_new, y_new, z_new) resampled arrays
    """
    x_new = np.zeros(int((len(x))/s_x))
    y_new = np.zeros(int((len(y))/s_y))
    z_new = np.zeros((len(y_new), len(x_new)))
    for i in range(len(x_new)):
        for j in range(len(y_new)):
            x_new[i] = x[int(i*s_x)]
            y_new[j] = y[int(j*s_y)]
            z_new[j,i] = z[int(j*s_y),int(i*s_x)]
    return x_new, y_new, z_new

def rotate_basis(rho, U_rot):
    """
    Rotates the basis of a density matrix rho using a unitary transformation U_rot:

    $\\rho_{rotated} = U_{rot} @ \\rho @ U_{rot}^\\dagger$

    It can for example be used if the system includes a magnetic field, which leads to mixing of the basis states.
    Then, the transformation matrix U_rot can be calculated from the eigenvectors of the Hamiltonian.
    These can, for example, be calculated using the ```dressed_states``` functions.
    For the six-level system:
    ```
    t, _,_,_,_,_, e_vectors, rho = sixls_linear_dressed_states(0, 1, p1, lindblad=True, bx=bx, return_eigenvectors=True)
    rho_rotated = rotate_basis(rho, e_vectors[0])
    ```


    Args:
        rho (np.ndarray): Density matrix to be rotated.
        U_rot (np.ndarray): Unitary transformation matrix.
        
    Returns:
        np.ndarray: Rotated density matrix.
    """
    return U_rot @ rho @ U_rot.conj().T


# def trunc_svd_inv(A, lam=1e-8):
#     U, s, Vh = np.linalg.svd(A, full_matrices=False)
#     s_reg = s / (s**2 + lam)
#     return Vh.T @ np.diag(s_reg) @ U.T.conj() @ B



# def truncated_svd_inv(A, lam=1e-8):
#     """
#     Computes the pseudo-inverse of a matrix using truncated SVD.
    
#     Args:
#         A (np.ndarray): Input matrix to be inverted.
#         lam (float): Regularization parameter to avoid division by zero.
        
#     Returns:
#         np.ndarray: Pseudo-inverse of the input matrix A.
#     """
#     u,s,vh = np.linalg.svd(A)
#     print("SVD singular values:", s)
#     A_inv = np.zeros_like(A)
#     for i in range(len(s)):
#         if s[i] > lam:
#             A_inv += (1/s[i]) * np.outer(vh[i,:], u[:,i])
#     return A_inv

# a = np.array([[1, 2,3,4], [3, 4,5,6],[1, 2,5,4],[1, 2,5,4.00000001]], dtype=complex)
# a_inv = np.linalg.inv(a)
# print("Inverse of a:\n", a_inv)

# # u,s,vh = np.linalg.svd(a)

# # a_rec = np.zeros_like(a,dtype=float)
# # for i in range(len(s)):
# #     a_rec += s[i] * np.outer(u[:,i], vh[i,:])
# # a_inv_svd = np.zeros_like(a,dtype=float)
# # for i in range(len(s)):
# #     a_inv_svd += 1/s[i] * np.outer(vh[i,:], u[:,i])

# # print("Reconstructed a from SVD:\n", a_rec)
# print("Inverse of a using SVD:\n", truncated_svd_inv(a, lam=1e-12))

# print(a@a_inv)
# print(a@truncated_svd_inv(a, lam=1e-18))

def calc_tl_dynmap_pseudo(dm, times, debug=False):
    """
    Calculate the time-local dynamical map for a given dynamical map.
    Singular matrices are handled by using the pseudo-inverse.
    The time-local dynamical map is defined as:
        _dm_tl[i] = E_ti+1,ti, i.e., _dm_tl[i]*rho(ti) = rho(ti+1)
    where E_ti+1,ti = E_ti+1,t0 * E_ti,t0^-1
    Args:
        dm (np.ndarray): Dynamical map array of shape (n_t, n_h^2, n_h^2), where n_h is the Hilbert space dimension 
        times (np.ndarray): Time array of shape (n_t)
        debug (bool): If True, print information if a singular matrix is encountered
    Returns:
        np.ndarray: Time-local dynamical map (shape (n_t-1, n_h^2, n_h^2))
    """
    times = np.round(times, 4)
    n = dm.shape[1]
    # input:
    # dm[i] = E_t(i+1),t0, i.e., dm[i]*rho0 = rho(ti+1)
    # most importantly, rho[1] = dm[0]*rho0
    # output:
    # _dm_tl[i] = E_ti+1,ti, i.e., _dm_tl[i]*rho(ti) = rho(ti+1)
    _dm_tl = np.zeros((len(times)-1,n,n),dtype=complex)
    # rho[1] = dm[0]*rho0, dm[0] = E_t1,t0
    # E_ti+1,ti = E_ti+1,t0 * E_ti,t0^-1
    # i.e., _dm_tl[i] = dm[i] * dm[i-1]^-1
    # with dm[-1] = identity so we can use dm[0] as the first element
    _dm_tl[0] = dm[0]
    for i in range(1,len(_dm_tl)):
        try:
            # _c = np.linalg.cond(dm[i])
            # if _c > 1e6:
            #_dm_tl[i] = np.dot(dm[i],truncated_svd_inv(dm[i-1], lam=1e-10))
            # else:
            _dm_tl[i] = np.dot(dm[i],np.linalg.pinv(dm[i-1],rcond=1e-12))
        except np.linalg.LinAlgError:
            _dm_tl[i] = np.dot(dm[i],np.linalg.pinv(dm[i-1]))
            if debug:
                print("Singular matrix at time = {}, index: {}, Tr(dm_tl[i])={}".format(times[i], i, np.real(np.trace(np.dot(_dm_tl[i],0.5*np.ones(4)).reshape(2,2)))))
    return _dm_tl

def extract_dms(dm, times, tau_c, t_MTOs):
    """
    Extracts the time-local dynamical map and the dynamical maps for the time before and after the MTO.
    
    Parameters:
    --------
        dm (np.ndarray): 
            Dynamical map array of shape (n_t, n_h^2, n_h^2), where n_h is the Hilbert space dimension 
        times (np.ndarray): 
            Time array of shape (n_t)
        tau_c (float): 
            Correlation time
        t_MTOs (list): 
            List of times at which the MTO is applied

    Returns:
    --------
        tl_map (np.ndarray): 
            Time-local dynamical map, shape (n_h^2, n_h^2)
        tl_dms (np.ndarray): 
            Dynamical maps for the time before and after the MTO, shape (n_tauc, n_h^2, n_h^2)
            where n_tauc is the number of time steps in the memory time tau_c. 
    """
    # extract the dynamical map for the first tau_c
    # dt = times[1] - times[0]
    i_timelocal = np.where(times > times[0]+tau_c)[0][0]
    len_tauc = i_timelocal  # int(tau_c/dt)
    # print(i_timelocal, len_tauc)

    # the MTO is included in the dynamical map at t_MTO
    # i.e., it first comes into effect to the density matrix at t_MTO+dt
    i_tmtos = []
    for t_MTO in t_MTOs:
        try:
            i_tmtos.append(np.where(times == t_MTO)[0][0])
        except IndexError:
            print(f"Available times: {times}")
            print(f"Requested t_MTO: {t_MTO}")
            raise ValueError(f"t_MTO {t_MTO} not found in times array. Make sure that t_MTO is included in the times array.")
            
    # i_tmto = np.where(times == t_MTO)[0][0]
    
    # extract the dynamical map for the time before the MTO
    tl_dms = []

    dm_1 = dm[:len_tauc]
    tl_dms.append(dm_1)
    # extract the dynamical map for the time after the MTO
    for i_tmto in i_tmtos:
        dm_2 = dm[i_tmto:i_tmto+len_tauc]
        tl_dms.append(dm_2)
    # dm_2 = dm[i_tmto:i_tmto+len_tauc]
    # extract the time-local dynamical map
    tl_map = dm[i_timelocal]
    # with np.printoptions(precision=4, suppress=True):
    #     print(dm[i_tmtos[0]-2])
    #     print(dm[i_tmtos[0]-1])
    #     print(dm[i_tmtos[0]])
    #     print(dm[i_tmtos[0]+1])
    return tl_map, tl_dms

def check_tl_map_params(tl_map, rho0):
    """
    checks the parameters for the time-local map
    tl_map: time-local dynamical map
    times: time array. note that we round the times to 5 decimals
    rho0: initial density matrix
    tau_c: correlation time

    the idea is to use:
    1) the dynamical map for the first tau_c
    2) the time-local map for the rest of the time 
    """
    # rho0 must be quadratic matrix
    n = int(rho0.shape[0])
    if rho0.shape[1] != n:
        raise ValueError("rho0 must be a {n}x{n} matrix")
    if tl_map.shape != (n**2, n**2):
        raise ValueError("tl_map must be a {}x{} matrix, is {}".format(n**2, n**2, np.shape(tl_map)))
    return n

def use_tl_map(tl_map, times, rho0):
    """
    inputs:
    tl_map: time-local dynamical map
    times: time array.
    rho0: initial density matrix
    tau_c: correlation time

    the idea is to use:
    1) the dynamical map for the first tau_c
    2) the time-local map for the rest of the time 
    """
    # rho0 must be quadratic matrix
    n = check_tl_map_params(tl_map, rho0)
    rho = np.zeros((len(times),n,n),dtype=complex)
    rho = rho.reshape(len(times),n**2)
    rho[0] = rho0.reshape(n**2)
    
    # from 0 to tau_c or t_MTO
    for i in range(len(times)-1):
        rho[i+1] = np.dot(tl_map,rho[i])
    return rho.reshape(len(times),n,n)

def use_dm_block(dm, rho0):
    """
    inputs:
    dm: dynamical map
    rho0: initial density matrix

    returns:
    rho: density matrix at each time step
    """
    # rho0 must be quadratic matrix
    n = check_tl_map_params(dm[0], rho0)
    rho = np.zeros((len(dm)+1,n,n),dtype=complex)
    rho = rho.reshape(len(dm)+1,n**2)
    rho[0] = rho0.reshape(n**2)
    
    # from 0 to tau_c or t_MTO
    for i in range(len(dm)):
        rho[i+1] = np.dot(dm[i],rho[i])
    return rho.reshape(len(dm)+1,n,n)

def tl_pad_stationary(tl_map, times, rho):
    n = check_tl_map_params(tl_map, rho[0])
    rho_complete = np.zeros((len(times),n,n),dtype=complex)
    rho_complete[:len(rho)] = rho
    rho_complete = rho_complete.reshape(len(times),n**2)

    for i in range(len(rho), len(times)):
        rho_complete[i] = np.dot(tl_map,rho_complete[i-1])
    return rho_complete.reshape(len(times),n,n)

def tl_pad_stationary_nsteps(tl_map, n_steps, rho):
    n = check_tl_map_params(tl_map, rho)
    rho_complete = np.zeros((n_steps,n,n),dtype=complex)
    rho_complete[:len(rho)] = rho
    rho_complete = rho_complete.reshape(n_steps,n**2)

    for i in range(len(rho), n_steps):
        rho_complete[i] = np.dot(tl_map,rho_complete[i-1])
    return rho_complete.reshape(n_steps,n,n)

def use_tl_map_mto(tl_map, dm_1, dm_2, times, rho0, t_MTO, debug=False):
    """
    inputs:
    tl_map: time-local dynamical map
    dm_1: dynamical map before the MTO. should have length tau_c/dt
    dm_2: dynamical map after the MTO
    times: time array. note that we round the times to 5 decimals
    rho0: initial density matrix
    tau_c: correlation time
    t_MTO: time at which the MTO is applied

    the idea is to use:
    1) the dynamical map for the first tau_c
    2) the time-local map for the rest of the time until the MTO is applied
    3) the MTO-dynamical map for t_MTO until t_MTO+tau_c
    4) the time-local map for the rest of the time 
    """
    # rho0 must be quadratic matrix
    n = check_tl_map_params(tl_map, rho0)
    n_sq = n**2
    rho = np.zeros((len(times),n,n),dtype=complex)
    rho = rho.reshape(len(times),n_sq)
    rho[0] = rho0.reshape(n_sq)
    times = np.round(times, 5)
    
    i_mto = np.where(times >= t_MTO)[0][0]

    if debug:
        print("info on piecewise application: ",i_mto, times[i_mto], len(dm_1), len(dm_2))

    i_dm1 = np.min([i_mto, len(dm_1)])
    if i_mto < len(dm_1):
        print("caution: t_MTO is smaller than tau_c")
    # from 0 to tau_c or t_MTO
    for i in range(i_dm1):
        rho[i+1] = np.dot(dm_1[i],rho[i])
    # from tau_c to t_MTO
    for i in range(i_dm1,i_mto):
        rho[i+1] = np.dot(tl_map,rho[i])
    # from t_MTO to t_MTO+tau_c
    for i in range(i_mto,i_mto+len(dm_2)):
        rho[i+1] = np.dot(dm_2[i-i_mto],rho[i])
    # from t_MTO+tau_c to the end
    for i in range(i_mto+len(dm_2),len(times)-1):
        rho[i+1] = np.dot(tl_map,rho[i])
    return rho.reshape(len(times),n,n)

def check_tlmap_frobenius(tl_map, times, filename="dynmap_tl_frobenius",xlim=25, check_against_i=None):
    norms_tl = np.zeros((len(times)-3),dtype=float)
    for i in range(len(times)-3):
        if check_against_i is not None:
            norms_tl[i] = np.linalg.norm(tl_map[i]-tl_map[check_against_i])
        else:
            norms_tl[i] = np.linalg.norm(tl_map[i]-tl_map[i+1])
    # relevant indices: 
    #print("len(times): ", len(times))
    #print("len(norms_tl): ", len(norms_tl))

    ix = np.where((times-times[0] > 0) & (times-times[0] < xlim))[0]
    #print(ix)
    plt.clf()
    plt.xlabel("Time")
    plt.ylabel("Norm")
    plt.title("difference of adjacent dynamical maps")
    # plt.plot(times[1:-2]-times[0], norms_tl)
    plt.plot(times[ix]-times[0], norms_tl[ix-1])
    # plt.legend(loc="upper right")
    # plt.xlim(1000,1150)
    # plt.ylim(0.8*np.min(norms_tl),1.2*np.max(norms_tl))
    plt.yscale('log')
    plt.xlim(0,xlim)
    plt.savefig(filename+"_diff.png")
    plt.clf()

    norms_tl = np.zeros((len(tl_map)),dtype=float)
    for i in range(len(tl_map)):
        norms_tl[i] = np.linalg.norm(tl_map[i])

    plt.clf()
    plt.xlabel("Time")
    plt.ylabel("Norm")
    plt.title("Norm of dynamical maps")
    plt.plot(times[ix]-times[0], norms_tl[ix])
    plt.yscale('log')
    plt.tight_layout()
    plt.xlim(0,xlim)
    plt.savefig(filename+"_norms.png")
    plt.clf()

    # calculate singular values of the dynamical maps
    sv_tl = np.zeros((len(tl_map),len(tl_map[0])),dtype=float)
    for i in range(len(tl_map)):
        sv_tl[i] = np.linalg.svd(tl_map[i], compute_uv=False)
    plt.clf()
    plt.xlabel("Time")
    plt.ylabel("Singular values")
    plt.title("Singular values of dynamical maps")
    for i in range(len(sv_tl[0])):
        plt.plot(times[ix]-times[0], sv_tl[ix,i], label=f"sv {i+1}")
    # plt.legend(loc="upper right")
    plt.yscale('log')
    # plt.tight_layout()
    plt.ylim(1e-30,1e2)
    plt.xlim(0,xlim)
    plt.savefig(filename+"_sv.png")

def nm_to_mev(lambda_light):
    _HBAR = 0.6582119514  # meV ps
    _c_light = 299.792e3  # nm/ps
    return _HBAR * 2*np.pi*_c_light / lambda_light

def mev_to_nm(energy_light):
    _HBAR = 0.6582119514  # meV ps
    _c_light = 299.792e3  # nm/ps
    return _HBAR * 2*np.pi*_c_light / energy_light

def ghz_to_mev(ghz):
    """
    Convert frequency in GHz to energy in meV.
    
    Parameters:
    ghz (float): Frequency in GHz.
    
    Returns:
    float: Energy in meV.
    """
    h = 2*np.pi * 0.6582119514  
    return ghz * h * 1e-3  # Convert GHz to meV using hbar

def mev_to_ghz(mev):
    """
    Convert energy in meV to frequency in GHz.
    
    Parameters:
    mev (float): Energy in meV.
    
    Returns:
    float: Frequency in GHz.
    """
    h = 2*np.pi * 0.6582119514
    return mev / (h * 1e-3)  # Convert meV to GHz using hbar

def with_filename(func):
    @wraps(func)
    def wrapper(
        start: float = 0.1,
        stop: float = 12,
        num: int = 101,
        nth: int = 10,
        get_inverse: bool = False,
        round_to: int = 8,
        filename: Optional[str] = None
    ):
        result = func(start, stop, num, nth, get_inverse, round_to)
        if filename is not None:
            suffix = "_inverse" if get_inverse else "_sparse"
            return result, filename + suffix
        return result
    return wrapper

@with_filename
def get_sparse_range(start=0.1, stop=12, num=101, nth=10, get_inverse=False,round_to=8):
    range_full = np.linspace(start, stop, num)
    range_sparse = range_full[::nth]
    if get_inverse:
        # returns range_full without the values in range_sparse
        # use sets: contain only unique values
        range_sparse_set = set(range_sparse)
        range_full_set = set(range_full)
        range_inverse = range_full_set - range_sparse_set  # set difference
        # range_inverse = [x for x in range_full if x not in range_sparse_set]
        return np.round(sorted(range_inverse),round_to)
    return range_sparse

def get_union(arr_x1, arr_x2, arr_z1, arr_z2, axis_z=None):
    # Get the union of arr_x1 and arr_x2 and sort the result.
    # array_z is the array of z values corresponding to arr_x1 and arr_x2
    # so array_z should also be sorted according to the union of arr_x1 and arr_x2.
    len_x1 = len(arr_x1)
    len_x2 = len(arr_x2)
    shape_z1 = arr_z1.shape
    shape_z2 = arr_z2.shape
    if len(shape_z1) == 1:
        arr_z1 = arr_z1.reshape((len_x1, 1))
        shape_z1 = arr_z1.shape
    if len(shape_z2) == 1:
        arr_z2 = arr_z2.reshape((len_x2, 1))
        shape_z2 = arr_z2.shape
    if axis_z is None:
        if shape_z1[0] == shape_z1[1]:
            return ValueError("Cannot determine axis for z arrays.")
        if shape_z1[0] == len_x1 and shape_z2[0] == len_x2:
            axis_z = 0
        elif shape_z1[1] == len_x1 and shape_z2[1] == len_x2:
            axis_z = 1
        else:
            raise ValueError("Cannot determine axis for z arrays.")
    arr_x = np.concatenate((arr_x1, arr_x2))
    arr_z = np.concatenate((arr_z1, arr_z2), axis=axis_z)
    arr_x, indices = np.unique(arr_x, return_index=True)
    arr_z = arr_z[indices]
    return arr_x, arr_z
