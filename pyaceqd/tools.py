import numpy as np

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

def construct_t(t0, tend, dt_small=0.1, dt_big=1.0, *pulses, factor_tau=4, simple_exp=False, gaussian_t=False):
    """
    constructs t-axis that has dt_small discretization during the pulses and dt_big otherwise.
    standard factor is 4, i.e., -4*tau_pulse,..,4*tau_pulse
    """
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
        _exp_part = np.exp(np.arange(np.log(intervals[0][1]),np.log(tend),dt_small))
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
    ts.append(np.array([tend]))
    return np.concatenate(ts,axis=0)

def simple_t_gaussian(t0, texp, tend, dt_small=0.1, dt_big=1.0, *pulses, decimals=2, exp_part=True):
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
        ts.append(np.arange(texp,tend,10*dt_small))
    ts.append(np.array([tend]))
    return np.round(np.concatenate(ts,axis=0), decimals=decimals)  

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

def output_ops_dm(dim=2):
    """
    returns the output operators for a system with dim levels
    """
    ops = []
    for i in range(dim):
        for j in range(i,dim):
            ops.append("|{}><{}|_{}".format(i,j,dim))
    return ops

def compose_dm(outputs, dim=2):
    """
    composes a density matrix from the output of ACE, with every output-array being the time dynamics for the corresponding output operator
    """
    # dim is the dimension of the system
    rho = np.zeros((len(outputs[0]),dim,dim),dtype=np.complex128)
    n = 1  # start at 0, as the zeroth output is the time axis
    for j in range(dim):
        for k in range(j,dim):
            rho[:,j,k] = outputs[n]
            rho[:,k,j] = np.conjugate(outputs[n])
            n += 1
    return np.real(outputs[0]), rho
