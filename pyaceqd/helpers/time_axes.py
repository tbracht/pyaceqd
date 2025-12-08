import numpy as np

def time_axis_to_ndt(t_array, dt):
    """
    takes an array of time values and a time step dt and returns an array of integers
    that represent the indices of the time values in units of dt, i.e.,
    t_array = [0.0, 0.1, 0.2, 0.3], dt=0.1 -> ndt_array = [0, 1, 2, 3]
    also removes duplicates that can occur due to rounding, i.e.,
    t_array = [0.04, 0.16, 0.24], dt=0.1 -> ndt_array = [0, 2]
    
    :param t_array: Description
    :param dt: Description
    """
    t_array = round_to_dt(t_array, dt)
    ndt_array = np.round(t_array/dt).astype(int)
    return ndt_array

def time_axis_to_ndiff_dt(t_array, dt):
    """
    takes an array of time values and a time step dt and returns an array of integers
    that represent the differences of the indices of the time values in units of dt, i.e.,
    t_array = [0.0, 0.1, 0.2, 0.3], dt=0.1 -> ndt_array = [0, 1, 1, 1]
    t_array = [0.0, 0.2, 0.5, 0.9], dt=0.1 -> ndt_array = [0, 2, 3, 4]
    
    :param t_array: Description
    :param dt: Description
    """
    ndt_array = time_axis_to_ndt(t_array, dt)
    if len(ndt_array) == 1:
        return ndt_array
    ndiff_dt_array = np.diff(ndt_array, prepend=ndt_array[0])
    return ndiff_dt_array

def n_dt_to_time_axis(ndt_array, dt):
    """
    takes an array of integers that represent time indices in units of dt and returns
    the corresponding time values, i.e.,
    ndt_array = [0, 1, 2, 3], dt=0.1 -> t_array = [0.0, 0.1, 0.2, 0.3]
    
    :param ndt_array: Description
    :param dt: Description
    """
    t_array = ndt_array.astype(float)*dt
    t_array = round_to_dt(t_array, dt)
    return t_array

def ndiff_dt_to_time_axis(ndiff_dt_array, dt):
    """
    takes an array of integers that represent differences of time indices in units of dt
    and returns the corresponding time values, i.e.,
    ndiff_dt_array = [0, 1, 1, 1], dt=0.1 -> t_array = [0.0, 0.1, 0.2, 0.3]
    ndiff_dt_array = [0, 2, 3, 4], dt=0.1 -> t_array = [0.0, 0.2, 0.5, 0.9]
    
    :param ndiff_dt_array: Description
    :param dt: Description
    """
    ndt_array = np.cumsum(ndiff_dt_array)
    t_array = n_dt_to_time_axis(ndt_array, dt)
    return t_array

def _merge_intervals(intervals):
    """
    assumes intervals sorted by their respective start-value.
    merges them, also merging if the bounds of one interval lie on the next, i.e.,
    [[0,1],[1,2]] -> [[0,2]]
    """
    if len(intervals) > 1:
        # sort the intervals by their start value
        intervals.sort(key=lambda x: x[0])
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
    # Handle both scalar and array inputs
    is_scalar = np.isscalar(t)
    t_array = np.atleast_1d(t)
    
    result = np.round(t_array/dt)*dt
    # remove duplicates that can occur due to rounding
    _, idx = np.unique(result, return_index=True)
    result = result[np.sort(idx)]
    
    # Return scalar if input was scalar
    return result[0] if is_scalar else result
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