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

def construct_t(t0, tend, dt_small=0.1, dt_big=1.0, *pulses, factor_tau=4, simple_exp=False):
    """
    constructs t-axis that has dt_small discretization during the pulses and dt_big otherwise.
    standard factor is 4, i.e., -4*tau_pulse,..,4*tau_pulse
    """
    # put t0 and tau in arrays to sort them
    t0s = []
    taus = []
    for _p in pulses:
        t0s.append(_p.t0)
        taus.append(_p.tau)
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
        ts.append(np.arange(intervals[0][0],intervals[0][1],dt_small))
        ts.append(np.exp(np.arange(np.log(intervals[0][1]),np.log(tend),dt_small)))
        return np.concatenate(ts,axis=0)
    for i in range(len(intervals)):
        if i > 0:
            ts.append(np.arange(intervals[i-1][1],intervals[i][0],dt_big))
        ts.append(np.arange(intervals[i][0],intervals[i][1],dt_small))
    ts.append(np.arange(intervals[-1][1],tend,dt_big))
    ts.append(np.array([tend]))
    return np.concatenate(ts,axis=0)



def export_csv(filename, *arg, precision=4, delimit=','):
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
        # print("[i] csv saved to {}".format(filename))
    except TypeError:
        print("TypeError occured")
        for arguments in arg:
            print(arguments)
