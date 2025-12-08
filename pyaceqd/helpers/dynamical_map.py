import numpy as np
import matplotlib.pyplot as plt

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
    # print(np.allclose(tl_map, dm_1[-1]))
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
    plt.clf()
