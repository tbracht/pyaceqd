from asyncio import futures
import subprocess
import numpy as np
import os
from pyaceqd.tools import export_csv, construct_t
import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from pyaceqd.two_level_system.tls import tls_
from pyaceqd.pulses import ChirpedPulse

HBAR = 0.6582119514  # meV ps

def G1_twols(t0=0, tend=600, tau0=0, tauend=600, dt=0.1, dtau=0.5, *pulses, ae=3.0, temperature=4, gamma_e=1/100, phonons=False, pt_file=None, workers=10, temp_dir='/mnt/temp_data/', coarse_t=False):
    # pulse file generation
    _t_pulse = np.arange(t0,tend+tauend,step=dtau)
    pulse_file = temp_dir + "tls_G1_pulse.dat"
    pulse = np.zeros_like(_t_pulse, dtype=complex)
    for _p in pulses:
        pulse = pulse + _p.get_total(_t_pulse)
    export_csv(pulse_file, _t_pulse, pulse.real, pulse.imag, precision=8, delimit=' ')
    output_ops = ["|1><1|_2","|1><0|_2"]  # first: special case tau=0:
                                          # Tr(sigma^dagger*sigma * rho) = x, G1(t,0) = x(t)
                                          # second: Tr(sigma^dagger * rho) =  <|x><g|> = pxg
    options = {"gamma_e": gamma_e, "phonons": phonons, "ae": ae, "temperature": temperature, "lindblad": True, "pt_file": pt_file, "temp_dir": temp_dir,
               "pulse_file": pulse_file, "stream": True, "output_ops": output_ops}
    multitime_op = {"operator": "|0><1|_2","applyFrom": "_left", "applyBefore": "false"}
    t, tau, g1 = G1_general(t0,tend,tau0,tauend,dt,dtau,*pulses,system=tls_,multitime_op=multitime_op,coarse_t=coarse_t,workers=workers,**options)
    os.remove(pulse_file)
    return t, tau, g1

def G1_general(t0=0, tend=600, tau0=0, tauend=600, dt=0.1, dtau=0.02, *pulses, system=tls_, multitime_op={"operator": "|1><0|_2","applyFrom": "left"}, coarse_t=False, workers=10, **options):
    # includes tend
    t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    n_tau = int((tauend-tau0)/dtau)
    tau = np.linspace(tau0, tauend, n_tau + 1)
    # on t-axis: good resolution during the pulses, less resolution after/between them
    if coarse_t:
        t = construct_t(t0, tend, dt, 10*dt, *pulses)

    if options["phonons"]:
        if options["pt_file"] is None or not os.path.exists(options["pt_file"]+"_initial"):
            print("calculating pt file for G1")
            system(0,40,*pulses,dt=dtau,verbose=True,**options)
        else:
            print("using pt_file {}".format(options["pt_file"]))

    _G1 = np.zeros([len(t),len(tau)],dtype=complex)
    # G1 part
    with tqdm.tqdm(total=len(t)) as tq:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(len(t)):
                # remember to only apply sigma from the left for G1
                multitime_op_new = dict(multitime_op)  # must make a copy of the dict
                multitime_op_new["time"] = t[i]
                _e = executor.submit(system, t0,t[i] + tauend, *pulses, dt=dtau, suffix=i, multitime_op=multitime_op_new, **options)
                _e.add_done_callback(lambda f: tq.update())
                futures.append(_e)
            # wait for all futures
            wait(futures)
        for i in range(len(futures)):
            # futures are still 'future' objects
            futures[i] = futures[i].result()
        # futures now contains [t,g,x,pgx,pxg] for every i
        for i in range(len(t)):
            # for TLS:
            # futurs[i] is [t,x,pxg]
            # futures[i][2] are the pgx values
            # futures[i][1] are the x values
            # special case tau=0:
            # as Tr(sigma^dagger*sigma * rho) = x, G1(t,0) = x(t), which is the value with index [-(n_tau+1)]
            _G1[i,0] = futures[i][1][-n_tau-1]
            # as Tr(sigma^dagger * rho) =  <|x><g|> = pxg
            _G1[i,1:] = futures[i][2][-n_tau:]  # the last n_tau values
    return t, tau, _G1

def pulsed_mollow_tls(pulse_tau, areas, tend=500, tauend=500, dt=0.2, dtau=0.02, gamma_e=1/100, ae=3.0, temperature=4, phonons=False, pt_file="tls_3.0nm_4k_th10_tmem20.48_dt0.02.ptr", workers=7, temp_dir='/mnt/temp_data/',save_dir=None):
    n_tau = int((tauend)/dtau)
    tau_axis = np.linspace(0, tauend, n_tau + 1)
    spectrums = np.empty([len(areas),2*len(tau_axis)-1])
    fft_freqs = -2*np.pi * HBAR * np.fft.fftfreq(2*len(tau_axis)-1,d=dtau)
    for i in range(len(areas)):
        print("{}/{}".format(i+1,len(areas)))
        p1 = ChirpedPulse(tau_0=pulse_tau, e_start=0, alpha=0, e0=areas[i], t0=pulse_tau*4)
        t_axis, tau_axis, g1 = G1_twols(0,tend,0,tauend,dt,dtau,p1,ae=ae,gamma_e=gamma_e,coarse_t=True,phonons=phonons, workers=workers, temperature=temperature, pt_file=pt_file, temp_dir=temp_dir)
        # plt.pcolormesh(t_axis,tau_axis,np.real(g1.transpose()),shading='auto')
        # plt.xlabel("t in ps")
        # plt.ylabel("tau in ps")
        # plt.title("area:{:.4f} pi".format(areas[i]))
        # plt.colorbar()
        # plt.savefig("data_test/g1_{}_r.png".format(i))
        # plt.clf()
        # plt.pcolormesh(t_axis,tau_axis,np.imag(g1.transpose()),shading='auto')
        # plt.xlabel("t in ps")
        # plt.ylabel("tau in ps")
        # plt.title("area:{:.4f} pi".format(areas[i]))
        # plt.colorbar()
        # plt.savefig("data_test/g1_{}_i.png".format(i))
        # plt.clf()
        # symmetric g1 for negative tau. for FFT
        g1_symm = np.empty([len(t_axis),2*len(tau_axis)-1],dtype=complex)
        g1_symm[:,:len(tau_axis)] = g1[:,::-1]
        g1_symm[:,-(len(tau_axis)-1):] = np.conj(g1[:,1:])
        spectra = np.empty([len(g1_symm),len(g1_symm[0])],dtype=complex)
        for j in range(len(g1_symm)):
            # do fft for every t, along the tau axis
            spectra[j] = np.fft.fftshift(np.fft.fft(g1_symm[j]))
        total = np.zeros_like(spectra[0],dtype=float)
        # integrate along the t axis
        total += np.real(np.trapz(spectra.transpose(), t_axis))
        spectrums[i] = total
        if save_dir is not None:
            _name = "_tau{:.2f}_lifet{:.1f}.npy".format(pulse_tau,1/gamma_e)
            np.save(save_dir+"x"+_name,np.fft.fftshift(fft_freqs))
            np.save(save_dir+"y"+_name,areas)
            np.save(save_dir+"z"+_name,spectrums)
    return np.fft.fftshift(fft_freqs), areas, spectrums
