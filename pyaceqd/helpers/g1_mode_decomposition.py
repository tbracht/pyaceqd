import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator


def decompose_G1(t, tau, g1_t_tau, n_modes=4, plot=False, dir=""):
    """
    takes a G1(t, tau) and transforms it to G1(t1, t2) with t1=t, t2=t+tau, 
    then eigendecomposes G1(t1,t2) to get the mode decomposition.
    Parameters:
    -------------
    t: time axis
    tau: tau axis
    g1_t_tau

    """
    # check if dir exists, if not create
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    # check dimension 
    if len(t) != np.shape(g1_t_tau)[0] or len(tau) != np.shape(g1_t_tau)[1]:
        raise UserWarning("g1_t_tau is shape {}, expected {} (len(t)),{} (len(tau))".format(np.shape(g1_t_tau,len(t),len(tau))))
    if plot:
        plt.clf()
        # use pcolorfast
        fig, ax = plt.subplots()
        c = ax.pcolormesh(t.real, tau.real, np.real(g1_t_tau).T, shading='auto')
        fig.colorbar(c, ax=ax)
        ax.set_xlabel("Time (a.u.)")
        ax.set_ylabel("Delay (a.u.)")
        ax.set_title("G1(t, tau)")
        plt.savefig(dir+"g1.png")

    # Transform to G1(t1, t2) with t1=t, t2=t+tau
    # Symmetry: G1(t, -tau) = conj(G1(t - tau, tau))
    # => G1(t1, tau<0) = conj(G1(t1 + tau, -tau)) = conj(G1(t2, t1 - t2))
    # => lower triangle (t2 < t1): G1(t1, t2) = conj(G1(t2, t1 - t2))
    rgi = RegularGridInterpolator((t.real, tau.real), g1_t_tau, method='linear', bounds_error=False, fill_value=0)
    
    limit = min(t.real.max(), tau.real.max())
    n = min(len(t), len(tau))
    t1_lin = np.linspace(t.real[0], limit, n)
    t2_lin = np.linspace(t.real[0], limit, n)
    T1, T2 = np.meshgrid(t1_lin, t2_lin, indexing='ij')
    TAU = T2 - T1
    
    # upper triangle: (t1, t2 - t1)
    upper = rgi(np.column_stack([T1.ravel(), TAU.ravel()])).reshape(n, n)
    # lower triangle: conj(G1(t2, t1 - t2))
    lower = np.conj(rgi(np.column_stack([T2.ravel(), (-TAU).ravel()])).reshape(n, n))
    
    g1_t1_t2 = np.where(TAU >= 0, upper, lower)

    if plot:
        fig, ax = plt.subplots()
        c = ax.pcolormesh(t1_lin, t2_lin, g1_t1_t2.real.T, shading='auto')
        fig.colorbar(c, ax=ax)
        ax.set_xlabel("t1 (a.u.)")
        ax.set_ylabel("t2 (a.u.)")
        ax.set_title("G1(t1, t2)")
        plt.savefig(dir+"g1_t1_t2.png")
        plt.close()

    # Eigendecompose G1(t1,t2) — Hermitian, so use eigh (eigenvalues real, sorted ascending)
    eigenvalues, eigenvectors = np.linalg.eigh(g1_t1_t2)
    # dominant eigenvector = last column (largest eigenvalue)
    dominant_vec = eigenvectors[:, -1]

    # Reconstruct G1 from the n_modes dominant eigenvectors
    n_modes = n_modes if n_modes <= len(eigenvalues) else len(eigenvalues)
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(range(len(eigenvalues)), eigenvalues[::-1])
        axes[0].set_xlabel("Mode index")
        axes[0].set_ylabel("Eigenvalue")
        axes[0].set_xlim(-1,15)
        axes[0].set_title("Eigenvalue spectrum")

        # plot n_modes dominant eigenvector
        for i in range(n_modes):
            axes[1].plot(t1_lin, eigenvectors[:, -1 - i].real, label=f"Mode {i+1} (λ={eigenvalues[-1 - i]:.4g})")
            # axes[1].plot(t1_lin, -eigenvectors[:, -1 - i].imag, label=f"Mode {i+1} Im (λ={eigenvalues
        # axes[1].plot(t1_lin, dominant_vec.real, label="Re")
        # axes[1].plot(t1_lin, -dominant_vec.imag, label="Im")
        axes[1].set_xlabel("t (a.u.)")
        axes[1].set_title(f"Dominant eigenvector (λ={eigenvalues[-1]:.4g})")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(dir+"g1_eigenvector.png")
        plt.clf()
        plt.close()


    vals_top = eigenvalues[-n_modes:]
    vecs_top = eigenvectors[:, -n_modes:]
    g1_reconstructed = (vecs_top * vals_top) @ vecs_top.conj().T
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        c0 = axes[0].pcolormesh(t1_lin, t2_lin, g1_t1_t2.real.T, shading='auto')
        fig.colorbar(c0, ax=axes[0])
        axes[0].set_xlabel("t1 (a.u.)")
        axes[0].set_ylabel("t2 (a.u.)")
        axes[0].set_title("G1(t1, t2) original")
        
        c1 = axes[1].pcolormesh(t1_lin, t2_lin, g1_reconstructed.real.T, shading='auto')
        fig.colorbar(c1, ax=axes[1])
        axes[1].set_xlabel("t1 (a.u.)")
        axes[1].set_ylabel("t2 (a.u.)")
        axes[1].set_title(f"G1 reconstructed ({n_modes} modes)")
        
        c2 = axes[2].pcolormesh(t1_lin, t2_lin, (g1_t1_t2 - g1_reconstructed).real.T, shading='auto')
        fig.colorbar(c2, ax=axes[2])
        axes[2].set_xlabel("t1 (a.u.)")
        axes[2].set_ylabel("t2 (a.u.)")
        axes[2].set_title("Residual")
    
        plt.tight_layout()
        plt.savefig(dir+"g1_reconstructed.png")
        plt.clf()
        plt.close()
    return eigenvalues, eigenvectors, g1_reconstructed, t1_lin, t2_lin


