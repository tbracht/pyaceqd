import numpy as np
import matplotlib.pyplot as plt
from pyaceqd.tools import export_csv, basis_states, compose_dm, output_ops_dm
import colorsys
from tabulate import tabulate

def hex_to_rgba(hex_code):
    hex_code = hex_code.lstrip('#')
    if len(hex_code) == 6:
        hex_code += 'FF'  # Append alpha channel if not provided
    decimal_value = int(hex_code, 16)
    rgba_tuple = (decimal_value >> 24 & 255, decimal_value >> 16 & 255, decimal_value >> 8 & 255, decimal_value & 255)
    return rgba_tuple

def select_equally_spaced_colors(n):
    colors = []
    hue_values = [i / n for i in range(n)]  # Equally spaced hue values

    for hue in hue_values:
        rgb = colorsys.hls_to_rgb(hue, 0.5, 1.0)  # Convert HSL to RGB
        hex_code = "#{:02X}{:02X}{:02X}".format(*[int(255 * c) for c in rgb])  # Convert RGB to hexadecimal color code
        colors.append(hex_code)
    
    return colors

def compose_dm_new(outputs, dim=2):
    """
    composes a density matrix from the output of ACE, with every output-array being the time dynamics for the corresponding output operator
    """
    # dim is the dimension of the system
    rho = np.zeros((len(outputs[0]),dim,dim),dtype=np.complex128)
    for i in range(len(outputs[0])):
        rho[i] = np.reshape(outputs[1:,i], (dim,dim))
    t = np.real(outputs[0])
    return t, rho

def dressed_states_new(system, dim, t_start, t_end, *pulses, plot=True, t_lim=None, e_lim=None, filename="dressed", firstonly=False, colors=None, visible_states=None, return_eigenvectors=False, print_states=None, no_pulse=False, **options):
    options["output_ops"] = []
    # firstonly is not used when calculating the density matrix, only for the composition of the dressed states
    # rho is the density matrix that is later transformed
    _,rho = compose_dm_new(system(t_start, t_end, *pulses, **options), dim=np.prod(dim))
    options["print_H"] = True
    options["firstonly"] = firstonly
    if no_pulse:
        # the 'no_pulse' option can be used if the underlying
        # system is not in its eigenstates, e.g., if an external 
        # magnetic field is applied in the hamiltonian.
        # Then, only the hamiltonian without the pulses is used for
        # the diagonalization.
        pulses = []
    # data is used to calculate the dressed states
    t, H_total = system(t_start, t_end, *pulses, **options)
    if colors is None:
        colors = select_equally_spaced_colors(n=np.prod(dim))
    return _dressed_states(t=t, Htot=H_total, dim=dim, rho=rho, colors=colors, filename=filename, plot=plot, t_lim=t_lim, e_lim=e_lim, visible_states=visible_states, return_eigenvectors=return_eigenvectors, print_states=print_states)

def _dressed_states(t, Htot, dim, rho, colors, filename, plot=False, t_lim=None, e_lim=None, visible_states=None, return_eigenvectors=False, print_states=None):
    _dim = np.prod(dim)
    if plot:
        plt.clf()
        plt.ylim(-0.1,1.1)
        labels = basis_states(dim)
        for i in range(_dim):
            plt.plot(t, rho[:,i,i].real, label=labels[i],color=colors[i])
        if t_lim is not None:
            plt.xlim(t_lim[0],t_lim[1])
        plt.xlabel("t (ps)")
        plt.ylabel("occupation")
        plt.legend()
        plt.savefig(filename + "_rho.png")
        plt.clf()
    e_vectors = np.zeros((len(t),_dim,_dim),dtype=np.complex128)
    e_values = np.zeros((len(t),_dim))

    for i in range(len(t)):
        e_values[i], e_vectors[i] = np.linalg.eigh(Htot[i])
    
    # first fix the phase of the eigenvectors
    for i in range(len(t)):
        # if first component of first EV is not real and smaller than 0:
        # multiply all EVs with exp(-1j*angle)
        angle=0
        if (np.imag(e_vectors[i,0,0]) !=0 or e_vectors[i,0,0] < 0):
            angle = np.angle(e_vectors[i,0,0])
        e_vectors[i,:,:] = e_vectors[i,:,:]*np.exp(-1j*angle)
    
    if print_states is not None:
        _t = print_states
        i = np.argmin(np.abs(t-_t))
        header = basis_states(dim)
        # add column in front for the dressed state index
        header.insert(0,"t:{:.2f}".format(t[i]))
        header.append("Energy")
        table = []
        for j in range(_dim):
            row = ["ds"+str(j+1)]
            row.extend(np.abs(e_vectors[i,j])**2)
            row.extend([e_values[i,j]])
            table.append(row)
        print(tabulate(table,headers=header,floatfmt=".2f"))
        # print(tabulate(np.abs(e_vectors[i])**2,headers=header,floatfmt=".2f"))

    n_colors = np.empty([_dim,e_values.shape[0]])  # for gnuplot
    if len(colors) != _dim:
        print("Error: Number of colors does not match number of dressed states.")
        return
    
    s_colors = []  # stores color values
    r_array = np.zeros(_dim)
    g_array = np.zeros(_dim)
    b_array = np.zeros(_dim)
    a_array = np.zeros(_dim)
    a_array_gp = np.zeros(_dim)  # for gnuplot
    for i in range(_dim):
        r_array[i] = hex_to_rgba(colors[i])[0]/255
        g_array[i] = hex_to_rgba(colors[i])[1]/255
        b_array[i] = hex_to_rgba(colors[i])[2]/255
        if visible_states is None:
            a_array[i] = hex_to_rgba(colors[i])[3]/255
            a_array_gp[i] = 1-hex_to_rgba(colors[i])[3]/255

    if visible_states is not None:
        # check that no value will be OOB
        if np.max(visible_states) > _dim-1:
            print("Error: Visible states out of bounds.")
            return
        a_array[visible_states] = 1
        a_array_gp[visible_states] = 0
    # r_array = np.array([0,255])/255
    # g_array = np.array([0,0])/255
    # b_array = np.array([255,0])/255
    # a_array = np.array([255,255])/255
    # a_array_gp = np.array([0,0])  # for gnuplot

    for i in range(_dim):
        colors = []
        for j in range(e_values.shape[0]):
            e = np.abs(e_vectors[j,i])**2
            r = int(np.clip(np.dot(r_array,e),0,1)*255)
            g = int(np.clip(np.dot(g_array,e),0,1)*255)
            b = int(np.clip(np.dot(b_array,e),0,1)*255)
            a = int(np.clip(np.dot(a_array,e),0,1)*255)
            agp = int(np.clip(np.dot(a_array_gp,e),0,1)*255)
            n_colors[i,j] = 65536*r + 256*g + b + agp*16777216
            colors.append("#{:02x}{:02x}{:02x}{:02x}".format(r,g,b,a))
        s_colors.append(colors)
        if plot:
            plt.scatter(t,e_values[:,i],c=colors)
    if plot:
        if t_lim is not None:
            plt.xlim(t_lim[0],t_lim[1])
        if e_lim is not None:
            plt.ylim(e_lim[0],e_lim[1])
        for i in range(_dim):
            plt.plot(t,e_values[:,i],label="ds{}".format(i+1))
        plt.legend()
        plt.xlabel("t (ps)")
        plt.ylabel("E (meV)")
        plt.savefig(filename + "_ds.png")
        plt.clf()

    # dressed state occupations
    # we use the following formula for the occupation of a dressed state |psi>:
    # <|psi><psi|> = sum_ij a_i * a_j^* * <|phi_i><phi_j|>
    # where |phi_i> are the states of the system and a_i are the components of |psi> in the basis of |phi_i>
    # <|phi_i><phi_j|> is the density matrix rho
    ds_occ = np.zeros([len(t),_dim])
    for i in range(len(t)):
        for j in range(_dim):
            ds_ij = e_vectors[i,j][:,None]*e_vectors[i,j].conj()  # ai * aj^*
            ds_occ[i,j] = np.sum(ds_ij*rho[i]).real  # sum_ij ai * aj^* * <|phi_i><phi_j|>
    if plot:
        plt.clf()
        plt.ylim(-0.1,1.1)
        if t_lim is not None:
            plt.xlim(t_lim[0],t_lim[1])
        for i in range(_dim):
            plt.scatter(t,ds_occ[:,i],c=s_colors[i])
        for i in range(_dim):
            plt.plot(t,ds_occ[:,i],label="ds{}".format(i+1))
        plt.xlabel("t (ps)")
        plt.ylabel("occupation (dressed state)")
        plt.legend()
        plt.savefig(filename + "_ds_occ.png")
        plt.clf()
    populations = np.diagonal(rho, axis1=1, axis2=2)
    if return_eigenvectors:
        return t, populations, e_values, ds_occ, s_colors, n_colors, e_vectors, rho
    return t, populations, e_values, ds_occ, s_colors, n_colors
