import numpy as np

def order_eigenstates(eigenvalues, eigenvectors):
    """
    fix order of eigenvalues/eigenvectors to match previous time step
    by computing overlaps between eigenvectors
    assumes some time axis t is given, and eigenvalues/eigenvectors are arrays of length len(t)
    
    :param eigenvalues: len(t) array un-ordered eigenvalues
    :param eigenvectors: len(t) array of un-ordered eigenvectors as given by numpy.linalg.eig
    """
    from scipy.optimize import linear_sum_assignment

    e_vectors_ordered = np.zeros_like(eigenvectors)
    e_values_ordered = np.zeros_like(eigenvalues)
    e_vectors_ordered[0] = eigenvectors[0]
    e_values_ordered[0] = eigenvalues[0]

    len_t = eigenvalues.shape[0]
    num_states = eigenvalues.shape[1]
    for i in range(1, len_t):
        # compute overlaps with previous step
        overlap = np.abs(e_vectors_ordered[i-1].conj().T @ eigenvectors[i])  # shape (dim, dim)

        # find best matching permutation (argmax in each row)
        # But ensure no duplicates: solve as assignment problem
        # (Hungarian algorithm implemented in scipy as linear_sum_assignment)
            
        row_ind, col_ind = linear_sum_assignment(-overlap)  # maximize overlap
        e_vectors_ordered[i] = eigenvectors[i][:, col_ind]
        e_values_ordered[i] = eigenvalues[i][col_ind]
    return e_values_ordered, e_vectors_ordered
