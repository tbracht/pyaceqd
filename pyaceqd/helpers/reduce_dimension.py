import numpy as np

def filter_n_excitations(total_excitation_matrix, max_excitations):
    total_excitations = np.copy(total_excitation_matrix)
    total_excitations = np.round(total_excitations, decimals=5)  # round to avoid floating point issues
    for i in range(total_excitations.shape[0]):
        if total_excitations[i,i] <= max_excitations:
            total_excitations[i,i] = 0  # set states with total excitations less than or equal to max_excitations to 0
    states_to_remove = get_remove_indices(total_excitations)  # get indices of states to remove based on max_excitations
    return states_to_remove

def get_union_indices(list1, list2):
    """
    Returns the sorted union of two lists of indices.
    """
    return sorted(set(list1) | set(list2))

def remove_dim_numbers(H_original, state_indices):
    """
    Removes the state from the Hamiltoian and returns the new Hamiltonian.

    Parameters
    ----------
    H_original : np.ndarray
        The original Hamiltonian in matrix form.
    state_indices : int or list of int
        index of state to be removed from Hamiltonian.
    """
    if len(state_indices) == 0:
        return H_original
    dim = H_original.shape[0]
    # Remove the state from the Hamiltonian
    # check if any of the state indices are out of bounds
    if isinstance(state_indices, int):
        state_indices = [state_indices]
    for index in state_indices:
        if index < 0 or index >= dim:
            raise ValueError(f"State index {index} is out of bounds for Hamiltonian of dimension {dim}.")
    H_reduced = np.delete(H_original, state_indices, axis=0)
    H_reduced = np.delete(H_reduced, state_indices, axis=1)
    return H_reduced

def get_remove_indices(state_matrix):
    state_matrix = np.round(state_matrix, decimals=5)  # round to avoid floating point issues
    dim = state_matrix.shape[0]
    if state_matrix.shape[1] != dim:
        raise ValueError(f"state_matrix must be square, but got shape {state_matrix.shape}.")
    states_to_remove = []
    for i in range(dim):
        if state_matrix[i,i] > 0:
            states_to_remove.append(i)
    return states_to_remove

def remove_dims(H_original, state_matrix):
    states_to_remove = get_remove_indices(state_matrix)
    return remove_dim_numbers(H_original, states_to_remove)

# H = np.diag([1, 2, 3, 4])
# print("Original Hamiltonian:")
# print(H)
# H_reduced = remove_dim_numbers(H, [1,2])
# print("Reduced Hamiltonian:")
# print(H_reduced)
