import itertools

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