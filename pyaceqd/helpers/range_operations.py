import numpy as np
from typing import Optional
from functools import wraps

def with_filename(func):
    @wraps(func)
    def wrapper(
        start: float = 0.1,
        stop: float = 12,
        num: int = 101,
        nth: int = 10,
        get_inverse: bool = False,
        round_to: int = 8,
        filename: Optional[str] = None
    ):
        result = func(start, stop, num, nth, get_inverse, round_to)
        if filename is not None:
            suffix = "_inverse" if get_inverse else "_sparse"
            return result, filename + suffix
        return result
    return wrapper

@with_filename
def get_sparse_range(start=0.1, stop=12, num=101, nth=10, get_inverse=False,round_to=8):
    range_full = np.linspace(start, stop, num)
    range_sparse = range_full[::nth]
    if get_inverse:
        # returns range_full without the values in range_sparse
        # use sets: contain only unique values
        range_sparse_set = set(range_sparse)
        range_full_set = set(range_full)
        range_inverse = range_full_set - range_sparse_set  # set difference
        # range_inverse = [x for x in range_full if x not in range_sparse_set]
        return np.round(sorted(range_inverse),round_to)
    return range_sparse

def get_union(arr_x1, arr_x2, arr_z1, arr_z2, axis_z=None):
    # Get the union of arr_x1 and arr_x2 and sort the result.
    # array_z is the array of z values corresponding to arr_x1 and arr_x2
    # so array_z should also be sorted according to the union of arr_x1 and arr_x2.
    len_x1 = len(arr_x1)
    len_x2 = len(arr_x2)
    shape_z1 = arr_z1.shape
    shape_z2 = arr_z2.shape
    if len(shape_z1) == 1:
        arr_z1 = arr_z1.reshape((len_x1, 1))
        shape_z1 = arr_z1.shape
    if len(shape_z2) == 1:
        arr_z2 = arr_z2.reshape((len_x2, 1))
        shape_z2 = arr_z2.shape
    if axis_z is None:
        if shape_z1[0] == shape_z1[1]:
            return ValueError("Cannot determine axis for z arrays.")
        if shape_z1[0] == len_x1 and shape_z2[0] == len_x2:
            axis_z = 0
        elif shape_z1[1] == len_x1 and shape_z2[1] == len_x2:
            axis_z = 1
        else:
            raise ValueError("Cannot determine axis for z arrays.")
    arr_x = np.concatenate((arr_x1, arr_x2))
    arr_z = np.concatenate((arr_z1, arr_z2), axis=axis_z)
    arr_x, indices = np.unique(arr_x, return_index=True)
    arr_z = arr_z[indices]
    return arr_x, arr_z
