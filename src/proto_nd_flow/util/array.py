import numpy as np
import numpy.lib.recfunctions as rfn


def mask2sel(arr: np.ma.masked_array) -> np.ndarray:
    """Given a masked array `arr`, converts its mask into a "selector", a simple
    N-D array of bools (for slicing other arrays, etc.). For structured arrays,
    whose masks are N-D arrays of tuples, this function will regard a mask on
    any column as a mask on the whole element (i.e. row).
    """

    return ~rfn.structured_to_unstructured(arr.mask).any(axis=-1)
