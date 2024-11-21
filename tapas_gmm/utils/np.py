from functools import lru_cache, wraps

import numpy as np


def np_cache(*lru_args, array_argument_index=0, **lru_kwargs):
    """
    https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75
    LRU cache implementation for functions whose parameter at ``array_argument_index`` is a numpy array of dimensions <= 2

    Example:
    >>> from sem_env.utils.cache import np_cache
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     return factor * array
    >>> multiply(array, 2)
    >>> multiply(array, 2)
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
    """

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            np_array = args[array_argument_index]
            if len(np_array.shape) > 2:
                raise RuntimeError(
                    f"np_cache is currently only supported for arrays of dim. less than 3 but got shape: {np_array.shape}"
                )
            hashable_array = tuple(map(tuple, np_array))
            args = list(args)
            args[array_argument_index] = hashable_array
            return cached_wrapper(*args, **kwargs)

        @lru_cache(*lru_args, **lru_kwargs)
        def cached_wrapper(*args, **kwargs):
            hashable_array = args[array_argument_index]
            array = np.array(hashable_array)
            args = list(args)
            args[array_argument_index] = array
            return function(*args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper

    return decorator


def get_indeces_of_duplicate_rows(
    matrix: np.ndarray, decimals: int | None = 2
) -> list[np.ndarray]:
    """
    Get the indeces of the duplicate rows in a matrix.
    https://stackoverflow.com/a/60876649
    https://stackoverflow.com/a/5427178

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to check for duplicate rows.
    decimals : int, optional
        The number of decimals to which to round.

    Returns
    -------
    list[np.ndarray]
        The indeces of the duplicate rows, grouped per value.
    """
    if decimals is not None:
        matrix = integer_rounding(matrix, decimals=decimals)

    unq, count = np.unique(matrix, axis=0, return_counts=True)
    repeated_groups = unq[count > 1]

    indeces = []

    for group in repeated_groups:
        repeated_idx = np.argwhere(np.all(matrix == group, axis=1))
        indeces.append(repeated_idx.ravel())

    return indeces


def integer_rounding(array: np.ndarray, decimals: int = 2) -> np.ndarray:
    """
    Round to a fixed number of leading decimal places (akin to floating point rounding).
    """
    exponents = np.floor(np.log10(np.abs(array)))
    scaled_down = array / (10**exponents)
    rounded = np.around(scaled_down, decimals)
    scaled_up = rounded * (10**exponents)

    return scaled_up
