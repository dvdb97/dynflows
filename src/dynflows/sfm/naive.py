import numpy as np

from typing import Set, Any, Tuple, Literal
from itertools import chain, combinations


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def lazy_sfm_naive(E: Set[Any], f, threshold: float) -> Tuple[Set[Any], float]:
    """Lazy submodular function minimization which stops once a solution has a value strictly below a certain threshold.
    if there is no such solution, the global minimizer is returned.
    
    Args:
        E (Set[Any]): The base set.
        f (_type_): The submodular function to minimize.
        threshold (float): A threshold below which a solution value has to fall in order to be considered minimum.

    Returns:
        Tuple[Set[Any], float]: Returns the minimizer and minimum.
    """
    min_sol = []
    min_val = np.inf

    # Go through all possible subsets of the terminal nodes.
    for A in powerset(E):
        A = set(A)
        val = f(A)

        if val < threshold:
            return A, val

        if val < min_val:
            min_sol = A
            min_val = val

    return min_sol, min_val


def sfm_naive(E: Set[Any], f, minimizer_type: Literal['min', 'max'] = 'min') -> Tuple[Set[Any], float]:
    """Submodular function minimization by naive enumeration of all subsets.

    Args:
        E (Set[Any]): The base set.
        f (_type_): The submodular function to minimize.

    Returns:
        Tuple[Set[Any], float]: Returns the minimizer and minimum.
    """
    min_sol = []
    min_val = np.inf

    # Go through all possible subsets of the terminal nodes.
    for A in powerset(E):
        A = set(A)
        val = f(A)

        if val > min_val:
            continue

        if val == min_val:
            if minimizer_type == 'min' and len(A) >= len(min_sol):
                continue

            if minimizer_type == 'max' and len(A) <= len(min_sol):
                continue

        min_sol = A
        min_val = val

    return min_sol, min_val
