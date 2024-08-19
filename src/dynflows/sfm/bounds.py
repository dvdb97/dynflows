from typing import Set, Tuple, Any


def bounds(E: Set[Any], f) -> Tuple[float, float]:
    """Computes a lower and upper bound of a submodular function on the given
    base set.

    Args:
        E (_type_): The base set.
        f (_type_): The submodular function.

    Returns:
        _type_: _description_
    """
    summed = sum(max(0, f({e}) - f(set())) for e in E)

    return f(set()) - summed, f(set()) + summed
