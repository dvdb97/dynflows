import numpy as np


def greedy(f, perm, id_to_elem):
    """
    Compute a vertex in the function's base polyhedron.

    :param f: A submodular function f with f({}) = 0.
    :param perm: A permutation of all elements in E representing a total order.
    :param id_to_elem: A mapping from integers to elements
    :return: Returns a point on the function's base polyhedron.
    """
    x = np.zeros(len(perm), dtype=np.float64)
    elems = list()
    costs = f(elems)

    for e in perm:
        elems.append(id_to_elem[e])

        x[e] = f(elems) - costs
        costs += x[e]

    return x