import numpy as np
import networkx as nx

from typing import List, Set, Any, Tuple, Literal
from numpy.typing import ArrayLike

from random import choice
from queue import PriorityQueue
from copy import deepcopy
from bisect import insort
from itertools import combinations, product

from scipy import linalg
from scipy.optimize import linprog, OptimizeResult

from dynflows.sfm.utils import gaussian_elimination
from dynflows.sfm.greedy import greedy
from dynflows.sfm.subgradients import sg1, sg2, sg3





class Lattice:
    def __init__(self, E: List[Any], f, custom_greedy=None) -> None:
        """A data structure which serves as an oracle for function calls.

        Args:
            E (List[Any]): The list of elements in their original representation (not index-based)
            f (_type_): The submodular function used for minimization.
            custom_greedy (_type_, optional): A custom function for the greedy algorithm. Defaults to None.
        """
        self.__E = E
        self.__f = f
        self.__custom_greedy = custom_greedy

        # The set of elemments that is guaranteed to be part of the set of minimizers.
        self.__min_subset = set() 
        self.__min_subset_value = 0

        # The graph representing the areas of the lattice that have already been explored.
        self.__graph = nx.DiGraph()
        self.__root = hash(())
        self.__graph.add_node(self.__root, **{'value': 0, 'set': set()})

        # The lowest value encountered so far and solution with this value.
        self.__minimum_ub = 0
        self.__minimizer = self.__root

    def update_elements(self, E):
        self.__E = E

    def get_minimum(self) -> int:
        return self.__min_subset_value + self.__minimum_ub
    
    def get_minimizer(self) -> Set:
        return self.__min_subset | self.__graph.nodes[self.__minimizer]['set']

    def lookup(self, elems: List[int] | Set[int]) -> int:
        """Lookup the value for a given set of elements.

        Args:
            elems (List[int] | Set[int]): The set of elements represented by their indices.

        Returns:
            int: The value for the given set.
        """
        if isinstance(elems, set):
            elems = list(elems)
            
        elems = sorted([self.__E[e] for e in elems])
        hashed = hash(tuple(elems))

        if self.__graph.has_node(hashed):
            return self.__graph.nodes[hashed]['value']
        
        value = self.__f(elems)
        self.__graph.add_node(hashed, **{'value': value, 'set': set(elems)})

        if value < self.__minimum_ub:
            self.__minimum_ub = value
            self.__minimizer = hashed
        
        # Since we are interested in the inclusion-maximal minimizer, we also update the current minimizer when we find a larger set with identical value.
        if value == self.__minimum_ub and len(elems) > len(self.__graph.nodes[self.__minimizer]['set']):
            self.__minimum_ub = value
            self.__minimizer = hashed

        return value

    def get_greedy(self, dist: List[int]) -> ArrayLike:
        """_summary_

        Args:
            dist (List[int]): A diststance function represented as a list. The elements have to be represented by their respective ids.

        Returns:
            ArrayLike: The vector that is the result of the Greedy Algorithm.
        """
        # Transform the distance function into a permutation.
        perm = _to_permutation(dist)

        if self.__custom_greedy is not None:
            return self.__custom_greedy(perm)

        elems = list()
        
        u = self.__root
        u_value = self.__graph.nodes[u]['value']

        x = np.zeros(len(dist), dtype=np.float64)

        for e in perm:
            insort(elems, self.__E[e])
            v = hash(tuple(elems))

            if self.__graph.has_node(v):
                # assert self.__graph.nodes[v]['set'] == set(elems), "Node set seems to be labeled incorrectly."

                if self.__graph.has_edge(u, v):
                    delta = self.__graph.edges[u, v]['delta']

                    v_value = u_value + delta

                    # assert self.__graph.edges[u, v]['element'] == self.__E[e], "The edge seems to be labeled incorrectly."
                else:
                    v_value = self.__graph.nodes[v]['value']
                    delta = v_value - u_value

                    self.__graph.add_edge(u, v, **{'element': self.__E[e], 'delta': delta})
            else:
                v_value = self.__f(elems)
                delta = v_value - u_value

                self.__graph.add_node(v, **{'value': v_value, 'set': set(elems)})
                self.__graph.add_edge(u, v, **{'element': self.__E[e], 'delta': delta})

                # If this reduces the current upper bound on the minimizer, update the minimum.
                if v_value < self.__minimum_ub:
                    self.__minimum_ub = v_value
                    self.__minimizer = v

                # Since we are interested in the inclusion-maximal minimizer, we also update the current minimizer when we find a larger set with identical value.
                if v_value == self.__minimum_ub and len(elems) > len(self.__graph.nodes[self.__minimizer]['set']):
                    self.__minimum_ub = v_value
                    self.__minimizer = v

            u = v
            u_value = v_value

            x[e] = delta

        return x
    
    def subgradient_descent(self, X: Set[Any], method) -> List[Any]:
        E = set(self.__E)

        sg_method = lambda Y, j: method(self.__f, E, Y, j)

        while True:
            subgradient = np.array([sg_method(X, e) for e in self.__E])
            X_ = {self.__E[e] for e in (subgradient < 0).nonzero()[0]}

            if X_ == X:
                break

            X = X_

        return X
    
    def reduce(self, verbose=False):
        A = self.subgradient_descent(set(), method=sg1)
        B = self.subgradient_descent(set(self.__E), method=sg2)
        E = list((set(self.__E) - A) & B)

        if verbose:
            print(f'{len(self.__E) - len(B)} elements can be removed.')
            print(f'{len(A)} elements are guaranteed to be minimizers.')
            print(f'Reduced n from {len(self.__E)} to {len(E)}')
        
        offset = self.lookup([idx for idx, e in enumerate(self.__E) if e in A])
        self.__E = E
        
        f_old = self.__f
        self.__f = lambda S: f_old(A | set(S)) - offset

        return E


def _to_permutation(dist):
    """
    Converts a distance function into a permutation over the ground set.

    :param dist: The distance function encoded as a list.
    :return: Returns the permutation of the elements represented by the given distance function.
    """

    # This implementation uses the fact that sorted is stable to break ties between elements.
    return sorted(range(len(dist)), key=lambda e: dist[e])


class OptimalityGraph:
    def __init__(self, dists) -> None:
        self.__graph = nx.DiGraph()
        self.__graph.add_nodes_from(range(len(dists[0])))

        self.__contains = set()

        self.add_dists(dists)

    def is_optimal(self, pos, zero, neg):
        if len(pos) == 0:
            return True, set(neg + zero)

        self.__graph.remove_nodes_from(['s', 't'])
        self.__graph.add_nodes_from(['s', 't'])
        self.__graph.add_edges_from([('s', p) for p in pos] + [(n ,'t') for n in neg])

        if not nx.has_path(self.__graph, 's', 't'):
            paths = nx.single_target_shortest_path(self.__graph, 't')

            solution = set(paths.keys())
            self.__graph.remove_nodes_from(['s', 't'])

            return True, solution
        
        return False, None   
        
    def add_dist(self, dist):
        if dist in self.__contains:
            return

        self.__contains.add(dist)
        perm = _to_permutation(dist)

        for i in range(len(perm)-1):
            a = perm[i]
            b = perm[i+1]

            if self.__graph.has_edge(a, b):
                self.__graph.edges[a, b]['count'] += 1
            else:
                self.__graph.add_edge(a, b, count=1)

    def add_dists(self, dists):
        for dist in dists:
            self.add_dist(dist)

    def remove_dist(self, dist):
        if dist not in self.__contains:
            return
        
        self.__contains.remove(dist)
        perm = _to_permutation(dist)
    
        for i in range(len(perm)-1):
            a = perm[i]
            b = perm[i+1]

            if self.__graph.has_edge(a, b):
                self.__graph.edges[a, b]['count'] -= 1

                if self.__graph.edges[a, b]['count'] == 0:
                    self.__graph.remove_edge(a, b)

    def remove_dists(self, dists):
        for dist in dists:
            self.remove_dist(dist)


def __get_element_labels(x) -> Tuple[List[int], List[int], List[int]]:
    """Determines all the elements that are positive, negative or zero.

    Returns:
        Tuple[List[int], List[int], List[int]]: A list for positive, zero and negative elements, respectively.
    """
    zero_mask = np.isclose(x, 0)
    pos_mask = np.greater(x, 0)

    zero = zero_mask.nonzero()[0].tolist()
    pos = (pos_mask & ~zero_mask).nonzero()[0].tolist()
    neg = (~pos_mask & ~zero_mask).nonzero()[0].tolist()

    return pos, zero, neg


def __is_valid(dists: List[Tuple[int]], prims: List[int], negs: List[int]) -> bool:
    """Check if the given distance functions and primaries are valid, that is they satisfy 
    a) d(e)=0 for every negative element e
    b) d(e) <= p(e), where p is the primary function for e.

    Args:
        dists (List[Tuple[int]]): The collection of distance functions.
        prims (List[int]): A list mapping each element to the index of its primary function.
        negs (List[int]): A list of all elements that are negative in x.

    Returns:
        bool: True if the distance functions are valid.
    """
    for dist in dists:
        for neg in negs:
            if dist[neg] != 0:
                return False

        for e in range(len(dist)):
            if dist[e] > dists[prims[e]][e] + 1:
                return False

    return True


def __to_secondary(dist: List[int], e: int) -> List[int]:
    """Transform a given distance function to the secondary distance function with respect to some element.

    Args:
        dist (List[int]): A distance function.
        e (int): An element from the ground set.

    Returns:
        List[int]: Returns a new distance function dist' with dist'(e) = dist(e)+1.
    """
    return tuple((dist[idx] if idx != e else dist[idx] + 1) for idx in range(len(dist)))


def __compute_gammas(deltas: ArrayLike, zeros: List[int], p: int) -> Tuple[ArrayLike, ArrayLike]:
    """Solves the linear equations for obtaining the gammas for an updating step
    in Orlin's algorithm.

    Args:
        deltas (ArrayLike): The delta vectors obtained by replacing each primary by its secondary.
        zeros (List[int]): The list of elements that are 0 in x.
        p (int): A positive element chosen for updating.

    Returns:
        Tuple[ArrayLike, ArrayLike]: The computed gammas and corresponding changes of x.
    """
    m, n = deltas.shape
    gammas = np.zeros(n, dtype=np.float64)

    if len(zeros) != 0:
        c = -np.ones(len(zeros)+1, dtype=np.float64)
        A_eq = deltas[zeros, :][:, zeros + [p]]
        b_eq = np.zeros(len(zeros), dtype=np.float64)
        bounds = (0, 1)

        rslt: OptimizeResult = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        gammas[zeros + [p]] = rslt['x']
        dx = deltas @ gammas
        
        # Set all entries close to zero to zero for numerical reasons.
        gammas[np.isclose(gammas, 0)] = 0

        assert np.all(np.greater_equal(gammas, 0)) and not np.allclose(gammas, 0), "The gammas should be nonnegative and not a zero vector."

        return gammas, dx
    else:
        gammas[p] = 1

        return gammas, deltas[:, p]


def __determine_alpha(x, lambdas, dx, gammas, p, dists, prims_idx) -> float:
    if not np.isclose(dx[p], 0):
        alpha = np.divide(-x[p], dx[p], dtype=np.float64)
    else:
        alpha = np.inf

    for idx, _ in enumerate(dists):
        is_primary_of = [e for e in range(len(x)) if prims_idx[e] == idx]

        summed = np.sum(gammas[is_primary_of], dtype=np.float64)

        if summed != 0:
            alpha = np.minimum(np.divide(lambdas[idx], summed, dtype=np.float64), alpha, dtype=np.float64)

            assert alpha >= 0, "Alpha should be positive!"

    return alpha


def __reduce(dists: List, xs: ArrayLike, lambdas: ArrayLike):
    A = np.vstack([np.ones((1, xs.shape[1])), xs])

    while True:
        _, n = xs.shape
        M, _, col_perm, dim = gaussian_elimination(A)

        if dim >= n:
            break

        w = np.zeros(n, dtype=np.float64)
        M = M[:, col_perm]
        w[0:dim] = M[0:dim, dim]
        w[dim] = -1
        w = w[col_perm]

        # update the lambdas by adding some multiple of w.
        lambdas += np.min(-lambdas[w < 0] / w[w < 0]) * w

        # Remove columns with zero-valued lambdas and update the dist functions correspondingly.
        keep = ~np.isclose(lambdas, 0)
        dists = [dist for idx, dist in enumerate(dists) if keep[idx]]
        xs = xs[:, keep]
        lambdas = lambdas[keep]
        A = A[:, keep]

    return dists, xs, lambdas


def __check_for_distance_gap(dists, prims_idx):
    """Checks for a distance gap in the distance functions in order to eliminate elements.

    Args:
        dists (_type_): The current collection of distance functions.
        prims_idx (_type_): The the indices of the primary functions.

    Returns:
        _type_: A list of elements to eliminate.
    """
    min_dists = [dists[prims_idx[e]][e] for e in range(len(prims_idx))]
    mins = set(min_dists)
    gap_k = None

    for k in range(1, len(prims_idx)):
        if k - 1 not in mins and k in mins:
            gap_k = k

    eliminate = list()

    if gap_k is not None:
        for e in range(len(prims_idx)):
            if min_dists[e] >= gap_k:
                eliminate.append(e)

    return eliminate


def __correct(xs: ArrayLike, lambdas: ArrayLike, pos: List[int], zero: List[int], neg: List[int]) -> Tuple[ArrayLike, ArrayLike]:
    """Correct numerical errors by computing a new convex combination of xs such that the positive elements remain positive, 
    zero elements remain zero and negative elements remain negative.

    Args:
        xs (_type_): _description_
        lambdas (_type_): _description_
        pos (_type_): _description_
        zero (_type_): _description_
        neg (_type_): _description_

    Returns:
        Tuple[ArrayLike, ArrayLike]: _description_
    """
    c = np.zeros(len(lambdas))
    b = np.zeros(len(xs))

    A_lb = np.vstack([-xs[pos], xs[neg]])
    b_lb = b[pos + neg]

    A_eq = np.vstack([xs[zero], np.ones(len(lambdas))])
    b_eq = np.append(b[zero], 1)

    rslt = linprog(c, A_lb, b_lb, A_eq, b_eq, bounds=(0, 1))
    lambdas = rslt['x']

    return lambdas, xs @ lambdas


def __preprocess(E: List[Any], f) -> List[Any]:
    E = set(E)
    val = f(E)

    return [e for e in E if val - f(E - {e}) <= 0]


def sfm_orlin(E: List[Any], f, lazy=False, integral=True) -> Tuple[Set[Any], int]:
    """Submodular function minimization using Orlin's algorithm.

    Args:
        E (List[Any]): The list of elements in the ground set.
        f (_type_): The submodular function to minimizat.
        lazy (bool, optional): If True, the algorithm terminates immediately once a set A with f(A) < 0 was found. Defaults to False.
        integral (bool, optional): If True, the algorithm will abort once a set A with f(A) > -1 was found since 0 has to be the minimumm then. Defaults to True.

    Returns:
        Tuple[Set[Any], int]: If lazy is False, the inclusion-maximal minimizer and its value is returned. Otherwise a set with a negative value or value of zero is returned.
    """
    if isinstance(E, set):
        E = list(E)

    # The lattice storing results for function and greedy evaluations.
    lattice = Lattice(E, f)
    E = lattice.reduce()

    # Initial distance functions and primary distance functions.
    dists = [tuple(0 for _ in E)]
    prims_idx = [0 for _ in E]
    criterion = OptimalityGraph(dists)

    d_mins = [0 for _ in E]

    lambdas = np.asarray([1], dtype=np.float64)
    xs = np.asarray([lattice.get_greedy(dist) for dist in dists], dtype=np.float64).T
    x = xs @ lambdas

    assert np.allclose(x, (xs.T)[-1]), "Something went wrong while initializing x."

    while True:
        while True:
            pos, zero, neg = __get_element_labels(x)
            x[zero] = 0

            assert __is_valid(dists, prims_idx, neg), "Distance functions are not valid."

            # Check optimality criterion.
            optimal, _ = criterion.is_optimal(pos, zero, neg)

            # Stop if optimality was proven. If this algorithm is running in lazy mode, it will already terminate, if a set A with f(A) < 0 has been found.
            if optimal or (lazy and lattice.get_minimum() < 0):
                solution = lattice.get_minimizer()

                return solution, f(solution)
            
            # If the function is integral and the algorithm runs in lazy mode, it suffices to find a lower bound >-1 of the dual problem.
            if lazy and integral and np.sum(x[neg]) > -1:
                return {}, 0

            #p = np.random.choice(pos, p=x[pos] / np.sum(x[pos]))
            #p = np.argmax(x[pos])
            p = choice(pos)
            # p = sorted(pos, key=lambda p: d_mins[p])[0]

            # Compute the deltas for the current primary and secondary functions.
            prims = [dists[idx] for idx in prims_idx]
            secs =  [__to_secondary(dists[idx], e) for e, idx in enumerate(prims_idx)]
            deltas = np.zeros((len(E), len(E)))

            for e in zero + [p]:
                deltas[:, e] = (lattice.get_greedy(secs[e]) - lattice.get_greedy(prims[e])).T

            # Solve the equation system giving changes for lambda and x.
            gammas, dx = __compute_gammas(deltas, zero, p)

            # Determine a maximum step size alpha for updating x += alpha * dx.
            alpha = __determine_alpha(x, lambdas, dx, gammas, p, dists, prims_idx)
            x += alpha * dx

            # Sometimes the update step results in p becoming negative very close to 0.
            x[p] = max(0, x[p])

            lambda_lookup = {dist: lbda for dist, lbda in zip(dists, lambdas)}

            for sec in secs:
                if sec not in lambda_lookup:
                    lambda_lookup[sec] = 0

            # Compute the new lambda values for the old distance functions and the newly added secondary functions.
            for dist in lambda_lookup.keys():
                is_primary_of = [e for e in range(len(E)) if dists[prims_idx[e]] == dist]
                is_secondary_of = [e for e in range(len(E)) if secs[e] == dist]

                lambda_lookup[dist] += alpha * (np.sum(gammas[is_secondary_of], dtype=np.float64) - np.sum(gammas[is_primary_of], dtype=np.float64))
                # assert lambda_lookup[dist] >= 0 or np.isclose(lambda_lookup[dist], 0), "Lambdas have to remain nonnegative!"

                if lambda_lookup[dist] < 0:
                    lambda_lookup[dist] = 0

            new_lambdas = []
            dists = []

            # Remove all distance functions with lambda = 0 and sort the distance functions by the sum of their distances.
            for dist, lbda in sorted(lambda_lookup.items(), key=lambda tup: np.sum(tup[0])):
                if not np.isclose(lbda, 0):
                    criterion.add_dist(dist)
                    dists.append(dist)
                    new_lambdas.append(lbda)
                else:
                    criterion.remove_dist(dist)

            # Update the new xs and lambdas.
            lambdas = np.asarray(new_lambdas, dtype=np.float64) 
            xs = np.asarray([lattice.get_greedy(dist) for dist in dists], dtype=np.float64).T

            assert len(zero) == 0 or np.allclose(x[zero], 0), "All zero elements of x have to remain zero."
            assert len(pos) == 0 or np.all(np.greater_equal(x[pos], 0)), "All positive have to remain nonnegative."

            # Check xs, lambdas and x for correctness. They can become incorrect due to numerical problems.
            if not (np.allclose(np.sum(lambdas, dtype=np.float64), 1) and np.all(lambdas >= 0) and np.allclose(xs @ lambdas, x)) or xs.shape[1] >= 2 * len(E):
                pos, zero, neg = __get_element_labels(x)

                # Compute a new set of lambdas to counter numerical errors.
                lambdas, x = __correct(xs, lambdas, pos, zero, neg)

                keep = ~np.isclose(lambdas, 0)
                lambdas = lambdas[keep]
                dists = [dists[i] for i in range(len(dists)) if keep[i]]
                xs = xs[:, keep]

                assert np.allclose(xs @ lambdas, x)

            prims_idx = [np.argmin([dist[e] for dist in dists]) for e in range(len(E))]

            # Check for a distance gap and update the elements accordingly.
            eliminate = __check_for_distance_gap(dists, prims_idx)

            if len(eliminate) != 0:
                print(f'Eliminating: {[E[e] for e in eliminate]}')

                # Eliminate all elements that are above the distance gap and update xs, dists and the primaries accordingly.
                idx_keep = [e for e, _ in enumerate(E) if e not in eliminate]
                E = [E[idx] for idx in idx_keep]
                d_mins = [d_mins[idx] for idx in idx_keep]

                dists = [tuple(d for e, d in enumerate(dist) if e not in eliminate) for dist in dists]
                prims_idx = [idx for e, idx in enumerate(prims_idx) if e not in eliminate]
                criterion = OptimalityGraph(dists)

                xs = xs[idx_keep]
                x = x[idx_keep]

                lattice.update_elements(E)

                break

            assert all([dists[prims_idx[e]][e] >= d_mins[e] for e in range(len(E))]), "D_min has to be nondecreasing!"
            d_mins = [dists[prims_idx[e]][e] for e in range(len(E))]
