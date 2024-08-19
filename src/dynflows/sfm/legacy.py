


def is_invertible(A):
    return A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]


def compute_gammas(deltas: np.array, zeros: List[int], e: int):
    m, n = deltas.shape
    gammas = np.zeros(n, dtype=np.float64)

    if len(zeros) != 0:
        A_ = deltas[zeros, :][:, zeros]
        b_ = -deltas[zeros, e]

        if is_invertible(A_):
            gammas[zeros] = np.linalg.solve(A_, b_)
            gammas[e] = 1
        else:
            # Find a column that could potentially be a zero column.
            idx = np.argmin(np.sum(np.abs(A_), axis=0))

            # Check if that column is a zero column.
            if np.allclose(A_[:, idx], 0):
                # If so, return a trivial solution.
                gammas = np.zeros(n)
                gammas[zeros[idx]] = 1

                return gammas, deltas[:, zeros[idx]]

            _, _, Vt = np.linalg.svd(A_)
            gammas[zeros] = Vt[-1]

            if np.allclose(np.sum(b_), 0):
                gammas[e] = 1
            else:
                gammas[e] = 0

        dx = deltas @ gammas

        return gammas, dx
    else:
        gammas[e] = 1

        return gammas, deltas[:, e]
    

def sfm_orlin(E, f, custom_greedy=None):
    if isinstance(E, set):
        E = list(E)

    dists = DFCollection(E, f, custom_greedy=custom_greedy)

    if custom_greedy is not None:
        greedy = custom_greedy

    lambdas = np.array([1], dtype=np.float64)
    _, _, xs = dists[:]
    x = xs @ lambdas

    while True:
        # Determine the positive, negative and zero elements.
        zero = [i for i in range(len(x)) if np.isclose(x[i], 0)]
        neg = [i for i in range(len(x)) if x[i] < 0 and not np.isclose(x[i], 0)]
        pos = [i for i in range(len(x)) if x[i] > 0 and dists.get_min_dist(i) != len(E)+1 and not np.isclose(x[i], 0)]

        assert dists.is_valid(neg), "Distance functions are not valid."

        # Check optimality criterion.
        if dists.is_optimal(pos, neg):
            return {e for idx, e in enumerate(E) if idx in set(neg) | set(zero)}, sum(x[neg])

        # Select a positive element.
        p = pos[np.argmax(x[pos])]

        prims = dists.get_primaries(as_perm=True)
        secs = dists.get_secondaries(as_perm=True)

        deltas = np.vstack([greedy(f, sec_perm, E) - greedy(f, prim_perm, E) for prim_perm, sec_perm in zip(prims, secs)]).T
        gammas, dx = compute_gammas(deltas, zero, p)

        if dx[p] != 0:
            alpha = -x[p] / dx[p]
        else:
            alpha = np.inf

        for idx, dist in enumerate(dists):
            summed = np.sum(gammas[dists.is_primary_of(dist)])

            if summed != 0:
                alpha = min(lambdas[idx] / summed, alpha)

                assert alpha >= 0, "Alpha should be positive!"

        dists.add_dists(dists.get_secondaries(), update_prims=False)
        lambdas = np.append(lambdas, np.zeros(len(dists) - len(lambdas), dtype=np.float64))

        for idx, dist in enumerate(dists):
            lambdas[idx] += alpha * (np.sum(gammas[dists.is_secondary_of(dist)]) - np.sum(gammas[dists.is_primary_of(dist)]))

        x += alpha * dx

        lambdas = dists.update(lambdas)
        _, _, xs = dists[:]

        assert np.allclose(sum(lambdas), 1), "Lambdas have to sum up to 1!"
        assert np.all(lambdas >= 0), "Lambdas have to be non-negative!"
        assert np.allclose(xs @ lambdas, x), "Incorrect convex combination!"

        assert len(zero) == 0 or np.allclose(x[zero], 0), "All zero entries of x have to remain zero."

        if xs.shape[1] >= 3 * len(E):
            xs, lambdas = dists.reduce(xs, lambdas)

        dists.check_for_distance_gap()




class DFCollection:
    def __init__(self, E, f, custom_greedy=None):
        self.__E = E
        self.__f = f

        if custom_greedy is not None:
            self.__greedy = custom_greedy
        else:
            self.__greedy = greedy

        self.__dists = np.array([[0] * len(E)])
        self.__perms = np.array([to_permutation(self.__dists[-1])])
        self.__xs = np.array([self.__greedy(f, perm, self.__E) for perm in self.__perms]).T

        self.__update_prims_and_secs()

    def __len__(self):
        return len(self.__dists)

    def __getitem__(self, index):
        return self.__dists[index], self.__perms[index], self.__xs[:, index]

    def __iter__(self):
        return iter(self.__dists)

    def __update_prims_and_secs(self):
        self.__prims = np.array([np.argmin(self.__dists[:, e]) for e in range(len(self.__E))])
        self.__secs = np.array([to_secondary(self.__dists[idx], e) for e, idx in enumerate(self.__prims)])
        self.__secs_perms = np.array([to_permutation(dist) for dist in self.__secs])

    def get_primary_of(self, e, as_perm=False):
        if as_perm:
            return self.__perms[self.__prims[e]]

        return self.__prims[e]

    def get_primaries(self, as_perm=False):
        if as_perm:
            return self.__perms[self.__prims]

        return self.__dists[self.__prims]

    def get_secondary_of(self, e, as_perm=False):
        if as_perm:
            return self.__secs_perms[e]

        return self.__secs[e]

    def get_secondaries(self, as_perm=False):
        if as_perm:
            return self.__secs_perms

        return self.__secs

    def is_primary_of(self, prim: List[int]):
        return [e for e, dist_idx in enumerate(self.__prims) if np.all(self.__dists[dist_idx] == prim)]

    def is_secondary_of(self, sec: List[int]):
        return [e for e, dist in enumerate(self.__secs) if np.all(dist == sec)]

    def get_min_dist(self, e: int):
        return self.__dists[self.__prims[e]][e]

    def add_dists(self, dist: np.array, update_prims=True):
        self.__dists = np.vstack([self.__dists, dist])
        _, indices = np.unique(self.__dists, axis=0, return_index=True)
        self.__dists = self.__dists[sorted(indices)]

        self.__perms = np.array([to_permutation(dist) for dist in self.__dists])
        self.__xs = np.array([self.__greedy(self.__f, perm, self.__E) for perm in self.__perms]).T

        if update_prims:
            self.__update_prims_and_secs()

    def is_optimal(self, pos, neg) -> bool:
        pos = set(pos)
        neg = set(neg)

        for perm in self.__perms:
            for a, b in combinations(perm, 2):
                if a in pos and b in neg:
                    return False

        return True

    def is_valid(self, negs) -> False:
        for dist in self.__dists:
            for neg in negs:
                if dist[neg] != 0:
                    return False

            for e in range(len(dist)):
                if dist[e] > self.__dists[self.__prims[e]][e] + 1:
                    return False

        return True

    def update(self, lambdas):
        # Remove all distance functions with lambda = 0.
        keep = lambdas > 0
        self.__dists = self.__dists[keep, :]
        self.__perms = self.__perms[keep, :]
        self.__xs = self.__xs[:, keep]
        lambdas = lambdas[keep]

        # Sort the remaining distance functions by total distance
        perm = np.argsort(np.sum(self.__dists, axis=1))
        self.__dists[:] = self.__dists[perm]
        self.__perms[:] = self.__perms[perm]
        self.__xs[:, :] = self.__xs[:, perm]

        # Also update the lambdas accordingly.
        lambdas = lambdas[perm]

        # Since we have new distance functions, we have to update the primary and secondary distance functions aswell.
        self.__update_prims_and_secs()

        return lambdas

    def has_distance_gap(self) -> int | None:
        n = len(self.__prims)
        mins = set([self.get_min_dist(e) for e in range(n)])

        for k in range(1, n):
            if k - 1 not in mins and k in mins:
                return k

        return None

    def check_for_distance_gap(self):
        k = self.has_distance_gap()
        n = len(self.__prims)

        if k is not None:
            for e in range(n):
                if self.get_min_dist(e) >= k:
                    for dist in self.__dists:
                        dist[e] = n + 1

                    for sec in self.__secs:
                        sec[e] = n + 1

    def reduce(self, xs: np.array, lambdas: np.array):
        A = np.vstack([np.ones((1, xs.shape[1])), xs])

        while True:
            m, n = xs.shape
            M, row_perm, col_perm, dim = gaussian_elimination(A)

            if dim >= n:
                break

            w = np.zeros(n)
            M = M[:, col_perm]
            w[0:dim] = M[0:dim, dim]
            w[dim] = -1
            w = w[col_perm]

            # update the lambdas by adding some multiple of w.
            lambdas += np.min(-lambdas[w < 0] / w[w < 0]) * w

            # Remove columns with zero-valued lambdas and update the dist functions correspondingly.
            keep = ~np.isclose(lambdas, 0)
            xs = xs[:, keep]
            lambdas = lambdas[keep]
            self.__dists = self.__dists[keep]
            A = A[:, keep]

        self.__update_prims_and_secs()

        return xs, lambdas
    
def is_optimal(dists, pos, neg, zero) -> Tuple[bool, Set[int]]:
    """The current list of distance functions is optimal by constructing a digraph
    from the permutations arising from the distance functions and checking if pos
    and neg are connected.

    Args:
        dists (_type_): _description_
        pos (_type_): _description_
        neg (_type_): _description_
        zero (_type_): _description_

    Returns:
        bool: Returns True if the solution is optimal.
    """

    # If no positive elements are available, the solution has to be optimal.
    if len(pos) == 0:
        return True, set(neg + zero)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(len(dists[0])))

    G.add_nodes_from(['s', 't'])
    G.add_edges_from([('s', p) for p in pos] + [(n ,'t') for n in neg])

    for dist in dists:
        perm = to_permutation(dist)

        for a, b in combinations(perm, 2):
            G.add_edge(a, b)

    if not nx.has_path(G, 's', 't'):
        paths = nx.single_target_shortest_path(G, 't')

        solution = set(paths.keys())
        solution.remove('t')

        return True, solution

    return False