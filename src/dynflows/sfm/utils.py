import numpy as np

from itertools import product


def gaussian_elimination(A: np.array):
    m, n = A.shape
    r = 0

    row_perm = list(range(0, m))
    col_perm = list(range(0, n))

    M = np.zeros((m, n + m), dtype=np.float64)
    M[:, :n] = A
    M[:, n:] = np.identity(m, dtype=np.float64)

    changed = True

    while changed:
        changed = False

        for p, q in product(range(r, m), range(r, n)):
            if not np.isclose(M[p, q], 0):
                if p != r:
                    M[[p, r]] = M[[r, p]]
                    row_perm[p], row_perm[r] = row_perm[r], row_perm[p]

                if q != r:
                    M[:, [q, r]] = M[:, [r, q]]
                    col_perm[q], col_perm[r] = col_perm[r], col_perm[q]

                for i in range(r+1, m):
                    alpha = M[i, r] / M[r, r]
                    M[i] -= alpha * M[r]

                r += 1

                changed = True
                break

    for k in range(r-1, 0, -1):
        for i in range(k):
            alpha = M[i, k] / M[k, k]
            M[i] -= alpha * M[k]

    for k in range(r):
        M[k] /= M[k, k]

    return M, row_perm, col_perm, r

