

def find_max_feasible(func, max_v) -> int:
    """Given a function mapping returning True for every i <= n
    and False for i > n, find n via binary search.

    Args:
        func (_type_): The function to find n for.
        max_v (_type_): A value that is guaranteed to be False.

    Returns:
        int: Returns the value n.
    """
    left, right = 0, max_v

    while left < right:
        v = (left + right) // 2 + 1
        feas = func(v)

        if feas:
            left, right = v, right
        else:
            left, right = left, v-1

    return left if func(left) else left-1


def find_min_feasible(func, max_v, verbose=False) -> int:
    """Given a function returning True for every i >= n
    and False for i < n, find n via binary search.

    Args:
        func (_type_): The function to find n for.
        max_v (_type_): A value that is guaranteed to be True.

    Returns:
        int: Returns the value n.
    """
    left, right = 0, max_v

    while left < right:
        if verbose:
            print(f'[{left}, {right}]\r')

        v = (left + right) // 2
        feas = func(v)

        if feas:
            left, right = left, v
        else:
            left, right = v+1, right

    return right if func(right) else right+1


def binary_search(func, max_v):
    v = max_v // 2

    while True:
        rslt = func(v)

        if rslt > 0:
            v = v + (max_v - v) // 2
        elif rslt < 0:
            v = 0 + v // 2
        else:
            return v
        

if __name__ == '__main__':
    def f(v):
        return v >= 32
    
    for i in range(100, 200):
        print(find_min_feasible(f, i))