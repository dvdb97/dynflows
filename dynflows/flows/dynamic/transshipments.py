import networkx as nx
import numpy as np
import sys

from typing import Any, Set, Tuple, Literal
from itertools import combinations
from random import choice
from tqdm import tqdm

from dynflows.flows.dynamic.flow import TemporallyRepeatedFlow
from dynflows.flows.dynamic.time_horizon import get_upper_bound_of_T
from dynflows.flows.dynamic.max_flow import max_flow_over_time
from dynflows.flows.dynamic.lex_max import lex_max_flow_over_time
from dynflows.search import find_max_feasible, find_min_feasible

from dynflows.sfm.naive import lazy_sfm_naive, sfm_naive
from dynflows.sfm.orlin import sfm_orlin

try:
    import mcf_python
    print('Using Rust backend for min-cost flows')
except:
    print('Did not find Rust backend. Using Python implementation.')


def get_out_flow(
        G: nx.DiGraph,
        A: Set[Any],
        T: int,
        balance='balance',
        capacity='capacity', 
        transit='transit') -> int:
    """Compute the maximum amount of flow that can be sent out of subset A of terminals within a given time horizon.

    Args:
        G (nx.DiGraph): The network of the dynamic transshipment instance.
        A (Set[Any]): A set of terminals.
        T (int): The time horizon of the dynamic transshipment instance.
        balance (str, optional): The node attribute representing node balances. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing arc capacities. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing arc transit times. Defaults to 'transit'.

    Returns:
        int: Returns the maximum flow value.
    """
    sources = {n for n, c in G.nodes(data=balance, default=0) if c > 0}
    sinks = {n for n, c in G.nodes(data=balance, default=0) if c < 0}

    if isinstance(A, list):
        A = set(A)

    return max_flow_over_time(G, sources & A, sinks - A, T, capacity=capacity, transit=transit, return_flow=False)


class MaxOutFlow:
    def __init__(self, G, balance='balance', capacity='capacity', transit='transit') -> None:
        self.__G = G
        self.__sources = {n for n, c in G.nodes(data=balance, default=0) if c > 0}
        self.__sinks = {n for n, c in G.nodes(data=balance, default=0) if c < 0}

        self.__nodes_idx = {n: idx for idx, n in enumerate(G.nodes)}

        # Replace infs with the maximum value for the Rust datatype isize.
        self.__caps = [attr.get(capacity, 9_223_372_036_854_775_807) for _, _, attr in G.edges(data=True)]
        self.__transits = [attr.get(transit, 0) for _, _, attr in G.edges(data=True)]
        self.__arcs = [(self.__nodes_idx[u], self.__nodes_idx[v]) for u, v in G.edges()]

    def __call__(self, A, T) -> Any:
        if isinstance(A, list):
            A = set(A)

        src_idx = [self.__nodes_idx[n] for n in (self.__sources & A)]
        snk_idx = [self.__nodes_idx[n] for n in (self.__sinks - A)]

        value = mcf_python.max_dynamic_flow_value(len(self.__G.nodes), self.__arcs, T, src_idx, snk_idx, self.__caps, self.__transits)
        assert value != -1, 'min-cost flow computation was unsuccessful.'

        return value
    

class Greedy:
    def __init__(self, G, balance='balance', capacity='capacity', transit='transit') -> None:
        self.__G = G
        self.__nodes_idx = {n: idx for idx, n in enumerate(G.nodes)}

        # self.__bals = [bal for _ bal]
        self.__caps = [attr.get(capacity, np.inf) for _, _, attr in G.edges(data=True)]
        self.__transits = [attr.get(transit, 0) for _, _, attr in G.edges(data=True)]
        self.__arcs = [(self.__nodes_idx[u], self.__nodes_idx[v]) for u, v in G.edges()]

    def __call__(self, perm, T) -> Any:
        pass


def get_net_balance(
        G: nx.DiGraph,
        A: Set[Any],
        T: int,
        balance='balance', 
        capacity='capacity', 
        transit='transit',
        max_flow_method: MaxOutFlow = None) -> int:
    """Compute the amount of flow that needs to be send out of a subset A of terminals.

    Args:
        G (nx.DiGraph): The network of the dynamic transshipment instance.
        A (Set[Any]): A set of terminals.
        T (int): The time horizon of the dynamic transshipment instance.
        balance (str, optional): The node attribute representing node balances. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing arc capacities. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing arc transit times. Defaults to 'transit'.

    Returns:
        int: Returns the net balance.
    """
    net_balance = sum(G.nodes[n].get(balance, 0) for n in A)

    if max_flow_method is not None:
        return max_flow_method(A, T) - net_balance

    return get_out_flow(G, A, T, balance, capacity, transit) - net_balance


def __is_tight(
        G: nx.DiGraph,
        A: Set[Any],
        T: int,
        balance='balance', 
        capacity='capacity', 
        transit='transit') -> bool:
    """Checks if a subset A of the termminals is tight, i.e. o^T(A) = b(A).

    Args:
        G (nx.DiGraph): The network of the dynamic transshipment instance.
        A (List[Any]): A set of terminals to check tightness for.
        T (int): The time horizon of the dynamic transshipment instance
        balance (str, optional): The node attribute representing node balances. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing arc capacities. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing arc transit times. Defaults to 'transit'.
    """
    return get_net_balance(G, A, T, balance=balance, capacity=capacity, transit=transit) == 0


def __feasibility_naive(
        G: nx.DiGraph, 
        T: int,
        balance='balance', 
        capacity='capacity', 
        transit='transit',
        return_minimizer=False,
        lazy=True,
        max_flow_method: MaxOutFlow = None) -> bool:
    """Check feasibility of a dynamic transshipment instance using the naive sfm algorithm.

    Args:
        G (nx.DiGraph): The network of the dynamic transshipment instance.
        T (int): The time horizon of the dynamic transshipment instance
        balance (str, optional): The node attribute representing node balances. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing arc capacities. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing arc transit times. Defaults to 'transit'.

    Returns:
        bool: Returns True iff the dynamic transshipment instance is feasible.
    """
    terminals = {n for n, b in G.nodes(data=balance, default=0) if b != 0}

    def f(A):
        if len(A) == 0:
            return 0

        return get_net_balance(G, A, T, balance=balance, capacity=capacity, transit=transit, max_flow_method=max_flow_method)
    
    # Find the solution to the SFM instance
    if lazy:
        sol, val = lazy_sfm_naive(terminals, f, 0)
    else:
        sol, val = sfm_naive(terminals, f)

    if return_minimizer:
        return val == 0, sol
    
    return val == 0


def __feasibility_orlin(
        G: nx.DiGraph, 
        T: int,
        balance='balance', 
        capacity='capacity', 
        transit='transit',
        return_minimizer=False,
        lazy=True,
        max_flow_method: MaxOutFlow = None) -> bool:
    """Determine feasibility of a dynamic transshipment instance using Orlin's algorithm for submodular function minimization.

    Args:
        G (nx.DiGraph): Check feasibility of a dynamic transshipment instance using Orlin's algorithm for SFM.
        T (int): The time horizon of the dynamic transshipment instance
        balance (str, optional): The node attribute representing node balances. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing arc capacities. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing arc transit times. Defaults to 'transit'.
        return_minimizer (bool, optional): Whether to return the set that violates the feasibility criterion. Defaults to False.
        lazy (bool, optional): If True, the algorithm returns the first violator that it encounters. Defaults to True.
        max_flow_method (MaxOutFlow, optional): The max flow over time method to use. Optional. Defaults to None.

    Returns:
        bool: True if feasible, False otherwise. Also returns the minimizer if return_minimizer is True.
    """
    # Dismiss any instance for which the sum of balances isn't 0.
    assert sum(b for _, b in G.nodes(data=balance, default=0)) == 0, "The sum of balances must be 0!"

    terminals = {n for n, b in G.nodes(data=balance, default=0) if b != 0}

    def f(A):
        if len(A) == 0:
            return 0

        return get_net_balance(G, A, T, balance=balance, capacity=capacity, transit=transit, max_flow_method=max_flow_method)

    # Find the solution to the SFM instance
    sol, val = sfm_orlin(terminals, f, lazy=lazy)

    assert np.isclose(f(sol), val), "Orlin's algorithm didn't work as intended."

    if return_minimizer:
        return np.isclose(val, 0), sol
    
    return np.isclose(val, 0)


def is_feasible(
        G: nx.DiGraph, 
        T: int,
        balance='balance', 
        capacity='capacity', 
        transit='transit',
        method: Literal['orlin', 'naive'] = 'orlin',
        return_violated: bool = False,
        lazy: bool = True,
        max_flow_method: MaxOutFlow = None) -> bool:
    """Check feasibility of a dynamic transshipment instance (N, b, T).

    Args:
        G (nx.DiGraph): The network of the dynamic transshipment instance.
        T (int): The time horizon of the dynamic transshipment instance.
        balance (str, optional): The node attribute representing node balances. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing arc capacities. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing arc transit times. Defaults to 'transit'.
        method (Literal'orlin', 'naive'], optional): The method to use for SFM. Defaults to 'orlin'.
        return_violated (bool, optional): Whether to return the violated subset of terminals. Defaults to False.

    Returns:
        bool: Returns True iff the dynamic transshipment instance is feasible. 
    """
    if method == 'naive':
        return __feasibility_naive(G, T, balance=balance, capacity=capacity, transit=transit, return_minimizer=return_violated, lazy=lazy, max_flow_method=max_flow_method)
    elif method == 'orlin':
        return __feasibility_orlin(G, T, balance=balance, capacity=capacity, transit=transit, return_minimizer=return_violated, lazy=lazy, max_flow_method=max_flow_method)
    else:
        raise NotImplementedError()


def __discrete_newton_capacity():
    pass


def discrete_newton(
        G: nx.DiGraph,
        func,
        param_init,
        max_flow_method=None,
        balance='balance', 
        capacity='capacity', 
        transit='transit') -> int:
    """_summary_

    Args:
        G (nx.DiGraph): _description_
        func (_type_): _description_
        param_init (_type_): _description_
        max_flow_method (_type_, optional): _description_. Defaults to None.
        balance (str, optional): _description_. Defaults to 'balance'.
        capacity (str, optional): _description_. Defaults to 'capacity'.
        transit (str, optional): _description_. Defaults to 'transit'.

    Returns:
        int: _description_
    """
    param = param_init

    while True:
        feas, minimizer = func(param, return_minimizer=True)

        if feas:
            assert not func(param-1, return_minimizer=False), "The identified parameter is not optimal."

            return param
        
        while True:
            if get_net_balance(G, minimizer, param, balance, capacity, transit, max_flow_method) >= 0:
                break
            
            param += 1

        print(f'Moving to T={param}')


def __tweak_capacity(
        G: nx.DiGraph,
        Q: Set[Any],
        u: Any,
        v: Any,
        is_source: bool,
        T: int, 
        balance='balance', 
        capacity='capacity', 
        transit='transit',
        method: Literal['orlin', 'naive'] = 'orlin',
        search_method: Literal['binary', 'newton'] = 'binary') -> Tuple[int, int]:
    """Determines a maximum alpha such that the modified dynamic transshipment instance is feasible. Alpha
    determines the arc's (u,v) capacity. This function uses binary search to find alpha.

    Args:
        G (nx.DiGraph): The network of the dynamic transshipment instance.
        Q (Set[Any]): The subset in the chain to add the new terminal to.
        u (Any): The starting node of the arc (u,v)
        v (Any): The end node of the arc (u,v)
        is_source (bool): If True, the nodes u and v are both sources. If not, both are sinks.
        T (int): The time horizon for the dynamic transshipment.
        balance (str, optional): The node balances. Missing values for a node are interpreted as 0. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing an arc's capacity. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing an arc's transit time. Defaults to 'transit'.
        method (Literal['orlin', 'naive'], optional): The method to use for SFM. Defaults to 'orlin'.

    Returns:
        int: The maximum feasible alpha.
    """
    node = u if is_source else v
    bal_changes = dict()

    # Get the first terminal for the new added terminal and set the maximum alpha to the absolute of it's balance.
    first_term = (node[0], 0)
    max_alpha = abs(G.nodes[first_term][balance])

    def func(alpha: int) -> bool:
        """Constructs a dynamic transshipment instance by adding alpha capacity to the arc (u, v) and
        modifying the balances of the first terminal and the new terminal correspondingly. This function
        determines feasibility of the resulting dynamic transshipment instance.

        Args:
            alpha (int): The choice of alpha to test.

        Returns:
            bool: Returns True iff the instance is feasible.
        """
        # Just to make sure that changes done in one call don't affect the other calls...
        G_ext = G.copy()

        # Update the arcs capacity to the new candidate value.
        G_ext.edges[u, v][capacity] = alpha

        # Compute how the out-flow changes if new termimal is added to the parametrized network with the current choice of alpha.
        G_ext.nodes[node][balance] = 1 if is_source else -1

        # If available, use the Rust backend.
        if 'mcf_python' in sys.modules:
            max_flow_method = MaxOutFlow(G_ext, balance, capacity, transit)
        else:
            max_flow_method = None

        Q_before = Q
        Q_after = Q | {node}
        
        b = get_out_flow(G_ext, Q_after, T, balance, capacity, transit) - get_out_flow(G_ext, Q_before, T, balance, capacity, transit)
                     
        bal_changes[alpha] = b

        # Setting the new terminals balance too high (for sources) or too low (for sinks) can't possibly result in a feasible instance.
        if (is_source and b > G_ext.nodes[first_term][balance]) or (not is_source and b < G_ext.nodes[first_term][balance]):
            return False

        # Setting the balance of the new terminal to 0 is always feasible.
        if b == 0:
            return True

        # Adapt the new terminal's and first terminal's balances correspondingly.
        G_ext.nodes[node][balance] = b
        G_ext.nodes[first_term][balance] += -b

        return is_feasible(G_ext, T, balance=balance, capacity=capacity, transit=transit, method=method, max_flow_method=max_flow_method)

    # Compute the optimal alpha and set the arc's capacity to it.
    if search_method == 'binary':
        alpha = find_max_feasible(func, max_alpha)
    else:
        pass

    # Fetch the computed balance for the new terminal.
    b = bal_changes[alpha]

    return alpha, b


def __tweak_transit(
        G: nx.DiGraph,
        Q: Set[Any],
        u: Any,
        v: Any,
        is_source: bool,
        T: int, 
        balance='balance', 
        capacity='capacity', 
        transit='transit',
        method: Literal['orlin', 'naive'] = 'orlin'):
    """Determines a minimum delta such that the modified dynamic transshipment instance is feasible. Delta
    determines the arc's (u,v) transit time. This function uses binary search to find delta.

    Args:
        G (nx.DiGraph): The network of the dynamic transshipment instance.
        Q (Set[Any]): The subset in the chain to add the new terminal to.
        u (Any): The starting node of the arc (u,v)
        v (Any): The end node of the arc (u,v)
        is_source (bool): If True, the nodes u and v are both sources. If not, both are sinks.
        T (int): The time horizon for the dynamic transshipment.
        balance (str, optional): The node balances. Missing values for a node are interpreted as 0. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing an arc's capacity. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing an arc's transit time. Defaults to 'transit'.
        method (Literal['orlin', 'naive'], optional): The method to use for SFM. Defaults to 'orlin'.
    """
    node = u if is_source else v
    bal_changes = dict()

    # Get the first terminal for the new added terminal and set the maximum alpha to the absolute of it's balance.
    first_term = (node[0], 0)
    max_delta = T+1

    def func(delta: int) -> bool:
        """Constructs a dynamic transshipment instance by adding alpha capacity to the arc (u, v) and
        modifying the balances of the first terminal and the new terminal correspondingly. This function
        determines feasibility of the resulting dynamic transshipment instance.

        Args:
            alpha (int): The choice of alpha to test.

        Returns:
            bool: Returns True iff the instance is feasible.
        """
        # Don't allow negative deltas
        if delta < 0:
            return False

        # Just to make sure that changes done in one call don't affect the other calls...
        G_ext = G.copy()

        # Update the arcs capacity to the new candidate value.
        G_ext.edges[u, v][transit] = delta

        # Compute how the out-flow changes if new termimal is added to the parametrized network with the current choice of alpha.
        G_ext.nodes[node][balance] = 1 if is_source else -1

        # If available, use the Rust backend.
        if 'mcf_python' in sys.modules:
            max_flow_method = MaxOutFlow(G_ext, balance, capacity, transit)
        else:
            max_flow_method = None

        Q_before = Q
        Q_after = Q | {node}
        b = get_out_flow(G_ext, Q_after, T, balance, capacity, transit) - get_out_flow(G_ext, Q_before, T, balance, capacity, transit)
         
        bal_changes[delta] = b
        
        # Setting the new terminals balance too high (for sources) or too low (for sinks) can't possibly result in a feasible instance.
        if (is_source and b > G_ext.nodes[first_term][balance]) or (not is_source and b < G_ext.nodes[first_term][balance]):
            return False
        
        # Setting the balance of the new terminal to 0 is always feasible.
        if b == 0:
            return True

        # Adapt the new terminal's and first terminal's balances correspondingly.
        G_ext.nodes[node][balance] = b
        G_ext.nodes[first_term][balance] += -b

        return is_feasible(G_ext, T, balance=balance, capacity=capacity, transit=transit, method=method, max_flow_method=max_flow_method)

    # Compute the optimal delta and set the arc's transit time to it.
    delta = find_min_feasible(func, max_delta)

    # Adapt the new terminal's and first terminal's balances correspondingly.
    b = bal_changes[delta]

    return delta, b


def __find_tight(
        G: nx.DiGraph,
        Q: Set[Any],
        u: int,
        v: int,
        is_source: bool,
        T: int, 
        balance='balance', 
        capacity='capacity', 
        transit='transit',
        method: Literal['orlin', 'naive'] = 'orlin') -> Set[Any]:
    """Subroutine for the dynamic transshipment algorithm that finds a tight subset for the given instance.
    Since we chose the transit time (u, v) such that reducing it leads to an infeasible transshipment instance,
    we find a tight set by reducing the arc's transit time and finding the infeasible set in that dynamic transshipment
    instance.

    Args:
        G (nx.DiGraph): The modified network at the current iteration of the algorithm.
        u (int): The starting node of the newly added arc.
        v (int): The end node of the newly added arc.
        T (int): The time horizon of the dynamic transshipment instance.
        balance (str, optional): The node balances. Missing values for a node are interpreted as 0. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing an arc's capacity. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing an arc's transit time. Defaults to 'transit'.
        method (Literal['orlin', 'naive'], optional): The method to use for SFM. Defaults to 'orlin'.

    Returns:
        int: A tight set.
    """
    node = u if is_source else v
    first_term = (node[0], 0)

    G_ext = G.copy()

    # Reset the network's balances to the original state.
    db = G_ext.nodes[node][balance]
    G_ext.nodes[node][balance] = 0
    G_ext.nodes[first_term][balance] += db

    # Decrease delta by one and adapt the balances correspondingly.
    G_ext.edges[u, v][transit] -= 1

    # Dummy balance to mark the node as a terminal node.
    G_ext.nodes[node][balance] = 1 if is_source else -1

    # If available, use the Rust backend.
    if 'mcf_python' in sys.modules:
        max_flow_method = MaxOutFlow(G_ext, balance, capacity, transit)
    else:
        max_flow_method = None

    Q_before = Q
    Q_after = Q | {node}
    b = get_out_flow(G_ext, Q_after, T, balance, capacity, transit) - get_out_flow(G_ext, Q_before, T, balance, capacity, transit)

    G_ext.nodes[node][balance] = b
    G_ext.nodes[first_term][balance] += -b

    feasible, tight_set = is_feasible(G_ext, T, balance, capacity, transit, method, return_violated=True, max_flow_method=max_flow_method)

    assert not feasible, "The constructed network has to be infeasible."

    return tight_set


def dynamic_transshipment(
        G: nx.DiGraph, 
        T: int, 
        balance='balance', 
        capacity='capacity', 
        transit='transit',
        method: Literal['orlin', 'naive'] = 'orlin') -> TemporallyRepeatedFlow:
    """Compute a dynamic transshipment with time horizon T satisfying the provided balances.

    Args:
        G (nx.DiGraph): The network to compute the flow in.
        T (int): The time horizon for the dynamic transshipment.
        balance (str, optional): The node balances. Missing values for a node are interpreted as 0. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing an arc's capacity. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing an arc's transit time. Defaults to 'transit'.
        method (Literal['orlin', 'naive'], optional): The method to use for SFM. Defaults to 'orlin'.

    Returns:
        TemporallyRepeatedFlow: Returns a solution to the dynamic transshipment problem (N, b, T).
    """
    G_ext = nx.DiGraph()
    G_ext.add_nodes_from(G.nodes())
    G_ext.add_edges_from(G.edges(data=True))

    sources = [n for n, b in G.nodes(data=balance, default=0) if b > 0]
    sinks = [n for n, b in G.nodes(data=balance, default=0) if b < 0]

    sources_, sinks_, terminals_ = [], [], []

    for s in sources:
        s0 = (s, 0)

        # Add a new source s0 with the capacity as s and connect s0 to s via infinite-capacity, zero-transit arc.
        G_ext.add_node(s0, **{balance: G.nodes[s].get(balance, 0)})
        G_ext.add_edge(s0, s)

        sources_.append(s0)
        terminals_.append(s0)

    for t in sinks:
        t0 = (t, 0)

        # Add a new sink t0 with the capacity as t and connect t to it via infinite-capacity, zero-transit arc.
        G_ext.add_node(t0, **{balance: G.nodes[t].get(balance, 0)})
        G_ext.add_edge(t, t0)

        sinks_.append(t0)
        terminals_.append(t0)

    # The chain we want to construct. Start with the obvious tight sets.
    chain = [set(), set(terminals_)]

    # Keept track of how many additional terminals were added for each terminal.
    counters = {n: 0 for n in sources + sinks}

    with tqdm(total=len(terminals_)) as pbar:
        while len(chain) - len(terminals_) != 1:
            # Find the next adjacent sets Q and R in the chain with |R \ Q| > 1.
            for i in range(len(chain)-1):
                if len(chain[i+1]) - len(chain[i]) > 1:
                    Q, R = chain[i], chain[i+1]

                    break

            # Pick a terminal that changes the out-flow as much as possible.
            candidates = list(R - Q)  
            s = candidates[0]
            
            n, _ = s
            is_source = G.nodes[n][balance] > 0

            assert s[-1] == 0, 'The terminal should be a first terminal. Invariant violated.'            

            is_tight = False

            if __is_tight(G_ext, Q | {s}, T, balance=balance, capacity=capacity, transit=transit):
                # If Q with s added is already tight, nothing more to do.
                chain.insert(i+1, Q | {s})
                is_tight = True

            if __is_tight(G_ext, R - {s}, T, balance=balance, capacity=capacity, transit=transit):
                # If R removed from R is already tight, nothing more to do
                chain.insert(i+1, R - {s})
                is_tight = True

            if is_tight:
                continue
            else:
                # Add the first new terminal.
                n1 = (n, counters[n]+1)
                G_ext.add_node(n1)

                # Connect both terminals via an arc and determine an optimal capacity for it.
                if is_source:
                    G_ext.add_edge(n1, n)

                    # Compute maximum capacity for the arc (n, n1) such that redistributing balance leads to a feasible network.
                    alpha, b = __tweak_capacity(G_ext, Q, n1, n, is_source, T, balance, capacity, transit, method)

                    # Add the computed alpha and balance changes to the extended network.
                    G_ext.edges[n1, n][capacity] = alpha
                    G_ext.nodes[n1][balance] = b
                    G_ext.nodes[(n, 0)][balance] += -b

                    sources_.append(n1)
                    terminals_.append(n1)

                    # After the adaptions to the network it's safe to add n1 to Q. Since Q is tight, we can add it to the chain.
                    Q_1 = Q | { n1 }
                    assert __is_tight(G_ext, Q_1, T, balance, capacity, transit), "Something went wrong while determining alpha."
                else:
                    G_ext.add_edge(n, n1)

                    # Compute maximum capacity for the arc (n, n1) such that redistributing balance leads to a feasible network.
                    alpha, b = __tweak_capacity(G_ext, R, n, n1, is_source, T, balance, capacity, transit, method)

                    # Add the computed alpha and balance changes to the extended network.
                    G_ext.edges[n, n1][capacity] = alpha
                    G_ext.nodes[n1][balance] = b
                    G_ext.nodes[(n, 0)][balance] += -b

                    sinks_.append(n1)
                    terminals_.append(n1)

                    # After the adaptions to the network it's safe to add n1 to R. Since R is tight, we can add it to the chain.
                    Q_1 = R | { n1 }
                    assert __is_tight(G_ext, Q_1, T, balance, capacity, transit), "Something went wrong while determining alpha."

                # Add the second new terminal.
                n2 = (n, counters[n]+2)
                G_ext.add_node(n2)

                if is_source:
                    G_ext.add_edge(n2, n, **{capacity: 1})
                    
                    delta, b = __tweak_transit(G_ext, Q_1, n2, n, is_source, T, balance, capacity, transit, method)

                    # Add the computed delta and balance changes to the extended network.
                    G_ext.edges[n2, n][transit] = delta
                    G_ext.nodes[n2][balance] = b
                    G_ext.nodes[(n, 0)][balance] += -b

                    assert __is_tight(G_ext, Q_1, T, balance, capacity, transit), "Q_1 is no longer tight."
                    assert __is_tight(G_ext, Q_1 | { n2 }, T, balance, capacity, transit), "Q_2 is no longer tight."

                    # Find a tight set by reducing the newly found delta by one.
                    W = __find_tight(G_ext, Q_1, n2, n, is_source, T, balance, capacity, transit, method=method)

                    assert __is_tight(G_ext, W, T, balance, capacity, transit), "The identified W is not tight!"

                    sources_.append(n2)
                    terminals_.append(n2)
                else:
                    G_ext.add_edge(n, n2, **{capacity: 1})
                    delta, b = __tweak_transit(G_ext, Q_1, n, n2, is_source, T, balance, capacity, transit, method)

                    # Add the computed delta and balance changes to the extended network.
                    G_ext.edges[n, n2][transit] = delta
                    G_ext.nodes[n2][balance] = b
                    G_ext.nodes[(n, 0)][balance] += -b

                    assert __is_tight(G_ext, Q_1, T, balance, capacity, transit), "Q_1 is no longer tight."
                    assert __is_tight(G_ext, Q_1 | { n2 }, T, balance, capacity, transit), "Q_2 is no longer tight."

                    # Find a tight set by reducing the newly found delta by one.
                    W = __find_tight(G_ext, Q_1, n, n2, is_source, T, balance, capacity, transit, method=method)

                    assert __is_tight(G_ext, W, T, balance, capacity, transit), "The identified W is not tight!"
                    
                    sinks_.append(n2)
                    terminals_.append(n2)

                # After the adaptations it's safe to add n1 to Q. Since Q is tight, we can add it to the chain.
                Q_2 = Q_1 | { n2 }
                assert __is_tight(G_ext, Q_2, T, balance, capacity, transit), "Something went wrong while determining delta."

                # Update all later elements in the chain to also contain the new terminals
                for k in range(i+1, len(chain)):
                    chain[k] = chain[k] | Q_2
                
                # Add the new sets to the chain.
                if is_source:
                    if Q_2 | (R & W) != R:
                        chain.insert(i+1, Q_2 | (R & W))

                    chain.insert(i+1, Q_2)
                    chain.insert(i+1, Q_1)
                else:
                    chain.insert(i+1, Q_1)
                    chain.insert(i+1, R)

                    if Q | (R & W) != R:
                        chain.insert(i+1, Q | (R & W))

                # Update the counter for this terminal
                counters[n] += 2
            
            pbar.update(1)
        pbar.update(1)

    # Check if the new balances are the same as the old balances but distributed across the new terminals
    for term in sources + sinks:
        bal = G.nodes[term][balance]
        new_bals = sum([G_ext.nodes[(term, c)][balance] for c in range(counters[term]+1)])

        assert bal == new_bals, f"New terminls for terminal {term} have incorrect balances!"

    # Check if the chain only contains tight sets.
    for A in chain:
        assert __is_tight(G_ext, A, T, balance, capacity, transit), "Not all sets in the chain are tight!"

    # Compute the permutation arising from the chain.
    perm = [list(chain[i+1] - chain[i])[0] for i in range(len(chain)-1)]
    assert len(perm) == len(chain)-1 and len(perm) == len(terminals_), "Some element is missing."

    # Compute a lexiciograpically maximum flow over time which will result in a solution of the given transshipment instance.
    flow = lex_max_flow_over_time(G_ext, T, perm, sources_, sinks_, capacity, transit)

    # Remove the newly added terminals.
    flow.remove_nodes(terminals_)

    return flow


def quickest_transshipment(
        G: nx.DiGraph, 
        balance='balance', 
        capacity='capacity', 
        transit='transit',
        sfm_method: Literal['orlin', 'naive'] = 'orlin',
        search_method: Literal['binary', 'newton'] = 'binary') -> Tuple[TemporallyRepeatedFlow, int]:
    """Compute a dynamic transshipment with minimum feasible time horizon for the given dynamic network.

    Args:
        G (nx.DiGraph): The network to compute the flow in.
        balance (str, optional): The node balances. Missing values for a node are interpreted as 0. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing an arc's capacity. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing an arc's transit time. Defaults to 'transit'.
        sfm_method (Literal['orlin', 'naive'], optional): The method to use for SFM. Defaults to 'orlin'.
        search_method (Literal['binary', 'newton'], optional): The method to use for the parametric search. Defaults to 'binary'.
    Returns:
        Tuple[DynamicFlow, int]: Returns the resulting flow over time and the corresponding minimum time horizon.
    """
    assert sum(b for _, b in G.nodes(data=balance, default=0)) == 0, "The sum of balances must be 0!"

    # If available, use the Rust backend.
    if 'mcf_python' in sys.modules:
        max_out_flow = MaxOutFlow(G, balance, capacity, transit)
    else:
        max_out_flow = None

    if sfm_method == 'naive':
        def feasibility_method(T: int, return_minimizer=False):
            return __feasibility_naive(G, T, balance=balance, capacity=capacity, transit=transit, max_flow_method=max_out_flow, return_minimizer=return_minimizer)
    elif sfm_method == 'orlin':
        def feasibility_method(T: int, return_minimizer=False):
            return __feasibility_orlin(G, T, balance=balance, capacity=capacity, transit=transit, max_flow_method=max_out_flow, return_minimizer=return_minimizer)
    else:
        raise NotImplementedError()
    
    if search_method == 'binary':
        # Get an upper bound on T*
        T_max = get_upper_bound_of_T(G, balance, transit)

        T = find_min_feasible(feasibility_method, T_max, verbose=True)
    elif search_method == 'newton':
        T = discrete_newton(G, feasibility_method, 0, max_out_flow, balance, capacity, transit)
    else:
        raise NotImplementedError()

    flow = dynamic_transshipment(G, T, balance, capacity, transit, sfm_method)

    return flow, T