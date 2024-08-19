from typing import List, Set, Any

import networkx as nx

from dynflows.flows.dynamic.max_flow import max_flow_over_time


def quickest_flow_naive(
        G: nx.DiGraph,
        source: Any,
        sink: Any,
        balance: int,
        capacity='capacity', 
        transit='transit',
        init_T=0) -> int:
    """Compute the quickest flow over time by naively increasing T from an initial value until
    it the max flow equals the balance.

    Args:
        G (nx.DiGraph): _description_
        source (Any): _description_
        sink (Any): _description_
        balance (int): _description_
        capacity (str, optional): _description_. Defaults to 'capacity'.
        transit (str, optional): _description_. Defaults to 'transit'.
        init_T (int, optional): _description_. Defaults to 0.

    Returns:
        int: The minimum feasibly time horizon for the given network and balance.
    """
    T = init_T

    while True:
        f = max_flow_over_time(G, source, sink, T, capacity, transit, return_flow=False)

        if f >= balance:
            return T
        
        T += 1
