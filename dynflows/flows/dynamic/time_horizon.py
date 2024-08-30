import networkx as nx

from itertools import product
from networkx.algorithms.shortest_paths import shortest_path_length


def get_upper_bound_of_T(G: nx.DiGraph, balance='balance', transit='transit') -> int:
    """Computes an upper bound on the minimum feasible time horizon by computing the shortest path between each source-sink pair and taking the 
    longest of those paths to estimate how long it would take to send all supply along this path, assuming that the path's capacity is 1.

    Args:
        G (nx.DiGraph): The network for which to estimate the time horizon.
        balance (str, optional): The node balances. Missing values for a node are interpreted as 0. Defaults to 'balance'.
        capacity (str, optional): The arc attribute representing an arc's capacity. Defaults to 'capacity'.
        transit (str, optional): The arc attribute representing an arc's transit time. Defaults to 'transit'.

    Returns:
        int: An upper bound on the minimum feasible time horizon for the given dynamic transshipment instance.
    """
    # Get the sum of all supply in G.
    supply = sum([max(G.nodes[n].get(balance, 0), 0) for n in G.nodes])

    sources = [n for n in G.nodes if G.nodes[n].get(balance, 0) > 0]
    sinks = [n for n in G.nodes if G.nodes[n].get(balance, 0) < 0]

    max_length = 0

    # Find the longest shortest path between a source and a sink.
    for source, sink in product(sources, sinks):
        length = shortest_path_length(G, source, sink, transit)

        if length > max_length:
            length = max_length

    # If we temporally repeat this path it takes length + supply time steps to satisfy all the supply.
    return length + supply
