import networkx as nx

from random import choice, sample
from typing import List, Tuple


def maximum_walk(G: nx.DiGraph, e: Tuple) -> List:
    """Compute a maximum walk containing e.

    Args:
        G (nx.DiGraph): A directed graph containing e.
        e (Tuple): The edge used as a starting point for the walk.

    Returns:
        List: _description_
    """
    u, v = e
    assert G.has_edge(u, v), "e isn't part of G."

    G_copy: nx.DiGraph = G.copy()
    G_copy.remove_edge(u, v)
    walk = [(u, v)]
    visited = {u, v}

    while G_copy.out_degree(v) != 0:
        for u, v in G_copy.out_edges(v):
            # We have returned to the starting vertex and can stop.
            if v == walk[0][0]:
                walk.append((u, v))

                return walk
            
            if v in visited:
                continue

        G_copy.remove_edge(u, v)
        walk.append((u, v))
        visited.add(v)

    u, v = e

    while G_copy.in_degree(u) != 0:
        for u, v in G_copy.in_edges(u):
            # We have reached the end vertex of the walk and can stop.
            if u == walk[-1][-1]:
                walk = [(u, v)] + walk

                return walk
            
            if u in visited:
                continue
            
        G_copy.remove_edge(u, v)
        walk = [(u, v)] + walk
        visited.add(u)

    return walk


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edges_from([(0, 1), (1, 2), (2, 5), (2, 3), (3, 4)])

    print(maximum_walk(G, (0, 1)))

    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edges_from([(0, 1), (1, 2), (2, 5), (5, 3), (2, 3), (3, 4), (4, 0)])

    print(maximum_walk(G, (0, 1)))
