import networkx as nx
import numpy as np

from copy import deepcopy
from typing import Set, Any
from itertools import combinations

from dynflows.sfm.greedy import greedy


def check_optimality(G: nx.DiGraph, pos, neg):
    new_edges = [('s', p) for p in pos] + [(n ,'t') for n in neg]

    G.add_nodes_from(['s', 't'])
    G.add_edges_from(new_edges)

    if not nx.has_path(G, 's', 't'):
        return None
    
    # Check what nodes are connected to the source node and keep the shortest paths.
    paths = nx.single_source_shortest_path(G, 's')

    G.remove_nodes_from(['s', 't'])

    return paths  


def sfm_schrijver(E: Set[Any], f, custom_greedy=None):
    """

    Args:
        E (_type_): _description_
        f (_type_): _description_
        greedy (_type_, optional): _description_. Defaults to None.
    """
    if custom_greedy is not None:
        greedy = custom_greedy

    idx_to_e = list(E)
    e_to_idx = {e: idx for idx, e in enumerate(idx_to_e)}
    indices = list(range(len(E)))

    # Construct an arbitrary partial order and compute the greedy result for it.
    orders = [deepcopy(indices)]
    x = greedy(f, orders[0], idx_to_e)

    G = nx.DiGraph()
    G.add_nodes_from(indices)

    for order in orders:
        for i, j in combinations(order, 2):
            if not G.has_edge(i, j):
                G.add_edge(i, j, **{'count': 1})
            else:
                G.edges[i][j]['count'] += 1

    while True:
        poss = [idx for idx in indices if x[idx] > 0]
        negs = [idx for idx in indices if x[idx] < 0]

        paths = check_optimality(G, poss, negs)
        t = next(neg for neg in negs if neg in paths) # TODO: Choose lex max s instead.
        s = 0 # TODO:

        i = np.argmax([order.index(t) - order.index(s) for order in orders])
        order = orders[i]









    



