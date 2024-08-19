import networkx as nx
import numpy as np

import mcf_python

from dynflows.flows.static.flow import StaticFlow


def min_cost_flow(G: nx.DiGraph, balance='balance', capacity='capacity', weight='weight') -> StaticFlow:
    nodes = list(G.nodes)
    nodes_idx = {n: idx for idx, n in enumerate(G.nodes)}

    bals = [attr.get(balance, 0) for _, attr in G.nodes(data=True)]

    caps = [attr.get(capacity, 10000) for _, _, attr in G.edges(data=True)]
    costs = [attr.get(weight, 0) for _, _, attr in G.edges(data=True)]

    arcs = []

    for u, v in G.edges:
        arcs.append((nodes_idx[u], nodes_idx[v]))

    try:
        flow = mcf_python.min_cost_flow(len(G.nodes), arcs, bals, caps, costs)
    except mcf_python.MFCException:
        print('Failed to compute a min cost flow.')

    flow_dict = dict()

    for (u, v), f in flow.items():
        u, v = nodes[u], nodes[v]

        if u not in flow_dict.keys():
            flow_dict[u] = dict()

        flow_dict[u][v] = f

    return flow_dict