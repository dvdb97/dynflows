import networkx as nx
import numpy as np


def zero_flow(_, __) -> int:
    return 0


def construct_resnet(G: nx.DiGraph, flow=zero_flow, capacity='capacity', weight='weight') -> nx.DiGraph:
    """Build a residual network for a network with respect to a flow.

    Args:
        G (nx.DiGraph): A flow network.
        flow (_type_, optional): A static flow in the given network. Defaults to zero_flow.
        capacity (str, optional): The arc attribute representing the arc's capacity. Defaults to 'capacity'.
        weight (str, optional): The arc attribute representing the arc's weight. Defaults to 'weight'.

    Returns:
        nx.DiGraph: A residual network with artificial nodes and arcs representing backwards arcs.
    """
    resnet = nx.DiGraph()
    resnet.add_nodes_from(G.nodes)

    for u, v, attr in G.edges(data=True):
        node = f'{u}_{v}'

        # Add the original arc to the residual network.
        resnet.add_edge(u, v, {weight: attr.get(weight, 0), capacity: attr.get(capacity, np.inf)-flow(u, v)})

        # Add the residual arc to the residual network.
        resnet.add_node(node, {'type': 'artifcial'})
        resnet.add_edge(v, node, {weight: attr.get(weight, 0), capacity: flow(u, v)})
        resnet.add_edge(node, u, {weight: 0})

    return resnet