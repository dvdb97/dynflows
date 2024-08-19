import networkx as nx

from networkx import DiGraph
from networkx.algorithms.flow.mincost import min_cost_flow

from dynflows.flows.static.flow import StaticFlow
from dynflows.flows.static.min_cost import min_cost_flow


def min_cost_circulation(graph: DiGraph, capacity='capacity', weight='weight') -> StaticFlow:
    """
    Computes the minimum cost circulation in a given network.

    :param graph: A network with arc attributes denoting the arc's capacity and weight.
    :param capacity: The name of the arc attribute denoting the capacity. Default is 'capacity'.
    :param weight: The name of the arc attribute denoting the weight or costs. Default is 'weight'.
    :return: Returns a
    """
    flow = nx.min_cost_flow(graph, demand='NONE', capacity=capacity, weight=weight)
    # flow = min_cost_flow(graph, balance='NONE', capacity=capacity, weight=weight)

    return StaticFlow(flow)


if __name__ == '__main__':
    """
    # Just some code for testing.
    G = DiGraph()
    G.add_node('s')
    G.add_node('t')
    G.add_node('v_1')
    G.add_node('v_2')

    G.add_edge('s', 'v_1', capacity=5, weight=1)
    G.add_edge('s', 'v_2', capacity=5, weight=1)
    G.add_edge('v_1', 't', capacity=5, weight=1)
    G.add_edge('v_2', 't', capacity=5, weight=1)
    G.add_edge('t', 's', weight=-5)

    print(min_cost_circulation(G))
    """

    G = DiGraph()
    T = 8

    G.add_nodes_from([1, 2, 3, 4, 5, 6])
    G.add_edges_from([
        (1, 2, {'capacity': 1, 'transit': 1}),
        (1, 4, {'capacity': 2, 'transit': 1}),
        (2, 3, {'capacity': 1, 'transit': 1}),
        (4, 5, {'capacity': 1, 'transit': 2}),
        (4, 3, {'capacity': 2, 'transit': 2}),
        (5, 3, {'capacity': 1, 'transit': 3}),
        (5, 6, {'capacity': 1, 'transit': 3}),
        (3, 6, {'capacity': 2, 'transit': 4}),
        (6, 1, {'transit': -T})
    ])

    print(min_cost_circulation(G, weight='transit'))

    G = DiGraph()
    T = 4

    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([
        (1, 2, {'capacity': 2, 'transit': 1}),
        (1, 3, {'capacity': 3, 'transit': 1}),
        (2, 1, {'capacity': 1, 'transit': 1}),
        (2, 4, {'capacity': 3, 'transit': 1}),
        (3, 4, {'capacity': 3, 'transit': 1}),
        (3, 2, {'capacity': 1, 'transit': 1}),
        (4, 1, {'transit': -T})
    ])

    print(min_cost_circulation(G, weight='transit'))


