import networkx as nx
import matplotlib.pyplot as plt

from itertools import product

from dynflows.flows.dynamic.flow import DynamicFlow
from dynflows.flows.static.flow import StaticFlow



def draw_network(R: nx.DiGraph, capacity='capacity', costs='weight'):
    node_labels = {n: n for n in R.nodes}
    edge_labels = {(u, v): (attr.get(capacity, 'inf'), attr.get(costs, 0)) for u, v, attr in R.edges(data=True)}

    print(capacity, costs)

    pos = nx.circular_layout(R)

    nx.draw_networkx_nodes(R, pos)
    nx.draw_networkx_labels(R, pos, node_labels)

    single_edges = [(u, v) for u, v in R.edges if not R.has_edge(v, u)]

    edges_left = set()
    edges_right = set()
    for u, v in R.edges:
        if R.has_edge(v, u) and len({(u, v), (v, u)} & (edges_left | edges_right)) == 0:
            edges_left.add((u, v))
            edges_right.add((v, u))

    nx.draw_networkx_edges(R, pos, single_edges)
    
    nx.draw_networkx_edges(R, pos, edges_right, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(R, pos, edges_left, connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_edge_labels(R, pos, edge_labels, font_size=7)

    plt.show()


def __draw_static_flow(flow: StaticFlow, G: nx.DiGraph):
    node_labels = {n: n for n in G.nodes}
    edge_labels = {(u, v): flow.get_flow_value(u, v) for u, v in G.edges() if flow.get_flow_value(u, v) != 0}

    F = nx.DiGraph()
    F.add_nodes_from(G.nodes())
    F.add_edges_from([(u, v) for u, v in G.edges() if (u, v) in edge_labels])

    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, node_labels)

    # nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_edges(F, pos, edge_color='blue')
    nx.draw_networkx_edge_labels(F, pos, edge_labels, font_color='blue', font_size=7)

    plt.show()


def __draw_dynamic_flow(flow: DynamicFlow, G: nx.DiGraph, transit='transit'):
    """Visualize a dynamic flow by visualizing the flow in the corresponding time-expanded network.

    Args:
        flow (DynamicFlow): _description_
        G (nx.DiGraph): _description_
        transit (str, optional): _description_. Defaults to 'transit'.
    """
    T = flow.get_time_horizon()
    dist = 1

    G_texp = nx.DiGraph()
    G_texp.add_nodes_from([str(n) + '_' + str(t) for n, t in product(G.nodes(), range(T+1))])

    pos = dict()

    for y in range(T+1):
        for x, node in enumerate(G.nodes):
            t = y
            y *= dist
            x *= dist

            node = f'{node}_{t}'
            pos[node] = (x, -y)

    for u, v in G.edges():
        dt = G.get_edge_data(u, v).get(transit)

        for t in range(dt, T+1):
            f = flow.get_flow_value(u, v, t-dt) 

            if f > 0:
                G_texp.add_edge(f'{u}_{t-dt}', f'{v}_{t}', flow=f)

    node_labels = {n: n for n in G_texp.nodes()}
    edge_labels = {(u, v): f for u, v, f in G_texp.edges(data='flow')}

    nx.draw_networkx_nodes(G_texp, pos, node_size=1, node_shape='.')
    nx.draw_networkx_labels(G_texp, pos, node_labels, font_size=6)

    # nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_edges(G_texp, pos, edge_color='blue')
    nx.draw_networkx_edge_labels(G_texp, pos, edge_labels, font_color='blue', font_size=7)

    plt.show()


def draw_flow(flow: StaticFlow | DynamicFlow, G: nx.DiGraph, transit='transit'):
    if isinstance(flow, StaticFlow):
        __draw_static_flow(flow, G)
    else:
        __draw_dynamic_flow(flow, G, transit=transit)
