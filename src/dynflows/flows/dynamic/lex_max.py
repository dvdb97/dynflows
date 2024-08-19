import networkx as nx
import numpy as np

from networkx.algorithms.flow import min_cost_flow, max_flow_min_cost
from typing import List

from dynflows.flows.static.flow import StaticFlow
from dynflows.flows.static.circulation import min_cost_circulation
from dynflows.flows.visualize import draw_flow, draw_network
from dynflows.flows.dynamic.max_flow import max_flow_over_time
from dynflows.flows.dynamic.flow import DynamicFlow, TemporallyRepeatedFlow


# noinspection PyPep8Naming
def lex_max_flow_over_time(
        G: nx.DiGraph,
        T: int,
        perm: List,
        sources: List,
        sinks: List,
        capacity='capacity',
        transit='transit') -> DynamicFlow:
    """Compute a lexicographically maximum flow over time.

    Args:
        G (nx.DiGraph): A network with capacities and transit times.
        T (int): An integral time horizon.
        perm (List): A permutation representing an order of the terminals. 
        sources (List): The set of sources in the network.
        sinks (List): The set of sinks in the network.
        capacity (str, optional): The attribute representing an arc's capacity. Defaults to 'capacity'.
        transit (str, optional): The attribute representing an arc's transit time. Defaults to 'transit'.

    Returns:
        DynamicFlow: Returns a lexicographically maximum flow over time with time horizon T.
    """
    assert len(set(sources) & set(sinks)) == 0, 'Source and sink set have to be disjoint.'

    # Add a super-source to the network.
    super_source = 'psi'
    G_ext = G.copy()
    G_ext.add_node(super_source)
    ignore_arcs = list()

    # Connect the super-source to all sources.
    for source in sources:
        G_ext.add_edge('psi', source, transit=0)
        ignore_arcs.append(('psi', source))

    # Start with a zero flow in the erxtended network.
    f = StaticFlow.zero_flow()
    paths = []

    for _, terminal in enumerate(reversed(perm)):
        if terminal in sources:
            # Remove the arc (Psi, s) for this source s
            G_ext.remove_edge(super_source, terminal)

            # Get the residual network of f in the newly constructed network G.
            R = f.get_resnet(G_ext, capacity=capacity, weight=transit)

            # Compute the cheapest maximum flow in the residual network. 
            # Use the super-source and the terminal as source and sink, respectively.
            g = StaticFlow(max_flow_min_cost(R, super_source, terminal, capacity=capacity, weight=transit))
        else:
            G_ext.add_edge(terminal, super_source, **{transit: -T})

            # Get the residual network of f in the newly constructed network G.
            R = f.get_resnet(G_ext, capacity=capacity, weight=transit)

            # Compute a minimum cost circulation in the residual network.
            g = min_cost_circulation(R, capacity=capacity, weight=transit)
            
        # Decompose the resulting flow into paths.
        paths += g.decompose(R, ignore_arcs=ignore_arcs, start_in=sources + ['psi'], costs=transit)

        # Remove the artificial nodes and arcs from the flow.
        g.remove_artificial(R, inplace=True, capacity=capacity)
        f += g

    return TemporallyRepeatedFlow(paths, T=np.inf, ignore_nodes=['psi'])
