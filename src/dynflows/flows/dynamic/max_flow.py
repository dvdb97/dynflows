import networkx as nx

from networkx import single_source_bellman_ford_path_length

from typing import List, Any, Set, Union, Tuple

from dynflows.flows.static.circulation import min_cost_circulation
from dynflows.flows.dynamic.flow import DynamicFlow, TemporallyRepeatedFlow
from dynflows.flows.dynamic.cut_over_time import CutOverTime


def construct_extended_network(
        G: nx.DiGraph, 
        sources: List[Any], 
        sinks: List[Any], 
        T: int, 
        transit='transit') -> nx.DiGraph:
    """Construct an extended network by connecting every source to a super-source and every sink to a super-sink. The super-sink is connected to
    the super-sink via an infinite-capacity, (-(T+1))-transit arc.

    Args:
        G (nx.DiGraph): The directed graph of the network.
        sources (List[Any]): The list of vertices that are sources.
        sinks (List[Any]): The list of vertices that are sinks.
        T (int): The time horizon.
        transit (str, optional): The arc attribute representing the transit time. Defaults to 'transit'.

    Returns:
        nx.DiGraph: Returns an extended network.
    """
    assert not G.has_node('super-source') and not G.has_node('super-sink'), \
            "Cannot construct extended network because 'super-sink' and 'super-source already exist."
    
    # Construct the extended graph.
    extended_network = G.copy()
    extended_network.add_nodes_from(['super-source', 'super-sink'])

    # Connect all sources to the super-source.
    for s in sources:
        extended_network.add_edge('super-source', s)

    # Connect all sinks to the super-sink.
    for t in sinks:
        extended_network.add_edge(t, 'super-sink')

    # Add the arc (t,s) with weight -T to the modified graph.
    extended_network.add_edge('super-sink', 'super-source', **{transit: -(T+1)})

    return extended_network


def max_flow_over_time(
        G: nx.DiGraph, 
        sources: Set[Any] | List[Any] | Any, 
        sinks: Set[Any] | List[Any] | Any, 
        T: int, 
        capacity='capacity', 
        transit='transit', 
        return_flow=True) -> Union[Tuple[int, TemporallyRepeatedFlow], int]:
    """Compute a maximum flow over time with time horizon T. Supports multi-source and multi-sink flows.

    Args:
        G (DiGraph): A network with attributes denoting capacities and costs.
        sources (List[Any]): The sources of the flow.
        sinks (List[Any]): The sinks of the flow.
        T (int): The time horizon.
        capacity (str, optional): The arc attribute denoting an arc's capacity.. Defaults to 'capacity'.
        transit (str, optional): The arc attribute denoting an arc's transit. Defaults to 'transit'.
        return_flow (bool, optional): Whether to compute and return the final flow or just the flow value. Defaults to True.

    Returns:
        Union[Tuple[int, TemporallyRepeatedFlow], int]: If return_flow=True, it returns a tuple (int, TemporallyRepeatedFlow), 
        otherwise the function only returns the value of the flow.
    """    
    if not isinstance(sources, list) and not isinstance(sources, set):
        sources = [sources]

    if not isinstance(sinks, list) and not isinstance(sinks, set):
        sinks = [sinks]

    # If there are no sources or sinks, it is impossible to send flow.
    if len(sources) == 0 or len(sinks) == 0:
        if return_flow:
            return TemporallyRepeatedFlow([], T)
        
        return 0
    
    # Construct the extended network by adding a nodes 'super-source' and 'super-sink'.
    extended_network = construct_extended_network(G, sources, sinks, T, transit)

    # Compute the minimum cost circulation in the modified graph.
    flow = min_cost_circulation(extended_network, capacity=capacity, weight=transit)

    # Remove the flow along the (t,s)-arc.
    flow.set_flow_value('super-sink', 'super-source', 0)

    # Compute the value of the temporally repeated flow.
    repetition_penalty = sum([flow.get_flow_value(u, v) * t for u, v, t in G.edges(data=transit, default=0)])
    value = (T+1) * flow.get_value('super-sink') - repetition_penalty

    if return_flow:
         # Transform the circulation into an (s,t)-flow.
        flow.remove_node('super-source', preds=['super-sink'])
        flow.remove_node('super-sink', preds=sinks)

        # Compute a flow over time by temporally repeating the static flow.
        paths = flow.decompose(G, costs=transit)
        dyn_flow = TemporallyRepeatedFlow(paths, T)

        return value, dyn_flow

    return value


def min_cut_over_time(
        G: nx.DiGraph, 
        sources: Set[Any] | List[Any] | Any, 
        sinks: Set[Any] | List[Any] | Any, 
        T: int, 
        capacity='capacity', 
        transit='transit') -> CutOverTime:
    
    if not isinstance(sources, list) and not isinstance(sources, set):
        sources = [sources]

    if not isinstance(sinks, list) and not isinstance(sinks, set):
        sinks = [sinks]

    # If there are no sources or sinks, it is impossible to send flow.
    if len(sources) == 0 or len(sinks) == 0:        
        return None
    
    # Construct the extended network by adding a nodes 'super-source' and 'super-sink'.
    extended_network = construct_extended_network(G, sources, sinks, T, transit)

    # Compute the minimum cost circulation in the modified graph.
    flow = min_cost_circulation(extended_network, capacity=capacity, weight=transit)

    # Get the residual network of the circulation in the extended network.
    resnet = flow.get_resnet(extended_network, capacity=capacity, costs=transit)

    alphas = single_source_bellman_ford_path_length(resnet, 'super-source', weight=transit)
    alphas = {node: alphas[node] for node in G.nodes if node in alphas.keys()}

    return CutOverTime(alphas)

