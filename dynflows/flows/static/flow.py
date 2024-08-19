import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from numpy import Inf
from copy import deepcopy
from itertools import chain
from typing import Dict, Any, List, Tuple, Set, Self
from enum import Enum

from dynflows.utils.walk import maximum_walk
from dynflows.utils.path import ArcType, Path


class StaticFlow:
    def __init__(self, flow_dict: Dict[Any, Dict[Any, int]] = dict()):
        """A data structure for representing static flows.

        Args:
            flow_dict (Dict[Any, Dict[Any, int]], optional): A dictionary representing a flow. Defaults to {}.
        """
        self.__flow = flow_dict

    @staticmethod
    def zero_flow():
        return StaticFlow()

    def __add__(self, other):
        flow = deepcopy(other)

        for u in self.__flow.keys():
            for v in self.__flow[u].keys():
                flow.add_flow_value(u, v, self.get_flow_value(u, v))

        return flow

    def add_flow_value(self, u: Any, v: Any, delta: int):
        """Increase the flow sent along an arc by a given value.

        Args:
            u (Any): The start node of the arc.
            v (Any): The end node of the arc.
            delta (int): The value by which to increase the flow.
        """
        if u in self.__flow:
            if v in self.__flow[u]:
                self.__flow[u][v] += delta
            else:
                self.__flow[u][v] = delta
        else:
            self.__flow[u] = {v: delta}        

    def set_flow_value(self, u: Any, v: Any, value: int):
        """Set the flow sent along a specific arc to a specific value.

        Args:
            u (Any): The start node of the arc.
            v (Any): The end node of the arc.
            value (int): The new flow value.
        """
        self.__flow[u][v] = value

    def get_flow_value(self, u: Any, v: Any) -> int:
        """
        Args:
            u (Any): The start node of the arc.
            v (Any): The end node of the arc.

        Returns:
            int: Retruns the value of the flow sent along a given arc.
        """
        if u in self.__flow and v in self.__flow[u]:
            return self.__flow[u][v]

        return 0
    
    def remove_node(self, node: Any, preds: List[Any]):
        """Completely remove a node from the flow and update the flow acordingly.

        Args:
            node (Any): The node to remove.
            preds (List[Any]): The nodes u such that there is an arc (u, node).
        """
        del self.__flow[node]

        for pred in preds:
            if pred in self.__flow:
                del self.__flow[pred][node]

    def get_excess(self, v: Any) -> int:
        """
        Args:
            v (Any): The node to compute the excess for.

        Returns:
            int: Returns the total excess (i.e. in-flow - out-flow) for a given node.
        """
        in_flow = 0
        out_flow = 0

        for u, f in self.__flow.items():
            if v in f:
                in_flow += f[v]

        for _, value in self.__flow[v].items():
            out_flow += value

        return in_flow - out_flow

    def get_value(self, sink: Any | List[Any]) -> int:
        """Get the value of the flow. 

        Args:
            sink (Any | List[Any]): The network's sink(s). Since the flow doesn't know what the network's sink is, this
        information as to be provided.

        Returns:
            int: The amount of flow sent to the given sinks.
        """
        if isinstance(sink, List):
            return sum([self.get_excess(t) for t in sink])

        return self.get_excess(sink)
    
    def get_resnet(self, G: nx.DiGraph, capacity='capacity', weight='weight') -> nx.DiGraph:
        """Construct a residual network for this flow in a given graph.

        Args:
            G (nx.DiGraph): The network for this flow.
            capacity (str, optional): The attribute denoting an arc's capacity. Defaults to 'capacity'.
            weight (str, optional): The attribute denoting an arc's weight. Defaults to 'weight'.

        Returns:
            nx.DiGraph: returns a residual network containing artificial nodes.
        """
        resnet = nx.DiGraph()
        resnet.add_nodes_from(G.nodes)

        for u, v, attr in G.edges(data=True):
            node = f'res_{u}_{v}'

            # Add the original arc to the residual network.
            resnet.add_edge(u, v, **{weight: attr.get(weight, 0), capacity: attr.get(capacity, np.inf)-self.get_flow_value(u, v)})

            # Add the residual arc to the residual network.
            resnet.add_node(node, **{'type': 'artificial'})
            resnet.add_edge(v, node, **{weight: -attr.get(weight, 0), capacity: self.get_flow_value(u, v), 'type': 'backwards'})
            resnet.add_edge(node, u, **{weight: 0, 'type': 'artificial'})

        return resnet
    
    def remove_artificial(self, G: nx.DiGraph, inplace=False, capacity='capacity'):
        """Collapse flow along artificial arcs.

        Args:
            R (nx.DiGraph): The network of this flow. This network may contain artificial nodes, forward, or backward arcs.
            inplace (bool, optional): Whether to change this flow or return a new instance. Defaults to False.
            capacity (str, optional): The attribute denoting an arc's capacity. Defaults to 'capacity'.
            costs (str, optional): The attribute denoting an arc's costs. Defaults to 'weight'.

        Returns:
            StaticFlow: Returns a StaticFlow instance. Note that this flow can be infeasible.
        """
        flow = self if inplace else deepcopy(self)
        
        for node, attr in G.nodes(data=True):
            if attr.get('type', '') == 'artificial':
                preds = list(G.predecessors(node))
                succs = list(G.successors(node))

                assert len(preds) == 1 and len(succs) == 1, 'Artificial node should only have a single predecessor and successor!'

                v, u = preds[-1], succs[-1]

                # The flow value is the difference between the flow sent along the forward arc and the flow sent along the backwards arc
                f = flow.get_flow_value(u, v) - self.get_flow_value(v, node)

                flow.set_flow_value(u, v, f)
                flow.remove_node(node, [v]) 

        return flow
    
    def decompose(self, G: nx.DiGraph, ignore_arcs: Tuple[Any, Any] | List[Tuple[Any, Any]] = [], ignore_nodes: Any | List[Any] = [], start_in: Any | List[Any] = [], costs='weight') -> List[Tuple[Path, int]]:
        """Decompose this flow into (s,t)-paths in the given network.

        Args:
            G (nx.DiGraph):  A valid network for this flow.
            ignore_arcs (Tuple[Any, Any] | List[Tuple[Any, Any]]): The arc or set of arcs to ignore during decomposition. Defaults to [].
            ignore_nodes (Any | List[Any], optional): The node or set of nodes to ignore during decomposition. Defaults to [].
            costs (str, optional): he arc attribute representing weights / costs. Defaults to 'weight'.

        Returns:
            List[Tuple[Path, int]]: A list of paths and values.
        """
        if not isinstance(ignore_nodes, list):
            ignore_nodes = [ignore_nodes]

        if not isinstance(ignore_arcs, list):
            ignore_arcs = [ignore_arcs]

        ignore_nodes = set(ignore_nodes)
        ignore_arcs = set(ignore_arcs)

        flow = deepcopy(self)

        graph = deepcopy(G)
        graph.remove_edges_from((u, v, d) for u, v, d in G.edges(data=True) if self.get_flow_value(u, v) <= 0)

        # graph.remove_nodes_from(ignore_nodes)
        # graph.remove_edges_from(ignore_arcs)

        paths = list()

        while len(graph.edges()) != 0:
            # Get an arbitrary arc from the network.
            if start_in != []:
                arc = next(chain(*[list(graph.out_edges(node)) for node in start_in]))
            else:
                arc = next(e for e in graph.edges())

            walk = maximum_walk(graph, arc)

            length = 0
            path_value = np.inf
            arcs = list()

            for idx, (u, v) in enumerate(walk):
                attr = graph.get_edge_data(u, v)

                if (u, v) in ignore_arcs:
                    continue

                if u in ignore_nodes or v in ignore_nodes:
                    continue

                length += attr.get(costs, 0)

                # Check if the current candidate flow value for the path also matches this arc's capacity.
                path_value = min(path_value, flow.get_flow_value(u, v))

                # Skip artificial arcs.
                if attr.get('type', None) == 'artificial':
                    continue
                
                # if the node v is an artificial node, we skip it.
                if G.nodes[v].get('type', None) == 'artificial':
                    _, v = walk[idx+1]

                # Check what type of arc (u, v) is.
                if attr.get('type', None) == 'backwards':
                    arc_type = ArcType.BACKWARD

                    # Swap the two nodes to obtain the actual arc in the graph.
                    u, v = v, u
                else:
                    arc_type = ArcType.FORWARD

                arcs.append((u, v, attr.get(costs, 0), arc_type))

            paths.append((Path(arcs), path_value))

            # Reduce the flow value along the path and remove arcs with no flow remaining.
            for u, v in walk:
                flow.set_flow_value(u, v, flow.get_flow_value(u, v) - path_value)

                # Reduce the flow value along the path.
                if flow.get_flow_value(u, v) <= 0:
                    graph.remove_edge(u, v)

        assert len(graph.edges()) == 0, "There shouldn't be any arcs with flow left after decomposition."

        return paths

    def __str__(self):
        out = ''

        for u in self.__flow.keys():
            for v in self.__flow[u].keys():
                if self.__flow[u][v] != 0:
                    out += f"{u} -> {v}: {self.__flow[u][v]}" + ' \n '

        return out


def draw_flow(flow: StaticFlow, G: nx.DiGraph):
    node_labels = {n: n for n in G.nodes}
    edge_labels = {(u, v): flow.get_flow_value(u, v) for u, v in G.edges() if flow.get_flow_value(u, v) != 0}

    F = nx.DiGraph()
    F.add_nodes_from(G.nodes())
    F.add_edges_from([(u, v) for u, v in G.edges() if (u, v) in edge_labels])

    print(edge_labels)

    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, node_labels)

    # nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_edges(F, pos, edge_color='blue')
    nx.draw_networkx_edge_labels(F, pos, edge_labels, font_color='blue', font_size=7)

    plt.show()


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([
        (1, 2, {'capacity': 2, 'cost': 3}),
        (2, 1, {'capacity': 3, 'cost': 4}),
        (1, 3, {'capacity': 1, 'cost': 1}),
        (2, 4, {'capacity': 6, 'cost': 11}),
        (3, 4, {'capacity': 2, 'cost': 1})
    ])

    flow = StaticFlow({1: {2: 1, 3: 1}, 2: {4: 1}, 3: {4: 1}})
    R = flow.get_resnet(G, capacity='capacity', costs='cost')

    draw_flow(flow, G)

    for path in flow.decompose(G, costs='cost'):
        print(path)

    draw_resnet(R, capacity='capacity', costs='cost')
