import networkx as nx
import numpy as np

from abc import abstractmethod
from typing import List, Dict, Any, Set, Tuple

from dynflows.utils.path import ArcType, Path
from dynflows.flows.dynamic.cut_over_time import CutOverTime


class DynamicFlow:
    def __init__(self, T: int) -> None:
        self._T = T

    def get_time_horizon(self) -> int:
        return self._T
    
    def set_time_horizon(self, T: int):
        self._T = T

    @abstractmethod
    def covers_edge(self, u, v) -> bool:
        pass

    @abstractmethod
    def get_flow_value(self, u, v, t: int) -> int:
        pass

    @abstractmethod
    def get_excess(self, v: Any, t: int) -> int:
        pass

    @abstractmethod
    def remove_nodes(self, nodes: List[Any] | Any):
        pass

    def get_net_value(self, nodes: List[Any] | Any, T: int = None) -> Dict[Any, int]:
        """Get the net amount of flow units stored in the given vertices at time T. Can be used to compute the flow's value.

        Args:
            nodes (List[Any] | Any): A list of vertices.
            T (int, optional): A point in time. If None, we take the provided time horizon. Defaults to None.

        Returns:
            Dict[Any, int]: Returns the net amount of flow stored in the given vertices at time T.
        """
        T = self._T if T is None else T

        if not isinstance(nodes, list):
            nodes = [nodes]

        return {node: self.get_excess(node, T) for node in nodes}


class TemporallyRepeatedFlow(DynamicFlow):
    def __init__(self, paths: List[Tuple[Path, int]], T: int, ignore_nodes: List[Any] = []) -> None:
        super().__init__(T)
        self.__paths = paths
        self.__ignore_nodes = set(ignore_nodes)

    def get_flow_value(self, u, v, t: int) -> int:
        """Get the flow value of an arc (u, v) at time t.

        Args:
            u (_type_): The starting point of the arc.
            v (_type_): The end point of the arc.
            t (int): A point in time.

        Returns:
            int: The amount of flow entering the arc (u, v) at time t.
        """
        summed = 0

        for path, value in self.__paths:
            if (u, v) in path:
                if path.get_dist_to(u) <= t and t + path.get_dist_from(u) <= self._T:
                    summed += value if path.get_arc_type(u, v) == ArcType.FORWARD else -value

        return summed

    def get_excess(self, u: Any, t: int) -> int:
        """Get the excess of a vertices u at time t.

        Args:
            u (Any): A vertex u.
            t (int): A point in time t.

        Returns:
            int: The excess of u at time t.
        """
        summed = 0

        # Get all arcs entering u or leaving u.
        in_arcs = {path.get_in_arc(u) for path, _ in self.__paths if path.has_node(u) and path.get_in_arc(u) != None}
        out_arcs = {path.get_out_arc(u) for path, _ in self.__paths if path.has_node(u) and path.get_out_arc(u) != None}

        for v, w, dt, type in in_arcs:
            if type == ArcType.BACKWARD:
                continue

            dt = abs(dt)

            if v not in self.__ignore_nodes:
                for t_ in range(t - dt + 1):
                    summed += self.get_flow_value(v, w, t_)

        for v, w, dt, type in out_arcs:
            if type == ArcType.BACKWARD:
                continue

            dt = abs(dt)

            if w not in self.__ignore_nodes:
                 for t_ in range(t + 1):
                    summed -= self.get_flow_value(v, w, t_)                

        return summed
    
    def remove_nodes(self, nodes: List[Any] | Any):
        for n in nodes:
            self.__ignore_nodes.add(n)

    def covers_edge(self, u, v) -> bool:
        return any(path.has_edge(u, v) for path, _ in self.__paths)
    
    def covers_node(self, u) -> bool:
        return any(path.has_node(u) for path, _ in self.__paths)
    
    def to_cut_over_time(self, G: nx.DiGraph):
        return CutOverTime({node: min([path.get_dist_to(node) for path, _ in self.__paths]) for node in G.nodes() if self.covers_node(node)})
    
    def __str__(self) -> str:
        return '\n'.join(str(path) + f' (value={value})' for path, value in self.__paths)
