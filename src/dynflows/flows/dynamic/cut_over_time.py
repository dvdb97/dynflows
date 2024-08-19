from types import UnionType
from typing import Dict, Any, List, Set, Self

from numpy import Inf
from networkx import DiGraph


class CutOverTime:
    def __init__(self, nodes: Dict[Any, int] | List[Any] | Set[Any]) -> None:
        """Construct a cut over time.

        Args:
            nodes (Dict[Any, int] | List[Any] | Set[Any]): Either a list of nodes or a dict mapping nodes to alpha values.
        """
        if isinstance(nodes, dict):
            self.__alphas = nodes
        else:
            self.__alphas = {node: 0 for node in nodes}

    def get_alpha(self, node: Any) -> int:
        """
        Args:
            node (Any): A node.

        Returns:
            int: Returns the alpha value for the given node.
        """
        return self.__alphas[node]

    def get_capacity(self, G: DiGraph, capacity='capacity', transit='transit') -> int:
        """Compute the capacity for this cut.

        Args:
            G (DiGraph): The underlying network.
            capacity (str, optional): The arc attribute representing the arc's capacity. Defaults to 'capacity'.
            transit (str, optional):  The arc attribute representing the arc's transit. Defaults to 'transit'.

        Returns:
            int: The capacity of this cut over time in the corresponding network
        """
        cut_cap = 0

        for u, v, attr in G.edges(data=True):
            if u in self.__alphas.keys() and v in self.__alphas.keys():
                alpha_u = self.__alphas[u]
                alpha_v = self.__alphas[v]

                old = cut_cap
                cut_cap += max(0, alpha_v - alpha_u - attr.get(transit, 0)) * attr.get(capacity, Inf)

                if old < cut_cap:
                    pass

        return cut_cap
    
    def get_overlap(self, beta: Self, G: DiGraph, capacity='capacity', transit='transit') -> int:
        overlap = 0

        for u, v, attr in G.edges(data=True):
            if u in self.__alphas.keys() and v in self.__alphas.keys():
                alpha_u = self.__alphas[u]
                alpha_v = self.__alphas[v]

                beta_u = beta.get_alpha(u)
                beta_v = beta.get_alpha(v)

                overlap += max(0, min(alpha_v, beta_v) - attr.get(transit, 0) - max(alpha_u, beta_u)) * attr.get(capacity, Inf)

        return overlap
    
    def __str__(self) -> str:
        return '\n'.join(f'{node}: {alpha}' for node, alpha in self.__alphas.items())
    
    def __contains__(self, other: Self) -> bool:
        return all(other.get_alpha(node) < other.get_alpha(node) for node in self.__alphas.keys())
    
    def __or__(self, other: Self) -> Self:
        return CutOverTime({node: min(self.get_alpha(node), other.get_alpha(node)) for node in self.__alphas.keys()})
    
    def __and__(self, other: Self) -> Self:
        return CutOverTime({node: max(self.get_alpha(node), other.get_alpha(node)) for node in self.__alphas.keys()})
