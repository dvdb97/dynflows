from enum import Enum
from typing import Tuple, Any


class ArcType(Enum):
    FORWARD = 0
    BACKWARD = 1


class Path:
    def __init__(self, arcs: Tuple[Any, Any, int, ArcType]) -> None:
        self.__path = arcs
        self.__arcs = dict()
        self.__nodes = dict()
        self.__arc_length = dict()
        self.__length = 0

        for idx, (u, v, dt, type) in enumerate(arcs):
            self.__arcs[(u, v)] = {'time': self.__length, 'type': type, 'dt': dt}
            
            if type == ArcType.FORWARD:
                self.__nodes[u] = idx
                self.__nodes[v] = idx+1
            else:
                self.__nodes[u] = idx+1
                self.__nodes[v] = idx

            self.__length += dt

            if type == ArcType.FORWARD:
                self.__arc_length[(u, v)] = dt
            else:
                self.__arc_length[(v, u)] = dt

    def __getitem__(self, index):
        return self.__path[index]
    
    def has_node(self, u) -> bool:
        return u in self.__nodes
    
    def has_edge(self, u, v) -> bool:
        return (u, v) in self.__arcs.keys()
    
    def __contains__(self, elem):
        if isinstance(elem, tuple):
            return elem in self.__arcs
        
        return elem in self.__nodes
    
    def get_arc_length(self, u: Any, v: Any):
        return self.__arc_length[(u, v)]

    def get_arc_type(self, u, v) -> ArcType:
        assert (u, v) in self.__arcs, "Path doesn't contain arc."

        return self.__arcs[(u, v)]['type']

    def get_dist_to(self, u: Any) -> int:
        assert u in self.__nodes, "Path doesn't contain node."

        if self.__nodes[u] == len(self.__path):
            return self.__length
        
        u, v, _, _ = self.__path[self.__nodes[u]]

        return self.__arcs[(u, v)]['time']
    
    def get_in_arc(self, u: Any) -> Tuple[Any] | None:
        assert u in self.__nodes, "Path doesn't contain node."

        if self.__nodes[u] == 0:
            return None
        
        return self.__path[self.__nodes[u]-1]
    
    def get_out_arc(self, u: Any) -> Tuple[Any] | None:
        assert u in self.__nodes, "Path doesn't contain node."

        if self.__nodes[u] == len(self.__path):
            return None
        
        return self.__path[self.__nodes[u]]

    def get_dist_from(self, u: Any) -> int:
        return self.__length - self.get_dist_to(u)
    
    def __str__(self) -> str:
        return ' -> '.join(map(str, self.__path))
