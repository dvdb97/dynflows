# dynflows
A Python implementation of various algorithms for network flows over time. This includes
- the Ford-Fulkerson algorithm for maximum flows over time,
- the algorithm for lex-max flows over time by Hoppe and Tardos,
- Orlin's algorithm for submodular function minimization,
- feasibility determination by minimizing the submodular function $`v^T`$,
- computing a minimum feasible time horizon using a binary search,
- the algorithm for integral dynamic transshipments by Hoppe and Tardos.

# Install
1. Clone this repository
2. Open a command line in the repository
3. Install via `pip install -r requirements.txt .`

# Usage

### Construct a network using NetworkX:
``` Python
import networkx as nx

G = nx.DiGraph()

# Add a source 0 with supply of 20 and a sink 1 with demand of -20
G.add_nodes_from([
    (0, {'balance': 20}),
    (1, {'balance': -20})
])

# Connect 0 and 1 using an arc with capacity of 4 and transit time of 6
G.add_edges_from([
    (0,  1, {'capacity': 4, 'transit': 6}), 
])

```

The attibute names "balance", "capacity" and "transit" are default names in every implementation. If you decide to name the attributes differently, you have to specify this when calling one of the algorithms in this library. For example:
``` Python
quickest_transshipment(G, balance='b', capacity='c', transit='t')
```

### Compute a maximum flow over time:
``` Python
from dynflows.flows.dynamic import max_flow_over_time

# ...
# ... construct network ...
# ...

# Compute the maximum flow over time in the given network. Returns both the value and the flow.
value, flow = max_flow_over_time(G, sources, sinks, T)

# How much does the flow send along an arc (0, 1) at time $`\theta=5`$?
flow.get_flow_value(0, 1, 5)

# What is the excess of vertex 5 at time $`\theta=8`$?
flow.get_excess(5, 8)

# If you are not interested in the actual flow, you can only compute the value which is faster.
value = max_flow_over_time(G, sources, sinks, T, return_flow=False)
```

### Decide feasibility of a dynamic transshipment instance:
``` Python
from dynflows.flows.dynamic import is_feasible

# ...
# ... construct network ...
# ...

# Check for the time horizon $`T=50`$
T = 50

# Check feasibility using Orlin's algorithm
rslt = is_feasible(G, T)

# Check feasibility using the brute force approach which tends to be faster for smaller instances.
rslt = is_feasible(G, T, method='naive')

# Sometimes we are also interested in the violated set that makes the dynamic transshipment instance infeasible, or the minimizer if feasible.
rslt, violated = is_feasible(G, T, return_violated=True)

# By setting lazy=False, the algorithm returns the global minimizer.
rslt, violated = is_feasible(G, T, return_violated=True, lazy=False)
```

### Compute an integral dynamic transshipment:
``` Python

# ...
# ... construct network ...
# ...

# Use the time horizon $`T = 50`$
T = 50

# Compute an integral dynamic transshipment using Orlin's algorithm for feasibility checks.
flow = dynamic_transshipment(G, T)

# Use the naive (brute-force) feasibility check instead:
flow = dynamic_transshipment(G, T, method='naive')

# How much does the flow send along an arc (0, 1) at time $`\theta=5`$?
flow.get_flow_value(0, 1, 5)

# What is the excess of vertex 5 at time $`\theta=8`$?
flow.get_excess(5, 8)
```

### Compute a quickest integral dyamic transshipment:
``` Python

# ...
# ... construct network ...
# ...

# Compute the quickest transshipment and the corresponding time horizon using Orlin's algorithm and a binary search.
flow, T = quickest_transshipment(G)

# Use the naive (brute-force) feasibility check instead:
flow, T = quickest_transshipment(G, sfm_method='naive)

# How much does the flow send along an arc (10, 1) at time $`\theta=15`$?
flow.get_flow_value(10, 1, 15)

# What is the excess of vertex 3 at time $`\theta=20`$?
flow.get_excess(3, 20)
```
