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

Compute a maximum flow over time:
``` Python
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
