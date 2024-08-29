import networkx as nx
import numpy as np

from tqdm import tqdm
from itertools import permutations

from dynflows.flows.dynamic import max_flow_over_time, dynamic_transshipment, quickest_transshipment
from dynflows.flows.dynamic.transshipments import MaxOutFlow


G = nx.DiGraph()

# Add all vertices from the graph using strings as names for improved readability.
G.add_nodes_from([
    ('HANSA', {'balance': 500}), 
    ('WEST', {'balance': 700}), 
    'BRIDGE_1_START', 
    ('CENTRAL', {'balance': 2000}), 
    'BRIDGE_2_START', 
    ('SOUTH', {'balance': 800}), 
    ('SEVERIN', {'balance': 200}), 
    'BRIDGE_3_START', 
    'BRIDGE_3_END', 
    'BRIDGE_2_END', 
    'BRIDGE_1_END', 
    ('STAGE 1', {'balance': -2000}), 
    ('STAGE 2', {'balance': -1000}),
    ('STAGE 3', {'balance': -2000}),
    ('DEUTZ', {'balance': 800})
])

# Add arcs with capacity of 100 and transit time of 1
G.add_edges_from([
    ('WEST', 'HANSA', {'transit': 6}), 
    ('HANSA', 'CENTRAL'), 
    ('WEST', 'CENTRAL'), 
    ('CENTRAL', 'BRIDGE_1_START'), 
    ('WEST', 'SOUTH'), 
    ('SOUTH', 'CENTRAL'), 
    ('SOUTH', 'SEVERIN'),
    ('SEVERIN', 'BRIDGE_3_START'), 
    ('BRIDGE_2_START', 'BRIDGE_3_START'),
    ('BRIDGE_3_START', 'BRIDGE_2_START'), 
    ('BRIDGE_1_START', 'BRIDGE_2_START'),
    ('BRIDGE_2_START', 'BRIDGE_1_START'), 
    ('BRIDGE_1_START', 'BRIDGE_1_END'), 
    ('BRIDGE_2_START', 'BRIDGE_2_END'), 
    ('BRIDGE_3_START', 'BRIDGE_3_END'), 
    ('BRIDGE_1_END', 'BRIDGE_2_END'),
    ('BRIDGE_2_END', 'BRIDGE_1_END'),
    ('BRIDGE_2_END', 'BRIDGE_3_END'),
    ('BRIDGE_3_END', 'BRIDGE_2_END'),
    ('BRIDGE_1_END', 'STAGE 1'), 
    ('BRIDGE_2_END', 'STAGE 3'), 
    ('BRIDGE_3_END', 'STAGE 2'), 
    ('SEVERIN', 'BRIDGE_2_START'), 
    ('DEUTZ', 'BRIDGE_2_END')
], capacity=100, transit=1)

terminals = ['HANSA', 'WEST', 'CENTRAL', 'SOUTH', 'SEVERIN', 'STAGE 1', 'STAGE 2', 'STAGE 3', 'DEUTZ']
sources = ['HANSA', 'WEST', 'CENTRAL', 'SOUTH', 'SEVERIN', 'DEUTZ']
sinks = ['STAGE 1', 'STAGE 2', 'STAGE 3']

T = 100

print(len(G.nodes))

value = max_flow_over_time(G, sources, sinks, T)
print(value)

outflow = MaxOutFlow(G)
print(outflow(sources, T))

flow = dynamic_transshipment(G, T, method='orlin')
print(flow.get_net_value(terminals, T))