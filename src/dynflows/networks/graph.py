import random
import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def construct_neighborhood_graph(nodes, node_dists):
    max_dist = 0.10

    while True:
        G = nx.Graph()
        G.add_nodes_from(nodes)

        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if np.linalg.norm(node_dists[i] - node_dists[j]) < max_dist:
                    G.add_edge(i, j) 
        
        if nx.connected.is_connected(G):
            break

        max_dist += 0.10

    return G


def random_flow_network(n: int, n_terminals: int, balance: int, dim=2, max_dist=0.1, transit_fac=100):
    node_dists = [np.random.random(size=dim) for _ in range(n)]
    nodes = list(range(n))

    n_graph = construct_neighborhood_graph(nodes, node_dists)

    terminals = random.sample(nodes, n_terminals)
    random.shuffle(terminals)

    term_split = random.randint(1, len(terminals)-1)
    sources = set(terminals[0:term_split])
    sinks = set(terminals[term_split:])

    for node in terminals:
        if random.random() < 0.5:
            sources.add(node)
        else:
            sinks.add(node)

    assert len(sources) != 0 and len(sinks) != 0

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    visited = set()
    source_list = list(sources)
    sink_list = list(sinks)
    connected = dict()

    for node in nodes:
        u = node

        while u not in sources:
            v = random.choice(list(n_graph.neighbors(u)))
            visited.add(v)

            if G.has_edge(u, v):
                G[u][v]['capacity'] += 1
            else:
                transit = int(np.floor(np.linalg.norm(node_dists[u] - node_dists[v]) * transit_fac))

                G.add_edge(v, u, **{'capacity': 1, 'transit': transit})

            u = v
                
        s = u
        u = node

        while u not in sinks:
            v = random.choice(list(n_graph.neighbors(u)))
            visited.add(v)

            if G.has_edge(u, v):
                G[u][v]['capacity'] += 1
            else:
                transit = int(np.floor(np.linalg.norm(node_dists[u] - node_dists[v]) * transit_fac))

                G.add_edge(u, v, **{'capacity': 1, 'transit': transit})

            u = v

        t = u

        if s not in connected:
            connected[s] = set()

        connected[s].add(t)

    for i in range(balance):
        s = random.choice(source_list)

        while len(connected[s]) == 0:
            s = random.choice(source_list)

        if 'balance' in G.nodes[s]:
            G.nodes[s]['balance'] += 1
        else:
            G.nodes[s]['balance'] = 1

        t = random.choice(list(connected[s]))

        if 'balance' in G.nodes[t]:
            G.nodes[t]['balance'] -= 1
        else:
            G.nodes[t]['balance'] = -1        

    assert sum(G.nodes[n].get('balance', 0) for n in G.nodes()) == 0, 'Total balance has to sum to zero.'

    return G


if __name__ == '__main__':
    G = random_flow_network(100, 10, 200, max_dist=0.2)

    pos = nx.spring_layout(G, weight='transit')
    nx.draw(G, pos)
    plt.show()

    print(G.nodes(data=True))

