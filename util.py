import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy.io import loadmat

def load_traffic_matrix(timestep=0):
    tm = loadmat('data/abilene_tm.mat')['X'][timestep, :].reshape(12, 12)
    return tm

def generate_traffic_matrix():
    tm = np.random.randint(low=0, high=100, size=[12, 12])
    tm = tm - tm * np.eye(12)
    return tm

def load_network_topology():
    # initialize graph
    G = nx.Graph()
    # add nodes
    for i in range(12):
        G.add_node(i)
    # load data from csv
    df       = pd.read_csv('data/abilene_edge.csv', delimiter=' ')
    # add weight, capacity, delay to edge attributes
    for _, row in df.iterrows():
        i = row.src
        j = row.dest
        G.add_edge(i, j, weight=row.weight,
                         capacity=row.bw,
                         delay=row.delay)
    return G

def draw_network_topology(G, position=None):
    if position == None:
        position = nx.spring_layout(G)
    nx.draw(G, position, node_size=1000, alpha=0.5)
    nx.draw_networkx_labels(G, position)
    return position

def shortest_path(G, source, target):
    return nx.shortest_path(G, source=source, target=target, weight='weight')

class Segment:

    def __init__(self, G, i, j, k):
        self.segment_ik = nx.Graph()
        self.segment_kj = nx.Graph()
        self.segment_ik.add_nodes_from(G)
        self.segment_kj.add_nodes_from(G)
        nx.add_path(self.segment_ik, shortest_path(G, i, k))
        nx.add_path(self.segment_kj, shortest_path(G, k, j))

def get_segments(G):
    n = G.number_of_nodes()
    segments = np.empty([n, n, n]).tolist()
    for i, j, k in itertools.product(range(n), range(n), range(n)):
        segments[i][j][k] = Segment(G, i, j, k)
    return segments

def draw_segment(G, segment, i, j, k):
    plt.subplot(131)
    position = draw_network_topology(G)
    plt.title('Network topology')
    plt.subplot(132)
    draw_network_topology(segment.segment_ik, position=position)
    plt.title('Segment path i={} k={}'.format(i, k))
    plt.subplot(133)
    draw_network_topology(segment.segment_kj, position=position)
    plt.title('Segment path k={} j={}'.format(k, j))

def draw_segment_pred(G, segment, i, j, k):
    plt.subplot(231)
    position = draw_network_topology(G)
    plt.title('Network topology')
    plt.subplot(232)
    draw_network_topology(segment.segment_ik, position=position)
    plt.title('Segment path i={} k={}'.format(i, k))
    plt.subplot(233)
    draw_network_topology(segment.segment_kj, position=position)
    plt.title('Segment path k={} j={}'.format(k, j))

def draw_segment_ground_truth(G, segment, i, j, k):
    plt.subplot(234)
    position = draw_network_topology(G)
    plt.title('Network topology')
    plt.subplot(235)
    draw_network_topology(segment.segment_ik, position=position)
    plt.title('Segment path i={} k={}'.format(i, k))
    plt.subplot(236)
    draw_network_topology(segment.segment_kj, position=position)
    plt.title('Segment path k={} j={}'.format(k, j))

def g(segment, u, v):
    value = 0
    if segment.segment_ik.has_edge(u, v):
        value += 1
    if segment.segment_kj.has_edge(u, v):
        value += 1
#    if value == 2:
#        value = 100
    return value

def flatten_index(i, j, k, num_node):
    return i * num_node ** 2 + j * num_node + k

def get_max_utilization(solver, tm):
    '''
    Calculate the utilization with traffic matrix "tm"
    which has been used to solve the routing problem
    '''
    solver.extract_utilization(tm)
    return np.max([solver.G[u][v]['utilization'] for u, v in solver.G.edges])

def get_max_utilization_v2(solver, tm):
    '''
    Use the exiting solution on solver.
    Calculate the utilization with a new traffic matrix "tm"
    '''
    solver.extract_utilization_v2(tm)
    return np.max([solver.G[u][v]['utilization'] for u, v in solver.G.edges])

def get_degree(G, i):
    return len([_ for _ in nx.neighbors(G, i)])

def get_nodes_sort_by_degree(G):
    nodes = np.array(G.nodes)
    degrees = np.array([get_degree(G, i) for i in nodes])
    idx = np.argsort(degrees)[::-1]
    nodes = nodes[idx]
    degrees = degrees[idx]
    return nodes, degrees

def get_node2flows(solver):
    # extract parameters
    n = solver.G.number_of_nodes()
    # initialize
    node2flows = {}
    for i in solver.G.nodes:
        node2flows[i] = []
    # enumerate all flows
    # for i, j in itertools.combinations(range(n), 2):
    for i, j in itertools.product(range(n), range(n)):
        for k, path in solver.get_paths(i, j):
            for node in solver.G.nodes:
                if node in path:
                    node2flows[node].append((i, j))
    return node2flows
