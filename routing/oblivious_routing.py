import itertools
import pulp as pl
import numpy as np
from . import util
from copy import deepcopy

class ObliviousRoutingSolver:

    def __init__(self, G):
        '''
        G: networkx Digraph, a network topology
        '''
        self.G = G
        self.segments = util.get_segments(G)

    def flatten_index(self, i, j, num_edge):
        return i * num_edge + j

    def create_problem(self):
        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        E = G.number_of_edges()
        segments = self.segments

        # 0) initialize lookup dictionary from index i to edge u, v
        edges_dictionary = {}
        for i, (u, v) in enumerate(G.edges):
            edges_dictionary[i] = (u, v)

        # 1) create optimization model of dual problem
        problem = pl.LpProblem('SegmentRouting', pl.LpMinimize)
        theta = pl.LpVariable(name='theta', lowBound=0, upBound=1, cat='Continuous')
        alpha = pl.LpVariable.dicts(name='alpha', lowBound=0, upBound=1, indexs=np.arange(N ** 3))
        pi = pl.LpVariable.dicts(name='pi', indexs=np.arange(E ** 2), lowBound=0)

        # 2) objective function
        # minimize maximum link utilization
        problem += theta

        # 3) constraint function 2
        for i, j in itertools.product(range(N), range(N)):
            for e_prime in edges_dictionary:
                u, v = edges_dictionary[e_prime]
                lb = pl.lpSum([util.g(segments[i][j][k], u, v) * alpha[util.flatten_index(i, j, k, N)] for k in range(N)])
                for m in range(N):
                    loads = []
                    for e in edges_dictionary:
                        u, v = edges_dictionary[e]
                        load = util.g(segments[i][j][m], u, v) * pi[self.flatten_index(e, e_prime, E)]
                        loads.append(load)
                    problem += pl.lpSum(loads) >= lb

        # 4) constraint function 3
        for e_prime in edges_dictionary: # for edge e'
            utilizations = []
            u, v = edges_dictionary[e_prime]
            capacity_e_prime = G.get_edge_data(u, v)['capacity']
            for e in edges_dictionary: # for edge e
                u, v = edges_dictionary[e]
                capacity_e = G.get_edge_data(u, v)['capacity']
                utilization = capacity_e * pi[self.flatten_index(e, e_prime, E)] / capacity_e_prime
                utilizations.append(utilization)
            problem += pl.lpSum(utilizations) <= theta

        # 3) constraint function 4
        for i, j in itertools.product(range(N), range(N)):
            problem += pl.lpSum(alpha[util.flatten_index(i, j, k, N)] for k in range(N)) == 1

        return problem

    def solve(self):
        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        problem = self.create_problem()
        problem.solve()
        self.extract_status(problem)
        self.extract_solution(problem)

    def extract_solution(self, problem):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        # extract solution
        d = {}
        for v in problem.variables():
            d[v.name] = v.varValue
        self.solution = np.empty([num_node, num_node, num_node])
        for i, j, k in itertools.product(range(num_node), range(num_node), range(num_node)):
            index = util.flatten_index(i, j, k, num_node)
            self.solution[i, j, k] = d['alpha_{}'.format(index)]

    def extract_status(self, problem):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        # extract status
        self.status = pl.LpStatus[problem.status]

    def extract_utilization_v2(self, tm):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        segments = self.segments
        # recompute the solution, proportional to new demand
        solution = deepcopy(self.solution)
        for i, j in itertools.product(range(num_node), range(num_node)):
            solution[i,j,:] = solution[i,j,:] * tm[i,j]

        # extract utilization
        utilizations = []
        for u, v in G.edges:
            load = sum([solution[i, j, k] * util.g(segments[i][j][k], u, v) for i,j,k in itertools.product(range(num_node), range(num_node), range(num_node))])
            capacity = G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            G[u][v]['utilization'] = utilization
