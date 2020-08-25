from copy import deepcopy
import networkx as nx
import numpy as np
import pulp as pl
import itertools
import warnings
import util

class SegmentRoutingSolver:

    def __init__(self, G):
        '''
        G: networkx Digraph, a network topology
        '''
        self.G = G
        self.segments = util.get_segments(G)
        self.tm = None

    def create_problem(self, tm):
        # save input traffic matrix
        self.tm = tm

        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        segments = self.segments

        # 1) create optimization model
        problem = pl.LpProblem('SegmentRouting', pl.LpMinimize)
        theta = pl.LpVariable(name='theta', lowBound=0, upBound=1, cat='Continuous')
        x = pl.LpVariable.dicts(name='x',
                                indexs=np.arange(num_node ** 3),
                                lowBound=0,
                                cat='Continuous')

        # 2) objective function
        # minimize maximum link utilization
        problem += theta

        # 3) constraint function
        for u, v in G.edges:
            capacity = G.get_edge_data(u, v)['capacity']
            utilization = pl.lpSum(x[util.flatten_index(i,j,k,num_node)] * util.g(segments[i][j][k], u, v) / capacity for i, j, k in itertools.product(range(num_node), range(num_node), range(num_node)))
            problem += utilization <= theta

        # 3) constraint function
        # ensure all traffic are routed
        for i, j in itertools.product(range(num_node), range(num_node)):
            problem += pl.lpSum(x[util.flatten_index(i, j, k, num_node)] for k in range(num_node)) >= tm[i, j]

        return problem, x

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
            self.solution[i, j, k] = d['x_{}'.format(index)]

    def extract_utilization(self, tm):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        segments = self.segments
        # extract utilization
        utilizations = []
        for u, v in G.edges:
            load = sum([self.solution[i,j,k] * util.g(segments[i][j][k], u, v) for i,j,k in itertools.product(range(num_node), range(num_node), range(num_node))])
            capacity = G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            G[u][v]['utilization'] = utilization

    def extract_utilization_v2(self, tm):
        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        segments = self.segments
        # recompute the solution, proportional to new demand
        solution = deepcopy(self.solution)

        for i, j in itertools.product(range(N), range(N)):
            if self.tm[i, j] != 0 and tm[i, j] != 0:
                solution[i, j, :] = solution[i, j, :] / self.tm[i, j] * tm[i, j]
            elif self.tm[i, j] == 0 and tm[i, j] != 0:
                # assume route this situation by shortest path
                solution[i, j, i] = tm[i, j]
                warnings.warn('tm[{},{}]={} while new tm\'[{},{}]={}, create a shotest path flow from {} to {} to accomodate the demand'.format(i, j, self.tm[i, j], i, j, tm[i, j], i, j),
                              RuntimeWarning)
            else:
                solution[i, j, :] = 0
        # extract utilization
        utilizations = []
        for u, v in G.edges:
            load = sum([solution[i, j, k] * util.g(segments[i][j][k], u, v) for i,j,k in itertools.product(range(N), range(N), range(N))])
            capacity = G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            G[u][v]['utilization'] = utilization

    def extract_status(self, problem):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        # extract status
        self.status = pl.LpStatus[problem.status]

    def solve(self, tm):
        '''
        t: numpy.ndarray, traffic matrix at a given timestep
        '''
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        problem, x = self.create_problem(tm)
#        solver = pl.get_solver('GUROBI')
#        print(solver)
#        problem.solve(solver)
        problem.solve()
        self.extract_status(problem)
        self.extract_solution(problem)
        self.extract_utilization(tm)

    def get_paths(self, i, j):
        G = self.G
        if i == j: list_k = [i]
        else: list_k = np.where(self.solution[i, j] > 0)[0]
        paths = []
        for k in list_k:
            path = []
            path += util.shortest_path(G, i, k)[:-1]
            path += util.shortest_path(G, k, j)
            paths.append((k, path))
        return paths
