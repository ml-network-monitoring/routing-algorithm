import util
from shortest_path_routing import ShortestPathRoutingSolver

def main():
    # load data
    tm = util.load_traffic_matrix(0)

    # load network topology
    G = util.load_network_topology()

    # solver
    solver = ShortestPathRoutingSolver(G)
    solver.solve(tm)

    # debug
    print('max utilization', util.get_max_utilization_v2(solver, tm))

if __name__ == '__main__':
    main()
