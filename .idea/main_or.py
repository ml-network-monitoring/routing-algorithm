import util
from oblivious_routing import ObliviousRoutingSolver

def main():
    # load data
    tm = util.load_traffic_matrix(0)

    # load network topology
    G = util.load_network_topology()

    # solver
    solver = ObliviousRoutingSolver(G)
    solver.solve()

    for t in range(100):
        tm = util.load_traffic_matrix(t)
        print('t={} u={}'.format(t, util.get_max_utilization_v2(solver, tm)))


if __name__ == '__main__':
    main()
