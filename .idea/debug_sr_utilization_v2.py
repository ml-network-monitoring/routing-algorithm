import util
from segment_routing import SegmentRoutingSolver

def main():
    # load data
    tm = util.load_traffic_matrix(0)

    # load network topology
    G = util.load_network_topology()

    # solver
    solver = SegmentRoutingSolver(G)
    solver.solve(tm)

    max_utilization = util.get_max_utilization(solver, tm)
    print('max_utilization for tm', max_utilization)

    tm_ = util.load_traffic_matrix(100)
    max_utilization = util.get_max_utilization_v2(solver, tm_)
    print('max_utilization for tm_', max_utilization)

    solver.solve(tm)
    max_utilization = util.get_max_utilization(solver, tm_)
    print('max_utilization for tm_ by re optimization', max_utilization)

if __name__ == '__main__':
    main()
