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

    # get paths
    paths = solver.get_paths(2, 8)
    for k, path in paths:
        print(k, path)


if __name__ == '__main__':
    main()
