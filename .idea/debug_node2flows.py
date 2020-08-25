import util
from segment_routing import SegmentRoutingSolver

def main():
    # load data
    tm = util.load_traffic_matrix(0)
    # tm = util.generate_traffic_matrix()

    # load network topology
    G = util.load_network_topology()

    # solver
    solver = SegmentRoutingSolver(G)
    solver.solve(tm)

    # get flows on each nodes
    from pprint import pprint
    node2flows = util.get_node2flows(solver)
    pprint(node2flows)
    for node in node2flows:
        print('node={} number of flow={}'.format(node, len(node2flows[node])))

if __name__ == '__main__':
    main()
