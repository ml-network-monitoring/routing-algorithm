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

    # get degree
    import matplotlib.pyplot as plt

    print('Get degree')
    for i in range(12):
        degree = util.get_degree(G, i)
        print('node={} degree={}'.format(i, degree))

    # get all degrees
    print('Get degree sorted')
    nodes, degrees = util.get_nodes_sort_by_degree(G)
    for node, degree in zip(nodes, degrees):
        print(node, degree)
    util.draw_network_topology(G)
    plt.show()



if __name__ == '__main__':
    main()
