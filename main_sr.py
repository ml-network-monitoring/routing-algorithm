import util
import numpy as np
from segment_routing import SegmentRoutingSolver

def generate_traffic_matrix_v1():
    tm = np.random.randint(low=1, high=100, size=[12, 12])
    tm = tm - tm * np.eye(12)
    return tm

def generate_traffic_matrix_v2():
    tm = np.random.randint(low=100, high=200, size=[12, 12])
    tm = tm - tm * np.eye(12)
    return tm

def main():
    import numpy as np
    from copy import deepcopy

    # load data
    t = np.random.randint(0, 40000)
#    tm_pred = util.load_traffic_matrix(t)
#    tm      = util.load_traffic_matrix(t+12)
#    tm_pred = util.generate_traffic_matrix()
#    tm      = util.generate_traffic_matrix()
    while 1:
        tm_pred = generate_traffic_matrix_v1()
        tm      = generate_traffic_matrix_v2()

        # load network topology
        G = util.load_network_topology()

        # solve by prediction
        solver = SegmentRoutingSolver(G)
        solver.solve(tm_pred)
        u_pred = util.get_max_utilization_v2(solver, tm)
        solver_pred = deepcopy(solver)

        # solve by ground truth
        solver = SegmentRoutingSolver(G)
        solver.solve(tm)
        u = util.get_max_utilization_v2(solver, tm)

        print('u_pred={} u={}'.format(u_pred, u))
        # debug
        import itertools
        import numpy as np
        import matplotlib.pyplot as plt

        for i, j in itertools.product(range(12), range(12)):
            k_pred = np.argmax(solver_pred.solution[i, j])
            k = np.argmax(solver.solution[i, j])
            if k != k_pred:
                print('i={} j={} k_pred={} k={}'.format(i, j, k_pred, k))

                if i == j: k = i
                else: k = np.argmax(solver_pred.solution[i, j])
                util.draw_segment_pred(G, solver_pred.segments[i][j][k], i, j, k)

                if i == j: k = i
                else: k = np.argmax(solver.solution[i, j])
                util.draw_segment_ground_truth(G, solver.segments[i][j][k], i, j, k)
                plt.show()

if __name__ == '__main__':
    main()
