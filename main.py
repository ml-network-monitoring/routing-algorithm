import util
from segment_routing import SegmentRoutingSolver
from oblivious_routing import ObliviousRoutingSolver
from shortest_path_routing import ShortestPathRoutingSolver

def main():
    # load data
    import numpy as np
    t = np.random.randint(0, 40000)
    tm_pred = util.load_traffic_matrix(t)
    tm      = util.load_traffic_matrix(t+12)

    # load network topology
    G = util.load_network_topology()

    # solver
    solver = ObliviousRoutingSolver(G)
    solver.solve()
    u_ob = util.get_max_utilization_v2(solver, tm)

    # solver
    solver = ShortestPathRoutingSolver(G)
    solver.solve(tm)
    u_sp = util.get_max_utilization_v2(solver, tm)

    # solve by prediction
    solver = SegmentRoutingSolver(G)
    solver.solve(tm_pred)
    u_srp = util.get_max_utilization_v2(solver, tm)

    # solve by ground truth
    solver = SegmentRoutingSolver(G)
    solver.solve(tm)
    u_sr = util.get_max_utilization_v2(solver, tm)

    # result
    print('Shortest Path Routing:', u_sp)
    print('Oblivious Routing:', u_ob)
    print('Segment Routing on Prediction TM:', u_srp)
    print('Segment Routing on Ground Truth TM:', u_sr)

if __name__ == '__main__':
    main()
