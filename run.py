import os
import joblib
import numpy as np
from joblib import delayed, Parallel
from scipy.io import loadmat, savemat
from routing import SegmentRoutingSolver, util

def main():
    # parameter
    for dataset in ['abilene', 'brain']:
        # load data
        X = util.load_all_traffic_matrix(dataset)
        G = util.load_network_topology(dataset)

        # limit data
        # X = X[:100, :, :]

        def f(x):
            solver = SegmentRoutingSolver(G)
            solver.solve(x)
            u = util.get_max_utilization(solver, x)
            return u

        U = Parallel(n_jobs=os.cpu_count())(delayed(f)(x) for x in X)

        # save data
        mdict = {'X': X, 'U': U}
        savemat('../../data/data/{}_tm_mlu.mat'.format(dataset), mdict=mdict)

if __name__ == '__main__':
    main()
