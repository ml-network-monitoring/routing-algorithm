import os
import util
import joblib
import numpy as np
from joblib import delayed, Parallel
from scipy.io import loadmat, savemat
from segment_routing import SegmentRoutingSolver

def main():
    # load data
    X = loadmat('data/abilene_tm.mat')['X'].reshape(-1, 12, 12)
    G = util.load_network_topology()

    # limit data for fast dev
    # X = X[:100, :, :]

    def f(x):
        solver = SegmentRoutingSolver(G)
        solver.solve(x)
        u = util.get_max_utilization(solver, x)
        return u

    U = Parallel(n_jobs=os.cpu_count())(delayed(f)(x) for x in X)

    # save data
    mdict = {'X': X, 'U': U}
    savemat('data/abilene_tm_utilization.mat', mdict=mdict)

if __name__ == '__main__':
    main()
