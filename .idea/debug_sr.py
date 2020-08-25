# debug
import itertools
import numpy as np
import matplotlib.pyplot as plt

print('traffic_matrix=', t)
print('status=', solver.status)
for i, j in itertools.product(range(12), range(12)):
    if i == j: k = i
    else: k = np.argmax(solver.solution[i, j])
    print('source={} sink={} middle={} demand={}', i, j, k, t[i, j])
    print('routing=', solver.solution[i, j])
    # visualization
    util.draw_segment(G, solver.segments[i][j][k], i, j, k)
    plt.show()
