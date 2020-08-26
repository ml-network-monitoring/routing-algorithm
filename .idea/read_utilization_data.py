from scipy.io import loadmat

mat = loadmat('data/abilene_tm_utilization.mat')

print(mat['X'].shape)
print(mat['U'].shape)
print(mat['U'][0])
