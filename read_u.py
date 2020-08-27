from scipy.io import loadmat

for dataset in ['abilene', 'brain']:
    mat = loadmat('../../data/data/{}_tm_mlu.mat'.format(dataset))
    print(dataset)
    print(mat['X'].shape)
    print(mat['U'].shape)
    print(mat['U'][0])
