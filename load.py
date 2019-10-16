import numpy as np


def load_data(test=False):
    test_data = 'adult/adult_test.npz'
    train_data = 'adult/adult_train.npz'
    if test:
        data = np.load(test_data)

    else:
        data = np.load(train_data)

    data = {'x': data['x'], 'y': data['y'], 'a':data['a']}
    data['y'] = data['y'].reshape(-1)
    data['a'] = data['a'].reshape(-1)
    return data

