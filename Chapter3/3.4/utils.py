import numpy as np


def importDataset(path):
    '''return the dataset'''
    ans = []
    dataset = np.loadtxt(path, dtype=np.str, delimiter=';')

    samples = 10*(dataset[1:, :-1].astype(np.float))
    labels = dataset[1:, -1].astype(np.int)

    for i in range(labels.shape[0]):
        if labels[i] > 5:
            labels[i] = 1
        else:
            labels[i] = 0

    return samples, labels


def split_data(dataset):
    labels = []
    samples = []

    for data in dataset:
        s = data[1:3]
        samples.append(s)
        labels.append(data[3])

    labels = np.array(labels)
    samples = np.array(samples)

    return samples, labels