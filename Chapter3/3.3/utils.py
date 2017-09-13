import numpy as np
def importDataset(path):
    '''return the dataset'''
    ans = []
    dataset_file = file(path, 'r')
    raw_data = dataset_file.readlines()
    dataset = []

    for line in raw_data:
        single_data = line.split()
        for i in range(len(single_data)):
            try:
                single_data[i] = int(single_data[i])
            except:
                single_data[i] = float(single_data[i])
        dataset.append(single_data)

    return split_data(dataset)

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