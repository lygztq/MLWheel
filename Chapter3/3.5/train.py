import matplotlib.pyplot as plt
import numpy as np
import utils
import LDA

DATASET_PATH = "watermelon3a.txt"
REPORT_EVERY = 10
LEARNING_LOOP = 50000
LEARNING_DECAY = True
LEARNING_RATE = 0.1
LEARNING_DECAY_RATE = 0.9

def saveModel(path):
    pass


def train():
    samples, labels = utils.importDataset(DATASET_PATH)
    model = LDA.LDA(samples,labels)

    f1 = plt.figure(1)
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.scatter(samples[:8,0],samples[:8,1],marker='o',s=100,color='g',label='good')
    plt.scatter(samples[8:,0],samples[8:,1],marker='o',s=100,color='k',label='bad')
    plt.legend(loc='upper right')
    #plt.show()


    indexx = np.arange(0.2,0.9,0.001)
    indexy = np.arange(0,0.5,0.001)
    values = np.zeros([indexy.shape[0],indexx.shape[0]])
    for x in range(values.shape[1]):
        for y in range(values.shape[0]):
            values[y,x] = model.getClass(np.array([indexx[x],indexy[y]]))
    plt.contour(indexx,indexy,values)
    plt.show()
    print indexx.shape,indexy.shape,values.shape

    # for i in range(labels.shape[0]):
    #     print samples[i,:],model.getClass(samples[i]),labels[i]
    # print model.getClass(np.array([0.719,0.103]))



if __name__ == "__main__":
    print "test"
    train()