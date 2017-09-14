import matplotlib.pyplot as plt
import numpy as np
import utils
import logistic_regression

DATASET_PATH = "W_dataset/watermelon_3alpha.txt"
REPORT_EVERY = 10
LEARNING_LOOP = 50000
LEARNING_DECAY = True
LEARNING_RATE = 0.01
LEARNING_DECAY_RATE = 0.9

def saveModel(path):
    pass


def train(loop_number,learning_decay = False, learning_rate = 0.01, learning_decay_rate = 0.9):
    samples, labels = utils.importDataset(DATASET_PATH)
    model = logistic_regression.LogisticRegression(samples, labels, learning_rate)

    f1 = plt.figure(1)
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.scatter(samples[:8,0],samples[:8,1],marker='o',s=100,color='g',label='good')
    plt.scatter(samples[8:,0],samples[8:,1],marker='o',s=100,color='k',label='bad')
    plt.legend(loc='upper right')
    #plt.show()

    for _ in range(loop_number):
        if not _%REPORT_EVERY:
            print "loop %d\n"%(_)
            print "weights:",model.weights,"\n"
            print "bias:",model.bias,"\n"
        model.adjust()
        if(learning_decay):
            model.learning_rate *= learning_decay_rate

    indexx = np.arange(0.2,0.9,0.01)
    indexy = np.arange(0,0.5,0.01)
    values = np.zeros([indexy.shape[0],indexx.shape[0]])
    for x in range(values.shape[1]):
        for y in range(values.shape[0]):
            values[y,x] = model.getclass([indexx[x],indexy[y]])
    plt.contour(indexx,indexy,values)
    plt.show()
    # print indexx.shape,indexy.shape,values.shape



if __name__ == "__main__":
    print "test"
    train(LEARNING_LOOP,LEARNING_DECAY,LEARNING_RATE,LEARNING_DECAY_RATE)