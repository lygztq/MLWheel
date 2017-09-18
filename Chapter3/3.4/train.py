import matplotlib.pyplot as plt
import numpy as np
import utils
import logistic_regression

DATASET_PATH = "UCI_dataset/wine_quality/winequality-red.csv"
REPORT_EVERY = 1000
LEARNING_LOOP = 10000
LEARNING_DECAY = True
LEARNING_RATE = 0.0001
LEARNING_DECAY_RATE = 0.9

def saveModel(path):
    pass


def hold_out_train(loop_number,learning_decay = False, learning_rate = 0.01, learning_decay_rate = 0.9):
    samples, labels = utils.importDataset(DATASET_PATH)
    train_samples = samples[:1000,:]
    train_labels = labels[:1000]
    test_samples = samples[1000:,:]
    test_labels = labels[1000:]
    model = logistic_regression.LogisticRegression(train_samples, train_labels, learning_rate)

    for _ in range(loop_number):
        if not _%REPORT_EVERY:
            print "loop %d\n"%(_)
            print "weights:",model.weights,"\n"
            print "bias:",model.bias,"\n"
            #print "learning_rate:",model.learning_rate,'\n'
        model.adjust()
        if(learning_decay):
            model.learning_rate *= learning_decay_rate

    err_count = 0
    for i in range(test_samples.shape[0]):
        y_hat = model.getclass(test_samples[i])
        if y_hat!=test_labels[i]:
            err_count+=1

    print "error rate:",err_count*1.0/test_labels.shape[0]



# def train(loop_number,learning_decay = False, learning_rate = 0.01, learning_decay_rate = 0.9):
#     samples, labels = utils.importDataset(DATASET_PATH)
#     model = logistic_regression.LogisticRegression(samples, labels, learning_rate)
#
#     for _ in range(loop_number):
#         if not _%REPORT_EVERY:
#             print "loop %d\n"%(_)
#             print "weights:",model.weights,"\n"
#             print "bias:",model.bias,"\n"
#             print "learning_rate:",model.learning_rate,'\n'
#         model.adjust()
#         if(learning_decay):
#             model.learning_rate *= learning_decay_rate



if __name__ == "__main__":
    print "test"
    hold_out_train(LEARNING_LOOP,LEARNING_DECAY,LEARNING_RATE,LEARNING_DECAY_RATE)