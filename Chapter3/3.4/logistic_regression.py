import numpy as np
from tools import *


class LogisticRegression:
    def __init__(self, samples, labels, learning_rate=0.01):
        x_dim = samples.shape[1]
        self._sample_number = samples.shape[0]
        self._w = np.random.random(size=x_dim)
        self._b = np.random.random(size=1)
        self._learning_rate = learning_rate
        self._samples = samples
        self._labels = labels

    @property
    def weights(self):
        return self._w

    @weights.setter
    def weights(self, value):
        if(value.shape[0]!=self._w.shape[0]):
            print "wrong dimension for the value feed to weight."
            return
        self._w = value

    @property
    def bias(self):
        return self._b
    @bias.setter
    def bias(self, value):
        self._b = value

    @property
    def learning_rate(self):
        return self._learning_rate
    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    def gradient(self):
        size = self._samples.shape[1] + 1
        ans = np.zeros([size])

        for i in range(self._sample_number):
            p1 = sigmoid(get_z(self._w, self._samples[i], self._b))
            augmented_x = np.append(self._samples[i], 1)
            ans += augmented_x * (self._labels[i] - p1)
        ans = -1 * ans

        return ans

    def adjust(self):
        grad = self.gradient()
        augment = np.append(self._w, self._b)
        augment -= self._learning_rate*grad

        self._w = augment[:-1]
        self._b = augment[-1]

    def regression(self,x):
        z = get_z(self._w,x,self._b)
        return sigmoid(z)

    def getclass(self,x):
        r = self.regression(x)
        if r > 0.5:
            return 1
        else:
            return 0







