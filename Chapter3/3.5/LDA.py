from __future__ import division
import numpy as np
import tools

class LDA:
    def __init__(self, samples, labels):
        self._positive_samples = []
        self._negative_samples = []
        self._x_dim = samples.shape[1]
        self._positive_mean = np.zeros([self._x_dim])
        self._negative_mean = np.zeros([self._x_dim])
        self._w = np.zeros([samples.shape[1]])

        for index in range(labels.shape[0]):
            if labels[index] == 1:
                self._positive_samples.append(samples[index, :])
                self._positive_mean += samples[index,:]
            else:
                self._negative_samples.append(samples[index, :])
                self._negative_mean += samples[index, :]

        self._positive_mean /= len(self._positive_samples)
        self._negative_mean /= len(self._negative_samples)

        self._positive_samples = np.array(self._positive_samples)
        self._negative_samples = np.array(self._negative_samples)

        self._Sw = np.matrix(np.zeros([self._x_dim,self._x_dim]))
        # Sb = np.matrix(np.zeros([self._x_dim,self._x_dim]))
        for pos_sample in self._positive_samples:
            self._Sw += tools.get_cov_matrix(pos_sample-self._positive_mean)
        for neg_sample in self._negative_samples:
            self._Sw += tools.get_cov_matrix(neg_sample-self._negative_mean)

        self._w = tools.get_inverse(self._Sw) * np.matrix((self._negative_mean - self._positive_mean)).T
        self._w = np.reshape(np.array(self._w.T), [self._x_dim])

        self._positive_center = np.dot(self._positive_mean, self._w)
        self._negative_center = np.dot(self._negative_mean, self._w)
        # self._w = np.reshape(np.array(self._w.T), [self._x_dim])

    def getClass(self,x):
        x_projection = tools.get_proj(self._w,x)
        positive_dis = tools.absolute(x_projection-self._positive_center)
        negative_dis = tools.absolute(x_projection-self._negative_center)

        if positive_dis > negative_dis:
            return 0
        else:
            return 1


    @property
    def weight(self):
        return self._w
    @property
    def positive_mean(self):
        return self._positive_mean
    @property
    def negative_mean(self):
        return self._negative_mean
    @property
    def positive_center(self):
        return self._positive_center
    @property
    def negative_center(self):
        return self._negative_center





