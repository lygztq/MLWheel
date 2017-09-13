import numpy as np



class LogisticRegression:
    def __init__(self, x_dim):
        self._w = np.random.random(size=x_dim)
        self._b = np.random.random(size=1)

    @property
    def weights(self):
        return self._w

    @property
    def bias(self):
        return self._b


