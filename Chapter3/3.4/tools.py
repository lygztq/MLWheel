import numpy as np


def get_z(w, x, b):
    return np.vdot(w, x)+b


def e_pow_z(w, x, b):
    return np.exp(np.vdot(w, x)+b)


def sigmoid(z):
    return np.longfloat(1 / (1 + np.exp(-z)))


def likelihood(x, y, w, b):
    z = get_z(w, x, b)
    epz = e_pow_z(w, x, b)

    return np.log(1 + epz) - y * z


# def gradient(samples, labels, w, b):
#     size = samples.shape[1] + 1
#     ans = np.zeros([size])
#     #    augmenteds.append(np.append(i, 1))
#
#     for sample, label in samples,labels:
#         p1 = sigmoid(get_z(w,sample,b))
#         augmented_x = np.append(sample,1)
#         ans = ans + augmented_x * (label - p1)
#     ans = -1 * ans
#
#     return ans




