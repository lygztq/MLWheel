import numpy as np


def get_z(w, x, b):
    return np.vdot(w, x)+b


def e_pow_z(w, x, b):
    return np.exp(np.vdot(w, x)+b)


def sigmoid(w, x, b):
    epz = e_pow_z(w, x, b)
    return epz / (1 + epz)


def likelihood(x, y, w, b):
    z = get_z(w, x, b)
    epz = e_pow_z(w, x, b)

    return np.log(1 + epz) - y * z


def gradient()

