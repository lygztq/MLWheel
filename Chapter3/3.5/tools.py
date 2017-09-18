import numpy as np


def get_cov_matrix(x):
    cal_x = np.matrix(x)
    return cal_x.T * cal_x


def get_inverse(M):
    U, sigma_line, VT = np.linalg.svd(M)

    sigma = np.matrix(np.zeros([M.shape[0], M.shape[1]]))
    for i in range(sigma_line.shape[0]):
        sigma[i,i] = sigma_line[i]

    V = VT.T
    UT = U.T
    sigma_inv = sigma.I

    return V * sigma_inv * UT


def get_negative(x):
    return ~x + 1


def absolute(x):
    if x > 0:
        return x
    else:
        return -x


def get_proj(w,x):
    ans = 0
    for i in range(x.shape[0]):
        ans += w[i]*x[i]
    return ans

def reverse(x):
    ans = np.zeros(x.shape)
    for i in range(x.shape[0]):
        ans[x.shape[0]-i-1] = x[i]
    return ans
