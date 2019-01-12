from collections import deque

import numpy as np


# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape): # shape = (11,)
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1: # Only the first time
            self._M[...] = x
        else: # From the second time ~ 
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def mean(self):
        return self._M

    @mean.setter
    def mean(self, M):
        self._M = M

    @property
    def sum_square(self):
        return self._S

    @sum_square.setter
    def sum_square(self, S):
        self._S = S

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0): # shape = (11,), clip = 5
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        # [ 1.25204271  0.00339264 -0.00230297 -0.00529779  0.00736752  0.01062921
        # -0.1001518   0.22747316  0.43029206 -0.39176969  1.4032258 ]
        # print("x", x)
        if update: self.rs.push(x)
        if self.demean:
            # [ 1.25223306  0.00293633 -0.00315636 -0.00452045  0.0045643   0.0031924
            # -0.04760268  0.1138981   0.21307846 -0.19402713  0.69992728]
            # print("self.rs.mean", self.rs.mean)
            x = x - self.rs.mean
            # [-1.90359424e-04  4.56310453e-04  8.53387150e-04 -7.77341865e-04
            # 2.80322564e-03  7.43681254e-03 -5.25491167e-02  1.13575060e-01
            # 2.17213602e-01 -1.97742558e-01  7.03298517e-01]
            # print("self.demean_x", x)

        if self.destd:
            # [2.69208879e-04 6.45320431e-04 1.20687168e-03 1.09932741e-03
            # 3.96435972e-03 1.05172412e-02 7.43156735e-02 1.60619390e-01
            # 3.07186421e-01 2.79650207e-01 9.94614301e-01]
            # print("self.rs.std", self.rs.std)
            x = x / (self.rs.std + 1e-8)
            # [-0.70708052  0.70709582  0.70710092 -0.70710035  0.707105    0.70710611
            # -0.70710669  0.70710674  0.70710676 -0.70710676  0.70710677]
            # print("self.destd_x", x)
            
        if self.clip:
            # 5
            # print("self.clip", self.clip)
            x = np.clip(x, -self.clip, self.clip)
            # [-0.70708052  0.70709582  0.70710092 -0.70710035  0.707105    0.70710611
            # -0.70710669  0.70710674  0.70710676 -0.70710676  0.70710677]
            # print("self.clip", x)

        # [-0.70708052  0.70709582  0.70710092 -0.70710035  0.707105    0.70710611
        # -0.70710669  0.70710674  0.70710676 -0.70710676  0.70710677]
        # print("x completion", x)
        return x

    # def output_shape(self, input_space):
    #     return input_space.shape
