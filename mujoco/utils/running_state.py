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
        # print("씹슈발 4")
        # ex) [ 1.25227644e+00 -4.76101915e-03 -4.70683807e-03 -4.60811685e-03
        #       4.24723880e-04 -1.29097427e-03  2.35219037e-03 -1.20900498e-03
        #       4.22748799e-03  2.70184498e-03 -1.58194091e-03]
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
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
        # [ 1.25242342e+00  2.48001792e-03 -4.00974886e-03 -3.74310984e-03
        # 1.76107279e-03 -4.24441739e-03  4.94643485e-03  3.23042089e-04
        # -4.13514140e-03  3.71542489e-03 -3.37123480e-03]
        # print("x", x)
        if update: self.rs.push(x)
        # [ 1.25242342e+00  2.48001792e-03 -4.00974886e-03 -3.74310984e-03
        # 1.76107279e-03 -4.24441739e-03  4.94643485e-03  3.23042089e-04
        # -4.13514140e-03  3.71542489e-03 -3.37123480e-03]
        # print("self.rs.push(x)", x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        # print("x completion", x)
        return x

    # def output_shape(self, input_space):
    #     return input_space.shape
