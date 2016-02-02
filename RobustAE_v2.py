"""
@author: Chong Zhou
first complete: 01/29/2016
Des:
    refere robust pca code: https://github.com/nwbirnie/rpca/blob/master/rpca.py
    X = L + S
    L is a non-linearly low rank matrix and S is a sparse matrix.
    argmin |g(h(L))|_2 + |S|_1
    lagrangian multiplier to train model
"""
import timeit
import numpy as np
import BasicAE_v2 as AutoEncoder
import math
import numpy.linalg as nplin

class RobustAE():
    def __init__(self, hidden_size, lambda_=1.0, error = 1.0e-5, penOfOverfitting = 0.1):
        self.lambda_=lambda_
        self.hidden_size=hidden_size
        self.error = error
        self.pennalty = penOfOverfitting
    def fit(self, X, learning_rate=0.1, iteration=20):
        numfea = X.shape[1]
        samples = X.shape[0]
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        self.Y = np.zeros(X.shape)
        self.AE = AutoEncoder.AutoEncoder_v2(self.hidden_size,numfea)
        print "X shape", X.shape
        penOfOverfitting = 0.1

        mu = (X.shape[0] * X.shape[1]) / (4.0 * self.L1Norm(X))

        lamb = max(X.shape) ** -0.5

        while not self.converged(X, self.L, self.S):
            self.L = X - self.S - (mu**-1) * self.Y
            self.AE.fit(self.L, self.pennalty, learning_rate=learning_rate ,iteration=iteration)
            self.L = self.AE.getRecon(self.L)
            self.S = self.shrink(X - self.L + (mu**-1) * self.Y, lamb * mu)
            self.Y = self.Y + mu * (X - self.L - self.S)
        return self.L, self.S

    def L1Norm(self, X):
        return max(np.sum(X, axis=0))
    def shrink(self, X, tau):
        V = np.copy(X).reshape(X.size)
        for i in xrange(V.size):
            V[i] = math.copysign( max(abs(V[i]) - tau, 0) , V[i])
            if V[i] == -0:
                V[i] = 0
        return V.reshape(X.shape)
    def FNorm(self, X):
        return nplin.norm(X,'fro')

    def converged(self, X, L, S):
        error = self.FNorm(X - L - S) / self.FNorm(X)
        print "error = ", error
        return error <= self.error
    def transform(self, X):
        L = X - self.S
        return self.AE.transform(L)


if __name__ == "__main__":
    #X = np.array(range(20*40)).reshape((20,40))
    from sklearn.datasets import load_digits
    digits=load_digits()
    X = digits.data
    Y = digits.target
    print X.shape
    rae = RobustAE(hidden_size = 5*5, lambda_=1.0, error = 0.0001)
    L , S = rae.fit(X)
    print rae.transform(X).shape
