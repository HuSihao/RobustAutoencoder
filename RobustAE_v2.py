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
import BasicAE_v2.AutoEncoder_v2 as AutoEncoder
import math
import numpy.linalg as nplin

class RobustAE():
    def __init__(self, hidden_size, lambda_=1.0, error = 1.0e-5 ):
        self.lambda_=lambda_
        self.hidden_size=hidden_size
        self.error = error
    def fit(self, X):
        numfea = X.shape[1]
        samples = X.shape[0]
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        self.Y = np.zeros(X.shape)
        self.AE = AutoEncoder(self.hidden_size,numfea)
        print "X shape", X.shape
        penOfOverfitting = 0.1

        mu = (M.shape[0] * M.shape[1]) / (4.0 * self.L1Norm(M))
        lamb = max(X.shape) ** -0.5

        while not self.converged(X, L, S):
            L = X - S - (mu**-1) * Y
            self.AE.fit.(L, penOfOverfitting, learning_rate=0.1 ,iteration=20)
            self.L = self.AE.getRecon(L)
            self.S = self.shrink(M - self.L + mu(**-1) * Y, lamb * mu)
            self.Y = Y + mu * (X - self.L - self.S)
        return L,S

    def L1Norm(self, M):
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
        error = self.FNorm(X - L - S) / self.FNorm(M)
        print "error = ", error
        return error <= self.error
    def transform(self, X):
        L = X - self.S
        return self.AE.transform(L)
