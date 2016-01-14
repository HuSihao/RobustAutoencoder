import timeit
import numpy as np

class RobustAE():
    def __init__(self,hidden_size,visible_size,sparse_ratio):
        r = np.sqrt(6) / np.sqrt(hidden_size + visible_size +1)
        self.W = np.random.random((visible_size, hidden_size)) * 2 * r - r
        self.b1 = np.zeros(hidden_size , dtype= np.float64)
        self.b2 = np.zeros(visible_size , dtype = np.float64)
        self.hidden_size = hidden_size
        self.visible_size = visible_size
        self.sparse_ratio = sparse_ratio## adding sparsity
        self.mask = np.random.random((visible_size, hidden_size)) * 2 * r - r
    def sigmoid(self , x):
        return 1./(1. + np.exp(-x))
    def sigmoid_prime(self, x):
        return self.sigmoid(x)*(1. - self.sigmoid(x))
    def split(self,x):
        sparse = x * self.mask
        return sparse , x - sparse
    def getParameter(self):
        return self.W,self.b1,self.b2,self.visibel_size,self.hidden_size,self.mask
    def cost(self , data , lambda_):
        ##lambda_ is weight of regularization term
        samples = data.shape[0]
        sparse , x = split(data)
        z2 = np.dot(x, self.W) + np.tile(self.b1,(samples,1))
        h = self.sigmoid(z2)
        z3= np.dot(h , self.W.T)+np.tile(self.b2,(samples,1))
        re_con = self.sigmoid(z3)

        cost = np.sum((re_con + sparse -data) ** 2) / samples + lambda_ * np.sum(self.W ** 2) + np.sum(sparse ** 2)
        delta3 = - (re_con + sparse - data) * self.sigmoid_prime(z3)
        delta2 = np.dot(delta3, self.W) * self.sigmoid_prime(z2)
