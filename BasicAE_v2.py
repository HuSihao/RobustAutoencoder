import numpy as np
import timeit
import BasicAE as AE

class AutoEncoder_v2(AE.AutoEncoder):
    def getRecon(self, data, lambda_):
        z2 = np.dot(data, self.W) + np.tile(self.b1,(samples,1))
        h = self.sigmoid(z2)
        z3= np.dot(h , self.W.T)+np.tile(self.b2,(samples,1))
        re_con = self.sigmoid(z3)
        return re_con
