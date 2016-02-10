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
import BasicAE_theano as AutoEncoder
import math
import numpy.linalg as nplin

def minShrink1Plus2Norm(A, E, lam, mu):
    """
    @author: Prof. Randy
    from https://bitbucket.org/rcpaffenroth/playground/src/
    19a8a39cfc5b40e7d51697263881132a9d371fc5/src/lib/minShrink1Plus2Norm.py?at=feature-MVU&fileviewer=file-view-default

    Compute a fast minimization of shrinkage plus Frobenius norm.

    The is computes the minium of the following objective.

    .. math::

        \lambda \| \mathcal{S}_{\epsilon}( S_{ij} ) \|_1 +
        \mu / 2 \| S_{ij} - A_{ij} \|_F^2

    Args:
        A: A numpy array.

        E: A numpy array of error bounds.

        lam: The value of :math:`\lambda`.

        mu: The value of :math:`\mu`.

    Returns:
        The value of :math:`S` that achieves the minimum.
    """
    assert len(A.shape) == 1, 'A can only be a vector'
    assert A.shape == E.shape, 'A and E have  to have the same size'
    # Note, while the derivative is always zero when you use the
    # formula below, it is only a minimum if the second derivative is
    # positive.  The second derivative happens to be \mu.
    assert mu >= 0., 'mu must be >= 0'

    S = np.zeros(A.shape)

    for i in range(len(A)):
        if (lam/mu + A[i]) < -E[i]:
            S[i] = lam/mu + A[i]
        elif -E[i] < A[i] < E[i]:
            S[i] = A[i]
        elif E[i] < (-lam/mu + A[i]):
            S[i] = -lam/mu + A[i]
        else:
            Sp = (mu/2.)*(E[i] - A[i])*(E[i] - A[i])
            Sm = (mu/2.)*(-E[i] - A[i])*(-E[i] - A[i])
            if Sp < Sm:
                S[i] = E[i]
            else:
                S[i] = -E[i]
    return S
def shrink(epsilon, x):
    """
    @author: Prof. Randy
    from https://bitbucket.org/rcpaffenroth/playground/src/
    19a8a39cfc5b40e7d51697263881132a9d371fc5/src/lib/shrink.py?at=feature-MVU&fileviewer=file-view-default
    The shrinkage operator.
    This implementation is intentionally slow but transparent as
    to the mathematics.

    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        x: the vector to shrink on

    Returns:
        The shrunk vector
    """
    output = np.array(x*0.)
    if np.isscalar(epsilon):
        epsilon = np.ones(x.shape)*epsilon

    for i in range(len(x)):
        if x[i] > epsilon[i]:
            output[i] = x[i] - epsilon[i]
        elif x[i] < -epsilon[i]:
            output[i] = x[i] + epsilon[i]
        else:
            output[i] = 0
    return output
class RobustAE():
    """
    @author: Chong Zhou
    first complete: 01/29/2016
    Des:
        refere robust pca code: https://github.com/nwbirnie/rpca/blob/master/rpca.py
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin |L - g(h(L))|_2 + |S|_1
        lagrangian multiplier to train model
    """
    def __init__(self, hidden_size, lambda_=1.0, error = 1.0e-5, penOfOverfitting = 0.1):
        self.lambda_=lambda_
        self.hidden_size=hidden_size
        self.error = error
        self.pennalty = penOfOverfitting
    def fit(self, X, learning_rate=0.1, iteration=20, verbose=False, inner_iteration= 20):

        numfea = X.shape[1]
        samples = X.shape[0]
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        self.Y = np.zeros(X.shape)
        self.AE = AutoEncoder.AutoEncoder(hidden_size=self.hidden_size, visible_size=numfea)
        print "X shape", X.shape
        print "L shape", self.L.shape
        print "S shape", self.S.shape
        print "Y shape", self.Y.shape
        penOfOverfitting = 0.1

        ## the intial value for the augmented Lagrangian parameter.
        mu = (X.shape[0] * X.shape[1]) / (4.0 * nplin.norm(X,1))
        ## the value of the coupling constant between L and S
        #lamb = 1. / np.sqrt(np.max([m,n]))
        ## rho: the growth factor for the augmented Lagrangian parameter.
#         rho_s = len(vecM)/float(m*n)
#         rho = 1.2172+1.8588*rho_s
        rho = 1.01
        LS0 = self.S + self.L
        E = (np.ones(X.shape) * 1e-3).reshape(X.size)
        XFnorm = nplin.norm(X,'fro')
        for it in xrange(iteration):
            if verbose:
                print "iteration: ", it
                print "rho", rho
                print "mu", mu
            self.L = X - self.S -  self.Y / mu
            #def fit(self, X , iterations=15, batch_size = 20 ,learning_rate=0.15)
            print "2 L shape" ,self.L.shape
            print "L type", type(self.L)
            self.AE.fit(self.L, iterations=inner_iteration, batch_size = 25, learning_rate=learning_rate)
            self.L = self.AE.getRecon(self.L)
            A = (X - self.L + self.Y / mu).reshape(X.size)
            self.S = minShrink1Plus2Norm(A, E ,self.lambda_ , mu).reshape(X.shape)

            self.Y = self.Y + mu * (X - self.L - self.S)
            if mu * nplin.norm(LS0-self.L-self.S,'fro') / XFnorm < self.error:
                mu = rho * mu

            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm

            if c1 < self.error and c2 < self.error :
                break
            LS0 = self.L + self.S
        self.S = shrink(E, self.S.reshape(X.size)).reshape(X.shape)
        return self.L, self.S
    def transform(self, X):
        L = X - self.S
        return self.AE.transform(L)
    def getRecon(self):
        return self.AE.getRecon(self.L)




def test1():
    from sklearn.datasets import load_digits
    digits=load_digits()
    X = digits.data
    Y = digits.target
    print X.shape

    image_X = Image.fromarray(I.tile_raster_images(X=X,img_shape=(8,8), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_X.save(r"X1.png")
    rae = RobustAE(hidden_size = 6*6, lambda_=1.0, error = 0.0001)
    L , S = rae.fit(X,verbose = True)
    print rae.transform(X).shape
    image_S = Image.fromarray(I.tile_raster_images(X=S,img_shape=(8,8), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_S.save(r"S1.png")
    R = rae.getRecon()
    image_R = Image.fromarray(I.tile_raster_images(X=R,img_shape=(8,8), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_R.save(r"Recon1.png")

def test2():
    X=np.load(r'/home/czhou2/Documents/train_x_small.pkl')
    print X.shape
    inputsize=(28,28)
    hidden_size=(20,20)

    image_X = Image.fromarray(I.tile_raster_images(X=X,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    rae = RobustAE(hidden_size = 20*20, lambda_=1000.0, error = 0.0001)
    L , S = rae.fit(X,verbose = True)
    print rae.transform(X).shape
    image_S = Image.fromarray(I.tile_raster_images(X=S,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_S.save(r"S.png")
    R = rae.getRecon()
    image_R = Image.fromarray(I.tile_raster_images(X=R,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_R.save(r"Recon.png")
if __name__ == "__main__":
    import PIL.Image as Image
    import ImShow as I
    test1()
    #test2()
