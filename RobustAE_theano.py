import numpy as np
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams
import timeit

class RobustAE(object):
    "basic auto-encoder"
    def __init__(self , visible_size , hidden_size , sparse_ratio=0.1, random_state = 0):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.sparse_ratio = sparse_ratio
        rng = np.random.RandomState(random_state)
        initial_W = np.asarray(rng.uniform(
            low=-4 * np.sqrt(6. / (hidden_size + visible_size)),
            high=4 * np.sqrt(6. / (hidden_size + visible_size)),
            size=(visible_size, hidden_size)),
            dtype = theano.config.floatX)
        self.W = theano.shared(value=initial_W, name='W' , borrow=True)

        self.b1 = theano.shared(value = np.zeros(visible_size,dtype=theano.config.floatX),name='b1',borrow=True)
        self.b2 = theano.shared(value=np.zeros(hidden_size,dtype=theano.config.floatX),name='b2',borrow=True)
        self.random_state = random_state
        #self.x = T.dmatrix(name = 'input')

        self.params = [self.W, self.b1, self.b2]

    def get_hidden_values(self, x):
        return T.nnet.sigmoid(T.dot(x , self.W) + self.b1)

    def get_reconstructed(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden , self.W.T) + self.b2)

    def split(self, x, index, batch_size):
        print batch_size, index
        sparse = x * self.mask[index * batch_size: (index + 1) * batch_size,:]
        return sparse , x - sparse

    def get_cost(self, x, index, batch_size):
        sparse , low_rank = self.split(x, index, batch_size)
        h = self.get_hidden_values(low_rank)
        re_con = self.get_reconstructed(h) + sparse
        cost = T.sum( (x - re_con) ** 2 )
        return cost

    def get_updates(self, x, index, batch_size, learning_rate):
        cost = self.get_cost(x, index, batch_size= batch_size)
        gparams = T.grad(cost, self.params)
        updates = [(param , param - learning_rate*gparam) for param,gparam in zip(self.params , gparams)]
        return updates

    def fit(self, X , iterations=15, batch_size = 20 ,learning_rate=0.1):
        index = T.lscalar()
        x = T.matrix('x')
        n_train_batches = X.shape[0] / batch_size
        rng = np.random.RandomState(self.random_state)
        mask = rng.normal(loc = 0., size = (X.shape[0], X.shape[1]))
        self.mask = theano.shared(value= mask, name='mask', borrow=True)
        print self.mask.get_value().shape
        X = theano.shared(np.asarray(X,dtype=theano.config.floatX))

        cost = self.get_cost(x , index, batch_size= batch_size)
        updates = self.get_updates(x, index, batch_size = batch_size, learning_rate=learning_rate)
        #one_step_training = theano.function(inputs = [x] , outputs = cost , updates = updates)
        train_da = theano.function([index], cost,
                                    updates=updates,
                                    givens={x: X[index * batch_size: (index + 1) * batch_size]})

        start_time = timeit.default_timer()
        for epoch in xrange(iterations):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))
            print 'Traing epoch %d, cost ' % epoch, np.mean(c)
        end_time = timeit.default_timer()
        training_time = float(end_time - start_time)

        print 'AutoEncoder run for %.2fm ' % (training_time / 60.)

    def transform(self , X):
        input=T.dmatrix(name='input')
        sparse , low_rank = self.split(x)
        get_hidden_data=theano.function([input] , self.get_hidden_values(low_rank))
        return get_hidden_data(X)

if __name__=='__main__':
    data=np.load(r'/home/czhou2/Documents/train_x_small.pkl')
    #y=np.load(r'C:\Users\zc\Documents\MNIST data\Tutorial\train_y.pkl')
    print data.shape
    #print y.shape
    inputsize=(28,28)
    hidden_size=(20,20)

    rae = RobustAE(hidden_size=hidden_size[0]*hidden_size[1], visible_size=inputsize[0]*inputsize[1])
    rae.fit(data,iterations=2)
    rae.transform(data)
    print rae
