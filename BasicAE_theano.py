import numpy as np
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams
import timeit

class AutoEncoder(object):
    "basic auto-encoder"
    def __init__(self , visible_size , hidden_size , random_state = 0):
        self.visible_size = visible_size
        self.hidden_size = hidden_size

        rng = np.random.RandomState(random_state)
        initial_W = np.asarray(rng.uniform(
            low=-4 * np.sqrt(6. / (hidden_size + visible_size)),
            high=4 * np.sqrt(6. / (hidden_size + visible_size)),
            size=(visible_size, hidden_size)),
            dtype = theano.config.floatX)
        self.W = theano.shared(value=initial_W, name='W' , borrow=True)

        self.b1 = theano.shared(value = np.zeros(hidden_size,dtype=theano.config.floatX),name='b1',borrow=True)
        self.b2 = theano.shared(value=np.zeros(visible_size,dtype=theano.config.floatX),name='b2',borrow=True)
        self.random_state = random_state
        #self.x = T.dmatrix(name = 'input')

        self.params = [self.W , self.b1 , self.b2]

    def get_hidden_values(self, x):
        return T.nnet.sigmoid(T.dot(x , self.W) + self.b1)
    def get_reconstructed(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden , self.W.T) + self.b2)

    def get_cost(self, x, learning_rate):
        h = self.get_hidden_values(x)
        re_con = self.get_reconstructed(h)
        cost = T.sum( (x - re_con) ** 2 )
        return cost

    def get_updates(self, x, learning_rate):
        cost = self.get_cost(x, learning_rate)
        gparams = T.grad(cost, self.params)
        updates = [(param , param - learning_rate*gparam) for param,gparam in zip(self.params , gparams)]
        return updates

    def fit(self, X , iterations=15, batch_size = 20 ,learning_rate=0.15):
        index = T.lscalar()
        x = T.matrix('x')
        n_train_batches = X.shape[0] / batch_size
        X = theano.shared(np.asarray(X, dtype=theano.config.floatX))

        cost = self.get_cost(x , learning_rate=learning_rate)
        updates = self.get_updates(x, learning_rate=learning_rate)

        #one_step_training = theano.function(inputs = [x] , outputs = cost , updates = updates)
        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: X[index * batch_size: (index + 1) * batch_size]
            }
        )
        print "type n_train_batches: ", type(n_train_batches)
        start_time = timeit.default_timer()
        for epoch in xrange(iterations):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))

            print 'Training epoch %d, cost ' % epoch, np.mean(c)
        end_time = timeit.default_timer()
        training_time = float(end_time - start_time)

        print 'AutoEncoder run for %.2fm ' % (training_time/60.)

    def transform(self , X):
        input=T.dmatrix(name='input')
        get_hidden_data=theano.function([input] , self.get_hidden_values(input))
        return get_hidden_data(X)

if __name__=='__main__':
    data=np.load(r'/home/zc/Documents/train_x.pkl')
    #y=np.load(r'C:\Users\zc\Documents\MNIST data\Tutorial\train_y.pkl')
    print data.shape
    #print y.shape
    inputsize=(28,28)
    hidden_size=(20,20)

    autoencoder = AutoEncoder(hidden_size=hidden_size[0]*hidden_size[1], visible_size=inputsize[0]*inputsize[1])
    autoencoder.fit(data,iterations=2)
    autoencoder.transform(data)
