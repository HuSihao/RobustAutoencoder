import timeit
import numpy as np

class AutoEncoder():
    def __init__(self,hidden_size,visible_size):
        r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
        self.W = np.random.random((visible_size,hidden_size)) * 2 * r - r
        self.b1 = np.zeros(hidden_size, dtype=np.float64)
        self.b2 = np.zeros(visible_size, dtype=np.float64)
        self.hidden_size = hidden_size
        self.visible_size= visible_size
    def sigmoid(self,x):
        return 1./(1.+np.exp(-x))

    def sigmoid_prime(self,x):
        return self.sigmoid(x)*(1.-self.sigmoid(x))

    def getParameter(self):
        return self.W,self.b1,self.b2,self.visibel_size,self.hidden_size


    def cost(self , data , lambda_):
        ##lambda_ is weight of regularization term

        samples = data.shape[0]
        z2 = np.dot(data, self.W) + np.tile(self.b1,(samples,1))
        h = self.sigmoid(z2)
        z3= np.dot(h , self.W.T)+np.tile(self.b2,(samples,1))
        re_con = self.sigmoid(z3)
        cost = np.sum((re_con-data)**2) / samples + lambda_ * np.sum(self.W ** 2)
        delta3 = -(data - re_con) * self.sigmoid_prime(z3)
        delta2 = np.dot(delta3, self.W) * self.sigmoid_prime(z2)

        Wgrad = np.dot(delta3.T, h) / samples + lambda_ * self.W
        ## also train again
        ## Wgrad_= np.dot(delta2.T,data) / samples + lambda_ * self.W
        b1grad = np.sum(delta2 , axis=0) / samples
        b2grad = np.sum(delta3 , axis=0) / samples

        grad = [Wgrad , b1grad , b2grad]
        return cost, grad

    def one_step_training(self,data, learning_rate ,lambda_):
        one_step_cost , grad = self.cost(data , lambda_)
        self.W = self.W - learning_rate * grad[0]
        self.b1 = self.b1 - learning_rate * grad[1]
        self.b2 = self.b2 - learning_rate * grad[2]
        return one_step_cost
    def fit(self,data, lambda_, learning_rate=0.1 ,iteration=25, verbose=False):
        if verbose:
            print "training..."
            print
        training_cost = []
        start_time=timeit.default_timer()
        for i in range(iteration):
            one_step_cost = self.one_step_training(data, 0.1 , lambda_)
            if verbose:
                print "iteration: ", i , " cost: ", one_step_cost
            training_cost.append(one_step_cost)
        end_time = timeit.default_timer()
        if verbose:
            print
            print "total training time: ", ((end_time - start_time) / 60.), "m"

    def transform(self,data):
        z2 = np.dot(data,self.W) + np.tile(self.b1,(data.shape[0],1))
        h = self.sigmoid(z2)
        return h
if __name__ == '__main__':
    data=np.load(r'C:\Users\zc\Documents\MNIST data\Tutorial\train_x.pkl')
    y=np.load(r'C:\Users\zc\Documents\MNIST data\Tutorial\train_y.pkl')
    print data.shape
    print y.shape
    inputsize=(28,28)
    hidden_size=(20,20)

    autoencoder = AutoEncoder(hidden_size=hidden_size[0]*hidden_size[1], visible_size=inputsize[0]*inputsize[1])
    autoencoder.fit(data,lambda_=0.5,learning_rate=0.1, iteration=2)
    autoencoder.transform(data)
