import numpy as np
from loss import cross_entropy_error
from activation import sigmoid, relu, softmax

def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    if x.ndim == 2: 
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                fx = f(x[i,j])
                tmp_val = x[i,j]
                x[i,j] = tmp_val + h
                fxh = f(x[i,j])
                grad[i,j] = (fxh - fx)/h
                x[i,j] = tmp_val
        return grad
    else:
        for i in range(x.size):
            tmp_val = x[i]
            x[i] = tmp_val + h
            fxh1 = f(x[i])
            x[i] = tmp_val - h
            fxh2 = f(x[i])
            grad[i] = (fxh1-fxh2)/2*h
            x[i] = tmp_val
        return grad

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        self.W = {}
        self.W['W1'] = np.random.randn(input_size,hidden_size)
        self.W['b1'] = np.random.randn(hidden_size)
        self.W['W2'] = np.random.randn(hidden_size,output_size)
        self.W['b2'] = np.random.randn(output_size)
        self.loss_val = []
    
    def predict(self,x):
        W1 = self.W['W1']
        W2 = self.W['W2']
        b1 = self.W['b1']
        b2 = self.W['b2']
        
        a1 = np.dot(x,W1) + b1 # 출력값
        z1 = relu(a1)
        a2 = np.dot(z1,W2) + b2
        out = softmax(a2)
        return out
    
    def loss(self,x,t):
        y = self.predict(x)
        loss = cross_entropy_error(y,t)
        return loss

    def numerical_gradient(self,x,t):
        f = lambda W: self.loss(x,t)
        
        grads = {}
        grads['W1'] = numerical_gradient(f, self.W['W1'])
        grads['b1'] = numerical_gradient(f, self.W['b1'])
        grads['W2'] = numerical_gradient(f, self.W['W2'])
        grads['b2'] = numerical_gradient(f, self.W['b2'])
        
        return grads

    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        acc = sum(y == t)/x.shape[0]
        return acc
    
    def train(self,epochs,lr,x,t):
        for epoch in range(epochs):
            grads = self.numerical_gradient(x,t)
            for key in grads.keys():
                self.W[key] -= lr*grads[key]
            self.loss_val.append(self.loss(x,t))

class OneLayer:
    def __init__(self,input_size,output_size):
        self.W = {}
        self.W['W1'] = np.random.randn(input_size,output_size)
        self.W['b'] = np.random.randn(output_size)
    
    def predict(self,x):
        W1, b = self.W['W1'], self.W['b']
        pred = softmax(np.dot(x,W1) + b)
        return pred
    
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
    
    def numerical_gradient(self,x,t):
        y = self.predict(x)
        f = lambda W: cross_entropy_error(y,t)
        grad = {}
        grad['W1'] = numerical_gradient(f,self.W['W1'])
        grad['b'] = numerical_gradient(f, self.W['b'])
        
        return grad
    
    def accuracy(self,x,t):
        y = self.predict(x)
        acc = np.sum(np.argmax(y,axis=1) == np.argmax(t,axis=1))/y.shape[0]
        return acc
    
    def fit(self,x,t,epochs=1000,lr=1e-3,verbos=1):
        for epoch in range(epochs):
            self.W['W1'] = self.W['W1'] - lr*self.numerical_gradient(x,t)['W1']
            self.W['b'] -= lr*self.numerical_gradient(x,t)['b']
            if verbos == 1:
                print("=========== loss ",self.loss(x,t), "======== acc ",self.accuracy(x,t))
        