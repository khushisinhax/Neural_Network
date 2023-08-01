#!/usr/bin/env python
# coding: utf-8


import numpy as np
def sigmoid (x): #function sigmoid(x) which computes the sigmoid activation func for a given input x
    
    return 1 / (1 + np.exp(-x)) #define sigmoid function

#Coding a Neuron


class Neuron:  #represents single neuron in a nn
    def __init__(self,weights,bias): # constructor method of neuron class. initializes the attributes of neuron. self parameter is a ref to the instance being created, and it is automaticallly passed to the method when an object is created
        self.weights = weights # Weights control the signal (or the strength of the connection) between two neurons. In other words, a weight decides how much influence the input will have on the output.
        self.bias = bias #offset term that is added to shift the output of neuron, to allow model to learn the best representation of the data.
        
    def feedforward(self,inputs): #feedforward computation of the neuron
        
        total = np.dot(self.weights,inputs) + self.bias #weighted sum of inputs and bias
        return sigmoid(total) #sigmoid activation func applied to the computed total value and returns the result
    
weights = np.array([0,1]) #w1= 0, w2,1
bias = 4
n = Neuron(weights, bias) #instance of Neuron class with specified weights and bias

x=np.array([4,5]) #numpy array created with input values 2,3
print(n.feedforward(x)


#Combining Neurons into a NN

#imagine a network with 2 inputs, a hidden layer with 2 neurons h1 and h2, and an output layer with 1 neuron o1.

class NeuralNetwork:
    
    def __init__(self):
        weights = np.array([0,1])
        bias = 0
        
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights,bias)
        self.o1 = Neuron(weights,bias)

    def feedforward(self,x):
        out_h1= self.h1.feedforward(x)
        out_h2= self.h2.feedforward(x)
        
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        
        return out_o1
network = NeuralNetwork()
x = np.array([4,5])
print(network.feedforward(x))





#Calculate MSE loss
import numpy as np

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

y_true = np.array([1,0,0,1])
y_pred = np.array([0,0,0,0])

print(mse_loss(y_true,y_pred))






