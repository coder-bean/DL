import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math

e=math.e
#part b1:

x=np.linspace(-2*np.pi, 2*np.pi,1000)
y=np.sin(x)

    #splitting into 4 equal parts:
split_x=np.split(x,4)
split_y=np.split(y,4)

#part b2:

x_validation = np.random.uniform(-2*np.pi,2*np.pi,300)
#ANN TO GENERATE Y-VALS

#part b3:
plot.plot(x,y, label='Sine Wave')
plot.legend()

#defining the activation functions and it's derivative
def tanh(x):
    return (e**x-e**(-x))/(e**x+e**(-x))
def ddxtanhx(x):
    return (1/np.cosh(x)**2)

class NeuralNetwork:
    #initialize weights
    def __init__(self, input_size, h1_size, output_size):
        self.weights_hidden_input=np.random.randn(input_size, hidden_size)
        self.weights_hidden_output=np.random.randn(output_size, hidden_size)
        self.bias_hidden=np.random.randn(input_size, hidden_size)
        self.bias_output=np.random.randn(output_size, hidden_size)


    def backward(self,x,y,learning_rate):
        #error calculation
        output_error = y - self.output
        output_delta = output_error * ddxtanhx(self.output)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * ddxtanhx(self.hidden_layer_output)

        #weight and bias update
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += x.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def forward(self, x):
        self.hidden_layer_input=np.dot(x, self.weights_hidden_input)+self.bias_hidden
        self.hidden_layer_output=tanh(self.hidden_layer_input)
        self.output_layer_input=np.dot(self.hidden_layer_output, self.weights_hidden_output)+self.bias_output
        self.output=tanh(self.output_layer_input)      
        return self.output
    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y, learning_rate)
            loss = np.mean(np.square(y - self.output))
            print(f'Loss: {loss}')
