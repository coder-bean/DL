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
    return (1/math.cosh(x)**2)

class NeuralNetwork:
    #initialize weights
    def __init__(self, input_size, h1_size, output_size):
        self.weights_hidden_input=np.random.randn(input_size, hidden_size)
        self.weights_hidden_output=np.random.randn(output_size, hidden_size)
        self.bias_hidden=np.random.randn(input_size, hidden_size)
        self.bias_output=np.random.randn(output_size, hidden_size)
        self.bias_output=np.random.randn(output_size, hidden_size)
