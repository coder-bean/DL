import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math
import os
import imageio
import imageio.v2 as imageio
output_dir = 'training_plots'
filenames=[]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
#steps 2 and 3: ANN Architecture, Backpropogation equations
class NeuralNetwork:
    #initialize weights
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_hidden_input=np.random.randn(input_size, hidden_size)
        self.weights_hidden_output=np.random.randn(hidden_size, output_size)
        self.bias_hidden=np.random.randn(1, hidden_size)
        self.bias_output=np.random.randn(1, output_size)


    def forward(self, x):
        self.hidden_layer_input=np.dot(x, self.weights_hidden_input)+self.bias_hidden
        self.hidden_layer_output=tanh(self.hidden_layer_input)
        self.output_layer_input=np.dot(self.hidden_layer_output, self.weights_hidden_output)+self.bias_output
        self.output=tanh(self.output_layer_input)      
        return self.output
    
    

    def backward(self,x,y,learning_rate):
        #error calculation
        output_error = y - self.output
        output_delta = output_error * ddxtanhx(self.output)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * ddxtanhx(self.hidden_layer_output)

        #weight and bias update
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_hidden_input += x.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            
            self.forward(x)
            self.backward(x, y, learning_rate)
            loss = np.mean(np.square(y - self.output))
            print(f'Loss: {loss}')
            if(epoch%10==0):
                plot.plot(x,y, label='True Sine wave')
                plot.plot(x,self.output, label='approximation at epoch {epoch}')
                filename = f'training_plots/epoch_{epoch}.png'
                plot.legend(loc='lower left')
                filenames.append(filename)
                plot.savefig(filename)
                plot.close()

#step 6: I/O Normalization
def normalize(data, actual_min, actual_max):
    virtual_min = actual_min - 0.05 * (actual_max - actual_min)
    virtual_max = actual_max + 0.05 * (actual_max - actual_min)
    return 1.8 * (data - virtual_min) / (virtual_max - virtual_min) - 0.9

def denormalize(data, actual_min, actual_max):
    virtual_min = actual_min - 0.05 * (actual_max - actual_min)
    virtual_max = actual_max + 0.05 * (actual_max - actual_min)
    return (data + 0.9) * (virtual_max - virtual_min) / 1.8 + virtual_min

x=np.linspace(-2*np.pi, 2*np.pi,1000).reshape(-1,1)
y=np.sin(x)

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

x_normalized = normalize(x, x_min, x_max)
y_normalized = normalize(y, y_min, y_max)
#part b1: splitting into 4 equal parts:
split_x=np.split(x,4)
split_y=np.split(y,4)

#part b2:

x_validation = np.random.uniform(-2*np.pi,2*np.pi,300)


#defining the activation functions and it's derivative
def tanh(x):
    return np.tanh(x)
def ddxtanhx(x):
    return (1/np.cosh(x)**2)


#creating nn
nn = NeuralNetwork(input_size=1, hidden_size=4, output_size=1)
nn.train(x_normalized,y_normalized, epochs=1000, learning_rate=0.001)
predicted_y_normalized = nn.forward(x_normalized)
predicted_y = denormalize(predicted_y_normalized, y_min, y_max)


# Create a GIF from the saved images
with imageio.get_writer('sine_wave_training.gif', mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up images after creating the GIF
for filename in filenames:
    os.remove(filename)
