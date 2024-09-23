from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math
import os
import imageio

def normalize(data, actual_min, actual_max):
    virtual_min = actual_min - 0.05 * (actual_max - actual_min)
    virtual_max = actual_max + 0.05 * (actual_max - actual_min)
    return 1.8 * (data - virtual_min) / (virtual_max - virtual_min) - 0.9

def denormalize(data, actual_min, actual_max):
    virtual_min = actual_min - 0.05 * (actual_max - actual_min)
    virtual_max = actual_max + 0.05 * (actual_max - actual_min)
    return (data + 0.9) * (virtual_max - virtual_min) / 1.8 + virtual_min

# fetch dataset 
combined_cycle_power_plant = fetch_ucirepo(id=294) 

# data (as pandas dataframes) 
x = combined_cycle_power_plant.data.features 
y = combined_cycle_power_plant.data.targets 

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

x_normalized = normalize(x, x_min, x_max)
y_normalized = normalize(y, y_min, y_max)

#Dividing Data into Test, Validation and Training sets
test_x=x.iloc[8612:]
test_y=y.iloc[8612:]
train_x=[]
train_y=[]
val_x=[]
val_y=[]
for i in range(0,8612):
    if(i%5==0):
        val_x=x[i]
        val_y=y[i]
    else:
        train_x=x[i]
        train_y=y[i]


#defining the activation functions and it's derivative
def tanh(x):
    return np.tanh(x)

def ddxtanhx(x):
    return (1/np.cosh(x)**2)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def ddxlogisticx(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def ddxrelux(x):
    return np.where(x > 0, 1, 0)

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
                plot.plot(x,y, label='CCPP')
                plot.plot(x,self.output, label='approximation at epoch {epoch}')
                filename = f'training_plots/epoch_{epoch}.png'
                plot.legend(loc='lower left')
                filenames.append(filename)
                plot.savefig(filename)
                plot.close()

#creating nn

nn = NeuralNetwork(input_size=4, hidden_size=4, output_size=1)
nn.train(x_normalized,y_normalized, epochs=1000, learning_rate=0.001)



predicted_y_normalized = nn.forward(x_normalized)
predicted_y = denormalize(predicted_y_normalized, y_min, y_max)
# access metadata
#print(CCPP.metadata.uci_id)
#print("   ")

#print(CCPP.metadata.num_instances)
#print("   ")

#print(CCPP.metadata.additional_info.summary)
#print("   ")

# access variable info in tabular format
#print(CCPP.variables)
#print("   ")
