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

combined_cycle_power_plant = fetch_ucirepo(id=294) 


# Convert pandas DataFrame to NumPy array
x = combined_cycle_power_plant.data.features.to_numpy()
y = combined_cycle_power_plant.data.targets.to_numpy()

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

# Normalize data
x_normalized = normalize(x, x_min, x_max)
y_normalized = normalize(y, y_min, y_max)

# Dividing data into training, validation, and test sets
test_x = x_normalized[8612:]
test_y = y_normalized[8612:]
train_x, train_y, val_x, val_y = [], [], [], []

for i in range(8612):
    if i % 5 == 0:
        val_x.append(x_normalized[i])
        val_y.append(y_normalized[i])
    else:
        train_x.append(x_normalized[i])
        train_y.append(y_normalized[i])

train_x = np.array(train_x)
train_y = np.array(train_y)
val_x = np.array(val_x)
val_y = np.array(val_y)

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

# Neural Network class with updated forward and backward methods
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_hidden_input = np.random.randn(input_size, hidden_size)
        self.weights_hidden_1_2 = np.random.randn(hidden_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_1 = np.random.randn(1, hidden_size)
        self.bias_hidden_2 = np.random.randn(1, hidden_size)
        self.bias_output = np.random.randn(1, output_size)

    def forward(self, x):
        # First hidden layer
        self.hidden_layer_1_input = np.dot(x, self.weights_hidden_input) + self.bias_hidden_1
        self.hidden_layer_1_output = tanh(self.hidden_layer_1_input)
        
        # Second hidden layer, updated with the correct input from the first hidden layer
        self.hidden_layer_2_input = np.dot(self.hidden_layer_1_output, self.weights_hidden_1_2) + self.bias_hidden_2
        self.hidden_layer_2_output = tanh(self.hidden_layer_2_input)
        
        # Output layer
        self.output_layer_input = np.dot(self.hidden_layer_2_output, self.weights_hidden_output) + self.bias_output
        self.output = tanh(self.output_layer_input)
        
        return self.output

    def backward(self, x, y, learning_rate):
        # Error and delta calculations
        output_error = y - self.output
        output_delta = output_error * ddxtanhx(self.output)

        hidden_2_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_2_delta = hidden_2_error * ddxtanhx(self.hidden_layer_2_output)

        hidden_1_error = hidden_2_delta.dot(self.weights_hidden_1_2.T)
        hidden_1_delta = hidden_1_error * ddxtanhx(self.hidden_layer_1_output)

        # Weight and bias updates
        self.weights_hidden_output += self.hidden_layer_2_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_1_2 += self.hidden_layer_1_output.T.dot(hidden_2_delta) * learning_rate
        self.bias_hidden_2 += np.sum(hidden_2_delta, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_input += x.T.dot(hidden_1_delta) * learning_rate
        self.bias_hidden_1 += np.sum(hidden_1_delta, axis=0, keepdims=True) * learning_rate

    def train(self, x, y, epochs, learning_rate):
        filenames = []
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y, learning_rate)
            
            loss = np.mean(np.square(y - self.output))
            print(f'Epoch {epoch}: Loss = {loss}')
            
            if epoch % 10 == 0:
                plot.plot(x, y, label='Actual')
                plot.plot(x, self.output, label=f'Prediction at epoch {epoch}')
                plot.legend(loc='lower left')
                filename = f'training_plots/epoch_{epoch}.png'
                filenames.append(filename)
                plot.savefig(filename)
                plot.close()

# Create the neural network and train it
nn = NeuralNetwork(input_size=4, hidden_size=4, output_size=1)
nn.train(train_x, train_y, epochs=1000, learning_rate=0.001)

# Predict and denormalize the results
predicted_y_normalized = nn.forward(test_x)
predicted_y = denormalize(predicted_y_normalized, y_min, y_max)
