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

    def train(self, x, y, epochs, learning_rate, batch_size):
        # Ensure that x and y are NumPy arrays
        x = np.array(x)
        y = np.array(y)

        data_size = len(x)
        for epoch in range(epochs):
            # Shuffle the data at the start of each epoch (shuffling rows)
            shuffled_indices = np.random.permutation(data_size)
            x_shuffled = x[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(0, data_size, batch_size):
                end = i + batch_size if i + batch_size <= data_size else data_size
                batch_x = x_shuffled[i:end]
                batch_y = y_shuffled[i:end]

                # Forward and backward propagation for the batch
                self.forward(batch_x)
                self.backward(batch_x, batch_y, learning_rate)

            # Calculate and print loss for the epoch
            self.forward(x)
            loss = np.mean(np.square(y - self.output))
            print(f'Loss at epoch {epoch}: {loss}')

            if epoch % 10 == 0:
                plot.plot(x, y, label='CCPP')
                plot.plot(x, self.output, label=f'Approximation at epoch {epoch}')
                if not os.path.exists('tplots'):
                    os.makedirs('tplots')

                filename = f'tplots/epoch_{epoch}.png'
                plot.legend(loc='lower left')
                filenames.append(filename)
                plot.savefig(filename)
                plot.close()


batch_sizes = [1, 64, 256, len(train_x)]
for batch_size in batch_sizes:
    print(f"\nTraining with batch size: {batch_size}")
    nn = NeuralNetwork(input_size=4, hidden_size=6, output_size=1)
    nn.train(x_normalized, y_normalized, epochs=1000, learning_rate=0.001, batch_size=batch_size)

# Predict and denormalize the results
predicted_y_normalized = nn.forward(test_x)
predicted_y = denormalize(predicted_y_normalized, y_min, y_max)

with imageio.get_writer('ccpp.gif', mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up images after creating the GIF
for filename in filenames:
    os.remove(filename)
