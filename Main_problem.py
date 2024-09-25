from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math
import os
import imageio

filenames = []

# Normalize and Denormalize functions
def normalization(arr):
    xmin = np.min(arr)
    xmax = np.max(arr)

    xmin_ = xmin - (xmax - xmin) * 0.05
    xmax_ = xmax + (xmax - xmin) * 0.05
    
    X = (2 * arr - (xmax_ + xmin_)) / (xmax_ - xmin_)
    
    return X

def denormalization(X, arr):
    xmin = np.min(arr)
    xmax = np.max(arr)

    xmin_ = xmin - (xmax - xmin) * 0.05
    xmax_ = xmax + (xmax - xmin) * 0.05

    arr_original = ((X * (xmax_ - xmin_)) + (xmax_ + xmin_)) / 2

    return arr_original

# Fetch dataset
combined_cycle_power_plant = fetch_ucirepo(id=294)

# Convert pandas DataFrame to NumPy array
x = combined_cycle_power_plant.data.features.to_numpy()
y = combined_cycle_power_plant.data.targets.to_numpy()

# Normalize data
x_normalized = normalization(x)
y_normalized = normalization(y)

# Split data into training, validation, and test sets
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

# Activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def ddxtanhx(x):
    return 1 - np.tanh(x) ** 2

def logistic(x):
    return 1 / (1 + np.exp(-x))

def ddxlogisticx(x):
    return x * (1 - x)

# He initialization for weights
def he_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_hidden_input = he_initialization((input_size, hidden_size))
        self.weights_hidden_1_2 = he_initialization((hidden_size, hidden_size))
        self.weights_hidden_output = he_initialization((hidden_size, output_size))
        self.bias_hidden_1 = np.zeros((1, hidden_size))
        self.bias_hidden_2 = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, x):
        # First hidden layer
        self.hidden_layer_1_input = np.dot(x, self.weights_hidden_input) + self.bias_hidden_1
        self.hidden_layer_1_output = tanh(self.hidden_layer_1_input)
        
        # Second hidden layer
        self.hidden_layer_2_input = np.dot(self.hidden_layer_1_output, self.weights_hidden_1_2) + self.bias_hidden_2
        self.hidden_layer_2_output = tanh(self.hidden_layer_2_input)
        
        # Output layer
        self.output_layer_input = np.dot(self.hidden_layer_2_output, self.weights_hidden_output) + self.bias_output
        self.output = logistic(self.output_layer_input)
        
        return self.output

    def backward(self, x, y, learning_rate):
        # Error and delta calculations
        output_error = y - self.output
        output_delta = output_error * ddxlogisticx(self.output)

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

    def train(self, x, y, epochs, learning_rate, batch_size):
        x = np.array(x)
        y = np.array(y)
        data_size = len(x)

        for epoch in range(epochs):
            # Shuffle the data at the start of each epoch
            shuffled_indices = np.random.permutation(data_size)
            x_shuffled = x[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            # Mini-batch gradient descent
            for i in range(0, data_size, batch_size):
                end = i + batch_size if i + batch_size <= data_size else data_size
                batch_x = x_shuffled[i:end]
                batch_y = y_shuffled[i:end]

                self.forward(batch_x)
                self.backward(batch_x, batch_y, learning_rate)

            # After each epoch, plot the full dataset predictions
            full_output = self.forward(x)
            loss = np.mean(np.square(y - full_output))
            print(f'Loss at epoch {epoch}: {loss}')

            if epoch % 100 == 0:
                plot.scatter(range(len(x)), y, label='True Data', color='blue', alpha=0.6)
                plot.plot(range(len(x)), full_output, label=f'Approximation at epoch {epoch}', color='red')
                if not os.path.exists('plots'):
                    os.makedirs('plots')

                filename = f'plots/epoch_{epoch}.png'
                plot.legend(loc='lower left')
                filenames.append(filename)
                plot.savefig(filename)
                plot.close()

# Training with different batch sizes
batch_sizes = [1, 64, 256, len(train_x)]
for batch_size in batch_sizes:
    print(f"\nTraining with batch size: {batch_size}")
    nn = NeuralNetwork(input_size=4, hidden_size=4, output_size=1)
    nn.train(train_x, train_y, epochs=1001, learning_rate=0.001, batch_size=batch_size)

# Predict and denormalize the results
predicted_y_normalized = nn.forward(test_x)
predicted_y = denormalization(predicted_y_normalized, test_y)

# Create a GIF from saved plots
with imageio.get_writer('ccpp.gif', mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up images after creating the GIF
for filename in filenames:
    os.remove(filename)
