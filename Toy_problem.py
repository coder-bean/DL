import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math
import os
import imageio

# Function to create a directory for plots if it doesn't exist
output_dir = 'training_plots'
filenames = []
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# He Initialization
def he_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])

# Neural Network class without L2 Regularization
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_hidden_input = he_initialization((input_size, hidden_size))
        self.weights_hidden_1_2 = he_initialization((hidden_size, hidden_size))
        self.weights_hidden_output = he_initialization((hidden_size, output_size))
        self.bias_hidden_1 = np.zeros((1, hidden_size))
        self.bias_hidden_2 = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, x):
        self.hidden_layer_1_input = np.dot(x, self.weights_hidden_input) + self.bias_hidden_1
        self.hidden_layer_1_output = tanh(self.hidden_layer_1_input)
        
        self.hidden_layer_2_input = np.dot(self.hidden_layer_1_output, self.weights_hidden_1_2) + self.bias_hidden_2
        self.hidden_layer_2_output = tanh(self.hidden_layer_2_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_2_output, self.weights_hidden_output) + self.bias_output
        self.output = tanh(self.output_layer_input)
        
        return self.output

    def backward(self, x, y, learning_rate):
        output_error = y - self.output
        output_delta = output_error * ddxtanhx(self.output)

        hidden_2_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_2_delta = hidden_2_error * ddxtanhx(self.hidden_layer_2_output)

        hidden_1_error = hidden_2_delta.dot(self.weights_hidden_1_2.T)
        hidden_1_delta = hidden_1_error * ddxtanhx(self.hidden_layer_1_output)

        self.weights_hidden_output += self.hidden_layer_2_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_1_2 += self.hidden_layer_1_output.T.dot(hidden_2_delta) * learning_rate
        self.bias_hidden_2 += np.sum(hidden_2_delta, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_input += x.T.dot(hidden_1_delta) * learning_rate
        self.bias_hidden_1 += np.sum(hidden_1_delta, axis=0, keepdims=True) * learning_rate

    def train(self, x, y, x_val, y_val, epochs, learning_rate, batch_size):
        x = np.array(x)
        y = np.array(y)
        data_size = len(x)
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(data_size)
            x_shuffled = x[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(0, data_size, batch_size):
                end = i + batch_size if i + batch_size <= data_size else data_size
                batch_x = x_shuffled[i:end]
                batch_y = y_shuffled[i:end]

                self.forward(batch_x)
                self.backward(batch_x, batch_y, learning_rate)

            # Calculate training loss (using MSE)
            full_output = self.forward(x)
            train_loss = np.mean(np.abs((full_output-y) / (full_output+0.01)))
            training_losses.append(train_loss)

            # Calculate validation loss (using MAPE)
            val_output = self.forward(x_val)
            val_loss = np.mean(np.abs((val_output-y_val) / (val_output+0.01))) * 100  # Add a small constant to avoid division by zero
            validation_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Training Loss: {train_loss}, Validation Loss (MAPE): {val_loss}')
        return training_losses, validation_losses
# Define activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def ddxtanhx(x):
    return (1 / np.cosh(x) ** 2)

# Normalize function
def normalize(data, minim, maxim):
    minim = np.min(data)
    maxim = np.max(data)
    return (2 * data - (maxim + minim)) / (maxim - minim)

def denormalize(normalized, minim, maxim):
    return (normalized * (maxim - minim) + (maxim + minim)) / 2

# Prepare the sine wave data
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
y = np.sin(x)

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

x_normalized = normalize(x, x_min, x_max)
y_normalized = normalize(y, y_min, y_max)

# Split the data for validation
split_index = int(len(x_normalized) * 0.8)  # 80% for training, 20% for validation
x_train, x_val = x_normalized[:split_index], x_normalized[split_index:]
y_train, y_val = y_normalized[:split_index], y_normalized[split_index:]

# Initialize the neural network and train
nn = NeuralNetwork(input_size=1, hidden_size=4, output_size=1)
training_losses, validation_losses = nn.train(x_train, y_train, x_val, y_val, epochs=1000, learning_rate=0.001, batch_size=32)

# Plot training vs validation loss
plot.figure(figsize=(10, 6))
plot.plot(training_losses, label='Training Loss', color='blue')
plot.plot(validation_losses, label='Validation Loss', color='orange')
plot.title('Training vs Validation Loss')
plot.xlabel('Epochs')
plot.ylabel('Loss')
plot.legend()
plot.grid()
plot.show()
