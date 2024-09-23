import numpy as np
import matplotlib.pyplot as plot
import os
import imageio
import imageio.v2 as imageio
from ucimlrepo import fetch_ucirepo

# Setup for saving plots
output_dir = 'training_plots'
filenames = []

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Fetch the dataset
combined_cycle_power_plant = fetch_ucirepo(id=294)

# Use data from the dataset
X = combined_cycle_power_plant.data.features
y = combined_cycle_power_plant.data.targets

# Neural Network with 2 hidden layers
class NeuralNetwork:
    # Initialize weights and biases
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.weights_hidden_input = np.random.randn(input_size, hidden_size1)
        self.weights_hidden_hidden = np.random.randn(hidden_size1, hidden_size2)
        self.weights_hidden_output = np.random.randn(hidden_size2, output_size)
        
        self.bias_hidden1 = np.random.randn(1, hidden_size1)
        self.bias_hidden2 = np.random.randn(1, hidden_size2)
        self.bias_output = np.random.randn(1, output_size)

    # Forward pass
    def forward(self, x):
        self.hidden_layer_input1 = np.dot(x, self.weights_hidden_input) + self.bias_hidden1
        self.hidden_layer_output1 = tanh(self.hidden_layer_input1)
        
        self.hidden_layer_input2 = np.dot(self.hidden_layer_output1, self.weights_hidden_hidden) + self.bias_hidden2
        self.hidden_layer_output2 = tanh(self.hidden_layer_input2)
        
        self.output_layer_input = np.dot(self.hidden_layer_output2, self.weights_hidden_output) + self.bias_output
        self.output = tanh(self.output_layer_input)
        
        return self.output

    # Backward pass (gradient descent)
    def backward(self, x, y, learning_rate):
        # Output layer error
        output_error = y - self.output
        output_delta = output_error * ddxtanhx(self.output)
        
        # Hidden layer 2 error
        hidden_error2 = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta2 = hidden_error2 * ddxtanhx(self.hidden_layer_output2)
        
        # Hidden layer 1 error
        hidden_error1 = hidden_delta2.dot(self.weights_hidden_hidden.T)
        hidden_delta1 = hidden_error1 * ddxtanhx(self.hidden_layer_output1)

        # Weight and bias updates
        self.weights_hidden_output += self.hidden_layer_output2.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_hidden_hidden += self.hidden_layer_output1.T.dot(hidden_delta2) * learning_rate
        self.bias_hidden2 += np.sum(hidden_delta2, axis=0, keepdims=True) * learning_rate
        
        self.weights_hidden_input += x.T.dot(hidden_delta1) * learning_rate
        self.bias_hidden1 += np.sum(hidden_delta1, axis=0, keepdims=True) * learning_rate

    # Training loop
    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y, learning_rate)
            loss = np.mean(np.square(y - self.output))
            print(f'Epoch {epoch}, Loss: {loss}')
            
            # Save plot at regular intervals
            if epoch % 10 == 0:
                plot.scatter(x[:, 0], y, label='True', color='blue')
                plot.scatter(x[:, 0], self.output, label=f'Prediction at epoch {epoch}', color='red')
                plot.legend(loc='lower left')
                filename = f'{output_dir}/epoch_{epoch}.png'
                filenames.append(filename)
                plot.savefig(filename)
                plot.close()

# Normalization function
def normalize(data, actual_min, actual_max):
    virtual_min = actual_min - 0.05 * (actual_max - actual_min)
    virtual_max = actual_max + 0.05 * (actual_max - actual_min)
    return 1.8 * (data - virtual_min) / (virtual_max - virtual_min) - 0.9

# Denormalization function
def denormalize(data, actual_min, actual_max):
    virtual_min = actual_min - 0.05 * (actual_max - actual_min)
    virtual_max = actual_max + 0.05 * (actual_max - actual_min)
    return (data + 0.9) * (virtual_max - virtual_min) / 1.8 + virtual_min

# Convert the pandas dataframe to numpy arrays
X = X.values
y = y.values

# Normalizing the data
x_min, x_max = np.min(X), np.max(X)
y_min, y_max = np.min(y), np.max(y)

X_normalized = normalize(X, x_min, x_max)
y_normalized = normalize(y, y_min, y_max)

# Define activation functions and derivatives
def tanh(x):
    return np.tanh(x)

def ddxtanhx(x):
    return 1 - np.tanh(x) ** 2

# Create and train the neural network with 2 hidden layers
input_size = X.shape[1]
nn = NeuralNetwork(input_size=input_size, hidden_size1=8, hidden_size2=6, output_size=1)
nn.train(X_normalized, y_normalized, epochs=100, learning_rate=0.01)

# Predict using the trained model
predicted_y_normalized = nn.forward(X_normalized)
predicted_y = denormalize(predicted_y_normalized, y_min, y_max)

# Create a GIF from the saved images
with imageio.get_writer('power_plant_training_2_hidden_layers.gif', mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up images after creating the GIF
for filename in filenames:
    os.remove(filename)

print("Training GIF saved as 'power_plant_training_2_hidden_layers.gif'.")
