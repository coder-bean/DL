import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math
import os
import imageio

output_dir = 'training_plots'
filenames=[]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#steps 2 and 3: ANN Architecture, Backpropogation equations
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
                plot.scatter(range(len(x)), y, label='True Data', alpha=0.6)
                plot.plot(range(len(x)), full_output, label=f'Approximation at epoch {epoch}', color='red')
                if not os.path.exists('plots'):
                    os.makedirs('plots')

                filename = f'plots/epoch_{epoch}.png'
                plot.legend(loc='lower left')
                filenames.append(filename)
                plot.savefig(filename)
                plot.close()

# Training with different batch sizes


#step 6: I/O Normalization
def normalize(data, minim, maxim):
    minim=np.min(data)
    maxim=np.max(data)
    return (2*data - (maxim+minim))/(maxim-minim)

def denormalize(normalized, minim, maxim):
    return (normalized * (maxim - minim) + (maxim + minim)) / 2


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

batch_sizes = [1, 64, 256, len(x_normalized)]
for batch_size in batch_sizes:
    print(f"\nTraining with batch size: {batch_size}")
    nn = NeuralNetwork(input_size=1, hidden_size=4, output_size=1)
    nn.train(x_normalized,y_normalized, epochs=1000, learning_rate=0.001, batch_size=batch_size)

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
