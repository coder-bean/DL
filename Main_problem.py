import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math
import os
import imageio

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
combined_cycle_power_plant = fetch_ucirepo(id=294) 
  
# data (as pandas dataframes) 
x = combined_cycle_power_plant.data.features 
y = combined_cycle_power_plant.data.targets 



# Function to create a directory for plots if it doesn't exist
output_dir = 'training_plots'
filenames = []
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# He Initialization
def he_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lambda_reg=0.01, beta=0.9):
        self.weights_hidden_input = he_initialization((input_size, hidden_size))
        self.weights_hidden_1_2 = he_initialization((hidden_size, hidden_size))
        self.weights_hidden_output = he_initialization((hidden_size, output_size))
        self.bias_hidden_1 = np.zeros((1, hidden_size))
        self.bias_hidden_2 = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.lambda_reg = lambda_reg
        self.beta = beta  # Momentum coefficient

        # Initialize momentum terms for weights and biases
        self.v_weights_hidden_input = np.zeros_like(self.weights_hidden_input)
        self.v_weights_hidden_1_2 = np.zeros_like(self.weights_hidden_1_2)
        self.v_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.v_bias_hidden_1 = np.zeros_like(self.bias_hidden_1)
        self.v_bias_hidden_2 = np.zeros_like(self.bias_hidden_2)
        self.v_bias_output = np.zeros_like(self.bias_output)


    def forward(self, x):
        self.hidden_layer_1_input = np.dot(x, self.weights_hidden_input) + self.bias_hidden_1
        self.hidden_layer_1_output = tanh(self.hidden_layer_1_input)
        
        self.hidden_layer_2_input = np.dot(self.hidden_layer_1_output, self.weights_hidden_1_2) + self.bias_hidden_2
        self.hidden_layer_2_output = tanh(self.hidden_layer_2_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_2_output, self.weights_hidden_output) + self.bias_output
        self.output = logistic(self.output_layer_input)
        
        return self.output

    def backward(self, x, y, learning_rate, lambda_reg=0.01):
        output_error = y - self.output
        output_delta = output_error * ddxlogistic(self.output)

        hidden_2_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_2_delta = hidden_2_error * ddxtanhx(self.hidden_layer_2_output)

        hidden_1_error = hidden_2_delta.dot(self.weights_hidden_1_2.T)
        hidden_1_delta = hidden_1_error * ddxtanhx(self.hidden_layer_1_output)

        # Calculate updates for weights and biases with L2 regularization and momentum
        # Update weights_hidden_output
        grad_weights_hidden_output = self.hidden_layer_2_output.T.dot(output_delta) - lambda_reg * self.weights_hidden_output
        self.v_weights_hidden_output = self.beta * self.v_weights_hidden_output + (1 - self.beta) * grad_weights_hidden_output
        self.weights_hidden_output += self.v_weights_hidden_output * learning_rate

        grad_bias_output = np.sum(output_delta, axis=0, keepdims=True)
        self.v_bias_output = self.beta * self.v_bias_output + (1 - self.beta) * grad_bias_output
        self.bias_output += self.v_bias_output * learning_rate

        # Update weights_hidden_1_2
        grad_weights_hidden_1_2 = self.hidden_layer_1_output.T.dot(hidden_2_delta) - lambda_reg * self.weights_hidden_1_2
        self.v_weights_hidden_1_2 = self.beta * self.v_weights_hidden_1_2 + (1 - self.beta) * grad_weights_hidden_1_2
        self.weights_hidden_1_2 += self.v_weights_hidden_1_2 * learning_rate

        grad_bias_hidden_2 = np.sum(hidden_2_delta, axis=0, keepdims=True)
        self.v_bias_hidden_2 = self.beta * self.v_bias_hidden_2 + (1 - self.beta) * grad_bias_hidden_2
        self.bias_hidden_2 += self.v_bias_hidden_2 * learning_rate

        # Update weights_hidden_input
        grad_weights_hidden_input = x.T.dot(hidden_1_delta) - lambda_reg * self.weights_hidden_input
        self.v_weights_hidden_input = self.beta * self.v_weights_hidden_input + (1 - self.beta) * grad_weights_hidden_input
        self.weights_hidden_input += self.v_weights_hidden_input * learning_rate

        grad_bias_hidden_1 = np.sum(hidden_1_delta, axis=0, keepdims=True)
        self.v_bias_hidden_1 = self.beta * self.v_bias_hidden_1 + (1 - self.beta) * grad_bias_hidden_1
        self.bias_hidden_1 += self.v_bias_hidden_1 * learning_rate


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

            # Calculate training loss using MAPE
            full_output = self.forward(x)
            train_loss = np.mean(np.abs((y - full_output) / np.where(np.abs(y) > 0.1000000000, y, 1))) * 100
            training_losses.append(train_loss)
            

            # Calculate validation loss (using MAPE)
            val_output = self.forward(x_val)
            val_loss = np.mean(np.abs((y_val - val_output) / np.where(np.abs(y_val) > 0.1000000000, y_val, 1))) * 100
            validation_losses.append(val_loss)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Training MAPE: {train_loss}, Validation MSE: {val_loss}')
                plot.scatter(x[:, 0], y[:, 0], label='True Data', alpha=0.6)
                plot.scatter(x[:,0], full_output, label=f'Approximation at epoch {epoch}', color='red')
                if not os.path.exists('plots'):
                    os.makedirs('plots')

                filename = f'plots/epoch_{epoch}.png'
                plot.legend(loc='lower left')
                filenames.append(filename)
                plot.savefig(filename)
                plot.close()
        return training_losses, validation_losses

# Define activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def ddxtanhx(x):
    return (1 / np.cosh(x) ** 2)
def logistic(x):
    return 1 / (1 + np.exp(-x))

def ddxlogistic(x):
    log = logistic(x)
    return log * (1 - log)
def relu(x):
    return np.maximum(0, x)

def ddxrelu(x):
    return np.where(x > 0, 1, 0)

# Normalize function
def normalize(data):
    col_min_max = {}
    if isinstance(data, pd.DataFrame):
        normalized_data = data.apply(lambda col: (2 * col - (col.max() + col.min())) / (col.max() - col.min()), axis=0)
        col_min_max = {col: (data[col].min(), data[col].max()) for col in data.columns}
        return normalized_data, col_min_max
    elif isinstance(data, np.ndarray):
        normalized_data = np.zeros_like(data)
        col_min_max = {}
        for i in range(data.shape[1]):
            col = data[:, i]
            maxim, minim = col.max(), col.min()
            normalized_data[:, i] = (2 * col - (maxim + minim)) / (maxim - minim)
            col_min_max[i] = (minim, maxim)
        return normalized_data, col_min_max
    else:
        raise TypeError("Input data must be a pandas DataFrame or numpy array.")

def denormalize(normalized_data, col_min_max):
    if isinstance(normalized_data, pd.DataFrame):
        denormalized_data = normalized_data.apply(
            lambda col: (col * (col_min_max[col.name][1] - col_min_max[col.name][0]) + 
                         (col_min_max[col.name][1] + col_min_max[col.name][0])) / 2, axis=0)
        return denormalized_data
    elif isinstance(normalized_data, np.ndarray):
        denormalized_data = np.zeros_like(normalized_data)
        for i in range(normalized_data.shape[1]):
            minim, maxim = col_min_max[i]
            denormalized_data[:, i] = (normalized_data[:, i] * (maxim - minim) + (maxim + minim)) / 2
        return denormalized_data
    else:
        raise TypeError("Input data must be a pandas DataFrame or numpy array.")
# Normalization function to scale data between 0 and 1
def log_normalize(data):
    col_min_max = {}
    if isinstance(data, pd.DataFrame):
        normalized_data = data.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
        col_min_max = {col: (data[col].min(), data[col].max()) for col in data.columns}
        return normalized_data, col_min_max
    elif isinstance(data, np.ndarray):
        normalized_data = np.zeros_like(data)
        col_min_max = {}
        for i in range(data.shape[1]):
            col = data[:, i]
            minim, maxim = col.min(), col.max()
            normalized_data[:, i] = (col - minim) / (maxim - minim)
            col_min_max[i] = (minim, maxim)
        return normalized_data, col_min_max
    else:
        raise TypeError("Input data must be a pandas DataFrame or numpy array.")

# Denormalization function to revert data to original scale
def log_denormalize(normalized_data, col_min_max):
    if isinstance(normalized_data, pd.DataFrame):
        denormalized_data = normalized_data.apply(
            lambda col: col_min_max[col.name][0] + col * (col_min_max[col.name][1] - col_min_max[col.name][0]), axis=0)
        return denormalized_data
    elif isinstance(normalized_data, np.ndarray):
        denormalized_data = np.zeros_like(normalized_data)
        for i in range(normalized_data.shape[1]):
            minim, maxim = col_min_max[i]
            denormalized_data[:, i] = minim + normalized_data[:, i] * (maxim - minim)
        return denormalized_data
    else:
        raise TypeError("Input data must be a pandas DataFrame or numpy array.")

x_normalized, x_min_max=normalize(x)
y_normalized, y_min_max=log_normalize(y)
test_x= x_normalized[8612:]
test_y = y_normalized[8612:]
train_x, train_y, val_x, val_y = [], [], [], []

for i in range(8612):
    if i % 5 == 0:
        val_x.append(x_normalized.iloc[i])
        val_y.append(y_normalized.iloc[i])
    else:
        train_x.append(x_normalized.iloc[i])
        train_y.append(y_normalized.iloc[i])

train_x = np.array(train_x)
train_y = np.array(train_y)
val_x = np.array(val_x)
val_y = np.array(val_y)


# Initialize the neural network and train
nn = NeuralNetwork(input_size=4, hidden_size=8, output_size=1)
training_losses, validation_losses = nn.train(train_x, train_y, val_x, val_y, epochs=1000, learning_rate=0.0001, batch_size=64)


# Plot training vs validation loss
plot.figure(figsize=(10, 6))
plot.plot(training_losses, label='Training MAPE', color='blue')
plot.plot(validation_losses, label='Validation MAPE', color='orange')
plot.title('Training vs Validation MAPE')
plot.xlabel('Epochs')
plot.ylabel('MAPE (%)')
plot.legend()
plot.grid()
plot.show()

# Create a GIF of the training process
with imageio.get_writer('ccpp.gif', mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


# Remove plot images
for filename in filenames:
    os.remove(filename)
