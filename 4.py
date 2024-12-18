import numpy as np

# Normalize input
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input dataset (XOR)
x = x / np.amax(x, axis=0)

# Target outputs for XOR
y = np.array([[0], [1], [1], [0]])

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize parameters
input_layer_neurons = 2  # Number of input features
hidden_layer_neurons = 3  # Number of neurons in hidden layer
output_layer_neurons = 1  # Output neuron (binary classification)

# Random initialization of weights and biases
np.random.seed(1)
wb = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bb = np.random.uniform(size=(1, hidden_layer_neurons))
wout = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
bout = np.random.uniform(size=(1, output_layer_neurons))

# Learning rate
lr = 0.1

# Training for a number of epochs
epochs = 5000
for i in range(epochs):
    # Feedforward
    hinp = np.dot(x, wb) + bb
    hlayer_act = sigmoid(hinp)
    outinp = np.dot(hlayer_act, wout) + bout
    output = sigmoid(outinp)

    # Backpropagation
    EO = y - output
    outgrad = sigmoid_derivative(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)
    hiddengrad = sigmoid_derivative(hlayer_act)

    # Update weights and biases
    wout += hlayer_act.T.dot(d_output) * lr
    wb += x.T.dot(EH * hiddengrad) * lr
    bb += np.sum(EH * hiddengrad, axis=0) * lr
    bout += np.sum(d_output, axis=0) * lr

# Print the results after training
print("Input:\n", x)
print("Actual Output:\n", y)
print("Predicted Output after training:\n", output)
