import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[0], [1], [1], [0]])


np.random.seed(42)
weights1 = np.random.uniform(size=(2, 4))
weights2 = np.random.uniform(size=(4, 1))


learning_rate = 0.5
epochs = 10000


for epoch in range(epochs):

    layer1_input = np.dot(input_data, weights1)
    layer1_output = sigmoid(layer1_input)

    output_layer_input = np.dot(layer1_output, weights2)
    predicted_output = sigmoid(output_layer_input)


    error = output_data - predicted_output


    d_predicted_output = error * sigmoid_derivative(predicted_output)
    d_weights2 = np.dot(layer1_output.T, d_predicted_output)

    d_layer1_output = d_predicted_output.dot(weights2.T) * sigmoid_derivative(layer1_output)
    d_weights1 = np.dot(input_data.T, d_layer1_output)


    weights1 += d_weights1 * learning_rate
    weights2 += d_weights2 * learning_rate


for i in range(len(input_data)):
    layer1_input = np.dot(input_data[i], weights1)
    layer1_output = sigmoid(layer1_input)
    output_layer_input = np.dot(layer1_output, weights2)
    predicted_output = sigmoid(output_layer_input)
    print(f"Input: {input_data[i]}, Predicted Output: {np.round(predicted_output[0])}")
