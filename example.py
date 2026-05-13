from nn_from_scratch.neuralnetwork import NeuralNetwork
from nn_from_scratch.layer import Layer
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork()
nn.add_layer(Layer(2, 4, activation = 'relu'))
nn.add_layer(Layer(4, 1, activation = 'sigmoid'))

nn.train(X, y, epochs = 5000, learning_rate = 0.1)
print("Predictions:\n", nn.forward(X))