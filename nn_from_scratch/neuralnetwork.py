import numpy as np
from nn_from_scratch.layer import Layer

class NeuralNetwork:
     def __init__(self):
          self.layers = []

     def add_layer(self, layer):
          self.layers.append(layer)

     def forward(self, X):
          output = X
          for layer in self.layers:
               output = layer.forward(output)
          return output
     
     def backward(self, output_error, learning_rate):
          error = output_error
          for layer in reversed(self.layers):
               error = layer.backward(error, learning_rate)

     def train(self, X, y, epochs, learning_rate):
          for epoch in range(epochs):
               output = self.forward(X)
               loss = np.mean((y - output) ** 2)
               output_error = 2 * (output - y) / y.size
               self.backward(output_error, learning_rate)
               if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss: .6f}")