import numpy as np
from nn_from_scratch.activation import Activation

class Layer:
     def __init__(self, input_size, output_size, activation):
          self.weights = np.random.rand(input_size, output_size) * 0.1
          self.biases = np.zeros((1, output_size))

          if activation == 'sigmoid':
               self.activation = Activation.sigmoid
               self.activation_derivative = Activation.sigmoid_derivative

          elif activation == 'relu':
               self.activation = Activation.relu
               self.activation_derivative = Activation.relu_derivative
          else:
               raise ValueError("Unsupported activation function")
          
     def forward(self, input_data):
          self.input = input_data
          self.z = np.dot(self.input, self.weights) + self.biases
          self.output = self.activation(self.z)
          return self.output
     
     def backward(self, output_error, learning_rate):
          delta = output_error * self.activation_derivative(self.output)
          input_error = np.dot(delta, self.weights.T)
          weights_error = np.dot(self.input.T, delta)
          self.weights -= learning_rate * weights_error
          self.biases -= learning_rate * np.sum(delta, axis = 0, keepdims = True)
          return input_error