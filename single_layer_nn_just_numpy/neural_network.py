from numpy import exp, array, random, dot
import pandas as pd

# This is a simple neural network implement logistic regression 
# https://www.youtube.com/watch?v=p69khggr1Jo&index=3&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3
class NeuralNetwork():
    def __init__(self, k):
        # seed to generate the same numbers, good for debugging
        random.seed(1)

        # We model a single neuron, with k input connections and 1 output connection.
        # We assign random weights to a k x 1 matrix, with values in the range -1 to 1 and mean 0.
        self.synaptic_weigths = 2 * random.random((k, 1)) - 1

    # activation function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_labels, number_of_iterations):
        for iteration in xrange(number_of_iterations):
            # pass the training set through the neural net
            predicted_label = self.predict(training_set_inputs)
            # calculate the error
            error = training_set_labels - predicted_label
            # multiply the error by the input and again by the gradient of the sigmoid curve
            adjusment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(predicted_label))
            # update weights (backpropagation)
            self.synaptic_weigths += adjusment

    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weigths))
