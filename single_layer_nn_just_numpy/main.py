from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # seed to generate the same numbers, good for debugging
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 3 matrix, with values in the range -1 to 1 and mean 0.
        self.synaptic_weigths = 2 * random.random((3, 1)) -1

    # activation function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_ouputs, number_of_iterations):
        for iteration in xrange(number_of_iterations):
            # pass the training set through the neural net
            output = self.predict(training_set_inputs)
            # calculate the error
            error = training_set_ouputs - output
            # multiply the error by the input and again by the gradient of the sigmoid curve
            adjusment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            # update weights (backpropagation)
            self.synaptic_weigths += adjusment

    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weigths))


def main():
    neural_network = NeuralNetwork()
    print "Random initial synaptic weights"
    print neural_network.synaptic_weigths

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ])
    training_set_ouputs = array([[0, 1, 1,0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_ouputs, 10000)

    print "New learnd synaptic weights"
    print neural_network.synaptic_weigths

    # Test the neural network with a new unseen example
    print "Considering new situation [1, 0, 0] -> ?"
    print neural_network.predict(array([1, 0, 0]))


if __name__ == '__main__':
    main()

