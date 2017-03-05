from numpy import array

from dataset import Dataset
from single_layer_nn_just_numpy.neural_network import NeuralNetwork

def main():
    dataset = Dataset()
    dataset.load("data/voicegender.csv")
    dataset.prepare_for_learning()

    neural_network = NeuralNetwork(20)
    # print "Random initial synaptic weights"
    # print neural_network.synaptic_weigths

    # # The training set. We have 4 examples, each consisting of 3 input values
    # # and 1 output value.
    training_set_inputs = array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ])
    training_set_labels = array([[0, 1, 1,0]]).T

    print "Random Trainset shape", training_set_inputs.shape
    print "Random Trainset labels shape", training_set_labels.shape

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    # neural_network.train(training_set_inputs, training_set_labels, 10000)
    neural_network.train(dataset.trainset, dataset.trainset_labels, 10000)

    # print "New learnd synaptic weights"
    # print neural_network.synaptic_weigths

    # Test the neural network with a new unseen example
    # print "Considering new situation [1, 0, 0] -> ?"
    # print neural_network.predict(array([1, 0, 0]))
    print neural_network.predict(dataset.testset[0, :])
    print dataset.testset_labels[0]


if __name__ == '__main__':
    main()

