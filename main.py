from numpy import array

from dataset import Dataset
from single_layer_nn_just_numpy.neural_network import NeuralNetwork

def main():
    dataset = Dataset()
    dataset.load("data/voicegender.csv")
    dataset.prepare_for_learning()

    neural_network = NeuralNetwork(20)
    print "Random initial synaptic weights"
    print neural_network.synaptic_weigths

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(dataset.trainset, dataset.trainset_labels, 10000)

    print "New learned synaptic weights"
    print neural_network.synaptic_weigths

    # Test the neural network with a new unseen example
    print neural_network.predict(dataset.testset[0, :])
    print dataset.testset_labels[0]


if __name__ == '__main__':
    main()

