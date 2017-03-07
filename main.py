from numpy import array

from dataset import Dataset
from single_layer_numpy_nn_classifier.neural_network import NeuralNetwork
from single_layer_tensorflow_nn_classifier.neural_network import SingleLayerTensorFlowNeuralNetwork

def main():
    print "Loading data"
    dataset = Dataset()
    dataset.load("data/voicegender.csv")
    dataset.prepare_for_learning()

    selected_example = show_menu_options()
    if selected_example == 1:
        example_1(dataset)
    elif selected_example == 2:
        example_2(dataset)


def show_menu_options():
    menu_options = [
        "Simple Neural Network",
        "Single layer TensorFlow NeuralNetwork Classifier",
        "Two layer TensorFlow NeuraNetwork Classifier",
        "Applying convolutions"
    ]
    for i in range(len(menu_options)):
        print i+1, menu_options[i]
    print "\n"

    selected_example = -1
    try:
        selected_example = int(raw_input('Select Example to run: '))
    except ValueError:
        print "Invalid option"
        exit(1)
    return selected_example


def example_1(dataset):
    neural_network = NeuralNetwork(20)
    # print "Random initial synaptic weights"
    # print neural_network.synaptic_weigths

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(dataset.trainset, dataset.trainset_labels, 40000)

    # print "New learned synaptic weights"
    # print neural_network.synaptic_weigths

    # Test the neural network with a new unseen example
    print "Score", neural_network.score(dataset.testset, dataset.testset_labels)

def example_2(dataset):
    num_samples, num_features = dataset.trainset.shape
    (_, num_classes) = dataset.trainset_labels.shape

    tensorflow_nn = SingleLayerTensorFlowNeuralNetwork()
    tensorflow_nn.create_model(num_samples, num_features, num_classes)
    tensorflow_nn.train(dataset.trainset, dataset.trainset_labels)
    tensorflow_nn.score(dataset.testset, dataset.testset_labels)



if __name__ == '__main__':
    main()

