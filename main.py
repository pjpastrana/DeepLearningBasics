from numpy import array

from dataset import Dataset
from single_layer_numpy_nn_classifier.neural_network import NeuralNetwork
from single_layer_tensorflow_nn_classifier.neural_network import SingleLayerTensorFlowNeuralNetwork
from two_layer_tensorflow_nn_classifier.neural_network import TwoLayerTensorFlowNeuralNetwork
import sys

sys.stdout.flush()

def main():
    dataset = Dataset()
    dataset.load("data/voicegender.csv")
    dataset.prepare_for_learning()
    
    # show_menu_options, and execute selected
    menu_options = [
        ["Single layer TensorFlow NeuralNetwork Classifier", single_nn],
        ["Two layer TensorFlow NeuraNetwork Classifier", two_nn],
        ["Applying convolutions", tflearn_conv]
    ]
    for i in range(len(menu_options)):
        print i, menu_options[i][0]
    print "\n"

    selected_example = -1
    try:
        selected_example = int(raw_input('Select Example to run: '))
        if selected_example < 0 or selected_example > len(menu_options):
            raise ValueError
        # call the selected function that would run the Deep Learning example
        menu_options[selected_example][1](dataset)
    except ValueError:
        print "Invalid option"
        exit(1)


# def simple_nn(dataset):
#     neural_network = NeuralNetwork(20)
#     # print "Random initial synaptic weights"
#     # print neural_network.synaptic_weigths

#     # Train the neural network using a training set.
#     # Do it 10,000 times and make small adjustments each time.
#     neural_network.train(dataset.trainset, dataset.trainset_labels, 40000)

#     # print "New learned synaptic weights"
#     # print neural_network.synaptic_weigths

#     # Test the neural network with a new unseen example
#     print "Score", neural_network.score(dataset.testset, dataset.testset_labels)

def single_nn(dataset):
    num_samples, num_features = dataset.trainset.shape
    (_, num_classes) = dataset.trainset_labels.shape

    tensorflow_nn = SingleLayerTensorFlowNeuralNetwork()
    tensorflow_nn.create_model(num_samples, num_features, num_classes)
    tensorflow_nn.train(dataset.trainset, dataset.trainset_labels)
    tensorflow_nn.score(dataset.testset, dataset.testset_labels)

def two_nn(dataset):
    num_samples, num_features = dataset.trainset.shape
    (_, num_classes) = dataset.trainset_labels.shape

    tensorflow_nn = TwoLayerTensorFlowNeuralNetwork()
    tensorflow_nn.create_model(num_samples, num_features, num_classes)
    tensorflow_nn.train(dataset.trainset, dataset.trainset_labels)
    tensorflow_nn.score(dataset.testset, dataset.testset_labels)

def tflearn_conv(dataset):
    print "tflearn_conv"



if __name__ == '__main__':
    main()

