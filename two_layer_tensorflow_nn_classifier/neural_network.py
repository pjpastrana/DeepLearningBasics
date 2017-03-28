import tensorflow as tf

# This is a simple neural network implement logistic regression using tensor flow
# y = Wx + b
class TwoLayerTensorFlowNeuralNetwork(object):
    def __init__(self):
        super(TwoLayerTensorFlowNeuralNetwork, self).__init__()
        # model arguments and functions
        self.x = None

        self.W_01 = None
        self.b_01 = None
        self.z = None
        self.W_12 = None
        self.b_12 = None

        self.y_values = None
        self.y = None
        self.y_ = None
        self.cost = None
        self.optimizer = None
        self.tfsession = None


    def create_model(self, num_samples, num_features, num_classes):
        # learning_rate = 0.000001
        learning_rate = 0.00001
        num_hidden_nodes = 1024

        # input layer
        self.x = tf.placeholder(tf.float32, [None, num_features])
        self.W_01 = tf.Variable(tf.zeros([num_features, num_hidden_nodes]))
        self.b_01 = tf.Variable(tf.zeros([num_hidden_nodes]))

        # hidden layer
        z = tf.add(tf.matmul(self.x, self.W_01), self.b_01)
        h = tf.nn.relu(z)

        # output layer
        self.W_12 = tf.Variable(tf.zeros([num_hidden_nodes, num_classes]))
        self.b_12 = tf.Variable(tf.zeros([num_classes]))

        self.y_values = tf.add(tf.matmul(h, self.W_12), self.b_12)

        # Then we use softmax as an "activation function" that translates the
        # numbers outputted by the previous layer into probability form
        self.y = tf.nn.softmax(self.y_values)
        
        # For training purposes, we'll also feed a matrix of labels
        self.y_ = tf.placeholder(tf.float32, [None, num_classes])

        # Cost function: Mean squared error
        self.cost = tf.reduce_sum(tf.pow(self.y_ - self.y, 2))/(2*num_samples)
        
        # Gradient descent
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

    def train(self, inputX, inputY):
        # Initialize variables and tensorflow session
        init = tf.initialize_all_variables()
        self.tfsession = tf.Session()
        self.tfsession.run(init)

        training_epochs = 100000
        display_step = 50

        for i in range(training_epochs):
            # Take a gradient descent step using our inputs and labels
            self.tfsession.run(self.optimizer, feed_dict={self.x: inputX, self.y_: inputY})

            # That's all! The rest of the cell just outputs debug messages. 
            # Display logs per epoch step
            if (i) % display_step == 0:
                cc = self.tfsession.run(self.cost, feed_dict={self.x: inputX, self.y_:inputY})
                print "Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc)

        print "Optimization Finished!"
        training_cost = self.tfsession.run(self.cost, feed_dict={self.x: inputX, self.y_: inputY})
        print "Training cost=", training_cost, "W=", self.tfsession.run(self.W_12), "b=", self.tfsession.run(self.b_12), '\n'

    # TODO: revisit
    def score(self, input, labels):
        classification_scores = self.tfsession.run(self.y, feed_dict={self.x:input})
        self.tfsession.close()
        classification_labels = classification_scores.argmax(1)
        print sum(labels[:, 0] - classification_labels)

