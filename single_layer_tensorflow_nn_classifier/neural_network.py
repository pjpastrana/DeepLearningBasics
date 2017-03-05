import tensorflow as tf

# This is a simple neural network implement logistic regression using tensor flow
# y = Wx + b
class SingleLayerTensorFlowNeuralNetwork(object):
    def __init__(self):
        super(SingleLayerTensorFlowNeuralNetwork, self).__init__()
        # model arguments and functions
        self.x = None
        self.W = None
        self.b = None
        self.y_values = None
        self.y = None
        self.y_ = None
        self.cost = None
        self.optimizer = None


    def create_model(self, num_samples, num_features, num_classes):
        # Okay TensorFlow, we'll feed you an array of examples. Each example will
        # be an array of num_features float values.
        # "None" means we can feed you any number of examples
        # Notice we haven't fed it the values yet
        self.x = tf.placeholder(tf.float32, [None, num_features])

        # Maintain a num_features x num_classes float matrix for the weights that we'll keep updating 
        # through the training process (make them all zero to begin with)
        self.W = tf.Variable(tf.zeros([num_features, num_classes]))

        # Also maintain num_classes bias values
        self.b = tf.Variable(tf.zeros([num_classes]))

        # The first step in calculating the prediction would be to multiply
        # the inputs matrix by the weights matrix then add the biases
        self.y_values = tf.add(tf.matmul(self.x, self.W), self.b)

        # Then we use softmax as an "activation function" that translates the
        # numbers outputted by the previous layer into probability form
        self.y = tf.nn.softmax(self.y_values)
        
        # For training purposes, we'll also feed a matrix of labels
        self.y_ = tf.placeholder(tf.float32, [None, num_classes])

        # Cost function: Mean squared error
        self.cost = tf.reduce_sum(tf.pow(self.y_ - self.y, 2))/(2*num_samples)
        
        # Gradient descent
        learning_rate = 0.000001
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

    def train(self, inputX, inputY):
        # Initialize variables and tensorflow session
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        training_epochs = 2000
        display_step = 50

        for i in range(training_epochs):
            # Take a gradient descent step using our inputs and labels
            sess.run(self.optimizer, feed_dict={self.x: inputX, self.y_: inputY})

            # That's all! The rest of the cell just outputs debug messages. 
            # Display logs per epoch step
            if (i) % display_step == 0:
                cc = sess.run(self.cost, feed_dict={self.x: inputX, self.y_:inputY})
                print "Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc) #, \"W=", sess.run(W), "b=", sess.run(b)

        print "Optimization Finished!"
        training_cost = sess.run(self.cost, feed_dict={self.x: inputX, self.y_: inputY})
        print "Training cost=", training_cost, "W=", sess.run(self.W), "b=", sess.run(self.b), '\n'