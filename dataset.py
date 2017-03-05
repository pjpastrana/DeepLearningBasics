import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset(object):
    """docstring for Dataset"""
    def __init__(self):
        super(Dataset, self).__init__()
        self.raw_data = None
        self.trainset = None
        self.testset = None
        self.validation = None

    def load(self, filename):
        self.raw_data = pd.read_csv(filename)

    def prepare_for_learning(self):
        self.trainset, self.testset = train_test_split(self.raw_data, test_size = 0.2)
        self.trainset_labels = np.reshape(self.trainset["label"].as_matrix(), (len(self.trainset["label"]), 1))
        self.testset_labels = np.reshape(self.testset["label"].as_matrix(), (len(self.testset["label"]), 1) )
        self.trainset = self.trainset.drop("label", 1).as_matrix()
        self.testset = self.testset.drop("label", 1).as_matrix()

        print "Trainset shape", self.trainset.shape
        print "Trainset labels shape", self.trainset_labels.shape
        print "Testset shape", self.testset.shape
        print "Testset labels shape", self.testset_labels.shape