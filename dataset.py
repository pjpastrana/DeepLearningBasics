import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset source: www.kaggle.com/primaryobjects/voicegender
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
        # Yes, the label matrix it's redundant in this example for binary classication
        # But learning to do it this way opens the door to Multiclass classification
        self.trainset_labels = pd.DataFrame({
            'y1' : self.trainset["label"].as_matrix(), 
            'y2' : (self.trainset["label"] == 0).astype(int).as_matrix()
        }).as_matrix()
        self.testset_labels = pd.DataFrame({
            'y1' : self.testset["label"].as_matrix(), 
            'y2' : (self.testset["label"] == 0).astype(int).as_matrix()
        }).as_matrix()
        self.trainset = self.trainset.drop("label", 1).as_matrix()
        self.testset = self.testset.drop("label", 1).as_matrix()

        print "Trainset shape", self.trainset.shape
        print "Trainset labels shape", self.trainset_labels.shape
        print "Testset shape", self.testset.shape
        print "Testset labels shape", self.testset_labels.shape
        print "\n"