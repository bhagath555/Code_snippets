#!/usr/bin/env python3
import numpy as np
import pandas as pd
from math import prod
from collections import defaultdict
####################################################
# Naive Bayes for Generative Authorship Detection
####################################################

def load_bow_feature_vectors(filename: str) -> np.array:
    """
    Load the Bag-of-words feature vectors from the given file and return
    them as a two-dimensional numpy array with shape (n, p), where n is the number
    of examples in the dataset and p is the number of features per example.
    """
    return np.load(filename)


def load_class_values(filename: str) -> np.array:
    """
    Load the class values from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """
    return np.ravel((pd.read_csv(filename, sep='\t', usecols=["is_human"]).to_numpy() > 0) * 1) 


def misclassification_rate(cs: np.array, ys: np.array) -> float:
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """
    if len(cs) == 0:
        return float('nan')
    else:
        hits = np.sum(cs == ys)
        return 1 - hits / len(cs)
    


def class_priors(cs: np.ndarray) -> dict:
    """Compute the prior probabilities P(C=c) for all the distinct classes c in the given dataset.

    Args:
        cs (np.ndarray): one-dimensional array of values c(x) for all examples x from the dataset D

    Returns:
        dict: a dictionary mapping each distinct class to its prior probability
    """
    classes, counts = np.unique(cs, return_counts=True)
    total_data = len(cs)
    priors = {}
    
    for i in range(len(classes)):
        priors[classes[i]] = counts[i]/total_data
    
    return priors

def extract_features(filename: str) -> np.array:
    """
    Load the TSV file from the given filename and return a numpy array with the
    features extracted from the text.
    """
    data = pd.read_csv(filename, sep='\t')


def conditional_probabilities(xs: np.ndarray, cs: np.ndarray) -> dict:
    """Compute the conditional probabilities P(B_j = x_j | C = c) for all combinations of feature B_j, feature value x_j and class c found in the given dataset.

    Args:
        xs (np.ndarray): n-by-p array with n points of p attributes each
        cs (np.ndarray): one-dimensional n-element array with values c(x)

    Returns:
        dict: nested dictionary d with d[c][B_j][x_j] = P(B_j = x_j | C=c)
    """
    # Initializing triple default dictionary
    conditionals = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    classes = np.unique(cs)
    
    # Looping through each unique class
    for c in classes:
        data = xs[cs == c] # extracting corresponding features
        # Loop over each feature type
        num_features = data.shape[1]
        data_counts = data.shape[0] # Number of feature sets.
        for f in range(num_features):
            # f_vals - feature values, f_counts - feature value counts
            f_vals, f_counts = np.unique(data[:,f], return_counts=True)
            for i in range(len(f_vals)):
                conditionals[c][f][f_vals[i]] = f_counts[i]/data_counts
            
    return conditionals


class NaiveBayesClassifier:
    
    def __init__(self):
        # Store prior probabilities
        self.prior = {}
        # Store likelihoods. 
        self.likelihood = {}
        
    def fit(self, xs: np.ndarray, cs: np.ndarray):
        """Fit a Naive Bayes model on the given dataset

        Args:
            xs (np.ndarray): n-by-p array of feature vectors
            cs (np.ndarray): n-element array of class values
        """
        self.prior = class_priors(cs)
        self.likelihood = conditional_probabilities(xs, cs)
    
    def predict(self, x: np.ndarray) -> str:
        """Generate a prediction for the data point x

        Args:
            x (np.ndarray): a p-dimensional feature vector

        Returns:
            str: the most probable class for x
        """
        ranks = {}
        
        num_features = len(x)

        for p in self.prior:
            # assign p(a_i) to rank.
            ranks[p] = self.prior[p]
            # each feature value likelihood
            for i in range(num_features):
                ranks[p] = ranks[p] * self.likelihood[p][i][x[i]]
            
        return max(ranks, key=ranks.get)
        


def train_and_predict(training_features_file_name: str, training_labels_file_name: str, 
                      validation_features_file_name: str, validation_labels_file_name: str,
                      test_features_file_name: str) -> np.ndarray:
    """Train a model on the given training dataset, and predict the class values
    for the given testing dataset. Report the misclassification rate on the training
    and validation sets.

    Return an array with the predicted class values, in the same order as the
    examples in the testing dataset.
    """
    train_feats = load_bow_feature_vectors(training_features_file_name)
    train_lables = load_class_values(training_labels_file_name)
    val_feats = load_bow_feature_vectors(val_features_file_name)
    val_lables = load_class_values(val_classes_file_name)
    test_feats = load_bow_feature_vectors(test_features_file_name)
    
    NBC = NaiveBayesClassifier()
    NBC.fit(train_feats, train_lables)
    
    # Predictions
    train_prediction = []
    val_prediction = []
    test_prediction = []
    
    for feat in train_feats:
        train_prediction.append(NBC.predict(feat))
        
    for feat in val_feats:
        val_prediction.append(NBC.predict(feat))
        
    for feat in test_feats:
        test_prediction.append(NBC.predict(feat))
        
    # Misclassification
    print('training misclassfication ', misclassification_rate(train_lables, train_prediction))
    print('validation misclassfication ', misclassification_rate(val_lables, val_prediction))
    
    return test_prediction

########################################################################
# Tests
import os
from pytest import approx

train_features_file_name = os.path.join(os.path.dirname(__file__), 'data/bow-features-train.npy')
train_classes_file_name = os.path.join(os.path.dirname(__file__), 'data/labels-train.tsv')
val_features_file_name = os.path.join(os.path.dirname(__file__), 'data/bow-features-val.npy')
val_classes_file_name = os.path.join(os.path.dirname(__file__), 'data/labels-val.tsv')
test_features_file_name = os.path.join(os.path.dirname(__file__), 'data/bow-features-test.npy')


def test_that_training_features_are_here():
    assert os.path.isfile(train_features_file_name), \
        "Please put the training dataset file next to this script!"

def test_that_training_classes_are_here():
    assert os.path.isfile(train_classes_file_name), \
        "Please put the validation dataset file next to this script!"
    
def test_that_validation_features_are_here():
    assert os.path.isfile(val_features_file_name), \
        "Please put the validation dataset file next to this script!"

def test_that_validation_classes_are_here():
    assert os.path.isfile(val_classes_file_name), \
        "Please put the validation dataset file next to this script!"

def test_that_test_features_are_here():
    assert os.path.isfile(test_features_file_name), \
        "Please put the test dataset file next to this script!"

def test_class_priors():
    cs = np.array(list('abcababa'))
    priors = class_priors(cs)
    assert priors == dict(a=0.5, b=0.375, c=0.125)

def test_conditional_probabilities():
    cs = np.array(list('aabb'))
    xs = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [2, 0, 0],
        [2, 1, 0]
    ])

    p = conditional_probabilities(xs, cs)

    assert p['a'][0][1] == 0.5
    assert p['a'][0][0] == 0.5
    assert p['b'][0][2] == 1
    assert p['a'][1][0] == 0.5
    assert p['a'][1][1] == 0.5
    assert p['b'][1][0] == 0.5
    assert p['b'][1][1] == 0.5
    assert p['a'][2][1] == 0.5
    assert p['a'][2][0] == 0.5
    assert p['b'][2][0] == 1

### example dataset from the lecture
xs_example = np.array([x.split() for x in """sunny hot high weak
sunny hot high strong
overcast hot high weak
rain mild high weak
rain cold normal weak
rain cold normal strong
overcast cold normal strong
sunny mild high weak
sunny cold normal weak
rain mild normal weak
sunny mild normal strong
overcast mild high strong
overcast hot normal weak
rain mild high strong""".split('\n')])

cs_example = np.array("no no yes yes yes no yes no yes yes yes yes yes no".split())

def test_classifier():
    clf = NaiveBayesClassifier()
    clf.fit(xs_example, cs_example)
    pred = clf.predict(np.array('sunny cold high strong'.split()))
    assert pred == 'no', 'should classify example from the lecture correctly'

def test_classifier_unknown_value():
    clf = NaiveBayesClassifier()
    clf.fit(xs_example, cs_example)
    pred = clf.predict(np.array('sunny hot dry none'.split()))
    assert pred == 'no', 'should handle unknown feature values'


########################################################################
# Main program for running against the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys
    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Running train_and_predict.")
    preds = train_and_predict(train_features_file_name, train_classes_file_name,
                              val_features_file_name, val_classes_file_name,
                              test_features_file_name)
    if preds is not None:
        print("Saving predictions.")
        pd.DataFrame(preds).to_csv(f"naive-bayes-predictions-test.tsv", header=False, index=False, sep='\t')