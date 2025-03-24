#!/usr/bin/env python3
import numpy as np
import pandas as pd
import csv
from random import randint
import matplotlib.pyplot as plt
from typing import Tuple, List

np.random.seed(42)

##################################################################################
# Logistic Model for Generative Authorship Detection
##################################################################################


def random_classifier(size):
    random_list = []
    for i in range(size):
        random_list.append(randint(0,1))
    return np.array(random_list)

def load_feature_vectors(filename: str) -> np.array:
    """
    Load the feature vectors from the dataset in the given file and return
    them as a numpy array with shape (number-of-examples, number-of-features + 1).
    """
    reader = pd.read_csv(filename, delimiter='\t', header=0)
    reader_list = reader.values.tolist()
    num_feat = len(reader_list[0])
    features = []
    for  d in reader_list:
        templist = [1]
        for i in range(1, num_feat):
            templist.append(d[i])
        features.append(templist)
        
    return np.array(features)

def load_class_values(filename: str) -> np.array:
    """
    Load the class values for is_human (class 0 for False and class 1
    for True) from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """

    reader = pd.read_csv(filename, delimiter='\t', header=0)
    reader_list = reader.values.tolist()
    num_feat = len(reader_list[0])
    features = []
    for  d in reader_list:
        if d[1] == True:
            features.append(1)
        else:
            features.append(0)
    return np.array(features)

def misclassification_rate(cs: np.array, ys: np.array) -> float:
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """
    if len(cs) == 0:
        return float('nan')
    else:
        counter = 0
        for i in range (len(cs)):
            if ys[i] != cs[i]:
                counter += 1
        return counter/len(cs) * 100


def logistic_function(w: np.array, x: np.array) -> float:
    """
    Return the output of a logistic function with parameter vector `w` on
    example `x`.
    Hint: use np.exp(np.clip(..., -30, 30)) instead of np.exp(...) to avoid
    divisions by zero
    """

    return 1/(1 + np.exp(np.clip(-1 * np.dot(w,x), -30, 30)))



def logistic_prediction(w: np.array, x: np.array) -> float:
    """
    Making predictions based on the output of the logistic function
    """
    
    if logistic_function(w,x) < 0.5:
        return 0
    else:
        return 1



def initialize_random_weights(p: int) -> np.array:
    """
    Generate a pseudorandom weight vector of dimension p.
    """
    weight_vector = []

    for i in range(p):
        weight_vector.append(randint(-1,1))
    return np.array(weight_vector)


def logistic_loss(w: np.array, x: np.array, c: int) -> float:
    """
    Calculate the logistic loss function
    """

    y = logistic_function(w,x)

    return -c * np.log(y) - (1-c) * np.log(1-y) 


def train_logistic_regression_with_bgd(xs: np.array, cs: np.array, eta: float=1e-8, iterations: int=2000, validation_fraction: float=0) -> Tuple[np.array, float, float]:
    """
    Fit a logistic regression model using the Batch Gradient Descent algorithm and
    return the learned weights as a numpy array.

    Arguments:
    - `xs`: feature vectors in the training dataset as a two-dimensional numpy array with shape (n, p+1)
    - `cs`: class values c(x) for every element in `xs` as a one-dimensional numpy array with length n
    - `eta`: the learning rate as a float value
    - `iterations': the number of iterations to run the algorithm for
    - 'validation_fraction': fraction of xs and cs used for validation (not for training)

    Returns:
    - the learned weights as a column vector, i.e. a two-dimensional numpy array with shape (1, p)
    - logistic loss value
    - misclassification rate of predictions on training part of xs/cs
    - misclassification rate of predictions on validation part of xs/cs
    """

    weight_vector = initialize_random_weights(len(xs[0]))
    # Slicing dataset into traiing and validation set with validation index
    validation_index = round(len(xs) * (1 - validation_fraction))
    # Training set
    xtrain = xs[:validation_index]
    ctrain = cs[:validation_index]
    # Validation set
    xtest = xs[validation_index:]
    ctest = cs[validation_index:]
    # Lists to store loss, missclassification
    loss_vector = []
    missclass_train = []
    missclass_test = []

    for t in range(iterations):
        deltaw = np.zeros(len(xs[0]))
        for i in range(len(xtrain)):
            y = logistic_function(weight_vector,xtrain[i])
            small_delta = ctrain[i] - y
            deltaw = deltaw + eta * small_delta * xtrain[i]
        
        weight_vector = weight_vector + deltaw
        
        # Trainig loss calculation
        trainingloss = 0
        for i in range(len(xtrain)):
            trainingloss += logistic_loss(weight_vector, xtrain[i] , ctrain[i])
        loss_vector.append(trainingloss) 
        
        # Model prediction for training data.
        ystrain = []
        for i in range(len(xtrain)):
            ystrain.append(logistic_prediction(weight_vector, xtrain[i]))
            
        # Model prediction for validation data.
        ystest = []
        for i in range(len(xtest)):
            ystest.append(logistic_prediction(weight_vector, xtest[i]))
        
        missclass_train.append(misclassification_rate(ctrain, ystrain))
        missclass_test.append(misclassification_rate(ctest, ystest))
        
    return [weight_vector, loss_vector, missclass_train, missclass_test]


def plot_loss_and_misclassification_rates(loss: List[float],
                                          train_misclassification_rates: List[float],
                                          validation_misclassification_rates: List[float]):
    """
    Plots the normalized loss (divided by max(loss)) and both misclassification rates
    for each iteration.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Loss and Misclassification rate')
    ax1.plot(loss)
    ax1.set_ylabel('Loss')
    ax1.legend()    
    
    ax2.plot(train_misclassification_rates)
    ax2.set_ylabel('Train Missclassification')
    ax2.legend() 
    
    ax3.plot(validation_misclassification_rates)
    ax3.set_ylabel('Validation Missclassification')
    ax3.set_xlabel('Itrations')
    ax3.legend() 
    
    plt.show()

########################################################################
# Tests
import os
from pytest import approx


def test_logistic_function():
    x = np.array([1, 1, 2])
    assert logistic_function(np.array([0, 0, 0]), x) == approx(0.5)
    assert logistic_function(np.array([1e2, 1e2, 1e2]), x) == approx(1)
    assert logistic_function(np.array([-1e2, -1e2, -1e2]), x) == approx(0)
    assert logistic_function(np.array([1e2, -1e2, 0]), x) == approx(0.5)


def test_bgd():
    xs = np.array([
        [1, -1],
        [1, 2],
        [1, -2],
    ])
    cs = np.array([0, 1, 0])

    w, _, _, _ = train_logistic_regression_with_bgd(xs, cs, 0.1, 100)
    assert w @ [1, -1] < 0 and w @ [1, 2] > 0
    w, _, _, _ = train_logistic_regression_with_bgd(-xs, cs, 0.1, 100)
    assert w @ [1, -1] > 0 and w @ [1, 2] < 0



########################################################################
# Main program for running against the training dataset

#python3 programming_exercise_logistic_model.py features-train.tsv labels-train.tsv features-test.tsv predictions-test.tsv

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

    # Traing and testing features and labels    
    train_features_file_name = 'features-train.tsv'
    train_classes_file_name = 'labels-train.tsv'
    test_features_file_name = 'features-test.tsv'
    test_predictions_file_name = 'predictions-test.tsv'
    
    print("(a)")
    # Training feature vector
    xs = load_feature_vectors(train_features_file_name)
    # Testing feature vector
    xs_test = load_feature_vectors(test_features_file_name)
    # Trainig class labels
    cs = load_class_values(train_classes_file_name)
    c1 = 0 # Human class
    c2 = 0 # AI class
    for i in cs:
        if i == 1:
            c1 += 1
        else:
            c2 += 1
    print("human: " , c1 , ", AI: " , c2 , "\n")

    print("(b)")
    test_list = random_classifier(len(cs))
    
    print("Missclassification rate of the random classifier: " , misclassification_rate(cs, test_list))

    print("(c)")
    test_c_result = pytest.main(['-k', 'test_logistic_function', '--tb=short', __file__])
    if test_c_result != 0:
        sys.exit(test_c_result)
    print("Test logistic function successful")

    print("(d)")
    test_d_result = pytest.main(['-k', 'test_bgd', '--tb=short', __file__])
    if test_d_result != 0:
        sys.exit(test_d_result)
    print("Test bgd successful")
    w, loss, train_misclassification_rates, validation_misclassification_rates = train_logistic_regression_with_bgd(xs, cs, validation_fraction = 0.2)
    
    print("(e)")
    plot_loss_and_misclassification_rates(loss, train_misclassification_rates, validation_misclassification_rates)

    print("(f)")
    # Predictiong the test feature vectors (xs_test) with obtained model
    prediction_vector = []
    for x in xs_test:
        prediction_vector.append(logistic_prediction(w, x))
    # Writing predictions into a file.
    with open(test_predictions_file_name, 'w', newline='') as csvfile:
        fieldnames = ['is_human']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        
        writer.writeheader()
        for i in range(0, len(prediction_vector)):
            writer.writerow({'is_human': prediction_vector[i]})